"""
train.py — Training loop for PhishingTransformer.

Features:
  - Mixed precision (AMP) with GradScaler for RTX 4050 6GB VRAM
  - Gradient accumulation (effective batch = 16 x 4 = 64)
  - AdamW with linear warmup + cosine decay LR schedule
  - Early stopping on validation macro-F1
  - Per-epoch metrics: loss, accuracy, per-class F1
  - Checkpoint saving (best model only)
  - Ablation mode support (flip config flags, rerun)

Usage:
  python train.py                          # full hybrid training
  python train.py --no-features            # CLS-only ablation
  python train.py --no-perplexity          # no perplexity feature
  python train.py --epochs 10              # override epoch count
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import argparse
import json

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

from config import Config
from dataset import build_dataloaders
from features import FeatureExtractor
from Model.model import PhishingTransformer
from tokenizer_setup import PhishingTokenizer

cfg = Config()


# ─────────────────────────────────────────────
#  LR SCHEDULER
# ─────────────────────────────────────────────

def build_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_metrics(all_preds, all_labels, label_names):
    accuracy  = accuracy_score(all_labels, all_preds)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None,    zero_division=0)

    metrics = {"accuracy": accuracy, "macro_f1": macro_f1}
    for i, name in enumerate(label_names):
        metrics[f"f1_{name}"] = per_class[i] if i < len(per_class) else 0.0
    return metrics


# ─────────────────────────────────────────────
#  EVALUATION LOOP
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloader, extractor, device, config, split="val"):
    model.eval()

    all_preds   = []
    all_labels  = []
    total_loss  = 0.0
    num_batches = 0
    label_names = config.labels.labels

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        features = None
        if config.training.use_handcrafted_features:
            features = extractor.extract_batch(batch["raw"], device=device)

        with autocast("cuda", enabled=config.training.use_amp):
            output = model(input_ids, attention_mask, token_type_ids, features, labels)

        total_loss  += output.loss.item()
        num_batches += 1
        preds = output.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    metrics  = compute_metrics(all_preds, all_labels, label_names)
    metrics["loss"] = avg_loss
    return metrics


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train(config: Config = cfg):

    # ── Setup ─────────────────────────────────────────────────────
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  PHISHING TRANSFORMER — TRAINING")
    print(f"{'='*55}")
    print(f"  Device     : {device}")
    if device.type == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  AMP        : {config.training.use_amp}")
    print(f"  Features   : {config.training.use_handcrafted_features}")
    print(f"  Perplexity : {config.training.use_perplexity_feature}")

    if not config.training.use_handcrafted_features:
        run_name = "no_features"
    elif not config.training.use_perplexity_feature:
        run_name = "no_perplexity"
    else:
        run_name = "full_model"

    print(f"  Run name   : {run_name}")

    # ── Data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading datasets...")
    tokenizer = PhishingTokenizer()
    train_dl, val_dl, test_dl, class_weights = build_dataloaders(
        config=config, tokenizer=tokenizer, device=device,
    )

    # ── Feature extractor ─────────────────────────────────────────
    print(f"\n[2/5] Initialising feature extractor...")
    extractor = FeatureExtractor(config=config)
    if config.training.use_perplexity_feature:
        extractor.perplexity_scorer._load()

    # ── Model ─────────────────────────────────────────────────────
    print(f"\n[3/5] Building model...")
    model = PhishingTransformer(
        config        = config,
        class_weights = class_weights.to(device) if config.training.use_class_weights else None,
    ).to(device)

    counts = model.count_parameters()
    print(f"  Total parameters : {counts['total']:,}  ({counts['total_M']}M)")

    # ── Optimiser & Scheduler ─────────────────────────────────────
    print(f"\n[4/5] Setting up optimiser...")

    no_decay = ["bias", "LayerNorm.weight", "layer_norm", "norm"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr    = config.training.learning_rate,
        betas = (config.training.adam_beta1, config.training.adam_beta2),
        eps   = config.training.adam_epsilon,
    )

    steps_per_epoch = len(train_dl) // config.training.grad_accumulation_steps
    total_steps     = steps_per_epoch * config.training.num_epochs
    warmup_steps    = int(total_steps * config.training.warmup_ratio)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler("cuda", enabled=config.training.use_amp)

    print(f"  Steps per epoch  : {steps_per_epoch}")
    print(f"  Total steps      : {total_steps}")
    print(f"  Warmup steps     : {warmup_steps}")
    print(f"  Peak LR          : {config.training.learning_rate}")

    # ── Dirs ──────────────────────────────────────────────────────
    os.makedirs(config.paths.log_dir,        exist_ok=True)
    os.makedirs(config.paths.results_dir,    exist_ok=True)
    os.makedirs(config.paths.checkpoint_dir, exist_ok=True)

    # ── Training ──────────────────────────────────────────────────
    print(f"\n[5/5] Training for {config.training.num_epochs} epochs...")
    print(f"      Early stopping patience: {config.training.early_stopping_patience}\n")

    best_val_f1      = 0.0
    patience_counter = 0
    global_step      = 0
    history          = []
    label_names      = config.labels.labels
    ckpt_path        = os.path.join(config.paths.checkpoint_dir, f"best_model_{run_name}.pt")

    for epoch in range(1, config.training.num_epochs + 1):
        model.train()
        epoch_loss    = 0.0
        epoch_preds   = []
        epoch_labels  = []
        epoch_start   = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_dl):

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            features = None
            if config.training.use_handcrafted_features:
                features = extractor.extract_batch(batch["raw"], device=device)

            with autocast("cuda", enabled=config.training.use_amp):
                output = model(input_ids, attention_mask, token_type_ids, features, labels)
                loss   = output.loss / config.training.grad_accumulation_steps

            scaler.scale(loss).backward()

            epoch_loss   += output.loss.item()
            epoch_preds.extend(output.logits.argmax(dim=-1).cpu().tolist())
            epoch_labels.extend(labels.cpu().tolist())

            if (batch_idx + 1) % config.training.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # ── End of epoch ──────────────────────────────────────────
        epoch_time    = time.time() - epoch_start
        avg_loss      = epoch_loss / len(train_dl)
        train_metrics = compute_metrics(epoch_preds, epoch_labels, label_names)
        val_metrics   = evaluate(model, val_dl, extractor, device, config, "val")

        print(f"Epoch {epoch:>2}/{config.training.num_epochs}  [{epoch_time:.0f}s]")
        print(f"  Train  loss={avg_loss:.4f}  acc={train_metrics['accuracy']:.4f}  F1={train_metrics['macro_f1']:.4f}")
        print(f"  Val    loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}  F1={val_metrics['macro_f1']:.4f}")
        print(f"  Per-class val F1:")
        for name in label_names:
            print(f"    {name:<20}: {val_metrics[f'f1_{name}']:.4f}")

        history.append({
            "epoch":      epoch,
            "train_loss": avg_loss,
            "train_f1":   train_metrics["macro_f1"],
            "val_loss":   val_metrics["loss"],
            "val_f1":     val_metrics["macro_f1"],
            "val_acc":    val_metrics["accuracy"],
        })

        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            model.save(ckpt_path)
            print(f"  ✓ New best val F1: {best_val_f1:.4f} — checkpoint saved → {ckpt_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.training.early_stopping_patience})")
            if patience_counter >= config.training.early_stopping_patience:
                print(f"\n  Early stopping triggered at epoch {epoch}.")
                break

        print()

    # ── Final evaluation on test set ──────────────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL EVALUATION ON TEST SET")
    print(f"{'='*55}")

    best_model = PhishingTransformer.load(
        ckpt_path,
        class_weights=class_weights.to(device) if config.training.use_class_weights else None,
        config=config,
    ).to(device)

    test_metrics = evaluate(best_model, test_dl, extractor, device, config, "test")

    print(f"\n  Test loss     : {test_metrics['loss']:.4f}")
    print(f"  Test accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Test macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"\n  Per-class F1:")
    for name in label_names:
        print(f"    {name:<20}: {test_metrics[f'f1_{name}']:.4f}")

    results = {
        "run_name":     run_name,
        "best_val_f1":  best_val_f1,
        "test_metrics": test_metrics,
        "history":      history,
        "config": {
            "use_handcrafted_features": config.training.use_handcrafted_features,
            "use_perplexity_feature":   config.training.use_perplexity_feature,
            "hidden_size":              config.model.hidden_size,
            "num_encoder_layers":       config.model.num_encoder_layers,
            "num_attention_heads":      config.model.num_attention_heads,
            "learning_rate":            config.training.learning_rate,
            "total_params":             counts["total"],
        }
    }

    results_path = os.path.join(config.paths.results_dir, f"training_results_{run_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved → {results_path}")
    print(f"  Best val F1   : {best_val_f1:.4f}")
    print(f"  Best model    → {ckpt_path}")
    print(f"\n{'='*55}")
    print(f"  Training complete. Next step: python evaluate.py")
    print(f"{'='*55}\n")

    return results


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train PhishingTransformer")
    parser.add_argument("--no-features",   action="store_true")
    parser.add_argument("--no-perplexity", action="store_true")
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=None)
    parser.add_argument("--batch-size",    type=int,   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    run_cfg = Config()

    if args.no_features:
        run_cfg.training.use_handcrafted_features = False
        run_cfg.training.use_perplexity_feature   = False
        print("  [Ablation] Handcrafted features DISABLED")

    if args.no_perplexity:
        run_cfg.training.use_perplexity_feature = False
        print("  [Ablation] Perplexity feature DISABLED")

    if args.epochs:
        run_cfg.training.num_epochs = args.epochs

    if args.lr:
        run_cfg.training.learning_rate = args.lr

    if args.batch_size:
        run_cfg.training.batch_size = args.batch_size

    train(run_cfg)
