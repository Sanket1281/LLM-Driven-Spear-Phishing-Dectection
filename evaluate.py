"""
evaluate.py — Full evaluation suite for PhishingTransformer.

Produces:
  - Per-class precision, recall, F1, support
  - Confusion matrix heatmap (saved as PNG)
  - ROC curves one-vs-rest (saved as PNG)
  - Per-class confidence distribution plot (saved as PNG)
  - Full classification report printed to terminal
  - All metrics saved to results/evaluation_report.json

Usage:
  python evaluate.py                        # evaluate best full model
  python evaluate.py --checkpoint checkpoints/best_model_no_features.pt
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.amp import autocast

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from dataset import build_dataloaders
from features import FeatureExtractor
from Model.model import PhishingTransformer
from tokenizer_setup import PhishingTokenizer

cfg = Config()


# ─────────────────────────────────────────────
#  INFERENCE — collect all predictions
# ─────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, dataloader, extractor, device, config):
    model.eval()

    all_preds   = []
    all_labels  = []
    all_probs   = []
    all_confs   = []

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        features = None
        if config.training.use_handcrafted_features:
            features = extractor.extract_batch(batch["raw"], device=device)

        with autocast("cuda", enabled=config.training.use_amp):
            output = model(input_ids, attention_mask, token_type_ids, features)

        probs = F.softmax(output.logits.float(), dim=-1)
        preds = probs.argmax(dim=-1)
        confs = probs.max(dim=-1).values

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_confs.extend(confs.cpu().tolist())

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_confs),
    )


# ─────────────────────────────────────────────
#  PLOT — Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, label_names, save_path):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[0], linewidths=0.5,
    )
    axes[0].set_title("Confusion Matrix (counts)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)

    # Normalised
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
    )
    axes[1].set_title("Confusion Matrix (normalised)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Label", fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")
    return cm.tolist()


# ─────────────────────────────────────────────
#  PLOT — ROC Curves
# ─────────────────────────────────────────────

def plot_roc_curves(labels, probs, label_names, save_path):
    colors = ["#2196F3", "#F44336", "#4CAF50"]
    fig, ax = plt.subplots(figsize=(8, 6))

    auc_scores = {}
    for i, (name, color) in enumerate(zip(label_names, colors)):
        binary_labels = (labels == i).astype(int)
        class_probs   = probs[:, i]

        fpr, tpr, _ = roc_curve(binary_labels, class_probs)
        auc         = roc_auc_score(binary_labels, class_probs)
        auc_scores[name] = auc

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — One-vs-Rest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curves saved       → {save_path}")
    return auc_scores


# ─────────────────────────────────────────────
#  PLOT — Confidence Distribution
# ─────────────────────────────────────────────

def plot_confidence_distribution(labels, confs, preds, label_names, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for i, (name, color) in enumerate(zip(label_names, colors)):
        mask        = labels == i
        correct     = confs[mask & (preds == labels)]
        incorrect   = confs[mask & (preds != labels)]

        ax = axes[i]
        if len(correct) > 0:
            ax.hist(correct,   bins=20, alpha=0.7, color=color,
                    label=f"Correct ({len(correct)})",   density=True)
        if len(incorrect) > 0:
            ax.hist(incorrect, bins=20, alpha=0.7, color="gray",
                    label=f"Incorrect ({len(incorrect)})", density=True)

        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Density" if i == 0 else "", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.grid(alpha=0.3)

    plt.suptitle("Prediction Confidence Distribution by Class",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confidence plot saved  → {save_path}")


# ─────────────────────────────────────────────
#  PLOT — Training History
# ─────────────────────────────────────────────

def plot_training_history(history_path, save_path):
    if not os.path.exists(history_path):
        print(f"  No training history found at {history_path} — skipping plot")
        return

    with open(history_path) as f:
        results = json.load(f)

    history = results.get("history", [])
    if not history:
        return

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_f1   = [h["train_f1"]   for h in history]
    val_f1     = [h["val_f1"]     for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(epochs, train_loss, "b-o", markersize=4, label="Train loss")
    axes[0].plot(epochs, val_loss,   "r-o", markersize=4, label="Val loss")
    axes[0].set_title("Loss Curve", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # F1 curve
    axes[1].plot(epochs, train_f1, "b-o", markersize=4, label="Train F1")
    axes[1].plot(epochs, val_f1,   "r-o", markersize=4, label="Val F1")
    axes[1].set_title("Macro F1 Curve", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim([0, 1.05])
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved  → {save_path}")


# ─────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────

def evaluate(checkpoint_path=None, config=cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  PHISHING TRANSFORMER — EVALUATION")
    print(f"{'='*55}")
    print(f"  Device : {device}")

    # ── Determine checkpoint ──────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.paths.checkpoint_dir, "best_model_full_model.pt"
        )
    print(f"  Checkpoint : {checkpoint_path}")

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading test data...")
    tokenizer = PhishingTokenizer()
    _, _, test_dl, class_weights = build_dataloaders(
        config=config, tokenizer=tokenizer, device=device,
    )

    # ── Load features ─────────────────────────────────────────────
    print(f"\n[2/4] Loading feature extractor...")
    extractor = FeatureExtractor(config=config)
    if config.training.use_perplexity_feature:
        extractor.perplexity_scorer._load()

    # ── Load model ────────────────────────────────────────────────
    print(f"\n[3/4] Loading model...")
    model = PhishingTransformer.load(
        checkpoint_path,
        class_weights=class_weights.to(device),
    ).to(device)
    print(f"  Model loaded ✓")

    # ── Run inference ─────────────────────────────────────────────
    print(f"\n[4/4] Running inference on test set...")
    preds, labels, probs, confs = run_inference(
        model, test_dl, extractor, device, config
    )

    label_names = config.labels.labels

    # ── Classification report ─────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  CLASSIFICATION REPORT")
    print(f"{'='*55}")
    report = classification_report(
        labels, preds,
        target_names=label_names,
        digits=4,
    )
    print(report)

    # ── Per-class breakdown ───────────────────────────────────────
    report_dict = classification_report(
        labels, preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )

    # ── Plots ─────────────────────────────────────────────────────
    print(f"\n  Generating plots...")
    os.makedirs(config.paths.results_dir, exist_ok=True)

    cm = plot_confusion_matrix(
        labels, preds, label_names,
        os.path.join(config.paths.results_dir, "confusion_matrix.png"),
    )

    auc_scores = plot_roc_curves(
        labels, probs, label_names,
        os.path.join(config.paths.results_dir, "roc_curves.png"),
    )

    plot_confidence_distribution(
        labels, confs, preds, label_names,
        os.path.join(config.paths.results_dir, "confidence_distribution.png"),
    )

    # Training history plot
    history_path = os.path.join(
        config.paths.results_dir, "training_results_full_model.json"
    )
    plot_training_history(
        history_path,
        os.path.join(config.paths.results_dir, "training_curves.png"),
    )

    # ── AUC scores ───────────────────────────────────────────────
    print(f"\n  ROC-AUC Scores (one-vs-rest):")
    for name, auc in auc_scores.items():
        print(f"    {name:<20}: {auc:.4f}")

    # ── Overall accuracy ─────────────────────────────────────────
    accuracy = (preds == labels).mean()
    print(f"\n  Overall accuracy : {accuracy:.4f}")

    # ── Save full report ─────────────────────────────────────────
    full_report = {
        "checkpoint":       checkpoint_path,
        "accuracy":         float(accuracy),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "auc_scores":       auc_scores,
        "num_test_samples": int(len(labels)),
        "per_class_counts": {
            name: int((labels == i).sum())
            for i, name in enumerate(label_names)
        },
    }

    report_path = os.path.join(config.paths.results_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n  Full report saved  → {report_path}")
    print(f"\n{'='*55}")
    print(f"  Evaluation complete.")
    print(f"  Plots saved to: {config.paths.results_dir}")
    print(f"{'='*55}\n")

    return full_report


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: best_model_full_model.pt)"
    )
    args = parser.parse_args()

    # Install seaborn if missing
    try:
        import seaborn
    except ImportError:
        os.system("pip install seaborn --quiet")

    evaluate(checkpoint_path=args.checkpoint)