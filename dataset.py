"""
dataset.py — PyTorch Dataset + DataLoader for the Phishing Transformer.
No HuggingFace datasets library — pure PyTorch + our own tokenizer.

What this file does:
  1. PhishingDataset  — reads .jsonl splits, tokenizes on-the-fly, returns tensors
  2. collate_fn       — pads a batch to the longest sequence in that batch
  3. build_dataloaders— returns train / val / test DataLoaders ready for training
  4. compute_class_weights — for weighted CrossEntropyLoss
  5. Smoke test       — loads all three splits, prints stats, checks one batch

Usage:
  python dataset.py                        # runs smoke test
  from dataset import build_dataloaders
  train_dl, val_dl, test_dl, class_weights = build_dataloaders(cfg)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config
from tokenizer_setup import PhishingTokenizer

cfg = Config()


# ─────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────

class PhishingDataset(Dataset):
    """
    Reads a .jsonl file where each line is a JSON record with fields:
        subject : str   — email subject line
        body    : str   — email body text
        label   : str   — "benign" | "phishing" | "spear-phishing"

    Returns per item:
        input_ids      : LongTensor (max_seq_len,)
        attention_mask : LongTensor (max_seq_len,)
        token_type_ids : LongTensor (max_seq_len,)
        label          : LongTensor scalar
        raw            : dict  — original record (for feature extractor later)
    """

    def __init__(
        self,
        jsonl_path:  str,
        tokenizer:   PhishingTokenizer,
        label2id:    Dict[str, int],
        max_seq_len: int  = cfg.tokenizer.max_seq_len,
        augment:     bool = False,
    ):
        self.tokenizer   = tokenizer
        self.label2id    = label2id
        self.max_seq_len = max_seq_len
        self.augment     = augment

        self.records = self._load(jsonl_path)
        print(f"  Loaded {len(self.records):,} records from {jsonl_path}")
        self._print_distribution()

    # ── Loading ───────────────────────────────────────────────────────

    def _load(self, path: str) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Validate required fields
                if "label" not in rec:
                    continue
                if rec["label"] not in self.label2id:
                    continue

                # Normalise text fields — handle missing gracefully
                rec["subject"] = str(rec.get("subject", "") or "").strip()
                rec["body"]    = str(rec.get("body",    "") or "").strip()

                records.append(rec)
        return records

    def _print_distribution(self):
        counts = Counter(r["label"] for r in self.records)
        total  = len(self.records)
        for label, count in sorted(counts.items()):
            print(f"    {label:<20}: {count:>4}  ({count/total*100:.1f}%)")

    # ── Augmentation (training only) ──────────────────────────────────

    def _augment(self, subject: str, body: str) -> Tuple[str, str]:
        """
        Light text augmentation to improve generalisation.
        Applied randomly during training only.

        Techniques used (all label-preserving):
          - Random word dropout in body (5% chance per word)
          - Randomly swap subject case (all-caps simulation)
          - Randomly truncate body to 70-100% of its length
        """
        # Word dropout in body
        if random.random() < 0.3 and body:
            words = body.split()
            words = [w for w in words if random.random() > 0.05]
            body  = " ".join(words)

        # Random body truncation
        if random.random() < 0.2 and body:
            words  = body.split()
            cutoff = random.randint(int(len(words) * 0.7), len(words))
            body   = " ".join(words[:cutoff])

        # Subject uppercasing simulation (phishing emails often shout)
        if random.random() < 0.15 and subject:
            subject = subject.upper()

        return subject, body

    # ── PyTorch Dataset interface ─────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec     = self.records[idx]
        subject = rec["subject"]
        body    = rec["body"]

        if self.augment:
            subject, body = self._augment(subject, body)

        encoded = self.tokenizer.encode(subject, body, return_tensors=True)

        return {
            "input_ids":      encoded["input_ids"],        # (max_seq_len,)
            "attention_mask": encoded["attention_mask"],   # (max_seq_len,)
            "token_type_ids": encoded["token_type_ids"],   # (max_seq_len,)
            "label":          torch.tensor(
                                  self.label2id[rec["label"]],
                                  dtype=torch.long
                              ),
            "raw": rec,   # kept for handcrafted feature extractor
        }


# ─────────────────────────────────────────────
#  COLLATE FUNCTION
# ─────────────────────────────────────────────

def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate: stacks tensors, keeps raw records as a list.

    Note: our tokenizer already pads to max_seq_len=512, so all
    tensors in a batch are the same length. The collate fn still
    handles the 'raw' field (list of dicts) which torch's default
    collate can't handle.
    """
    input_ids      = torch.stack([b["input_ids"]      for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    token_type_ids = torch.stack([b["token_type_ids"] for b in batch])
    labels         = torch.stack([b["label"]          for b in batch])
    raws           = [b["raw"] for b in batch]

    return {
        "input_ids":      input_ids,       # (B, max_seq_len)
        "attention_mask": attention_mask,  # (B, max_seq_len)
        "token_type_ids": token_type_ids,  # (B, max_seq_len)
        "labels":         labels,          # (B,)
        "raw":            raws,            # list[dict], length B
    }


# ─────────────────────────────────────────────
#  CLASS WEIGHTS
# ─────────────────────────────────────────────

def compute_class_weights(
    dataset:   PhishingDataset,
    num_labels: int = cfg.labels.num_labels,
    device:    torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted CrossEntropyLoss.
    Compensates for the mild class imbalance in our dataset.

    Formula: weight_c = total_samples / (num_classes × count_c)
    """
    counts = Counter(r["label"] for r in dataset.records)
    label2id = cfg.labels.label2id
    total    = len(dataset.records)

    weights = []
    for i in range(num_labels):
        label = cfg.labels.id2label[i]
        count = counts.get(label, 1)   # avoid div-by-zero
        w     = total / (num_labels * count)
        weights.append(w)

    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return weight_tensor


# ─────────────────────────────────────────────
#  DATALOADERS
# ─────────────────────────────────────────────

def build_dataloaders(
    config:        Config              = cfg,
    tokenizer:     Optional[PhishingTokenizer] = None,
    device:        torch.device        = torch.device("cpu"),
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train / val / test DataLoaders and compute class weights.

    Returns:
        train_loader   : DataLoader  (shuffled, augmented)
        val_loader     : DataLoader  (ordered, no augment)
        test_loader    : DataLoader  (ordered, no augment)
        class_weights  : Tensor (num_labels,) on device
    """
    if tokenizer is None:
        tokenizer = PhishingTokenizer()

    label2id = config.labels.label2id

    print("\n── Train split ──────────────────────────────")
    train_ds = PhishingDataset(
        config.paths.train_file,
        tokenizer,
        label2id,
        augment=True,
    )

    print("\n── Val split ────────────────────────────────")
    val_ds = PhishingDataset(
        config.paths.val_file,
        tokenizer,
        label2id,
        augment=False,
    )

    print("\n── Test split ───────────────────────────────")
    test_ds = PhishingDataset(
        config.paths.test_file,
        tokenizer,
        label2id,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = config.training.batch_size,
        shuffle     = True,          
        collate_fn  = collate_fn,
        num_workers = 0,            
        pin_memory  = True,          # faster CPU→GPU transfer
        drop_last   = False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = config.training.eval_batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size  = config.training.eval_batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
        pin_memory  = True,
    )

    class_weights = compute_class_weights(train_ds, device=device)

    print(f"\n── Class weights (for loss) ─────────────────")
    for i, w in enumerate(class_weights.tolist()):
        print(f"    {config.labels.id2label[i]:<20}: {w:.4f}")

    return train_loader, val_loader, test_loader, class_weights


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  DATASET SMOKE TEST")
    print("=" * 55)

    tokenizer = PhishingTokenizer()
    train_dl, val_dl, test_dl, class_weights = build_dataloaders(
        tokenizer=tokenizer
    )

    print(f"\n── DataLoader stats ─────────────────────────")
    print(f"  Train batches : {len(train_dl)}")
    print(f"  Val batches   : {len(val_dl)}")
    print(f"  Test batches  : {len(test_dl)}")

    # Inspect one batch
    print(f"\n── First training batch ─────────────────────")
    batch = next(iter(train_dl))

    print(f"  input_ids shape      : {batch['input_ids'].shape}")
    print(f"  attention_mask shape : {batch['attention_mask'].shape}")
    print(f"  token_type_ids shape : {batch['token_type_ids'].shape}")
    print(f"  labels shape         : {batch['labels'].shape}")
    print(f"  labels in batch      : {batch['labels'].tolist()}")
    print(f"  label names          : {[cfg.labels.id2label[l.item()] for l in batch['labels']]}")

    # Check dtypes
    assert batch["input_ids"].dtype      == torch.long,  "input_ids must be Long"
    assert batch["attention_mask"].dtype == torch.long,  "attention_mask must be Long"
    assert batch["labels"].dtype         == torch.long,  "labels must be Long"

    # Check shapes
    B = cfg.training.batch_size
    L = cfg.tokenizer.max_seq_len
    assert batch["input_ids"].shape      == (B, L), f"Expected ({B},{L}), got {batch['input_ids'].shape}"
    assert batch["attention_mask"].shape == (B, L)
    assert batch["token_type_ids"].shape == (B, L)

    # Check CLS token is always first
    assert (batch["input_ids"][:, 0] == cfg.tokenizer.cls_token_id).all(), \
        "First token must always be [CLS]=101"

    # Check raw records came through
    assert len(batch["raw"]) == B
    print(f"\n  Sample raw record:")
    raw = batch["raw"][0]
    print(f"    label   : {raw['label']}")
    print(f"    subject : {raw['subject'][:60]}")
    print(f"    body    : {raw['body'][:80]}...")

    pad_counts = (batch["input_ids"] == cfg.tokenizer.pad_token_id).sum(dim=1)
    print(f"\n  Pad tokens per record in batch: {pad_counts.tolist()}")

    print("\n  ✅ All assertions passed — dataset is working correctly.")
    print("=" * 55)
    print("\n  Next step: build model/embeddings.py")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    smoke_test()
