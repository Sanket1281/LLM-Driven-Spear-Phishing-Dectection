"""
model/classifier.py — Classification Head with Handcrafted Feature Fusion.

Takes two inputs:
  1. cls_output  : (B, H)    — [CLS] token from encoder (semantic understanding)
  2. features    : (B, F)    — handcrafted features from features.py (F=26)

Fuses them and produces a 3-class probability distribution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cls_output (B, 256)          features (B, 26)
       │                              │
  Linear(256→128)              Linear(26→64)
  + LayerNorm + GELU           + LayerNorm + GELU
       │                              │
       └──────── Concat ─────────────┘
                   │
              (B, 192)           ← 128 + 64
                   │
           Linear(192→128)
           + LayerNorm + GELU
           + Dropout(0.3)
                   │
           Linear(128→3)         ← 3 classes
                   │
              log_softmax        ← for NLLLoss, or raw logits for CrossEntropyLoss

Why two separate projection branches before fusion?
  The CLS embedding and handcrafted features live in very different
  spaces — one is a 256-dim dense semantic vector, the other is 26
  structured numerical features (URL count, perplexity score, etc.).
  Projecting each into a common 128/64-dim space before concatenating
  lets the network learn how to align them before mixing.
  Direct concatenation without projection tends to let one modality
  dominate (usually the larger CLS vector).

Why Dropout(0.3) only on the final layer?
  Higher dropout on the classification head than the encoder (0.1)
  prevents the head from memorising training examples while the
  encoder is still learning general representations.

Ablation support:
  If use_handcrafted_features=False, the feature branch is bypassed
  and the classifier runs on CLS output only. This lets you run
  BERT-only ablation by flipping one config flag.

Usage:
  from model.classifier import ClassifierHead
  head = ClassifierHead(config)
  logits = head(cls_output, features)        # (B, 3)
  logits = head(cls_output, features=None)   # ablation mode
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import Config

cfg = Config()


# ─────────────────────────────────────────────
#  CLASSIFIER HEAD
# ─────────────────────────────────────────────

class ClassifierHead(nn.Module):
    """
    Two-branch fusion classifier:
      Branch A: CLS embedding  → project → 128-dim
      Branch B: Handcrafted features → project → 64-dim
      Fusion:   concat → Linear → GELU → Dropout → Linear → logits
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        m = config.model
        self.use_handcrafted = config.training.use_handcrafted_features

        # ── Branch A: CLS embedding projection ───────────────────
        self.cls_proj = nn.Sequential(
            nn.Linear(m.hidden_size, m.classifier_hidden),   # 256 → 128
            nn.LayerNorm(m.classifier_hidden, eps=m.layer_norm_eps),
            nn.GELU(),
        )

        # ── Branch B: Handcrafted feature projection ──────────────
        if self.use_handcrafted:
            feat_hidden = m.fusion_hidden // 2               # 64
            self.feat_proj = nn.Sequential(
                nn.Linear(m.num_handcrafted_features, feat_hidden),  # 26 → 64
                nn.LayerNorm(feat_hidden, eps=m.layer_norm_eps),
                nn.GELU(),
            )
            fusion_input_dim = m.classifier_hidden + feat_hidden     # 128 + 64 = 192
        else:
            self.feat_proj   = None
            fusion_input_dim = m.classifier_hidden                   # 128 only

        # ── Fusion layers ─────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, m.fusion_hidden),    # 192 → 128
            nn.LayerNorm(m.fusion_hidden, eps=m.layer_norm_eps),
            nn.GELU(),
            nn.Dropout(m.classifier_dropout),                # 0.3
            nn.Linear(m.fusion_hidden, m.num_labels),        # 128 → 3
        )

        self._init_weights(config)

    def _init_weights(self, config: Config):
        std = config.model.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        cls_output: torch.Tensor,             # (B, H=256)
        features:   Optional[torch.Tensor],   # (B, F=26) or None
    ) -> torch.Tensor:                        # (B, 3) logits
        """
        Forward pass.

        Returns raw logits (not softmaxed) — use with
        nn.CrossEntropyLoss which applies log-softmax internally.

        For inference probabilities: F.softmax(logits, dim=-1)
        """

        # ── Branch A ──────────────────────────────────────────────
        cls_repr = self.cls_proj(cls_output)    # (B, 128)

        # ── Branch B (optional) ──────
        if self.use_handcrafted and features is not None and self.feat_proj is not None:
            feat_repr = self.feat_proj(features.float())   # (B, 64)
            fused     = torch.cat([cls_repr, feat_repr], dim=-1)  # (B, 192)
        else:
            fused = cls_repr                               # (B, 128) ablation mode

        logits = self.fusion(fused)             # (B, 3)

        return logits


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  CLASSIFIER SMOKE TEST")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    B = 16
    H = cfg.model.hidden_size             # 256
    F_dim = cfg.model.num_handcrafted_features  # 26
    num_labels = cfg.model.num_labels     # 3

    # Simulate encoder output and handcrafted features
    cls_output = torch.randn(B, H, device=device)
    features   = torch.randn(B, F_dim, device=device)

    # ── Mode 1: Full hybrid (CLS + features) ─────────────────────
    print(f"\n── Mode 1: Hybrid (CLS + handcrafted features) ──")
    head = ClassifierHead().to(device)

    with torch.no_grad():
        logits = head(cls_output, features)

    print(f"  cls_output shape : {list(cls_output.shape)}")
    print(f"  features shape   : {list(features.shape)}")
    print(f"  logits shape     : {list(logits.shape)}  (expected [{B}, {num_labels}])")

    assert logits.shape == (B, num_labels), f"Wrong logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(),   "NaN in logits!"
    assert not torch.isinf(logits).any(),   "Inf in logits!"
    print(f"  Shape check  : ✓")
    print(f"  No NaN/Inf   : ✓")

    # Check softmax probabilities sum to 1
    probs = F.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert (prob_sums - 1.0).abs().max().item() < 1e-5, "Probs don't sum to 1"
    print(f"  Prob sums    : {prob_sums.min().item():.6f} / {prob_sums.max().item():.6f}  ✓")

    # Sample predictions
    preds = logits.argmax(dim=-1)
    pred_names = [cfg.labels.id2label[p.item()] for p in preds]
    print(f"  Sample preds : {pred_names[:4]}")

    # ── Mode 2: Ablation (CLS only, no features) ─────────────────
    print(f"\n── Mode 2: Ablation (CLS only, features=None) ───")

    # Temporarily disable handcrafted features
    head_ablation = ClassifierHead.__new__(ClassifierHead)
    cfg.training.use_handcrafted_features = False
    head_ablation.__init__(cfg)
    head_ablation = head_ablation.to(device)
    cfg.training.use_handcrafted_features = True   # restore

    with torch.no_grad():
        logits_abl = head_ablation(cls_output, features=None)

    print(f"  logits shape (ablation) : {list(logits_abl.shape)}  (expected [{B}, {num_labels}])")
    assert logits_abl.shape == (B, num_labels)
    probs_abl = F.softmax(logits_abl, dim=-1).sum(dim=-1)
    assert (probs_abl - 1.0).abs().max().item() < 1e-5
    print(f"  Ablation mode works     : ✓")

    # ── Loss computation check ────────────────────────────────────
    print(f"\n── Loss computation check ───────────────────────")
    labels      = torch.randint(0, num_labels, (B,), device=device)
    criterion   = nn.CrossEntropyLoss()
    loss        = criterion(logits, labels)

    print(f"  Random label loss : {loss.item():.4f}  (expected ~{torch.log(torch.tensor(float(num_labels))):.4f} for random)")
    assert not torch.isnan(loss),  "NaN loss!"
    assert loss.item() > 0,        "Loss should be positive"
    print(f"  Loss is valid     : ✓")

    # ── Parameter count ───────────────────────────────────────────
    print(f"\n── Parameter count ──────────────────────────────")
    total     = sum(p.numel() for p in head.parameters())
    cls_p     = sum(p.numel() for p in head.cls_proj.parameters())
    feat_p    = sum(p.numel() for p in head.feat_proj.parameters()) if head.feat_proj else 0
    fusion_p  = sum(p.numel() for p in head.fusion.parameters())
    print(f"  CLS projection branch   : {cls_p:>8,}")
    print(f"  Feature projection branch: {feat_p:>7,}")
    print(f"  Fusion layers           : {fusion_p:>8,}")
    print(f"  Total classifier params : {total:>8,}")

    print(f"\n  ✅ All classifier assertions passed.")
    print("=" * 55)
    print("\n  Next step: build model/model.py")


if __name__ == "__main__":
    smoke_test()
