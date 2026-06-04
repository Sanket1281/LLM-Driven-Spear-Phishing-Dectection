"""
model/model.py — Full PhishingTransformer assembly.

Wires together:
  PhishingEncoder    (embeddings + 6× EncoderLayer)
  ClassifierHead     (CLS proj + feature proj + fusion → 3 logits)

This is the single object you import everywhere else:
  train.py, evaluate.py, inference scripts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL FORWARD PASS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  input_ids      (B, L)  ─┐
  attention_mask (B, L)   ├─→ PhishingEncoder ─→ cls_output (B, H)
  token_type_ids (B, L)  ─┘                          │
                                                       ├─→ ClassifierHead ─→ logits (B, 3)
  features       (B, F)  ────────────────────────────┘

  Output: ModelOutput(logits, loss, cls_output, last_hidden_state)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SAVE / LOAD
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  model.save(path)         — saves state_dict + config
  PhishingTransformer.load(path) — restores full model

Usage:
  from model.model import PhishingTransformer
  model = PhishingTransformer()
  output = model(input_ids, attention_mask, token_type_ids, features, labels)
  loss, logits = output.loss, output.logits
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from config import Config
from .encoder import PhishingEncoder
from .classifier import ClassifierHead

cfg = Config()


# ─────────────────────────────────────────────
#  OUTPUT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class ModelOutput:
    """
    Structured output from PhishingTransformer.forward().

    Fields:
        logits            : (B, 3)    — raw class scores (use for loss + argmax)
        loss              : scalar    — CrossEntropyLoss (None if labels not provided)
        cls_output        : (B, H)    — [CLS] embedding before classifier
        last_hidden_state : (B, L, H) — all token representations
        probs             : (B, 3)    — softmax probabilities
    """
    logits:             torch.Tensor
    loss:               Optional[torch.Tensor]
    cls_output:         torch.Tensor
    last_hidden_state:  torch.Tensor
    probs:              torch.Tensor


# ─────────────────────────────────────────────
#  FULL MODEL
# ─────────────────────────────────────────────

class PhishingTransformer(nn.Module):
    """
    DeBERTa-inspired transformer for 3-class phishing email detection.

    Classes:
        0 — benign
        1 — phishing        (traditional)
        2 — spear-phishing  (LLM-generated)

    Args:
        config       : Config dataclass (defaults to global cfg)
        class_weights: Optional (3,) tensor for weighted CrossEntropyLoss.
                       Pass the output of compute_class_weights() from dataset.py.
    """

    def __init__(
        self,
        config:        Config              = cfg,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.config = config

        # ── Sub-modules ───────────────────────────────────────────
        self.encoder    = PhishingEncoder(config)
        self.classifier = ClassifierHead(config)

        self.criterion = nn.CrossEntropyLoss(
            weight     = class_weights,
            label_smoothing = 0.1,   # prevents overconfident predictions
                                     # helps generalisation on small dataset
        )

    def forward(
        self,
        input_ids:      torch.Tensor,            # (B, L)
        attention_mask: torch.Tensor,            # (B, L)
        token_type_ids: torch.Tensor,            # (B, L)
        features:       Optional[torch.Tensor],  # (B, F) 
        labels:         Optional[torch.Tensor] = None,  # (B,)
        return_weights: bool = False,
    ) -> ModelOutput:

        # ── 1. Encode ─────────────────────────────────────────────
        last_hidden_state, cls_output, _ = self.encoder(
            input_ids,
            attention_mask,
            token_type_ids,
            return_weights=return_weights,
        )
        # last_hidden_state : (B, L, H)
        # cls_output        : (B, H)

        # ── 2. Classify ───────────────────────────────────────────
        logits = self.classifier(cls_output, features)   # (B, 3)

        # ── 3. Loss ─────────────────────
        loss = None
        if labels is not None:
            if self.criterion.weight is not None:
                self.criterion.weight = self.criterion.weight.to(logits.device)
            loss = self.criterion(logits, labels)

        # ── 4. Probabilities ──────────────────────────────────────
        probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)

        return ModelOutput(
            logits            = logits,
            loss              = loss,
            cls_output        = cls_output,
            last_hidden_state = last_hidden_state,
            probs             = probs,
        )

    # ── Convenience methods ───────────────────────────────────────

    def predict(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        features:       Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Inference-mode forward. Returns predictions with class names.
        No gradient computation.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(
                input_ids, attention_mask, token_type_ids, features
            )
        preds      = output.logits.argmax(dim=-1)
        pred_names = [self.config.labels.id2label[p.item()] for p in preds]
        confs      = output.probs.max(dim=-1).values

        return {
            "predictions":  pred_names,
            "class_ids":    preds.tolist(),
            "confidences":  confs.tolist(),
            "probs":        output.probs.tolist(),
        }

    def count_parameters(self) -> dict:
        """Return parameter counts broken down by component."""
        enc_params  = sum(p.numel() for p in self.encoder.parameters())
        cls_params  = sum(p.numel() for p in self.classifier.parameters())
        total       = enc_params + cls_params
        return {
            "encoder":    enc_params,
            "classifier": cls_params,
            "total":      total,
            "total_M":    round(total / 1e6, 2),
        }

    def save(self, path: str):
        """Save model state dict to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
        }, path)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str, class_weights=None, config=None) -> "PhishingTransformer":
        """Load model from saved checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        use_config = config if config is not None else cfg
        model = cls(config=use_config, class_weights=class_weights)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Model loaded ← {path}")
        return model


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  FULL MODEL SMOKE TEST")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    B, L   = 4, 128
    H      = cfg.model.hidden_size
    F_dim  = cfg.model.num_handcrafted_features
    n_cls  = cfg.model.num_labels

    input_ids      = torch.randint(0, cfg.tokenizer.vocab_size, (B, L), device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    attention_mask[:, -20:] = 0
    token_type_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    token_type_ids[:, 64:] = 1
    features       = torch.randn(B, F_dim, device=device)
    labels         = torch.randint(0, n_cls, (B,), device=device)

    class_weights  = torch.tensor([1.13, 0.95, 0.94], device=device)

    model = PhishingTransformer(class_weights=class_weights).to(device)

    print(f"\n── Training forward pass ────────────────────────")
    model.train()
    output = model(input_ids, attention_mask, token_type_ids, features, labels)

    print(f"  logits shape            : {list(output.logits.shape)}  (expected [{B}, {n_cls}])")
    print(f"  loss                    : {output.loss.item():.4f}  (should be ~1.1 for random)")
    print(f"  cls_output shape        : {list(output.cls_output.shape)}")
    print(f"  last_hidden_state shape : {list(output.last_hidden_state.shape)}")
    print(f"  probs shape             : {list(output.probs.shape)}")

    assert output.logits.shape            == (B, n_cls),  "Wrong logits shape"
    assert output.cls_output.shape        == (B, H),      "Wrong CLS shape"
    assert output.last_hidden_state.shape == (B, L, H),   "Wrong hidden shape"
    assert output.loss is not None,                        "Loss should not be None"
    assert not torch.isnan(output.loss),                   "NaN loss!"
    print(f"  All shape checks  : ✓")
    print(f"  Loss is valid     : ✓")

    print(f"\n── Gradient flow check ──────────────────────────")
    output.loss.backward()

    enc_grad  = model.encoder.layers[0].attention.q_proj.weight.grad
    cls_grad  = model.classifier.fusion[-1].weight.grad
    emb_grad  = model.encoder.embeddings.token_embeddings.weight.grad

    assert enc_grad  is not None and enc_grad.abs().sum()  > 0, "No grad in encoder attention"
    assert cls_grad  is not None and cls_grad.abs().sum()  > 0, "No grad in classifier"
    assert emb_grad  is not None and emb_grad.abs().sum()  > 0, "No grad in embeddings"
    print(f"  Encoder attention grad norm : {enc_grad.norm().item():.4f}")
    print(f"  Classifier output grad norm : {cls_grad.norm().item():.4f}")
    print(f"  Embedding grad norm         : {emb_grad.norm().item():.4f}")
    print(f"  Gradients flow end-to-end   : ✓")

    print(f"\n── Inference forward pass ───────────────────────")
    result = model.predict(input_ids, attention_mask, token_type_ids, features)
    print(f"  Predictions  : {result['predictions']}")
    print(f"  Confidences  : {[f'{c:.3f}' for c in result['confidences']]}")
    assert len(result["predictions"]) == B
    assert all(p in cfg.labels.labels for p in result["predictions"])
    print(f"  Predict API  : ✓")

    print(f"\n── Save / Load check ────────────────────────────")
    save_path = "checkpoints/smoke_test_model.pt"
    model.save(save_path)

    loaded = PhishingTransformer.load(save_path, class_weights=class_weights)
    loaded = loaded.to(device)
    loaded.eval()

    with torch.no_grad():
        out_orig   = model(input_ids, attention_mask, token_type_ids, features)
        out_loaded = loaded(input_ids, attention_mask, token_type_ids, features)

    assert torch.allclose(out_orig.logits, out_loaded.logits, atol=1e-4), \
        "Loaded model produces different outputs!"
    print(f"  Saved and reloaded outputs match : ✓")
    os.remove(save_path)

    print(f"\n── Full model parameter count ───────────────────")
    counts = model.count_parameters()
    print(f"  Encoder params    : {counts['encoder']:>12,}")
    print(f"  Classifier params : {counts['classifier']:>12,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total params      : {counts['total']:>12,}  ({counts['total_M']}M)")

    # Estimated VRAM during training (rough)
    param_mb   = counts["total"] * 4 / 1024**2
    grad_mb    = param_mb                   
    adam_mb    = param_mb * 2              
    activation_mb = B * L * H * 4 * cfg.model.num_encoder_layers / 1024**2
    total_mb   = param_mb + grad_mb + adam_mb + activation_mb
    print(f"\n── Estimated VRAM usage (float32) ───────────────")
    print(f"  Parameters  : {param_mb:.0f} MB")
    print(f"  Gradients   : {grad_mb:.0f} MB")
    print(f"  Adam state  : {adam_mb:.0f} MB")
    print(f"  Activations : {activation_mb:.1f} MB  (batch={B}, seq={L})")
    print(f"  Estimated   : ~{total_mb:.0f} MB total")
    print(f"  With AMP (float16) : ~{total_mb/1.6:.0f} MB  ← well within 6GB")

    print(f"\n  ✅ Full model smoke test passed.")
    print("=" * 55)
    print("\n  Next step: build features.py")


if __name__ == "__main__":
    smoke_test()
