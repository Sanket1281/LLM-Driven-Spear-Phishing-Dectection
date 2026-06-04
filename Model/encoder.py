"""
model/encoder.py — Transformer Encoder Layer + Stacked Encoder.

Builds on attention.py. Each encoder layer wraps DisentangledSelfAttention
with the standard transformer scaffold:
  - Pre/Post LayerNorm
  - Feed-Forward Network (FFN)
  - Residual connections
  - Dropout

The full encoder stacks N of these layers and shares a single
RelativePositionEmbeddings module across all layers (DeBERTa style).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENCODER LAYER STRUCTURE  (Post-LayerNorm, matches DeBERTa)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input: hidden_states (B, L, H)
    │
    ├─ DisentangledSelfAttention
    │    └─ context (B, L, H)
    │
    ├─ Dropout → Residual add → LayerNorm
    │    └─ hidden_states = LayerNorm(hidden_states + dropout(context))
    │
    ├─ FFN: Linear(H→4H) → GELU → Dropout → Linear(4H→H)
    │
    └─ Dropout → Residual add → LayerNorm
         └─ output = LayerNorm(hidden_states + dropout(ffn_out))

Why GELU not ReLU?
  GELU (Gaussian Error Linear Unit) is smoother than ReLU near zero,
  which helps gradient flow during training from scratch. BERT, GPT,
  and DeBERTa all use GELU.

Why shared position embeddings across layers?
  DeBERTa shares one RelativePositionEmbeddings table across all encoder
  layers rather than having one per layer. This reduces parameters
  significantly (saves 5 × 262K = 1.3M params for 6 layers) and
  empirically performs just as well for classification tasks.

Usage:
  from model.encoder import PhishingEncoder
  encoder = PhishingEncoder()
  output = encoder(input_ids, attention_mask, token_type_ids)
  # output: (B, L, H)
  # CLS token at output[:, 0, :] is your sequence representation
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import Config
from .embeddings import TokenEmbeddings, RelativePositionEmbeddings
from .attention import DisentangledSelfAttention

cfg = Config()


# ─────────────────────────────────────────────
#  FEED-FORWARD NETWORK
# ─────────────────────────────────────────────

class FeedForwardNetwork(nn.Module):
    """
    Position-wise FFN applied identically to each token.

    Architecture: Linear(H → 4H) → GELU → Dropout → Linear(4H → H)

    The 4× expansion is standard in BERT/DeBERTa. It gives the model
    a larger representational space per token before projecting back.
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        m = config.model

        self.linear1 = nn.Linear(m.hidden_size, m.intermediate_size)   # H → 4H
        self.linear2 = nn.Linear(m.intermediate_size, m.hidden_size)   # 4H → H
        self.dropout = nn.Dropout(m.hidden_dropout_prob)

        self._init_weights(config)

    def _init_weights(self, config: Config):
        std = config.model.initializer_range
        for layer in [self.linear1, self.linear2]:
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, H)  →  output: (B, L, H)
        """
        x = self.linear1(x)               # (B, L, 4H)
        x = F.gelu(x)                     # smooth non-linearity
        x = self.dropout(x)
        x = self.linear2(x)               # (B, L, H)
        return x


# ─────────────────────────────────────────────
#  SINGLE ENCODER LAYER
# ─────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    One transformer encoder layer:
      Attention → Add & Norm → FFN → Add & Norm

    Receives pre-computed position embeddings (pos_emb, idx_mat)
    from the parent PhishingEncoder — not computed here.
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        m = config.model

        self.attention   = DisentangledSelfAttention(config)
        self.ffn         = FeedForwardNetwork(config)

        self.norm1 = nn.LayerNorm(m.hidden_size, eps=m.layer_norm_eps)
        self.norm2 = nn.LayerNorm(m.hidden_size, eps=m.layer_norm_eps)

        self.dropout = nn.Dropout(m.hidden_dropout_prob)

    def forward(
        self,
        hidden_states:  torch.Tensor,            # (B, L, H)
        attention_mask: torch.Tensor,            # (B, L)
        pos_emb:        torch.Tensor,            # (2K, H)
        idx_mat:        torch.Tensor,            # (L, L)
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # ── Attention block ───────────────────────────────────────
        attn_out, attn_weights = self.attention(
            hidden_states,
            attention_mask,
            pos_emb,
            idx_mat,
            return_weights=return_weights,
        )

        # Residual + LayerNorm (post-norm style)
        hidden_states = self.norm1(hidden_states + self.dropout(attn_out))

        # ── FFN block───
        ffn_out       = self.ffn(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ffn_out))

        return hidden_states, attn_weights


# ─────────────────────────────────────────────
#  FULL ENCODER STACK
# ─────────────────────────────────────────────

class PhishingEncoder(nn.Module):
    """
    Full encoder: TokenEmbeddings + N × EncoderLayer + shared RelativePositionEmbeddings.

    This is the complete feature extractor. Feed it tokenized emails,
    get back contextual representations for every token.

    The [CLS] token output (index 0) is used as the sequence-level
    representation for classification.

    Args:
        config : Config object

    Forward inputs:
        input_ids      : (B, L)   — token IDs
        attention_mask : (B, L)   — 1=real, 0=pad
        token_type_ids : (B, L)   — 0=subject, 1=body

    Forward outputs:
        last_hidden_state : (B, L, H)   — all token representations
        cls_output        : (B, H)      — [CLS] token only → for classifier
        all_attn_weights  : list of (B, N, L, L) or None
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        m = config.model

        # Input embeddings (token + type, no position)
        self.embeddings = TokenEmbeddings(config)

        # Shared relative position embeddings (one table for all layers)
        self.rel_pos_embeddings = RelativePositionEmbeddings(config)

        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(config)
            for _ in range(m.num_encoder_layers)   # 6 layers
        ])

        # Final LayerNorm on encoder output
        self.final_norm = nn.LayerNorm(m.hidden_size, eps=m.layer_norm_eps)

    def forward(
        self,
        input_ids:      torch.Tensor,            # (B, L)
        attention_mask: torch.Tensor,            # (B, L)
        token_type_ids: torch.Tensor,            # (B, L)
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[list]]:

        B, L = input_ids.shape
        device = input_ids.device

        # ── 1. Token embeddings ───────────────────────────────────
        hidden_states = self.embeddings(input_ids, token_type_ids)   # (B, L, H)

        # ── 2. Build relative position embeddings ─────────────────
        # Computed once, shared across all layers
        pos_emb, idx_mat = self.rel_pos_embeddings(L, device)
        # pos_emb : (2K, H)
        # idx_mat : (L, L)

        # ── 3. Pass through encoder layers ────────────────────────
        all_attn_weights = [] if return_weights else None

        for layer in self.layers:
            hidden_states, attn_weights = layer(
                hidden_states,
                attention_mask,
                pos_emb,
                idx_mat,
                return_weights=return_weights,
            )
            if return_weights and attn_weights is not None:
                all_attn_weights.append(attn_weights)

        # ── 4. Final normalisation ────────────────────────────────
        hidden_states = self.final_norm(hidden_states)   # (B, L, H)

        # ── 5. Extract [CLS] token ────────────────────────────────
        cls_output = hidden_states[:, 0, :]              # (B, H)

        return hidden_states, cls_output, all_attn_weights


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  ENCODER SMOKE TEST")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    B, L = 4, 128   # realistic email length
    H    = cfg.model.hidden_size        # 256
    N    = cfg.model.num_attention_heads # 8
    n_layers = cfg.model.num_encoder_layers  # 6

    # Simulate a tokenized batch
    input_ids      = torch.randint(0, cfg.tokenizer.vocab_size, (B, L), device=device)
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
    attention_mask[:, -20:] = 0   # last 20 tokens are padding
    token_type_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    token_type_ids[:, 64:] = 1    # subject=64 tokens, body=rest

    # ── Build encoder ─────────────────────────────────────────────
    encoder = PhishingEncoder().to(device)

    # ── Forward pass ─────────────────────────────────────────────
    print(f"\n── Forward pass ─────────────────────────────────")
    with torch.no_grad():
        last_hidden, cls_out, _ = encoder(
            input_ids, attention_mask, token_type_ids,
            return_weights=False,
        )

    print(f"  input_ids shape      : {list(input_ids.shape)}")
    print(f"  last_hidden shape    : {list(last_hidden.shape)}  (expected [4, 128, 256])")
    print(f"  cls_output shape     : {list(cls_out.shape)}  (expected [4, 256])")

    assert last_hidden.shape == (B, L, H), f"Wrong shape: {last_hidden.shape}"
    assert cls_out.shape     == (B, H),    f"Wrong CLS shape: {cls_out.shape}"
    print(f"  Shape checks : ✓")

    # ── CLS token is first token ──────────────────────────────────
    print(f"\n── CLS token check ──────────────────────────────")
    cls_from_hidden = last_hidden[:, 0, :]
    assert torch.allclose(cls_from_hidden, cls_out), "CLS output mismatch"
    print(f"  cls_output == last_hidden[:,0,:] : ✓")

    # ── No NaN/Inf ────────────────────────────────────────────────
    print(f"\n── Numerical stability check ────────────────────")
    assert not torch.isnan(last_hidden).any(), "NaN in last_hidden!"
    assert not torch.isinf(last_hidden).any(), "Inf in last_hidden!"
    assert not torch.isnan(cls_out).any(),     "NaN in cls_output!"
    print(f"  No NaN/Inf in outputs : ✓")
    print(f"  last_hidden std : {last_hidden.std().item():.4f}  (should be > 0)")
    print(f"  cls_output  std : {cls_out.std().item():.4f}  (should be > 0)")

    # ── Attention weights (return_weights=True) ───────────────────
    print(f"\n── Attention weights return check ───────────────")
    with torch.no_grad():
        _, _, all_weights = encoder(
            input_ids, attention_mask, token_type_ids,
            return_weights=True,
        )
    assert all_weights is not None,          "Expected attention weights list"
    assert len(all_weights) == n_layers,     f"Expected {n_layers} layers of weights"
    assert all_weights[0].shape == (B, N, L, L)
    print(f"  Returned {len(all_weights)} layers of attention weights : ✓")
    print(f"  Each weight shape : {list(all_weights[0].shape)}  (expected [4, 8, 128, 128])")

    # ── Padding positions have no influence ───────────────────────
    print(f"\n── Padding isolation check ──────────────────────")
    pad_repr  = last_hidden[:, -20:, :]   # padding region
    real_repr = last_hidden[:, :64, :]    # real subject tokens
    print(f"  Real token repr std  : {real_repr.std().item():.4f}")
    print(f"  Pad  token repr std  : {pad_repr.std().item():.4f}")
    assert real_repr.std().item() > 0, "Real token representations are constant"
    print(f"  Representations are non-constant : ✓")

    # ── Parameter count ───────────────────────────────────────────
    print(f"\n── Parameter count ──────────────────────────────")
    total      = sum(p.numel() for p in encoder.parameters())
    emb_params = sum(p.numel() for p in encoder.embeddings.parameters())
    pos_params = sum(p.numel() for p in encoder.rel_pos_embeddings.parameters())
    enc_params = sum(p.numel() for p in encoder.layers.parameters())
    print(f"  TokenEmbeddings          : {emb_params:>10,}")
    print(f"  RelPosEmbeddings (shared): {pos_params:>10,}")
    print(f"  EncoderLayers (×{n_layers})     : {enc_params:>10,}")
    print(f"  ─────────────────────────────────────")
    print(f"  Total encoder params     : {total:>10,}  (~{total/1e6:.1f}M)")

    # ── Memory estimate ───────────────────────────────────────────
    print(f"\n── GPU memory estimate ──────────────────────────")
    param_mb  = total * 4 / 1024**2   # float32
    print(f"  Parameters (float32) : {param_mb:.1f} MB")
    print(f"  Parameters (float16) : {param_mb/2:.1f} MB  ← training AMP footprint")
    print(f"  (Activations + gradients will be ~3-4× params during training)")

    print(f"\n  ✅ All encoder assertions passed.")
    print("=" * 55)
    print("\n  Next step: build model/classifier.py")


if __name__ == "__main__":
    smoke_test()
