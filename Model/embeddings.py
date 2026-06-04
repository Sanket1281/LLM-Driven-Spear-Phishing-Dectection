"""
model/embeddings.py — Token embeddings + Disentangled Relative Position Embeddings.

This is the first DeBERTa-specific component. Unlike BERT which adds absolute
position embeddings to token embeddings at the input layer, DeBERTa keeps
content and position completely separate throughout the entire encoder.

Two embedding modules live here:

  1. TokenEmbeddings
     - Token embeddings (vocab_size → hidden_size)
     - Token type embeddings (segment 0/1 → hidden_size)
     - LayerNorm + Dropout
     - NO position embedding added here (unlike BERT)

  2. RelativePositionEmbeddings
     - Learns a (2 * max_relative_positions, hidden_size) embedding table
     - Given a sequence length, returns position embeddings for all
       relative distances: for position i, key position j, the relative
       distance is clip(i - j, -K, K) where K = max_relative_positions
     - Used inside the attention mechanism (not at input)

Usage:
  from model.embeddings import TokenEmbeddings, RelativePositionEmbeddings
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from config import Config

cfg = Config()


# ─────────────────────────────────────────────
#  TOKEN EMBEDDINGS
# ─────────────────────────────────────────────

class TokenEmbeddings(nn.Module):
    """
    Converts token IDs and segment IDs into continuous vectors.

    Input:
        input_ids      : (B, L)  — token IDs from tokenizer
        token_type_ids : (B, L)  — 0 = subject segment, 1 = body segment

    Output:
        embeddings     : (B, L, hidden_size)

    Deliberately does NOT add position embeddings here.
    Position is handled separately inside each attention layer (DeBERTa style).
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        m = config.model
        t = config.tokenizer

        # Core lookup tables
        self.token_embeddings = nn.Embedding(
            num_embeddings = t.vocab_size,      # 30,522
            embedding_dim  = m.hidden_size,     # 256
            padding_idx    = t.pad_token_id,    # 0 → always zero vector
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings = 2,                 # segment 0 or 1
            embedding_dim  = m.hidden_size,
        )

        self.layer_norm = nn.LayerNorm(
            m.hidden_size,
            eps = m.layer_norm_eps,
        )

        self.dropout = nn.Dropout(m.hidden_dropout_prob)

        # Initialise weights
        self._init_weights(config)

    def _init_weights(self, config: Config):
        std = config.model.initializer_range  # 0.02
        nn.init.normal_(self.token_embeddings.weight,      mean=0.0, std=std)
        nn.init.normal_(self.token_type_embeddings.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.token_embeddings.weight[config.tokenizer.pad_token_id].fill_(0)

    def forward(
        self,
        input_ids:      torch.Tensor,   # (B, L)
        token_type_ids: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:                  # (B, L, hidden_size)

        tok_emb  = self.token_embeddings(input_ids)       # (B, L, H)
        type_emb = self.token_type_embeddings(token_type_ids)  # (B, L, H)

        embeddings = tok_emb + type_emb                   # (B, L, H)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# ─────────────────────────────────────────────
#  RELATIVE POSITION EMBEDDINGS
# ─────────────────────────────────────────────

class RelativePositionEmbeddings(nn.Module):
    """
    Disentangled relative position embedding table (DeBERTa-style).

    Stores a learned embedding for each possible relative distance
    between two tokens in a sequence, clipped to [-K, K] where
    K = max_relative_positions.

    Table size: (2 * K,  hidden_size)
    Index 0        → relative distance -K   (far left)
    Index 2*K - 1  → relative distance +K-1 (far right)

    Key design choice:
      - We learn ONE table used for both key-side (c2p) and
        query-side (p2c) attention, projected separately inside
        the attention module via linear layers.
      - This matches DeBERTa's shared position embedding approach.

    get_position_embeddings(seq_len):
      Returns position embeddings for all pairwise relative distances
      in a sequence of length seq_len.
      Output shape: (2*K, hidden_size) — the attention module indexes
      into this using the relative distance matrix.

    build_relative_distance_matrix(seq_len):
      Returns an integer matrix of shape (seq_len, seq_len) where
      entry [i, j] = clip(i - j + K, 0, 2K-1)
      This maps to an index into the position embedding table.
    """

    def __init__(self, config: Config = cfg):
        super().__init__()
        m = config.model

        self.K          = m.max_relative_positions   # 512
        self.hidden_size = m.hidden_size              # 256
        table_size      = 2 * self.K                  # 1024

        self.position_embeddings = nn.Embedding(
            num_embeddings = table_size,
            embedding_dim  = m.hidden_size,
        )

        self.layer_norm = nn.LayerNorm(m.hidden_size, eps=m.layer_norm_eps)

        nn.init.normal_(
            self.position_embeddings.weight,
            mean = 0.0,
            std  = config.model.initializer_range,
        )

    def build_relative_distance_matrix(
        self,
        seq_len: int,
        device:  torch.device,
    ) -> torch.Tensor:
        """
        Build integer index matrix for relative distances.

        For query position i and key position j:
            raw_dist  = i - j                  (range: -(L-1) to +(L-1))
            clipped   = clip(raw_dist, -K, K-1)
            idx       = clipped + K            (range: 0 to 2K-1)

        Returns: (seq_len, seq_len)  dtype=long
        """
        positions = torch.arange(seq_len, device=device)   # (L,)

        dist_matrix = positions.unsqueeze(1) - positions.unsqueeze(0)   # (L, L)

        dist_matrix = dist_matrix.clamp(-self.K, self.K - 1)

        idx_matrix  = dist_matrix + self.K   # (L, L)

        return idx_matrix.long()

    def get_position_embeddings(self) -> torch.Tensor:
        """
        Return the full position embedding table after LayerNorm.
        Shape: (2*K, hidden_size)

        The attention module will index into this using the
        relative distance matrix.
        """
        # All indices: 0, 1, ..., 2K-1
        all_indices = torch.arange(
            2 * self.K,
            device = self.position_embeddings.weight.device,
        )
        pos_emb = self.position_embeddings(all_indices)   # (2K, H)
        pos_emb = self.layer_norm(pos_emb)
        return pos_emb

    def forward(
        self,
        seq_len: int,
        device:  torch.device,
    ) -> tuple:
        """
        Convenience forward used by the attention module.

        Returns:
            pos_emb  : (2*K, hidden_size)  — full embedding table
            idx_mat  : (seq_len, seq_len)  — relative distance indices
        """
        pos_emb = self.get_position_embeddings()
        idx_mat = self.build_relative_distance_matrix(seq_len, device)
        return pos_emb, idx_mat


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  EMBEDDINGS SMOKE TEST")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── TokenEmbeddings ──────────────────────────────────────────
    print("\n── TokenEmbeddings ──────────────────────────────")
    tok_emb = TokenEmbeddings().to(device)

    B, L = 4, 512
    input_ids      = torch.randint(0, 30522, (B, L), device=device)
    token_type_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    token_type_ids[:, 64:] = 1   # simulate subject=64 tokens, rest=body

    out = tok_emb(input_ids, token_type_ids)

    print(f"  Input  shape : {list(input_ids.shape)}")
    print(f"  Output shape : {list(out.shape)}  (expected [4, 512, 256])")
    assert out.shape == (B, L, cfg.model.hidden_size), \
        f"Shape mismatch: {out.shape}"

    # Padding index should produce zero token embedding (before type + norm)
    pad_ids   = torch.zeros(1, 1, dtype=torch.long, device=device)
    pad_types = torch.zeros(1, 1, dtype=torch.long, device=device)
    pad_out   = tok_emb.token_embeddings(pad_ids)
    assert pad_out.abs().sum().item() == 0.0, "Padding embedding should be all zeros"
    print(f"  Padding idx zero check : ✓")

    # Output should not be all zeros (sanity)
    assert out.abs().sum().item() > 0
    print(f"  Non-zero output check  : ✓")

    # ── RelativePositionEmbeddings ───────────────────────────────
    print("\n── RelativePositionEmbeddings ───────────────────")
    rel_emb = RelativePositionEmbeddings().to(device)

    seq_len = 64   # use a shorter seq for clarity
    pos_emb, idx_mat = rel_emb(seq_len, device)

    print(f"  pos_emb shape  : {list(pos_emb.shape)}  (expected [1024, 256])")
    print(f"  idx_mat shape  : {list(idx_mat.shape)}  (expected [64, 64])")
    assert pos_emb.shape == (2 * cfg.model.max_relative_positions, cfg.model.hidden_size)
    assert idx_mat.shape == (seq_len, seq_len)

    # Check index range is valid [0, 2K-1]
    K = cfg.model.max_relative_positions
    assert idx_mat.min().item() >= 0,       f"Index below 0: {idx_mat.min()}"
    assert idx_mat.max().item() <= 2*K - 1, f"Index above {2*K-1}: {idx_mat.max()}"
    print(f"  Index range    : [{idx_mat.min().item()}, {idx_mat.max().item()}]  ✓")

    # Diagonal should be K (distance 0 → index K)
    diag_vals = torch.diagonal(idx_mat)
    assert (diag_vals == K).all(), f"Diagonal should all be {K}"
    print(f"  Diagonal = K={K} (distance 0) : ✓")

    # Spot check: position 0 looking at position 5 → dist = -5 → idx = K-5
    assert idx_mat[0, 5].item() == K - 5, \
        f"idx[0,5] should be {K-5}, got {idx_mat[0,5].item()}"
    print(f"  Spot check idx[0,5] = K-5 = {K-5} : ✓")

    # ── Parameter count ──────────────────────────────────────────
    print("\n── Parameter counts ─────────────────────────────")
    tok_params = sum(p.numel() for p in tok_emb.parameters())
    rel_params = sum(p.numel() for p in rel_emb.parameters())
    print(f"  TokenEmbeddings          : {tok_params:>10,} params")
    print(f"  RelativePositionEmbeddings: {rel_params:>9,} params")
    print(f"  Total embeddings          : {tok_params + rel_params:>9,} params")

    print("\n  ✅ All embedding assertions passed.")
    print("=" * 55)
    print("\n  Next step: build model/attention.py")


if __name__ == "__main__":
    smoke_test()
