"""
config.py — Single source of truth for all hyperparameters, paths, and settings.
Phishing Transformer (DeBERTa-inspired) | Research Project
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
from dataclasses import dataclass, field
from typing import List

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
@dataclass
class PathConfig:
    root: str = os.path.abspath(os.path.dirname(__file__))

    train_file:      str = ""
    val_file:        str = ""
    test_file:       str = ""
    vocab_file:      str = ""
    tokenizer_cache: str = ""
    checkpoint_dir:  str = ""
    best_model:      str = ""
    final_model:     str = ""
    log_dir:         str = ""
    results_dir:     str = ""

    def __post_init__(self):
        self.train_file      = os.path.join(self.root, "data", "train.jsonl")
        self.val_file        = os.path.join(self.root, "data", "val.jsonl")
        self.test_file       = os.path.join(self.root, "data", "test.jsonl")
        self.vocab_file      = os.path.join(self.root, "tokenizer", "vocab.txt")
        self.tokenizer_cache = os.path.join(self.root, "tokenizer", "word2idx.json")
        self.checkpoint_dir  = os.path.join(self.root, "checkpoints")
        self.best_model      = os.path.join(self.root, "checkpoints", "best_model.pt")
        self.final_model     = os.path.join(self.root, "checkpoints", "final_model.pt")
        self.log_dir         = os.path.join(self.root, "logs")
        self.results_dir     = os.path.join(self.root, "results")


# ─────────────────────────────────────────────
#  TOKENIZER
# ─────────────────────────────────────────────
@dataclass
class TokenizerConfig:
    vocab_size:         int = 30_522      
    max_seq_len:        int = 512         
    pad_token:          str = "[PAD]"
    unk_token:          str = "[UNK]"
    cls_token:          str = "[CLS]"
    sep_token:          str = "[SEP]"
    mask_token:         str = "[MASK]"
    pad_token_id:       int = 0
    unk_token_id:       int = 100
    cls_token_id:       int = 101
    sep_token_id:       int = 102
    mask_token_id:      int = 103

    truncation_strategy: str = "body_first"   # truncate body before subject


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    # Core dimensions — sized for RTX 4050 6GB VRAM
    hidden_size:        int = 256         # BERT-base is 768; we scale down for scratch training
    num_attention_heads:int = 8           # must divide hidden_size evenly → 256 / 8 = 32 per head
    num_encoder_layers: int = 6           # BERT-base has 12; 6 is sufficient for classification
    intermediate_size:  int = 1024        # FFN hidden dim = 4× hidden_size (standard ratio)
    max_position_embeddings: int = 512    # must match tokenizer max_seq_len

    # DeBERTa-specific: relative position encoding
    max_relative_positions: int = 512     # range of relative distances tracked
    pos_att_type:       str = "c2p|p2c"  # content-to-position and position-to-content attention
                                          # full DeBERTa uses c2p|p2c|p2p; we skip p2p for efficiency

    # Regularisation
    hidden_dropout_prob:        float = 0.1
    attention_probs_dropout_prob: float = 0.1

    # Classification head
    num_labels:         int = 3           # benign | phishing | spear-phishing
    classifier_dropout: float = 0.3       # higher dropout on head than encoder
    classifier_hidden:  int = 128         # intermediate dim in 2-layer classifier head

    # Handcrafted feature branch (fed into fusion layer alongside [CLS] embedding)
    num_handcrafted_features: int = 26    # see features.py for full list
    fusion_hidden:      int = 128         # dim of fusion layer before final classifier

    # Layer normalisation epsilon
    layer_norm_eps:     float = 1e-7

    # Weight initialisation
    initializer_range:  float = 0.02      # std dev for normal init (matches BERT)

    @property
    def head_size(self) -> int:
        """Dimension per attention head."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}"
        return self.hidden_size // self.num_attention_heads

    @property
    def total_params_estimate(self) -> str:
        """Rough parameter count estimate for quick sanity check."""
        embedding   = TokenizerConfig().vocab_size * self.hidden_size
        attn_layer  = 4 * self.hidden_size * self.hidden_size   # Q, K, V, O projections
        ffn_layer   = 2 * self.hidden_size * self.intermediate_size
        per_layer   = attn_layer + ffn_layer
        total       = embedding + self.num_encoder_layers * per_layer
        return f"~{total / 1e6:.1f}M parameters"


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    batch_size:                 int = 16
    grad_accumulation_steps:    int = 4
    eval_batch_size:            int = 32  

    # Epochs
    num_epochs:                 int = 20
    early_stopping_patience:    int = 5    # stop if val F1 doesn't improve for 5 epochs

    # Optimiser — AdamW (standard for transformers)
    learning_rate:              float = 3e-4  
    weight_decay:               float = 0.01
    adam_epsilon:               float = 1e-8
    adam_beta1:                 float = 0.9
    adam_beta2:                 float = 0.999
    max_grad_norm:              float = 1.0    # gradient clipping

    # LR scheduler
    scheduler_type:             str   = "linear_warmup_cosine"
    warmup_ratio:               float = 0.1

    # Mixed precision (AMP) — critical for 6GB VRAM
    use_amp:                    bool  = True
    amp_dtype:                  str   = "float16" 
    
    use_class_weights:          bool  = True

    # Reproducibility
    seed:                       int   = 42

    # Logging & checkpointing
    log_every_n_steps:          int   = 50
    eval_every_n_epochs:        int   = 1
    save_best_only:             bool  = True

    # Feature extractor
    use_handcrafted_features:   bool  = True   
    use_perplexity_feature:     bool  = True   # GPT-2 perplexity score (LLM detection signal)


# ─────────────────────────────────────────────
#  LABELS
# ─────────────────────────────────────────────
@dataclass
class LabelConfig:
    labels:     List[str] = field(default_factory=lambda: [
        "benign",
        "phishing",
        "spear-phishing"
    ])

    @property
    def label2id(self) -> dict:
        return {l: i for i, l in enumerate(self.labels)}

    @property
    def id2label(self) -> dict:
        return {i: l for i, l in enumerate(self.labels)}

    num_labels: int = 3


# ─────────────────────────────────────────────
#  FEATURES  (handcrafted feature extractor settings)
# ─────────────────────────────────────────────
@dataclass
class FeatureConfig:
    # GPT-2 perplexity (for LLM-generated email detection)
    perplexity_model:   str   = "gpt2"           # loaded once, cached
    perplexity_max_len: int   = 512
    perplexity_device:  str   = "cpu"         

    # URL features
    max_urls_clip:      int   = 10               # clip URL count at 10 to avoid outlier skew

    # Urgency lexicon (used for urgency score feature)
    urgency_keywords: List[str] = field(default_factory=lambda: [
        "urgent", "immediately", "asap", "action required", "verify now",
        "account suspended", "limited time", "expires", "deadline",
        "respond now", "critical", "warning", "alert", "confirm immediately"
    ])

    # Authority lexicon
    authority_keywords: List[str] = field(default_factory=lambda: [
        "ceo", "director", "management", "hr department", "it department",
        "security team", "helpdesk", "admin", "official", "authorized"
    ])


# ─────────────────────────────────────────────
#  MASTER CONFIG  (single import point)
# ─────────────────────────────────────────────
@dataclass
class Config:
    paths:      PathConfig      = field(default_factory=PathConfig)
    tokenizer:  TokenizerConfig = field(default_factory=TokenizerConfig)
    model:      ModelConfig     = field(default_factory=ModelConfig)
    training:   TrainingConfig  = field(default_factory=TrainingConfig)
    labels:     LabelConfig     = field(default_factory=LabelConfig)
    features:   FeatureConfig   = field(default_factory=FeatureConfig)

    def __post_init__(self):
        # Auto-create directories on instantiation
        for path in [
            self.paths.checkpoint_dir,
            self.paths.log_dir,
            self.paths.results_dir,
            os.path.dirname(self.paths.vocab_file),
        ]:
            os.makedirs(path, exist_ok=True)

    def summary(self):
        """Print a human-readable config summary."""
        m = self.model
        t = self.training
        print("=" * 55)
        print("  PHISHING TRANSFORMER — CONFIG SUMMARY")
        print("=" * 55)
        print(f"  Architecture")
        print(f"    Hidden size          : {m.hidden_size}")
        print(f"    Attention heads      : {m.num_attention_heads} (head_size={m.head_size})")
        print(f"    Encoder layers       : {m.num_encoder_layers}")
        print(f"    FFN intermediate     : {m.intermediate_size}")
        print(f"    Max sequence length  : {m.max_position_embeddings}")
        print(f"    Position att type    : {m.pos_att_type}")
        print(f"    Estimated params     : {m.total_params_estimate}")
        print(f"  Training")
        print(f"    Batch size           : {t.batch_size} × {t.grad_accumulation_steps} accum = {t.batch_size * t.grad_accumulation_steps} effective")
        print(f"    Learning rate        : {t.learning_rate}")
        print(f"    Epochs               : {t.num_epochs} (patience={t.early_stopping_patience})")
        print(f"    Mixed precision      : {t.use_amp} ({t.amp_dtype})")
        print(f"    Handcrafted features : {t.use_handcrafted_features}")
        print(f"    Perplexity feature   : {t.use_perplexity_feature}")
        print(f"  Labels")
        for i, l in self.labels.id2label.items():
            print(f"    {i} → {l}")
        print("=" * 55)


# ─────────────────────────────────────────────
#  USAGE
# ─────────────────────────────────────────────
# from config import Config
# cfg = Config()
# cfg.summary()
#
# Access anywhere:
#   cfg.model.hidden_size
#   cfg.training.learning_rate
#   cfg.labels.label2id["phishing"]  → 1
#   cfg.paths.best_model             → "checkpoints/best_model.pt"

if __name__ == "__main__":
    cfg = Config()
    cfg.summary()
