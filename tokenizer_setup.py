"""
tokenizer_setup.py — Download BERT WordPiece vocab and build our own tokenizer.
No HuggingFace Tokenizer class used — pure scratch implementation.

What this file does:
  1. Downloads bert-base-uncased vocab.txt (30,522 tokens)
  2. Builds word2idx / idx2word mappings and saves them
  3. Implements WordPiece tokenization from scratch
  4. Implements encode() — text → padded token ID tensor
  5. Smoke-tests everything on a sample email

Usage:
  python tokenizer_setup.py          # downloads vocab, runs smoke test
  from tokenizer_setup import PhishingTokenizer
"""

import os
import re
import json
import unicodedata
import urllib.request
from typing import List, Tuple, Dict

from config import Config

cfg = Config()

# ─────────────────────────────────────────────
#  STEP 1 — Download BERT vocab.txt
# ─────────────────────────────────────────────

VOCAB_URL = (
    "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
)

def download_vocab(save_path: str = cfg.paths.vocab_file) -> None:
    """Download bert-base-uncased vocab.txt if not already present."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"  vocab.txt already exists at {save_path} — skipping download.")
        return

    print(f"  Downloading BERT vocab from HuggingFace...")
    urllib.request.urlretrieve(VOCAB_URL, save_path)
    print(f"  Saved to {save_path}")


# ─────────────────────────────────────────────
#  STEP 2 — Build word2idx / idx2word
# ─────────────────────────────────────────────

def build_vocab_mappings(
    vocab_path: str = cfg.paths.vocab_file,
    save_path:  str = cfg.paths.tokenizer_cache,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Read vocab.txt and build bidirectional mappings.
    Each line in vocab.txt is one token; line number = token ID.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]

    word2idx = {token: idx for idx, token in enumerate(tokens)}
    idx2word = {idx: token for idx, token in enumerate(tokens)}

    # Save word2idx as JSON for fast reload
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f)

    print(f"  Vocabulary size : {len(word2idx):,} tokens")
    print(f"  Saved word2idx  → {save_path}")

    return word2idx, idx2word


# ─────────────────────────────────────────────
#  STEP 3 — WordPiece Tokenizer (scratch)
# ─────────────────────────────────────────────

class PhishingTokenizer:
    """
    Hand-built WordPiece tokenizer that mirrors bert-base-uncased behaviour.

    Pipeline per input string:
      1. Unicode normalisation (NFD) + accent stripping
      2. Lowercase
      3. Whitespace + punctuation tokenisation → raw tokens
      4. WordPiece segmentation against BERT vocab
      5. Convert to IDs, add [CLS] / [SEP], pad / truncate to max_seq_len

    Email-specific behaviour:
      encode(subject, body) formats as:
        [CLS] <subject tokens> [SEP] <body tokens> [SEP] [PAD]...
      Body is truncated first if the combined length exceeds max_seq_len.
    """

    def __init__(
        self,
        vocab_path:   str = cfg.paths.vocab_file,
        cache_path:   str = cfg.paths.tokenizer_cache,
        max_seq_len:  int = cfg.tokenizer.max_seq_len,
    ):
        self.max_seq_len = max_seq_len

        # Special token IDs (fixed in BERT vocab)
        self.pad_id  = cfg.tokenizer.pad_token_id   # 0
        self.unk_id  = cfg.tokenizer.unk_token_id   # 100
        self.cls_id  = cfg.tokenizer.cls_token_id   # 101
        self.sep_id  = cfg.tokenizer.sep_token_id   # 102
        self.mask_id = cfg.tokenizer.mask_token_id  # 103

        # Load vocab
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.word2idx: Dict[str, int] = json.load(f)
        else:
            self.word2idx, _ = build_vocab_mappings(vocab_path, cache_path)

        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    # ── Internal helpers ──────────────────────────────────────────────

    def _normalise(self, text: str) -> str:
        """Unicode NFD normalisation + accent removal + lowercase."""
        text = unicodedata.normalize("NFD", text)
        text = "".join(
            ch for ch in text
            if unicodedata.category(ch) != "Mn"   # strip combining marks
        )
        return text.lower()

    def _is_punctuation(self, ch: str) -> bool:
        """True if character is punctuation (mirrors BERT's definition)."""
        cp = ord(ch)
        # ASCII punctuation ranges
        if (33 <= cp <= 47) or (58 <= cp <= 64) or \
           (91 <= cp <= 96) or (123 <= cp <= 126):
            return True
        return unicodedata.category(ch).startswith("P")

    def _tokenise_raw(self, text: str) -> List[str]:
        """
        Split on whitespace and around punctuation characters.
        Returns a flat list of raw string tokens.
        """
        text = self._normalise(text)
        tokens = []
        for word in text.strip().split():
            chars = []
            for ch in word:
                if self._is_punctuation(ch):
                    if chars:
                        tokens.append("".join(chars))
                        chars = []
                    tokens.append(ch)
                else:
                    chars.append(ch)
            if chars:
                tokens.append("".join(chars))
        return tokens

    def _wordpiece(self, token: str) -> List[str]:
        """
        Greedy longest-match WordPiece segmentation.
        Unknown characters map to [UNK].
        """
        if token in self.word2idx:
            return [token]

        sub_tokens = []
        start = 0
        while start < len(token):
            end = len(token)
            cur_substr = None
            while start < end:
                substr = token[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.word2idx:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return ["[UNK]"]
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def tokenise(self, text: str) -> List[str]:
        """Text → list of WordPiece string tokens (no special tokens)."""
        raw_tokens = self._tokenise_raw(text)
        wp_tokens  = []
        for tok in raw_tokens:
            wp_tokens.extend(self._wordpiece(tok))
        return wp_tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Token strings → IDs, unknown tokens map to unk_id."""
        return [self.word2idx.get(t, self.unk_id) for t in tokens]

    # ── Public API ────────────────────────────────────────────────────

    def encode(
        self,
        subject: str,
        body:    str,
        return_tensors: bool = True,
    ) -> dict:
        """
        Encode subject + body into model-ready tensors.

        Format:  [CLS] <subject> [SEP] <body> [SEP] [PAD]...
        Length:  exactly max_seq_len tokens

        Truncation strategy (body_first):
          Available slots = max_seq_len - 3 (for CLS + 2×SEP)
          Subject tokens kept in full (up to available // 4, max 64).
          Remaining slots given to body.

        Returns dict with:
          input_ids      : List[int] or torch.Tensor  shape (max_seq_len,)
          attention_mask : List[int] or torch.Tensor  shape (max_seq_len,)
          token_type_ids : List[int] or torch.Tensor  shape (max_seq_len,)
                           0 = subject segment, 1 = body segment
        """
        subject_tokens = self.tokenise(subject or "")
        body_tokens    = self.tokenise(body    or "")

        # Available positions (subtract CLS + 2×SEP)
        available = self.max_seq_len - 3

        # Subject gets up to 64 tokens; body gets the rest
        max_subject = min(64, available // 4)
        max_body    = available - min(len(subject_tokens), max_subject)

        subject_tokens = subject_tokens[:max_subject]
        body_tokens    = body_tokens[:max_body]

        # Build token list and segment IDs
        tokens      = ["[CLS]"] + subject_tokens + ["[SEP]"] + body_tokens + ["[SEP]"]
        token_types = (
            [0] * (len(subject_tokens) + 2) +   # 0 for CLS + subject + SEP
            [1] * (len(body_tokens) + 1)          # 1 for body + SEP
        )

        # Convert to IDs
        input_ids = self.convert_tokens_to_ids(tokens)

        # Attention mask (1 = real token, 0 = pad)
        attention_mask = [1] * len(input_ids)

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        input_ids      += [self.pad_id] * pad_len
        attention_mask += [0]          * pad_len
        token_types    += [0]          * pad_len

        assert len(input_ids) == self.max_seq_len, \
            f"Length mismatch: {len(input_ids)} != {self.max_seq_len}"

        if return_tensors:
            import torch
            return {
                "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_types,    dtype=torch.long),
            }
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_types,
        }

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """IDs → readable string (for debugging)."""
        special = {self.pad_id, self.cls_id, self.sep_id, self.mask_id}
        tokens = []
        for i in ids:
            tok = self.idx2word.get(i, "[UNK]")
            if skip_special_tokens and i in special:
                continue
            tokens.append(tok)
        # Merge WordPiece ## subwords
        text = " ".join(tokens)
        text = re.sub(r" ##", "", text)
        return text.strip()


# ─────────────────────────────────────────────
#  STEP 4 — Smoke Test
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  TOKENIZER SMOKE TEST")
    print("=" * 55)

    tokenizer = PhishingTokenizer()

    # Sample phishing email
    subject = "URGENT: Your Microsoft 365 account will be suspended"
    body    = (
        "Dear user, we have detected suspicious activity on your account. "
        "Please verify your credentials immediately by clicking the link below. "
        "Failure to act within 24 hours will result in permanent suspension. "
        "Click here: http://malicious-link.com/verify?token=abc123"
    )

    encoded = tokenizer.encode(subject, body)

    input_ids      = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]

    real_tokens = attention_mask.sum().item()
    seg0        = (token_type_ids == 0).sum().item()
    seg1        = (token_type_ids == 1).sum().item()

    print(f"\n  Input text")
    print(f"    Subject : {subject[:60]}...")
    print(f"    Body    : {body[:60]}...")
    print(f"\n  Encoded output")
    print(f"    input_ids shape      : {list(input_ids.shape)}")
    print(f"    attention_mask shape : {list(attention_mask.shape)}")
    print(f"    token_type_ids shape : {list(token_type_ids.shape)}")
    print(f"\n  Token breakdown")
    print(f"    Real tokens (non-pad): {real_tokens}")
    print(f"    Segment 0 (subject)  : {seg0} tokens")
    print(f"    Segment 1 (body)     : {seg1} tokens")
    print(f"    Pad tokens           : {512 - real_tokens}")
    print(f"\n  First 15 token IDs   : {input_ids[:15].tolist()}")
    print(f"  CLS token check      : {input_ids[0].item()} (expected 101) ✓" if input_ids[0] == 101 else "  ✗ CLS token wrong!")

    # Decode back
    decoded = tokenizer.decode(input_ids.tolist())
    print(f"\n  Decoded (first 100 chars):")
    print(f"    {decoded[:100]}...")

    # Edge cases
    print(f"\n  Edge case — empty subject:")
    enc2 = tokenizer.encode("", body)
    print(f"    input_ids[0] = {enc2['input_ids'][0].item()} (CLS=101)")
    print(f"    input_ids[1] = {enc2['input_ids'][1].item()} (SEP=102, subject empty)")

    print(f"\n  Edge case — very long body (should truncate cleanly):")
    long_body = "word " * 1000
    enc3 = tokenizer.encode(subject, long_body)
    print(f"    Total length   : {len(enc3['input_ids'])} (expected 512)")
    print(f"    Real tokens    : {enc3['attention_mask'].sum().item()}")

    print("\n  ✅ Smoke test passed — tokenizer is working correctly.")
    print("=" * 55)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[1/3] Downloading BERT vocab...")
    download_vocab()

    print("\n[2/3] Building word2idx mappings...")
    build_vocab_mappings()

    print("\n[3/3] Running smoke test...")
    smoke_test()

    print("\n✅ Tokenizer setup complete.")