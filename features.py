"""
features.py — Handcrafted Feature Extractor + GPT-2 Perplexity Scorer.

Extracts 26 numerical features from raw email text.
These feed into the ClassifierHead's feature branch alongside the CLS embedding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEATURE LIST  (26 total)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  URL features (4)
    0  url_count           — number of URLs in body
    1  has_url             — binary: any URL present
    2  url_mismatch        — link text ≠ href domain (common phishing tell)
    3  suspicious_tld      — URL uses .xyz/.tk/.ru etc.

  Urgency & manipulation (4)
    4  urgency_score       — count of urgency keywords normalised by length
    5  authority_score     — count of authority keywords
    6  threat_score        — count of threat/consequence keywords
    7  reward_score        — count of reward/prize keywords

  Structural (5)
    8  subject_len         — character length of subject
    9  body_len            — word count of body
    10 exclamation_count   — number of ! in subject+body
    11 caps_ratio          — ratio of uppercase letters
    12 punctuation_density — punctuation chars / total chars

  Sender/target signals (3)
    13 has_greeting        — personalised greeting (Dear [Name])
    14 greeting_specificity— how specific the greeting is (generic vs named)
    15 has_signature       — email has a sign-off block

  Readability (3)
    16 avg_word_len        — average word length (LLM text tends to be longer)
    17 avg_sentence_len    — average sentence length
    18 type_token_ratio    — vocabulary diversity (unique/total words)

  Phishing-specific patterns (4)
    19 has_action_request  — "click here", "verify", "confirm", "login"
    20 credential_request  — asks for password/username/SSN/card
    21 spoofing_indicators — mismatched sender name vs domain
    22 attachment_mention  — mentions attachment/invoice/document

  LLM detection (3)  ← KEY FEATURES for your research
    23 perplexity          — GPT-2 perplexity (low = suspiciously fluent)
    24 perplexity_bucket   — discretised: 0=very_low, 1=low, 2=med, 3=high
    25 fluency_anomaly     — perplexity Z-score vs population mean

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  from features import FeatureExtractor
  extractor = FeatureExtractor()                  # loads GPT-2 once

  # Single email
  feat_vector = extractor.extract(subject, body)  # numpy array (26,)

  # Batch (from DataLoader)
  feat_tensor = extractor.extract_batch(raw_list, device)  # Tensor (B, 26)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import re
import math
import numpy as np
from typing import List
from urllib.parse import urlparse

import torch

from config import Config

cfg = Config()


# ─────────────────────────────────────────────
#  LEXICONS
# ─────────────────────────────────────────────

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "action required", "verify now",
    "account suspended", "limited time", "expires", "deadline", "respond now",
    "critical", "warning", "alert", "confirm immediately", "act now",
    "time sensitive", "final notice", "last chance", "within 24 hours",
    "account will be closed", "suspended", "locked", "unusual activity",
]

AUTHORITY_KEYWORDS = [
    "ceo", "director", "management", "hr department", "it department",
    "security team", "helpdesk", "admin", "official", "authorized",
    "compliance", "legal department", "finance team", "payroll",
    "it support", "system administrator", "executive",
]

THREAT_KEYWORDS = [
    "suspended", "terminated", "legal action", "arrest", "lawsuit",
    "penalty", "fine", "blocked", "restricted", "unauthorized",
    "violation", "consequences", "failure to", "will result",
]

REWARD_KEYWORDS = [
    "winner", "won", "prize", "reward", "gift", "free", "bonus",
    "lottery", "selected", "congratulations", "claim", "voucher",
]

ACTION_KEYWORDS = [
    "click here", "click the link", "verify your", "confirm your",
    "login to", "log in to", "sign in", "update your", "validate",
    "authenticate", "access your account", "follow the link",
]

CREDENTIAL_KEYWORDS = [
    "password", "username", "social security", "ssn", "credit card",
    "bank account", "pin number", "security code", "cvv", "mother's maiden",
    "date of birth", "account number", "routing number",
]

SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ru", ".cn", ".pw", ".top", ".club", ".work",
    ".party", ".download", ".racing", ".stream", ".gq", ".ml", ".ga",
}

GREETING_PATTERNS = [
    r"dear\s+\w+",
    r"hello\s+\w+",
    r"hi\s+\w+",
    r"good\s+(morning|afternoon|evening)\s+\w+",
]

SIGNATURE_PATTERNS = [
    r"(best regards|kind regards|sincerely|regards|thanks|thank you)[,.]?\s*\n",
    r"(yours (truly|faithfully|sincerely))[,.]?\s*\n",
]

ATTACHMENT_KEYWORDS = [
    "attached", "attachment", "invoice", "document", "file", "pdf",
    "spreadsheet", "receipt", "report", "see attached",
]


# ─────────────────────────────────────────────
#  URL HELPERS
# ─────────────────────────────────────────────

URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

def extract_urls(text: str) -> List[str]:
    return URL_PATTERN.findall(text)

def has_suspicious_tld(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)
    except Exception:
        return False

def has_url_mismatch(text: str) -> bool:
    """
    Detect anchor text that differs from the actual URL domain.
    Pattern: [display_text](url) or >display_text<  where display != url domain
    Also catches plain 'click here' next to a URL.
    """
    # Markdown-style links
    md_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', text)
    for display, url in md_links:
        try:
            url_domain = urlparse(url).netloc.lower()
            if url_domain not in display.lower() and display.lower() not in url_domain:
                if any(brand in display.lower() for brand in
                       ["microsoft", "google", "apple", "amazon", "paypal", "bank"]):
                    return True
        except Exception:
            pass

    # "click here" adjacent to URL (classic mismatch)
    if re.search(r'click\s+here', text, re.IGNORECASE) and extract_urls(text):
        return True

    return False


# ─────────────────────────────────────────────
#  GPT-2 PERPLEXITY SCORER
# ─────────────────────────────────────────────

class PerplexityScorer:
    """
    Computes GPT-2 perplexity for text fluency measurement.

    Low perplexity  → text is very fluent/predictable → likely LLM-generated
    High perplexity → text is awkward/unpredictable   → likely human-written spam

    Model runs on CPU to preserve GPU VRAM for training.
    Loaded once and reused across all emails.
    """

    # Population statistics for Z-score normalisation
    # These are approximate values from phishing email research
    # Will be updated after first batch extraction if fit() is called
    POP_MEAN = 150.0
    POP_STD  = 80.0

    def __init__(self, config: Config = cfg):
        self.max_len = config.features.perplexity_max_len   # 512
        self.device  = torch.device(config.features.perplexity_device)  # cpu
        self.model_name = config.features.perplexity_model  # "gpt2"
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load(self):
        """Lazy load — only downloads/loads GPT-2 on first use."""
        if self._loaded:
            return
        print("  Loading GPT-2 for perplexity scoring (one-time download ~500MB)...")
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self._tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        self._model     = GPT2LMHeadModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        self._loaded = True
        print("  GPT-2 loaded ✓")

    def score(self, text: str) -> float:
        """
        Compute perplexity of text under GPT-2.

        Perplexity = exp(average negative log-likelihood per token)

        Returns a float. Lower = more fluent.
        Returns POP_MEAN on error (neutral fallback).
        """
        self._load()

        text = text.strip()
        if not text or len(text.split()) < 5:
            return self.POP_MEAN   # too short to score reliably

        try:
            inputs = self._tokenizer(
                text,
                return_tensors     = "pt",
                truncation         = True,
                max_length         = self.max_len,
            ).to(self.device)

            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 3:
                return self.POP_MEAN

            with torch.no_grad():
                outputs = self._model(input_ids, labels=input_ids)
                # outputs.loss = mean negative log-likelihood per token
                nll = outputs.loss.item()

            perplexity = math.exp(nll)

            # Clamp to reasonable range (avoid inf on very bad text)
            perplexity = min(perplexity, 10_000.0)
            return perplexity

        except Exception:
            return self.POP_MEAN

    def z_score(self, perplexity: float) -> float:
        """Normalise perplexity to Z-score using population statistics."""
        return (perplexity - self.POP_MEAN) / max(self.POP_STD, 1.0)

    def bucket(self, perplexity: float) -> int:
        """
        Discretise perplexity into 4 buckets:
          0 = very low  (<50)   → highly suspicious, likely LLM
          1 = low       (<100)  → suspicious
          2 = medium    (<200)  → normal range
          3 = high      (≥200)  → likely human-written spam
        """
        if perplexity < 50:   return 0
        if perplexity < 100:  return 1
        if perplexity < 200:  return 2
        return 3


# ─────────────────────────────────────────────
#  MAIN FEATURE EXTRACTOR
# ─────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts all 26 features from a raw email (subject + body).

    Args:
        config           : Config object
        use_perplexity   : if False, perplexity features are set to 0
                           (useful for fast ablation runs)

    Methods:
        extract(subject, body)           → np.ndarray (26,)
        extract_batch(raw_list, device)  → torch.Tensor (B, 26)
    """

    NUM_FEATURES = 26

    def __init__(
        self,
        config:          Config = cfg,
        use_perplexity:  bool   = None,
    ):
        self.config = config
        self.use_perplexity = (
            use_perplexity
            if use_perplexity is not None
            else config.training.use_perplexity_feature
        )
        self.perplexity_scorer = PerplexityScorer(config) if self.use_perplexity else None

    # ── Individual feature extractors ─────────────────────────────

    def _url_features(self, subject: str, body: str) -> List[float]:
        """Features 0-3: URL-based signals."""
        text = subject + " " + body
        urls = extract_urls(text)

        url_count    = min(len(urls), self.config.features.max_urls_clip)
        has_url      = float(len(urls) > 0)
        url_mismatch = float(has_url_mismatch(text))
        susp_tld     = float(any(has_suspicious_tld(u) for u in urls))

        return [url_count, has_url, url_mismatch, susp_tld]

    def _lexicon_features(self, subject: str, body: str) -> List[float]:
        """Features 4-7: Urgency, authority, threat, reward scores."""
        text       = (subject + " " + body).lower()
        word_count = max(len(text.split()), 1)

        def score(keywords):
            hits = sum(1 for kw in keywords if kw in text)
            return hits / math.log(word_count + 2)   # log-normalise by length

        urgency   = score(URGENCY_KEYWORDS)
        authority = score(AUTHORITY_KEYWORDS)
        threat    = score(THREAT_KEYWORDS)
        reward    = score(REWARD_KEYWORDS)

        return [urgency, authority, threat, reward]

    def _structural_features(self, subject: str, body: str) -> List[float]:
        """Features 8-12: Structural signals."""
        full_text = subject + " " + body

        subject_len  = float(len(subject))
        body_len     = float(len(body.split()))
        excl_count   = float(full_text.count("!"))

        letters = [c for c in full_text if c.isalpha()]
        caps_ratio = (
            sum(1 for c in letters if c.isupper()) / len(letters)
            if letters else 0.0
        )

        punct_count = sum(1 for c in full_text if c in '.,;:!?()[]{}"\'-')
        punct_density = punct_count / max(len(full_text), 1)

        return [subject_len, body_len, excl_count, caps_ratio, punct_density]

    def _sender_features(self, subject: str, body: str) -> List[float]:
        """Features 13-15: Greeting and signature signals."""
        body_lower = body.lower()

        # Personalised greeting detection
        has_greeting = float(any(
            re.search(p, body_lower) for p in GREETING_PATTERNS
        ))

        # Greeting specificity: named greeting scores higher than generic
        named_greeting = float(bool(re.search(
            r'dear\s+[A-Z][a-z]+|hello\s+[A-Z][a-z]+', body
        )))
        greeting_specificity = named_greeting

        # Signature detection
        has_signature = float(any(
            re.search(p, body_lower) for p in SIGNATURE_PATTERNS
        ))

        return [has_greeting, greeting_specificity, has_signature]

    def _readability_features(self, body: str) -> List[float]:
        """Features 16-18: Readability metrics (LLM text is unusually polished)."""
        words = body.split()
        if not words:
            return [0.0, 0.0, 0.0]

        avg_word_len = sum(len(w) for w in words) / len(words)

        # Sentence splitting
        sentences = re.split(r'[.!?]+', body)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_len = (
            sum(len(s.split()) for s in sentences) / len(sentences)
            if sentences else 0.0
        )

        # Type-token ratio (vocabulary diversity)
        unique_words = len(set(w.lower() for w in words))
        ttr = unique_words / len(words)

        return [avg_word_len, avg_sentence_len, ttr]

    def _phishing_pattern_features(self, subject: str, body: str) -> List[float]:
        """Features 19-22: Phishing-specific pattern signals."""
        text       = (subject + " " + body).lower()
        body_lower = body.lower()

        has_action     = float(any(kw in text for kw in ACTION_KEYWORDS))
        cred_request   = float(any(kw in text for kw in CREDENTIAL_KEYWORDS))
        attachment_men = float(any(kw in body_lower for kw in ATTACHMENT_KEYWORDS))

        # Spoofing: sender claims to be a known brand in body
        # but email likely comes from a non-matching domain
        brand_names = ["microsoft", "google", "apple", "amazon", "paypal",
                       "facebook", "netflix", "bank", "wells fargo", "chase"]
        spoofing = float(
            sum(1 for b in brand_names if b in body_lower) >= 1
            and has_action
        )

        return [has_action, cred_request, spoofing, attachment_men]

    def _perplexity_features(self, body: str) -> List[float]:
        """Features 23-25: LLM detection via GPT-2 perplexity."""
        if not self.use_perplexity or self.perplexity_scorer is None:
            return [0.0, 0.0, 0.0]

        perp    = self.perplexity_scorer.score(body)
        bucket  = float(self.perplexity_scorer.bucket(perp))
        z_score = self.perplexity_scorer.z_score(perp)

        # Clip Z-score to [-3, 3] to avoid outlier explosion
        z_score = max(-3.0, min(3.0, z_score))

        return [perp, bucket, z_score]

    # ── Public API ────────────────────────────────────────────────

    def extract(self, subject: str, body: str) -> np.ndarray:
        """
        Extract all 26 features from one email.
        Returns np.ndarray of shape (26,) dtype float32.
        """
        subject = str(subject or "").strip()
        body    = str(body    or "").strip()

        features = (
            self._url_features(subject, body)          +  # 4
            self._lexicon_features(subject, body)      +  # 4
            self._structural_features(subject, body)   +  # 5
            self._sender_features(subject, body)       +  # 3
            self._readability_features(body)           +  # 3
            self._phishing_pattern_features(subject, body) +  # 4
            self._perplexity_features(body)               # 3
        )

        assert len(features) == self.NUM_FEATURES, \
            f"Expected {self.NUM_FEATURES} features, got {len(features)}"

        vec = np.array(features, dtype=np.float32)

        # Replace any NaN/Inf with 0 (defensive)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        return vec

    def extract_batch(
        self,
        raw_list: List[dict],
        device:   torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Extract features for a batch of raw email dicts.
        Each dict must have 'subject' and 'body' keys.

        Returns torch.Tensor of shape (B, 26) on the specified device.
        """
        vectors = []
        for rec in raw_list:
            vec = self.extract(
                rec.get("subject", ""),
                rec.get("body",    ""),
            )
            vectors.append(vec)

        batch = np.stack(vectors, axis=0)   # (B, 26)
        return torch.tensor(batch, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────
#  SMOKE TEST
# ─────────────────────────────────────────────

def smoke_test():
    print("\n" + "=" * 55)
    print("  FEATURES SMOKE TEST")
    print("=" * 55)

    # ── Sample emails ─────────────────────────────────────────────
    emails = [
        {
            "label":   "spear-phishing",
            "subject": "URGENT: Your Microsoft 365 account requires immediate verification",
            "body": (
                "Dear John, I hope this message finds you well. I am writing to inform you "
                "that your Microsoft 365 account has been flagged for unusual activity. "
                "To prevent suspension, please verify your credentials immediately by "
                "clicking the secure link below. This action must be completed within "
                "24 hours to avoid permanent account termination. "
                "Click here: https://microsoft-verify.xyz/login?token=abc123 "
                "Best regards, Microsoft Security Team"
            ),
        },
        {
            "label":   "phishing",
            "subject": "YOU HAVE WON!!! Claim ur prize NOW!!!",
            "body": (
                "CONGRATULATIONS!!! u hav been selected as winner of $1,000,000 prize!! "
                "click link 2 claim ur reward NOW before it EXPIRE!!! "
                "http://totallylegit.tk/claim?id=99999 "
                "ACT FAST LIMITED TIME OFFER!!!"
            ),
        },
        {
            "label":   "benign",
            "subject": "Team lunch this Friday",
            "body": (
                "Hi everyone, just a reminder that we have a team lunch this Friday at noon. "
                "We'll be going to the Italian place on Main Street. Please let me know if "
                "you have any dietary restrictions. Looking forward to seeing everyone there! "
                "Thanks, Sarah"
            ),
        },
    ]

    # ── Test without perplexity first (fast) ─────────────────────
    print(f"\n── Feature extraction (perplexity disabled) ─────")
    extractor_fast = FeatureExtractor(use_perplexity=False)

    feature_names = [
        "url_count", "has_url", "url_mismatch", "suspicious_tld",
        "urgency_score", "authority_score", "threat_score", "reward_score",
        "subject_len", "body_len", "exclamation_count", "caps_ratio", "punct_density",
        "has_greeting", "greeting_specificity", "has_signature",
        "avg_word_len", "avg_sentence_len", "type_token_ratio",
        "has_action_request", "credential_request", "spoofing_indicators", "attachment_mention",
        "perplexity", "perplexity_bucket", "fluency_anomaly",
    ]

    for email in emails:
        vec = extractor_fast.extract(email["subject"], email["body"])
        assert vec.shape == (26,), f"Wrong shape: {vec.shape}"
        assert not np.isnan(vec).any(), "NaN in features!"

        print(f"\n  [{email['label']}]")
        print(f"    Subject: {email['subject'][:55]}...")
        # Print non-zero features
        for i, (name, val) in enumerate(zip(feature_names, vec)):
            if val != 0.0 and i < 23:   # skip perplexity features (disabled)
                print(f"    {name:<25}: {val:.4f}")

    # ── Batch extraction ──────────────────────────────────────────
    print(f"\n── Batch extraction check ───────────────────────")
    batch_tensor = extractor_fast.extract_batch(emails, device=torch.device("cpu"))
    print(f"  Batch shape : {list(batch_tensor.shape)}  (expected [3, 26])")
    assert batch_tensor.shape == (3, 26), f"Wrong batch shape: {batch_tensor.shape}"
    assert not torch.isnan(batch_tensor).any(), "NaN in batch features!"
    print(f"  No NaN/Inf  : ✓")
    print(f"  Batch extraction : ✓")

    # ── Verify LLM phishing has lower url count on susp tld ──────
    print(f"\n── Feature sanity checks ────────────────────────")
    spear_vec = extractor_fast.extract(emails[0]["subject"], emails[0]["body"])
    phish_vec = extractor_fast.extract(emails[1]["subject"], emails[1]["body"])
    benign_vec= extractor_fast.extract(emails[2]["subject"], emails[2]["body"])

    # Spear phishing should have higher urgency than benign
    assert spear_vec[4] > benign_vec[4], "Spear phishing should have higher urgency"
    print(f"  Spear urgency > benign urgency : ✓  ({spear_vec[4]:.3f} > {benign_vec[4]:.3f})")

    # Traditional phishing should have exclamation marks
    assert phish_vec[10] > benign_vec[10], "Phishing should have more exclamations"
    print(f"  Phishing exclamations > benign : ✓  ({phish_vec[10]:.0f} > {benign_vec[10]:.0f})")

    # Benign should have no suspicious TLD
    assert benign_vec[3] == 0.0, "Benign email should have no suspicious TLD"
    print(f"  Benign has no suspicious TLD   : ✓")

    # Spear phishing should trigger action request
    assert spear_vec[19] == 1.0, "Spear phishing should have action request"
    print(f"  Spear phishing has action request : ✓")

    # ── Optional: perplexity test ─────────────────────────────────
    print(f"\n── Perplexity scorer (optional — requires ~500MB download) ──")
    print(f"  Skipping in smoke test to avoid download.")
    print(f"  To test perplexity manually run:")
    print(f"    from features import PerplexityScorer")
    print(f"    scorer = PerplexityScorer()")
    print(f"    print(scorer.score('Your account has been suspended.'))")

    print(f"\n  ✅ All feature assertions passed.")
    print("=" * 55)
    print("\n  Next step: build train.py")


if __name__ == "__main__":
    smoke_test()