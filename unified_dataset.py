import json
import csv
import os
import re
import random
import email
from email import policy
from pathlib import Path
from tqdm import tqdm

random.seed(42)  # reproducibility for your paper

# ---------- Config ----------
ENRON_CSV = "emails.csv"  # from Kaggle Enron dump
KAGGLE_PHISH_CSV = "Phishing_Email.csv"  # from Kaggle phishing dataset
SPEAR_JSONL = "spear_phishing_cleaned.jsonl"  # your cleaned synthetic

OUTPUT_FILE = "unified_dataset.jsonl"

# Target counts per class
TARGET_BENIGN = 1000
TARGET_PHISHING = 1000


# Spear-phishing: use everything you have

# ---------- Helpers ----------
def clean_text(text):
    """Remove extra whitespace, control chars."""
    if not text: return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_body(body, max_words=400):
    """Limit body length to keep model inputs manageable."""
    words = body.split()
    return " ".join(words[:max_words])


def normalize_record(from_, to, subject, body, label, source, extra=None):
    """Standardized record format for all 3 classes."""
    return {
        "from": clean_text(from_)[:200],
        "to": clean_text(to)[:200],
        "subject": clean_text(subject)[:300],
        "body": truncate_body(clean_text(body)),
        "label": label,
        "source": source,
        "metadata": extra or {}
    }


# ---------- 1. Process Enron (Benign) ----------
def load_enron(csv_path, target_count):
    print(f"\n📥 Processing Enron benign emails...")
    records = []

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        # Enron CSV has 'file' and 'message' columns
        # 'message' contains full raw email
        for row in tqdm(reader, desc="Parsing Enron"):
            try:
                msg = email.message_from_string(row["message"], policy=policy.default)

                from_ = str(msg.get("From", ""))
                to = str(msg.get("To", ""))
                subject = str(msg.get("Subject", ""))

                # Extract body
                if msg.is_multipart():
                    body_parts = []
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body_parts.append(part.get_content())
                    body = " ".join(body_parts)
                else:
                    body = msg.get_content() if hasattr(msg, "get_content") else str(msg.get_payload())

                # Filter: skip very short / very long / forwarded chains
                word_count = len(body.split())
                if word_count < 30 or word_count > 800:
                    continue
                if "-----Original Message-----" in body:  # skip forwards/replies
                    continue
                if not from_ or not subject:
                    continue

                records.append(normalize_record(
                    from_, to, subject, body,
                    label="benign",
                    source="enron"
                ))

                if len(records) >= target_count * 3:  # over-sample then random pick
                    break
            except Exception:
                continue

    # Randomly sample target_count to avoid temporal bias
    random.shuffle(records)
    records = records[:target_count]
    print(f"✓ Collected {len(records)} benign emails from Enron")
    return records


# ---------- 2. Process Kaggle Phishing ----------
def load_kaggle_phishing(csv_path, target_count):
    records = []
    skipped = 0

    csv.field_size_limit(10_000_000)

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        # Read raw lines, skip any line that's clearly corrupted
        lines = []
        header = None
        for i, line in enumerate(f):
            if i == 0:
                header = line
                continue
            # Skip lines that are suspiciously large (>500KB = corrupt)
            if len(line) > 500_000:
                skipped += 1
                continue
            lines.append(line)

    print(f"  Skipped {skipped} oversized lines")

    # Now parse the cleaned lines
    import io
    content = header + "".join(lines)
    reader = csv.DictReader(io.StringIO(content))

    # Auto-detect columns
    rows = list(reader)
    if not rows:
        print("  ERROR: No rows loaded")
        return []

    print(f"  Columns found: {list(rows[0].keys())}")
    print(f"  Total rows: {len(rows)}")

    # Detect text and label columns
    text_col = None
    label_col = None
    for col in rows[0].keys():
        cl = col.lower()
        if any(x in cl for x in ["text", "body", "email", "content", "message"]):
            if text_col is None: text_col = col
        if any(x in cl for x in ["type", "label", "class", "category"]):
            if label_col is None: label_col = col

    print(f"  Using text_col={text_col!r}, label_col={label_col!r}")

    if not text_col or not label_col:
        print("  ERROR: Could not detect columns. Available:", list(rows[0].keys()))
        return []

    for row in tqdm(rows, desc="Parsing Kaggle phishing"):
        try:
            text = row.get(text_col, "") or ""
            label_val = (row.get(label_col, "") or "").lower()

            # Accept rows labelled as phishing
            if not any(x in label_val for x in ["phish", "spam", "malicious", "1"]):
                continue

            text = text.strip()
            if not text:
                continue

            # Split into subject / body heuristically
            lines = text.split("\n", 1)
            if len(lines) > 1 and len(lines[0]) < 150:
                subject = lines[0].strip()
                body = lines[1].strip()
            else:
                subject = "(no subject)"
                body = text

            word_count = len(body.split())
            if word_count < 20 or word_count > 800:
                continue

            records.append(normalize_record(
                from_="(unknown sender)",
                to="(unknown recipient)",
                subject=subject,
                body=body,
                label="phishing",
                source="kaggle_phishing"
            ))

        except Exception:
            continue

    random.shuffle(records)
    print(f"  ✓ Collected {min(len(records), target_count)} phishing emails")
    return records[:target_count]


# ---------- 3. Process Your Spear-Phishing ----------
def load_spear_phishing(jsonl_path):
    print(f"\n📥 Processing spear-phishing emails...")
    records = []
    with open(jsonl_path) as f:
        for line in tqdm(f, desc="Loading spear-phishing"):
            try:
                e = json.loads(line)
                records.append(normalize_record(
                    from_=e.get("from", ""),
                    to=e.get("to", ""),
                    subject=e.get("subject", ""),
                    body=e.get("body", ""),
                    label="spear-phishing",
                    source="llm_generated",
                    extra={
                        "generator_model": e.get("generator_model", "unknown"),
                        **e.get("metadata", {})
                    }
                ))
            except Exception:
                continue
    print(f"✓ Loaded {len(records)} spear-phishing emails")
    return records


# ---------- Build Unified Dataset ----------
all_records = []

if os.path.exists(ENRON_CSV):
    all_records.extend(load_enron(ENRON_CSV, TARGET_BENIGN))
else:
    print(f"⚠️  {ENRON_CSV} not found — skipping benign class")

if os.path.exists(KAGGLE_PHISH_CSV):
    all_records.extend(load_kaggle_phishing(KAGGLE_PHISH_CSV, TARGET_PHISHING))
else:
    print(f"⚠️  {KAGGLE_PHISH_CSV} not found — skipping phishing class")

if os.path.exists(SPEAR_JSONL):
    all_records.extend(load_spear_phishing(SPEAR_JSONL))

random.shuffle(all_records)

# ---------- Save ----------
with open(OUTPUT_FILE, "w") as f:
    for r in all_records:
        f.write(json.dumps(r) + "\n")

# ---------- Summary ----------
from collections import Counter

label_counts = Counter(r["label"] for r in all_records)
source_counts = Counter(r["source"] for r in all_records)

print(f"\n{'=' * 50}")
print(f"✅ Saved {len(all_records)} total records → {OUTPUT_FILE}")
print(f"\n📊 Class distribution:")
for label, count in label_counts.most_common():
    print(f"   {label:20s}: {count:5d} ({count / len(all_records) * 100:.1f}%)")
print(f"\n📂 Source distribution:")
for source, count in source_counts.most_common():
    print(f"   {source:20s}: {count:5d}")