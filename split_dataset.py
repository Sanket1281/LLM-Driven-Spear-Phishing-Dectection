import json
import random
import hashlib
from collections import defaultdict, Counter

random.seed(42)

INPUT_FILE = "unified_dataset.jsonl"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/val.jsonl"
TEST_FILE = "data/test.jsonl"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------- Load ----------
records = []
with open(INPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))

print(f"📥 Loaded {len(records)} records")


# ---------- Leakage prevention: hash-based dedup ----------
# Hash on (subject + first 100 chars of body) — catches reformatted duplicates
def content_hash(r):
    key = (r["subject"][:100] + r["body"][:200]).lower()
    return hashlib.md5(key.encode()).hexdigest()


seen = set()
unique = []
for r in records:
    h = content_hash(r)
    if h not in seen:
        seen.add(h)
        unique.append(r)
records = unique
print(f"✓ After cross-class dedup: {len(records)} records")

# ---------- Stratified split ----------
by_class = defaultdict(list)
for r in records:
    by_class[r["label"]].append(r)

train, val, test = [], [], []

for label, items in by_class.items():
    random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train.extend(items[:n_train])
    val.extend(items[n_train:n_train + n_val])
    test.extend(items[n_train + n_val:])

    print(f"   {label:20s}: train={n_train}, val={n_val}, test={n - n_train - n_val}")

# Shuffle final splits
random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

# ---------- Save ----------
for path, data in [(TRAIN_FILE, train), (VAL_FILE, val), (TEST_FILE, test)]:
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

print(f"\n✅ Splits saved:")
print(f"   {TRAIN_FILE}: {len(train)} records")
print(f"   {VAL_FILE}:   {len(val)} records")
print(f"   {TEST_FILE}:  {len(test)} records")

# ---------- Verify class balance ----------
print(f"\n📊 Final class distributions:")
for name, split in [("train", train), ("val", val), ("test", test)]:
    counts = Counter(r["label"] for r in split)
    print(f"\n  {name}:")
    for label, count in counts.most_common():
        print(f"    {label:20s}: {count:5d} ({count / len(split) * 100:.1f}%)")