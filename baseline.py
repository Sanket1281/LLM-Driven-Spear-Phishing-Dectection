"""
baselines.py — Classical ML baselines for comparison against PhishingTransformer.

Trains and evaluates 3 baseline models on the same train/test splits:
  1. Logistic Regression + TF-IDF
  2. Random Forest + TF-IDF
  3. Random Forest + Handcrafted Features (26-dim)

All results saved to results/baseline_results.json
Comparison table printed to terminal.

Usage:
  python baselines.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import json
import time
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from features import FeatureExtractor

cfg = Config()


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_text(records):
    """Combine subject + body into single string for TF-IDF."""
    texts  = []
    labels = []
    for r in records:
        subject = str(r.get("subject", "") or "").strip()
        body    = str(r.get("body",    "") or "").strip()
        texts.append(f"{subject} {body}".strip())
        labels.append(r["label"])
    return texts, labels


def prepare_features(records, extractor):
    """Extract 26 handcrafted features for each record."""
    features = []
    labels   = []
    for r in tqdm(records, desc="  Extracting features"):
        vec = extractor.extract(
            r.get("subject", ""),
            r.get("body",    ""),
        )
        features.append(vec)
        labels.append(r["label"])
    return np.array(features), labels


# ─────────────────────────────────────────────
#  EVALUATION HELPER
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, label_names):
    preds    = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    per_class = f1_score(y_test, preds, average=None, zero_division=0)

    report = classification_report(
        y_test, preds,
        target_names=label_names,
        digits=4,
        output_dict=True,
    )

    result = {
        "accuracy":  accuracy,
        "macro_f1":  macro_f1,
    }
    for i, name in enumerate(label_names):
        result[f"f1_{name}"] = float(per_class[i]) if i < len(per_class) else 0.0

    result["classification_report"] = report
    return result, preds


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_baselines(config=cfg):
    print(f"\n{'='*55}")
    print(f"  BASELINE MODELS")
    print(f"{'='*55}")

    label_names = config.labels.labels
    label2id    = config.labels.label2id

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n[1/5] Loading data...")
    train_records = load_jsonl(config.paths.train_file)
    test_records  = load_jsonl(config.paths.test_file)

    print(f"  Train: {len(train_records)} records")
    print(f"  Test : {len(test_records)} records")

    # Text + labels for TF-IDF models
    train_texts, train_labels = prepare_text(train_records)
    test_texts,  test_labels  = prepare_text(test_records)

    all_results = {}

    # ── Baseline 1: Logistic Regression + TF-IDF ─────────────────
    print(f"\n[2/5] Training Logistic Regression + TF-IDF...")
    t0 = time.time()

    lr_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features   = 50_000,
            ngram_range    = (1, 2),
            sublinear_tf   = True,
            min_df         = 2,
            strip_accents  = "unicode",
        )),
        ("clf", LogisticRegression(
            max_iter       = 1000,
            C              = 1.0,
            class_weight   = "balanced",
            random_state   = 42,
            solver         = "lbfgs",
            multi_class    = "multinomial",
        )),
    ])

    lr_pipeline.fit(train_texts, train_labels)
    lr_result, lr_preds = evaluate_model(
        lr_pipeline, test_texts, test_labels, label_names
    )
    lr_result["train_time_s"] = round(time.time() - t0, 2)
    all_results["logistic_regression_tfidf"] = lr_result

    print(f"  Done in {lr_result['train_time_s']}s")
    print(f"  Accuracy : {lr_result['accuracy']:.4f}")
    print(f"  Macro F1 : {lr_result['macro_f1']:.4f}")
    for name in label_names:
        print(f"    {name:<20}: {lr_result[f'f1_{name}']:.4f}")

    # ── Baseline 2: Random Forest + TF-IDF ───────────────────────
    print(f"\n[3/5] Training Random Forest + TF-IDF...")
    t0 = time.time()

    # from sklearn.pipeline import Pipeline
    from scipy.sparse import hstack

    rf_tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features  = 30_000,
            ngram_range   = (1, 2),
            sublinear_tf  = True,
            min_df        = 2,
            strip_accents = "unicode",
        )),
        ("clf", RandomForestClassifier(
            n_estimators  = 300,
            max_depth     = None,
            class_weight  = "balanced",
            random_state  = 42,
            n_jobs        = -1,
        )),
    ])

    rf_tfidf_pipeline.fit(train_texts, train_labels)
    rf_tfidf_result, _ = evaluate_model(
        rf_tfidf_pipeline, test_texts, test_labels, label_names
    )
    rf_tfidf_result["train_time_s"] = round(time.time() - t0, 2)
    all_results["random_forest_tfidf"] = rf_tfidf_result

    print(f"  Done in {rf_tfidf_result['train_time_s']}s")
    print(f"  Accuracy : {rf_tfidf_result['accuracy']:.4f}")
    print(f"  Macro F1 : {rf_tfidf_result['macro_f1']:.4f}")
    for name in label_names:
        print(f"    {name:<20}: {rf_tfidf_result[f'f1_{name}']:.4f}")

    # ── Baseline 3: Random Forest + Handcrafted Features ─────────
    print(f"\n[4/5] Training Random Forest + Handcrafted Features...")
    print(f"  (extracting 26 features per email — may take a few minutes)")
    t0 = time.time()

    extractor = FeatureExtractor(config=config, use_perplexity=False)

    print(f"\n  Training set:")
    train_feats, train_feat_labels = prepare_features(train_records, extractor)
    print(f"  Test set:")
    test_feats,  test_feat_labels  = prepare_features(test_records,  extractor)

    scaler = StandardScaler()
    train_feats_scaled = scaler.fit_transform(train_feats)
    test_feats_scaled  = scaler.transform(test_feats)

    rf_feat_clf = RandomForestClassifier(
        n_estimators = 500,
        max_depth    = None,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )
    rf_feat_clf.fit(train_feats_scaled, train_feat_labels)

    class Wrapper:
        def __init__(self, clf): self.clf = clf
        def predict(self, X): return self.clf.predict(X)

    rf_feat_result, _ = evaluate_model(
        Wrapper(rf_feat_clf),
        test_feats_scaled, test_feat_labels, label_names
    )
    rf_feat_result["train_time_s"] = round(time.time() - t0, 2)

    # Feature importance
    importances = rf_feat_clf.feature_importances_
    feat_names  = [
        "url_count", "has_url", "url_mismatch", "suspicious_tld",
        "urgency_score", "authority_score", "threat_score", "reward_score",
        "subject_len", "body_len", "exclamation_count", "caps_ratio",
        "punctuation_density", "has_greeting", "greeting_specificity",
        "has_signature", "avg_word_len", "avg_sentence_len",
        "type_token_ratio", "has_action_request", "credential_request",
        "spoofing_indicators", "attachment_mention",
        "perplexity", "perplexity_bucket", "fluency_anomaly",
    ]
    top_features = sorted(
        zip(feat_names, importances),
        key=lambda x: x[1], reverse=True
    )[:10]
    rf_feat_result["top_10_features"] = [
        {"feature": n, "importance": round(float(v), 4)}
        for n, v in top_features
    ]
    all_results["random_forest_handcrafted"] = rf_feat_result

    print(f"  Done in {rf_feat_result['train_time_s']}s")
    print(f"  Accuracy : {rf_feat_result['accuracy']:.4f}")
    print(f"  Macro F1 : {rf_feat_result['macro_f1']:.4f}")
    for name in label_names:
        print(f"    {name:<20}: {rf_feat_result[f'f1_{name}']:.4f}")
    print(f"\n  Top 10 most important features:")
    for feat in rf_feat_result["top_10_features"]:
        print(f"    {feat['feature']:<25}: {feat['importance']:.4f}")

    # ── Summary comparison table ──────────────────────────────────
    print(f"\n[5/5] Summary comparison table")
    print(f"\n{'='*75}")
    print(f"  {'Model':<35} {'Accuracy':>9} {'Macro F1':>9} {'Spear-F1':>9}")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*9}")

    model_display = {
        "logistic_regression_tfidf":   "LR + TF-IDF",
        "random_forest_tfidf":         "RF + TF-IDF",
        "random_forest_handcrafted":   "RF + Handcrafted Features",
    }

    for key, display in model_display.items():
        r = all_results[key]
        print(f"  {display:<35} "
              f"{r['accuracy']:>9.4f} "
              f"{r['macro_f1']:>9.4f} "
              f"{r[f'f1_spear-phishing']:>9.4f}")

    # Add our model for comparison
    results_path = os.path.join(
        config.paths.results_dir, "training_results_full_model.json"
    )
    if os.path.exists(results_path):
        with open(results_path) as f:
            our_results = json.load(f)
        tm = our_results["test_metrics"]
        print(f"  {'Custom DeBERTa Hybrid (ours)':<35} "
              f"{tm['accuracy']:>9.4f} "
              f"{tm['macro_f1']:>9.4f} "
              f"{tm['f1_spear-phishing']:>9.4f}")

    print(f"{'='*75}\n")

    # ── Save all results ──────────────────────────────────────────
    os.makedirs(config.paths.results_dir, exist_ok=True)
    save_path = os.path.join(config.paths.results_dir, "baseline_results.json")

    # Remove non-serializable classification_report nested dicts
    save_results = {}
    for key, val in all_results.items():
        save_results[key] = {
            k: v for k, v in val.items()
            if k != "classification_report"
        }

    with open(save_path, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"  Results saved → {save_path}")
    print(f"\n  Next step: python evaluate.py")

    return all_results


if __name__ == "__main__":
    run_baselines()