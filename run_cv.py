"""
run_cv.py — Fast 3-Seed Cross-Validation Wrapper (Live Terminal Logging)
"""

import subprocess
import json
import os
import sys
import numpy as np

# Reduced to 3 standard academic seeds to save hardware time
SEEDS = [42, 123, 456]
RESULTS_FILE = "results/training_results_full_model.json"

def run_cross_validation():
    print("=" * 55)
    print("  STARTING 3-SEED CROSS-VALIDATION (LIVE OUTPUT)")
    print("=" * 55)

    macro_f1_scores = []
    spear_phishing_f1_scores = []

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n[{'='*15} RUN {i}/3 | SEED {seed} {'='*15}]")

        env = os.environ.copy()
        env["EXPERIMENT_SEED"] = str(seed)
        env["PYTHONUNBUFFERED"] = "1"

        # -u forces Python to print every log instantly without buffering
        subprocess.run([sys.executable, "-u", "train.py"], env=env)

        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                data = json.load(f)

            test_metrics = data.get("test_metrics", {})
            macro_f1 = test_metrics.get("macro_f1", 0) * 100
            spear_f1 = test_metrics.get("f1_spear-phishing", 0) * 100

            macro_f1_scores.append(macro_f1)
            spear_phishing_f1_scores.append(spear_f1)

            print(f"\n--> COMPLETED RUN {i}/3 | Seed {seed} | Macro F1: {macro_f1:.2f}% | Spear-Phish F1: {spear_f1:.2f}%")
        else:
            print(f"\n[!] Error: Could not locate {RESULTS_FILE} after Run {i}.")

    # Calculate final averages if runs were successful
    if macro_f1_scores:
        print("\n" + "=" * 55)
        print("  CROSS-VALIDATION FINAL RESULTS")
        print("=" * 55)
        print(f"Macro F1 Mean        : {np.mean(macro_f1_scores):.2f}% ± {np.std(macro_f1_scores):.2f}%")
        print(f"Spear-Phishing Mean  : {np.mean(spear_phishing_f1_scores):.2f}% ± {np.std(spear_phishing_f1_scores):.2f}%")

if __name__ == "__main__":
    run_cross_validation()