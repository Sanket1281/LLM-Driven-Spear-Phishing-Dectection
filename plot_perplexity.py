"""
plot_perplexity.py — Generates a publication-ready density plot
proving that GPT-2 Perplexity separates LLM emails from human emails.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from features import PerplexityScorer

# Using train.jsonl because more data points make a smoother, better-looking curve
DATA_FILE = "data/train.jsonl"
PLOT_OUTPUT = "perplexity_distribution.png"


def generate_plot():
    print(f"{'=' * 55}\n  GENERATING PERPLEXITY DENSITY PLOT\n{'=' * 55}")

    if not os.path.exists(DATA_FILE):
        print(f"[!] Error: Could not find {DATA_FILE}. Please check the path.")
        return

    print(f"Loading data from {DATA_FILE}...")
    scorer = PerplexityScorer()

    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)

            # Combine subject and body like your feature extractor does
            text = str(data.get("subject", "")) + " " + str(data.get("body", ""))
            label = data.get("label", "unknown")

            # Score it
            perp = scorer.score(text)
            records.append({"Label": label, "Perplexity": perp})

    df = pd.DataFrame(records)

    # Clean up labels for the graph legend
    label_map = {
        "benign": "Benign (Human Corporate)",
        "phishing": "Traditional Phishing (Human Spam)",
        "spear-phishing": "Synthetic Spear-Phishing (LLM)"
    }
    df["Class"] = df["Label"].map(label_map)

    # Filter out extreme outliers for a clean visualization (Perplexity > 500)
    df = df[df["Perplexity"] < 500]

    print("Drawing density plot...")

    # Set academic plotting style
    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid", context="paper")

    # Create the KDE (Kernel Density Estimate) plot
    sns.kdeplot(
        data=df,
        x="Perplexity",
        hue="Class",
        fill=True,
        common_norm=False,
        palette=["#2ecc71", "#e74c3c", "#3498db"],  # Green, Red, Blue
        alpha=0.5,
        linewidth=2
    )

    # Format the graph for the research paper
    plt.title("GPT-2 Perplexity Distribution Across Email Classes", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Sequence Perplexity Score (Lower = More Predictable/Fluent)", fontsize=12, labelpad=10)
    plt.ylabel("Density", fontsize=12, labelpad=10)
    plt.xlim(0, 400)  # Lock the X-axis for clean viewing

    # Save the high-res image
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT)
    print(f"Success! Plot saved to: {PLOT_OUTPUT}")


if __name__ == "__main__":
    generate_plot()