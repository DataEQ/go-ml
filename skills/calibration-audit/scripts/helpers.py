"""Helper utilities for the calibration audit.

These are provided for you — no TODOs here. Use them in your phase scripts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "..", "references", "predictions.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")

BINARY_HEADS = [
    {"name": "is_complaint",  "prob": "prob_is_complaint",  "label": "label_is_complaint"},
    {"name": "test1pass",     "prob": "prob_test1pass",     "label": "label_test1pass"},
    {"name": "test2pass",     "prob": "prob_test2pass",     "label": "label_test2pass"},
    {"name": "test3pass",     "prob": "prob_test3pass",     "label": "label_test3pass"},
    {"name": "test4pass",     "prob": "prob_test4pass",     "label": "label_test4pass"},
    {"name": "test5pass",     "prob": "prob_test5pass",     "label": "label_test5pass"},
    {"name": "vulnerability", "prob": "prob_vulnerability", "label": "label_vulnerability"},
]


def load_predictions():
    """Load the predictions CSV and return a DataFrame."""
    return pd.read_csv(PREDICTIONS_PATH)


def ensure_output_dir():
    """Create the outputs directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_calibration_curves(head_results, title, filename):
    """Plot reliability diagrams for multiple heads.
    
    Args:
        head_results: list of dicts with keys:
            - name: head name
            - bin_midpoints: array of bin center values (x axis)
            - bin_accuracies: array of actual positive rates per bin (y axis)
            - bin_counts: array of sample counts per bin
            - ece: Expected Calibration Error (float)
        title: plot title
        filename: output filename (saved to outputs/)
    """
    ensure_output_dir()
    n_heads = len(head_results)
    cols = 4
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for idx, result in enumerate(head_results):
        ax = axes[idx]
        midpoints = result["bin_midpoints"]
        accuracies = result["bin_accuracies"]
        counts = result["bin_counts"]

        # Bar chart of actual accuracy per bin
        bar_width = 0.8 / len(midpoints) if len(midpoints) > 0 else 0.08
        ax.bar(midpoints, accuracies, width=bar_width, alpha=0.7, color="#4C72B0", label="Actual")
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Actual positive rate")
        ax.set_title(f"{result['name']}\nECE={result['ece']:.3f}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8)

        # Add sample counts as text
        for mp, acc, cnt in zip(midpoints, accuracies, counts):
            if cnt > 0:
                ax.text(mp, acc + 0.03, f"n={int(cnt)}", ha="center", fontsize=6, color="gray")

    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_before_after_calibration(before_results, after_results, filename):
    """Plot before/after calibration curves side by side per head.
    
    Args:
        before_results: list of dicts (same format as plot_calibration_curves)
        after_results: list of dicts (same format, after calibration)
        filename: output filename
    """
    ensure_output_dir()
    n_heads = len(before_results)
    fig, axes = plt.subplots(2, n_heads, figsize=(3.5 * n_heads, 7))
    if n_heads == 1:
        axes = axes.reshape(2, 1)

    for idx in range(n_heads):
        for row, (results, label) in enumerate([(before_results, "Before"), (after_results, "After")]):
            ax = axes[row, idx]
            r = results[idx]
            bar_width = 0.8 / len(r["bin_midpoints"]) if len(r["bin_midpoints"]) > 0 else 0.08
            color = "#D65F5F" if row == 0 else "#4C72B0"
            ax.bar(r["bin_midpoints"], r["bin_accuracies"], width=bar_width, alpha=0.7, color=color)
            ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"{label}: {r['name']}\nECE={r['ece']:.3f}", fontsize=10)
            if idx == 0:
                ax.set_ylabel("Actual positive rate")
            if row == 1:
                ax.set_xlabel("Predicted probability")

    fig.suptitle("Calibration: Before vs After Platt Scaling", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def print_comparison_table(rows):
    """Print a formatted comparison table.
    
    Args:
        rows: list of dicts with keys matching the header
    """
    if not rows:
        return
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    
    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "  ".join("-" * col_widths[h] for h in headers)
    
    print(header_line)
    print(separator)
    for row in rows:
        print("  ".join(str(row[h]).ljust(col_widths[h]) for h in headers))
