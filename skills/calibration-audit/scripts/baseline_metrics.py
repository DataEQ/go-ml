"""Compute baseline metrics for the complaints model predictions at threshold=0.5.

Run this first to see where the model stands before any calibration work.

Usage:
    python scripts/baseline_metrics.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, "..", "references", "predictions.csv")

BINARY_HEADS = [
    ("is_complaint", "prob_is_complaint", "label_is_complaint"),
    ("test1pass", "prob_test1pass", "label_test1pass"),
    ("test2pass", "prob_test2pass", "label_test2pass"),
    ("test3pass", "prob_test3pass", "label_test3pass"),
    ("test4pass", "prob_test4pass", "label_test4pass"),
    ("test5pass", "prob_test5pass", "label_test5pass"),
    ("vulnerability", "prob_vulnerability", "label_vulnerability"),
]

MULTICLASS_HEADS = [
    ("complaint_outcome", "complaint_outcome_pred_idx", "complaint_outcome_true_idx"),
    ("sentiment", "sentiment_pred_idx", "sentiment_true_idx"),
    ("tone", "tone_pred_idx", "tone_true_idx"),
    ("severity", "severity_pred_idx", "severity_true_idx"),
]


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    print(f"Loaded {len(df)} predictions from {PREDICTIONS_PATH}\n")

    # --- Binary heads ---
    print("=" * 70)
    print("BINARY HEADS (threshold = 0.5)")
    print("=" * 70)
    for name, prob_col, label_col in BINARY_HEADS:
        y_true = df[label_col].values
        y_prob = df[prob_col].values
        m = binary_metrics(y_true, y_prob)
        pos_rate = y_true.mean()
        mean_prob = y_prob.mean()
        print(f"\n  {name}")
        print(f"    Positive rate: {pos_rate:.1%}  |  Mean predicted prob: {mean_prob:.3f}")
        print(f"    Accuracy: {m['accuracy']:.3f}  |  F1: {m['f1']:.3f}  |  AUROC: {m['auroc']:.3f}")
        print(f"    Precision: {m['precision']:.3f}  |  Recall: {m['recall']:.3f}")

    # --- Multiclass heads ---
    print(f"\n{'=' * 70}")
    print("MULTICLASS HEADS")
    print("=" * 70)
    for name, pred_col, true_col in MULTICLASS_HEADS:
        y_true = df[true_col].values
        y_pred = df[pred_col].values
        acc = accuracy_score(y_true, y_pred)
        print(f"\n  {name}")
        print(f"    Accuracy: {acc:.3f}")

    # --- Regression ---
    print(f"\n{'=' * 70}")
    print("REGRESSION HEAD")
    print("=" * 70)
    y_true = df["conduct_score_true"].values
    y_pred = df["conduct_score_pred"].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"\n  conduct_score")
    print(f"    MAE: {mae:.2f}  |  RMSE: {rmse:.2f}")

    # --- Quick calibration hint ---
    print(f"\n{'=' * 70}")
    print("CALIBRATION HINT")
    print("=" * 70)
    print("\n  Compare mean predicted probability vs actual positive rate per head.")
    print("  A well-calibrated model should have these roughly equal.\n")
    for name, prob_col, label_col in BINARY_HEADS:
        pos_rate = df[label_col].mean()
        mean_prob = df[prob_col].mean()
        gap = mean_prob - pos_rate
        flag = " ⚠️" if abs(gap) > 0.10 else " ✓"
        print(f"    {name:20s}  actual={pos_rate:.3f}  predicted={mean_prob:.3f}  gap={gap:+.3f}{flag}")


if __name__ == "__main__":
    main()
