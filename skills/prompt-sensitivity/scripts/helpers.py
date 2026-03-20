"""Helper utilities for the prompt sensitivity audit.

Provided — no changes needed.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_PATH = os.path.join(SCRIPT_DIR, "..", "references", "signals.json")
LABELED_PATH = os.path.join(SCRIPT_DIR, "..", "references", "signals_labeled.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs")


def load_signals():
    """Load signals without labels (for classification)."""
    with open(SIGNALS_PATH) as f:
        return json.load(f)["signals"]


def load_labeled_signals():
    """Load signals with ground truth labels (for evaluation)."""
    with open(LABELED_PATH) as f:
        return json.load(f)["signals"]


def ensure_output_dir():
    """Create outputs directory if needed."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_results(results, filename):
    """Save classification results to outputs/."""
    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {path}")
    return path


def evaluate_binary(predictions, ground_truth, label_key="is_complaint"):
    """Evaluate binary classification accuracy.
    
    Args:
        predictions: list of dicts with 'id' and the label_key
        ground_truth: list of dicts with 'id' and 'labels' containing the label_key
        label_key: which label to evaluate
    
    Returns:
        dict with accuracy, precision, recall, f1, and per-signal details
    """
    truth_map = {s["id"]: s["labels"][label_key] for s in ground_truth}
    
    tp = fp = tn = fn = 0
    details = []
    
    for pred in predictions:
        sid = pred["id"]
        predicted = pred.get(label_key, pred.get("predicted", None))
        actual = truth_map.get(sid)
        
        if actual is None:
            continue
            
        correct = bool(predicted) == bool(actual)
        
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
        
        if not correct:
            details.append({
                "id": sid,
                "predicted": predicted,
                "actual": actual,
                "text_preview": next((s["text"][:80] for s in ground_truth if s["id"] == sid), ""),
            })
    
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {
        "accuracy": (tp + tn) / total if total else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "errors": details,
        "total": total,
    }


def compare_runs(run_a, run_b, label_key="is_complaint"):
    """Compare two classification runs and find where they disagree.
    
    Args:
        run_a: list of prediction dicts from variant A
        run_b: list of prediction dicts from variant B
        label_key: which label to compare
    
    Returns:
        dict with agreement stats and disagreement details
    """
    map_a = {p["id"]: p.get(label_key, p.get("predicted")) for p in run_a}
    map_b = {p["id"]: p.get(label_key, p.get("predicted")) for p in run_b}
    
    common_ids = set(map_a.keys()) & set(map_b.keys())
    agree = 0
    disagree = []
    
    for sid in sorted(common_ids):
        if bool(map_a[sid]) == bool(map_b[sid]):
            agree += 1
        else:
            disagree.append({
                "id": sid,
                "variant_a": map_a[sid],
                "variant_b": map_b[sid],
            })
    
    total = len(common_ids)
    return {
        "agreement_rate": agree / total if total else 0,
        "total_compared": total,
        "agreements": agree,
        "disagreements": len(disagree),
        "disagreement_details": disagree,
    }


def print_eval_summary(name, eval_result):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    c = eval_result["confusion"]
    print(f"  Accuracy:  {eval_result['accuracy']:.1%}")
    print(f"  Precision: {eval_result['precision']:.1%}")
    print(f"  Recall:    {eval_result['recall']:.1%}")
    print(f"  F1:        {eval_result['f1']:.1%}")
    print(f"  Confusion: TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}")
    
    if eval_result["errors"]:
        print(f"\n  Errors ({len(eval_result['errors'])}):")
        for err in eval_result["errors"]:
            direction = "FP" if err["predicted"] else "FN"
            print(f"    [{direction}] {err['id']}: {err['text_preview']}...")


def print_comparison_summary(name_a, name_b, comparison):
    """Pretty-print comparison between two runs."""
    print(f"\n{'=' * 60}")
    print(f"  {name_a} vs {name_b}")
    print(f"{'=' * 60}")
    print(f"  Agreement: {comparison['agreement_rate']:.1%} ({comparison['agreements']}/{comparison['total_compared']})")
    print(f"  Disagreements: {comparison['disagreements']}")
    
    if comparison["disagreement_details"]:
        print(f"\n  Disagreement details:")
        for d in comparison["disagreement_details"]:
            print(f"    {d['id']}: A={d['variant_a']} vs B={d['variant_b']}")
