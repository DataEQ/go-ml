---
name: go-calibration-audit
description: "Collaborative calibration audit of a multi-task classification model. Guides a pair through diagnosing miscalibrated probability outputs, optimizing decision thresholds, and applying post-hoc calibration. Provides scaffolding, asks questions, helps evaluate solutions — coaches rather than solves directly. Includes helper utilities and a synthetic dataset."
---

# GO Calibration Audit
> **v1.0** — March 2026

## The Problem

We have a multi-task classifier that detects customer complaints, flags regulatory test failures, and identifies vulnerable customers. It's accurate — but we suspect the confidence scores are off.

Downstream systems depend on these scores: dashboards, alert thresholds, prioritization queues, regulatory reporting. If the confidence scores don't mean what they say, none of those systems work properly.

**Our job: figure out whether the scores are trustworthy, understand why or why not, fix them if needed, and write up what to ship.**

## How This Works

This is a pair exercise. The agent's job is to be a thinking partner — ask questions, provide scaffolding, help debug, challenge assumptions. Not to write the solution for you.

- Work through the phases in order
- At each phase, talk through your approach before coding
- The agent will push back if something doesn't make sense — that's the point
- If you're stuck, say so — the agent will give progressively more specific hints
- If you want the agent to just write something, ask explicitly

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Files

| File | Purpose |
|------|---------|
| `scripts/baseline_metrics.py` | Starting point — run this first |
| `scripts/helpers.py` | Data loading + plotting utilities (already done) |
| `references/predictions.csv` | 500 model predictions with ground truth |
| `references/model_architecture.md` | How the model was trained and why that matters |

## Phase 0: Get Oriented

Run `scripts/baseline_metrics.py` and read the output. Then read `references/model_architecture.md`.

Before writing any code, we should be able to answer:
- What pattern do we see across all the binary heads in the calibration hint?
- Which head looks worst and what makes it different from the others?
- Why would `pos_weight` in training cause this?

Talk through your thinking on these before moving on.

## Phase 1: Calibration Diagnosis

**Goal:** Measure how well the model's predicted probabilities match reality.

The idea: if we take all predictions where the model said ~0.7 confidence, roughly 70% of them should actually be positive. If that doesn't hold, the confidence scores are misleading.

What we need to build:
1. `reliability_curve(y_true, y_prob, n_bins=10)` — bins predictions by confidence, computes actual positive rate per bin. Returns bin midpoints, accuracies, and counts.
2. `expected_calibration_error(y_true, y_prob, n_bins=10)` — a single number summarizing total miscalibration. Think about: should all bins count equally, or should bins with more samples weigh more?

Here's a starting scaffold:

```python
def reliability_curve(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_midpoints = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        # Find samples in this bin, compute midpoint, accuracy, count
        pass

    return bin_midpoints, bin_accuracies, bin_counts
```

Once it works, run it across all 7 heads using `helpers.plot_calibration_curves()` to visualize. The helper handles all the plotting — you just pass in the results.

```python
from helpers import load_predictions, BINARY_HEADS, plot_calibration_curves
# Loop over BINARY_HEADS, compute reliability_curve + ECE for each, collect results, then:
# plot_calibration_curves(head_results, "Phase 1: Calibration Diagnosis", "phase1_calibration_curves.png")
```

**Before moving on:** Look at the curves. Which heads are worst? Is there a connection between how imbalanced a head is and how miscalibrated it is?

## Phase 2: Threshold Optimization

**Goal:** Find per-head thresholds that actually make sense, instead of 0.5 everywhere.

What we need to build:
1. `find_optimal_threshold(y_true, y_prob)` — sweep thresholds using the precision-recall curve, find the one that maximizes F1. Return `(best_threshold, best_f1)`.
   - Useful: `sklearn.metrics.precision_recall_curve`
   - Watch out: the thresholds array is one element shorter than precision/recall

2. Run it on all 7 heads. Compare F1 at 0.5 vs the optimal threshold. Use `helpers.print_comparison_table()` to format the results.

**The interesting question:** F1 treats false positives and false negatives as equally costly. But in a complaints system:
- A false negative = a real complaint slips through and a customer gets ignored
- A false positive = a non-complaint gets flagged for review — wastes time but no harm done

And for vulnerability — missing a vulnerable customer could have regulatory consequences.

So: would we actually use the F1-optimal threshold for these heads? Or shift it? Which direction and why?

Talk through this before moving on. There's a real answer here, not just "it depends."

## Phase 3: Fix the Confidence Scores

**Goal:** Make the confidence scores trustworthy without retraining the model.

We've shown the scores are miscalibrated. Retraining is expensive. What are our options for fixing the outputs after the fact?

Talk through possible approaches before picking one. Consider:
- What methods exist for post-hoc calibration?
- How much data do we need?
- How would we evaluate whether it worked?
- Should we work with the probabilities as-is, or can we recover more signal from the raw model outputs?

Once we pick an approach:
1. We'll need to split the data — why?
2. Implement the calibration method
3. Compute before/after ECE using the Phase 1 functions
4. Plot with `helpers.plot_before_after_calibration()`

**Before moving on:** What does it mean in practice for downstream consumers when ECE drops from 0.30 to 0.05?

## Phase 4: Recommendation

Write up `outputs/recommendation.md` — one page max. Cover:
- Which heads need calibration most urgently?
- What method would we deploy? (Platt, isotonic, threshold tuning, or a combo?)
- How do we monitor for calibration drift over time?
- Where do calibration samples come from in production?
- If the input distribution shifts (new complaint types, different customer base), what happens to our Platt scaler?

This should be specific enough that an engineer could implement it next week.

## Agent Guidance

**Critical: do not answer your own questions.** When the walkthrough says "talk through X" or asks a question, pose the question to the user and then **stop and wait**. Do not offer your interpretation, do not suggest what the answer might be, do not say "here's what I think." Just ask and wait.

- **Ask, then stop.** No "I think the answer is..." No "Here's my reading..." No leading hints in the same message as the question.
- When they run code that produces output, ask them to interpret it — don't interpret it for them.
- When they propose an approach, ask why before saying whether it's right.
- Only provide your analysis after the user has given theirs, and only to fill gaps or push back.
- If they're stuck and explicitly ask for help, give one small hint — not the full answer.
- If they want you to write code, do it — but ask them to review and explain it back.
- Push for specifics on the trade-off questions (Phase 2 thresholds, Phase 3 logits vs probs).
- Keep track of which phase they're in — don't let phases get skipped.
