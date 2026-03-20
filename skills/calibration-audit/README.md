# GO Calibration Audit

## The Problem

We have a multi-task classifier that detects customer complaints, flags regulatory test failures, and identifies vulnerable customers. It's accurate — but we suspect the confidence scores are off.

Downstream systems depend on these scores: dashboards, alert thresholds, prioritization queues, regulatory reporting. If the confidence scores don't mean what they say, none of those systems work properly.

**Our job: figure out whether the scores are trustworthy, understand why or why not, fix them if needed, and write up what to ship.**

## Setup

```bash
git clone git@github.com:DataEQ/go-ml.git
cd go-ml/skills/calibration-audit
pip install pandas numpy scikit-learn matplotlib
```

## Getting Started

1. Open this repo in your editor with your preferred AI coding agent
2. Run `python scripts/baseline_metrics.py` to see the current state
3. Your agent will pick up `SKILL.md` and guide you through 4 phases:

| Phase | What | Goal |
|-------|------|------|
| 0 | Orientation | Understand the baseline and why the scores are off |
| 1 | Calibration Diagnosis | Measure how far off the confidence scores are |
| 2 | Threshold Optimization | Find per-head decision boundaries that actually work |
| 3 | Calibration Fix | Make the confidence scores mean what they say |
| 4 | Recommendation | Write up what to deploy |

## Files

```
├── SKILL.md                          # Agent instructions + walkthrough
├── scripts/
│   ├── baseline_metrics.py           # Run first — shows current metrics
│   └── helpers.py                    # Data loading + plotting (provided)
└── references/
    ├── predictions.csv               # 500 model predictions with ground truth
    └── model_architecture.md         # How the model was trained
```

## What You'll Need

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`
- An AI coding agent (Claude Code, Codex, Cursor, etc.)
- No GPU, no model weights, no external APIs — everything is self-contained
