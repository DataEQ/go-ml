# GO Prompt Sensitivity Audit

## The Problem

We use LLMs to classify customer signals — detecting complaints, identifying vulnerability, scoring severity. It works, but we don't know how fragile it is.

Small prompt changes sometimes flip classifications: reword the system instruction, reorder the few-shot examples, change "classify" to "determine" — and suddenly edge cases go the other way. We need to understand how sensitive our classifier is to prompt variations, find the failure modes, and build a more robust prompt.

**Our job: systematically test prompt variations against a labeled dataset, measure what breaks, understand why, and produce a prompt that's demonstrably more robust than what we started with.**

## Setup

```bash
git clone git@github.com:DataEQ/go-ml.git
cd go-ml/skills/prompt-sensitivity
pip install openai anthropic pandas
```

You'll need an API key: set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Getting Started

1. Open this repo in your editor with your preferred AI coding agent
2. Run the baseline classifier:
   ```bash
   python scripts/baseline_classify.py --provider openai --model gpt-4o-mini
   ```
3. Your agent will pick up `SKILL.md` and guide you through 4 phases:

| Phase | What | Goal |
|-------|------|------|
| 0 | Establish Baseline | Run the starter prompt and understand where it fails |
| 1 | Design Variations | Create hypothesis-driven prompt variants to test |
| 2 | Run Experiments | Execute variants and collect results |
| 3 | Analyze Failures | Find patterns in what's sensitive and why |
| 4 | Build Robust Prompt | Combine learnings into a better, tested prompt |

## Files

```
├── SKILL.md                          # Agent instructions + walkthrough
├── scripts/
│   ├── baseline_classify.py          # Run first — baseline prompt + classifier
│   └── helpers.py                    # Evaluation + comparison utilities (provided)
└── references/
    ├── signals.json                  # 30 customer signals (no labels)
    └── signals_labeled.json          # Same signals with ground truth
```

## What You'll Need

- Python 3.8+
- `openai` or `anthropic` Python package
- An API key for one of the above
- An AI coding agent (Claude Code, Codex, Cursor, etc.)
- 30 signals × a few variants = modest API costs (~$0.50 with gpt-4o-mini)
