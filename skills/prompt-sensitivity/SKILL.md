---
name: go-prompt-sensitivity
description: "Collaborative prompt sensitivity audit for an LLM-based complaint classifier. Guides a pair through testing how prompt variations affect classification accuracy — rephrasing, few-shot examples, system prompt tone, output format, and more. Provides a labeled dataset, baseline classifier, and evaluation helpers. Coaches rather than solves directly."
---

# GO Prompt Sensitivity Audit
> **v1.0** — March 2026

## The Problem

We use LLMs to classify customer signals — detecting complaints, identifying vulnerability, scoring severity. It works, but we don't know how fragile it is.

Small prompt changes sometimes flip classifications: reword the system instruction, reorder the few-shot examples, change "classify" to "determine" — and suddenly edge cases go the other way. We need to understand how sensitive our classifier is to prompt variations, find the failure modes, and build a more robust prompt.

**Our job: systematically test prompt variations against a labeled dataset, measure what breaks, understand why, and produce a prompt that's demonstrably more robust than what we started with.**

## How This Works

This is a pair exercise. The agent's job is to be a thinking partner — ask questions, provide scaffolding, help debug, challenge assumptions. Not to write the solution for you.

- Work through the phases in order
- At each phase, talk through your approach before coding
- The agent will push back if something doesn't make sense
- If you're stuck, say so — progressively more specific hints are available
- If you want the agent to just write something, ask explicitly

## Setup

```bash
pip install openai anthropic   # at least one
pip install pandas              # for analysis
```

You'll need an API key: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

## Files

| File | Purpose |
|------|---------|
| `scripts/baseline_classify.py` | Starting point — runs baseline prompt against all signals |
| `scripts/helpers.py` | Data loading, evaluation, comparison utilities (provided) |
| `references/signals.json` | 30 customer signals (text + channel, no labels) |
| `references/signals_labeled.json` | Same signals with ground truth labels |

## Phase 0: Establish Baseline

Run the baseline classifier:

```bash
python scripts/baseline_classify.py --provider openai --model gpt-4o-mini
```

Look at the results. Before doing anything else:
- Which signals did the baseline get wrong? Why?
- Are the errors random, or is there a pattern?
- Look at the baseline prompt in `scripts/baseline_classify.py` — what's weak about it?

Talk through your observations before moving on.

## Phase 1: Hypothesis-Driven Variation Design

**Goal:** Design a set of prompt variations that test specific hypotheses about sensitivity.

Don't just randomly change words. Each variation should test a specific hypothesis. Some starting points to consider — but come up with your own too:

**Instruction framing:**
- Does it matter if you say "classify" vs "determine" vs "analyze"?
- What if the system prompt uses an authoritative tone vs a neutral one?
- Does adding "think step by step" change anything?

**Few-shot examples:**
- What happens when you add 2-3 examples to the prompt?
- Does the order of examples matter? (complaint first vs non-complaint first)
- What if all examples are clear-cut vs including an edge case?

**Definition sensitivity:**
- The baseline defines a complaint loosely. What if we use the FCA's formal definition?
- What if we explicitly list what is NOT a complaint?

**Output format:**
- JSON vs free text — does structured output change classification?
- What if you ask for reasoning before the classification vs after?

For each variation, write down:
1. What you're testing (the hypothesis)
2. What you expect to happen
3. The modified prompt

**Before coding anything:** talk through at least 3 hypotheses. The agent should push back on weak hypotheses and ask about the reasoning.

## Phase 2: Run the Experiments

**Goal:** Execute your prompt variants and collect results.

The baseline script supports a `--prompt` flag for custom prompt templates:

```bash
python scripts/baseline_classify.py --provider openai --model gpt-4o-mini --prompt my_variant.txt
```

Or modify the script to run multiple variants in sequence. The helpers have comparison tools:

```python
from helpers import compare_runs, print_comparison_summary
# compare_runs(variant_a_predictions, variant_b_predictions)
```

Things to think about while running:
- Are you controlling for everything except the one variable you're testing?
- Temperature is set to 0 — is that sufficient for determinism? 
- How many runs would you need to be confident a difference is real vs noise?

**Before moving on:** collect results for at least 3 variants plus the baseline.

## Phase 3: Analyze the Failure Modes

**Goal:** Understand *why* certain signals are sensitive and others aren't.

This is the interesting part. Look across all your runs:
- Which signals flip classification depending on the prompt? Those are your sensitive signals.
- Which signals are classified correctly every time? Those are robust.
- Is there a pattern? (e.g., polite complaints are sensitive, explicit complaints are robust)

Use `helpers.evaluate_binary()` and `helpers.compare_runs()` to quantify.

Questions to investigate:
- Do the sensitive signals share characteristics? (length, tone, ambiguity, channel?)
- For few-shot variants: does the choice of examples have more impact than the instruction wording?
- If you asked for reasoning-first (chain of thought), did the reasoning quality predict the classification accuracy?
- Are there signals where the LLM is right and the ground truth label is arguably wrong?

**Before moving on:** you should be able to say "these types of signals are fragile because X, and these prompt elements have the most impact on accuracy."

## Phase 4: Build a Robust Prompt

**Goal:** Combine what you learned into a single prompt that's demonstrably better than the baseline.

This isn't about making the biggest prompt — it's about making the most robust one. Consider:
- Which variations improved accuracy on the sensitive signals without hurting the robust ones?
- Can you combine techniques (e.g., better definition + few-shot + chain of thought)?
- What's the trade-off between prompt length/cost and robustness?

Evaluate your final prompt against the full dataset. Compare to baseline:
- Overall accuracy improvement
- Improvement specifically on the previously-sensitive signals
- Any regressions?

Write up `outputs/findings.md` covering:
- Which prompt elements had the biggest impact and why
- Your final prompt and why you chose each element
- What's still fragile — what class of signals would you want to test next?
- How would you set up ongoing sensitivity monitoring in production?

## Agent Guidance

**Critical: do not answer your own questions.** When the walkthrough asks a question, pose it to the user and then **stop and wait**. Do not offer your interpretation. Just ask and wait.

- **Ask, then stop.** No "I think the answer is..." No "Here's my reading..."
- When they run experiments and see results, ask them to interpret first.
- When they propose a hypothesis, ask them to predict the outcome before running it.
- Push back on hypotheses that aren't specific enough ("I want to try a different prompt" → "What specifically are you changing and what do you expect to happen?")
- If they design a variant that doesn't isolate a single variable, point that out.
- Only provide analysis after the user has given theirs.
- If they're stuck, give one hint — not the full answer.
- Keep track of which phase they're in — don't let phases get skipped.
