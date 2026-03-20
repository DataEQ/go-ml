"""Baseline classifier — run this first to see how a simple prompt performs.

Uses a straightforward classification prompt against the signal dataset.
Results are saved to outputs/ for comparison with your variants.

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.

Usage:
    python scripts/baseline_classify.py --provider openai --model gpt-4o-mini
    python scripts/baseline_classify.py --provider anthropic --model claude-sonnet-4-20250514
"""

import argparse
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from helpers import load_signals, load_labeled_signals, save_results, evaluate_binary, print_eval_summary

BASELINE_PROMPT = """You are a customer complaint classifier for a UK financial services firm.

Analyze the following customer message and determine:
1. Is this a complaint? (A complaint is any expression of dissatisfaction that expects or implies resolution is needed)
2. What is the sentiment? (negative/neutral/positive)
3. How severe is this on a scale of 0-100?

Customer message:
{text}

Channel: {channel}

Respond in JSON:
{{
  "is_complaint": true/false,
  "sentiment": "negative"/"neutral"/"positive",
  "severity": 0-100,
  "reasoning": "brief explanation"
}}"""


def classify_with_openai(signals, model, prompt_template):
    """Classify signals using OpenAI API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("pip install openai")
        sys.exit(1)
    
    client = OpenAI()
    results = []
    
    for i, signal in enumerate(signals):
        prompt = prompt_template.format(text=signal["text"], channel=signal["channel"])
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            parsed["id"] = signal["id"]
            results.append(parsed)
            print(f"  [{i+1}/{len(signals)}] {signal['id']}: complaint={parsed.get('is_complaint')}")
        except Exception as e:
            print(f"  [{i+1}/{len(signals)}] {signal['id']}: ERROR — {e}")
            results.append({"id": signal["id"], "is_complaint": None, "error": str(e)})
        
        time.sleep(0.2)  # rate limit courtesy
    
    return results


def classify_with_anthropic(signals, model, prompt_template):
    """Classify signals using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("pip install anthropic")
        sys.exit(1)
    
    client = anthropic.Anthropic()
    results = []
    
    for i, signal in enumerate(signals):
        prompt = prompt_template.format(text=signal["text"], channel=signal["channel"])
        
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text
            # Strip markdown code fences if present
            if content.strip().startswith("```"):
                content = content.strip().split("\n", 1)[1].rsplit("```", 1)[0]
            parsed = json.loads(content)
            parsed["id"] = signal["id"]
            results.append(parsed)
            print(f"  [{i+1}/{len(signals)}] {signal['id']}: complaint={parsed.get('is_complaint')}")
        except Exception as e:
            print(f"  [{i+1}/{len(signals)}] {signal['id']}: ERROR — {e}")
            results.append({"id": signal["id"], "is_complaint": None, "error": str(e)})
        
        time.sleep(0.3)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline classification on signal dataset")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--prompt", help="Override prompt template (path to .txt file)")
    args = parser.parse_args()
    
    signals = load_signals()
    ground_truth = load_labeled_signals()
    
    prompt_template = BASELINE_PROMPT
    if args.prompt:
        with open(args.prompt) as f:
            prompt_template = f.read()
    
    print(f"Classifying {len(signals)} signals with {args.provider}/{args.model}...")
    
    if args.provider == "openai":
        results = classify_with_openai(signals, args.model, prompt_template)
    else:
        results = classify_with_anthropic(signals, args.model, prompt_template)
    
    # Save raw results
    run_name = f"baseline_{args.provider}_{args.model.replace('/', '_')}"
    save_results({
        "run_name": run_name,
        "provider": args.provider,
        "model": args.model,
        "prompt_template": prompt_template,
        "predictions": results,
    }, f"{run_name}.json")
    
    # Evaluate
    eval_result = evaluate_binary(results, ground_truth)
    print_eval_summary(f"Baseline ({args.provider}/{args.model})", eval_result)


if __name__ == "__main__":
    main()
