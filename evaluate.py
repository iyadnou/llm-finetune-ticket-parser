"""
STEP 2 & 4 — Measure how good the model is.

Rule #1 of fine-tuning: ALWAYS measure before and after. Without a number,
you can't tell if fine-tuning helped, hurt, or did nothing.

Usage:
  python evaluate.py                          # base model (before)
  python evaluate.py --adapter adapters       # fine-tuned model (after)
  python evaluate.py --limit 20 --show 5      # quick run, print 5 examples

Metrics:
  - valid_json : % of answers that parse as JSON with the right keys
  - per-field accuracy : % of test tickets where that field is exactly right
  - all_correct : % of tickets where EVERY field is right (the strict one)
"""

import argparse
import json

from mlx_lm import generate, load

FIELDS = ["category", "urgency", "product", "order_id", "customer_name"]


def parse_json(text):
    """Models sometimes wrap JSON in ```json fences or add chatter. Be lenient."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None, help="path to LoRA adapters")
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=3, help="examples to print")
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = [json.loads(line) for line in open(args.data)]
    if args.limit:
        rows = rows[: args.limit]

    n_valid, n_all = 0, 0
    field_ok = {f: 0 for f in FIELDS}

    for i, row in enumerate(rows):
        system, user, gold_msg = row["messages"]
        gold = json.loads(gold_msg["content"])

        # Build the prompt exactly like during training (system + user),
        # and let the model write the assistant turn.
        prompt = tokenizer.apply_chat_template(
            [system, user], add_generation_prompt=True, tokenize=False
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=120)
        pred = parse_json(out)

        if pred is not None and all(k in pred for k in FIELDS):
            n_valid += 1
        pred = pred or {}
        correct = {f: pred.get(f) == gold[f] for f in FIELDS}
        for f in FIELDS:
            field_ok[f] += correct[f]
        n_all += all(correct.values())

        if i < args.show:
            print(f"\n--- example {i + 1} ---")
            print("INPUT :", user["content"])
            print("MODEL :", out.strip())
            print("GOLD  :", gold_msg["content"])
        print(f"\r{i + 1}/{len(rows)} evaluated", end="", flush=True)

    n = len(rows)
    print(f"\n\n=== Results ({'fine-tuned' if args.adapter else 'base'} model, {n} tickets) ===")
    print(f"valid_json     : {100 * n_valid / n:5.1f}%")
    for f in FIELDS:
        print(f"{f:15s}: {100 * field_ok[f] / n:5.1f}%")
    print(f"ALL CORRECT    : {100 * n_all / n:5.1f}%")


if __name__ == "__main__":
    main()
