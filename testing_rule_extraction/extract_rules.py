"""
extract_rules.py
=================
Branch: angie
Folder: testing_rule_extraction/

Takes probe results from ngram_probe.py, sends them to Featherless AI,
and generates a learned_rules.md using the structured template.

Usage:
    set FEATHERLESS_API_KEY=your_key        (Windows CMD)
    $env:FEATHERLESS_API_KEY="your_key"     (PowerShell)

    python extract_rules.py --probe_results ./probe_results/ngram_probe_results.json
"""

import json
import os
import argparse
from openai import OpenAI


# ── Featherless client — initialised at module level so env var is read once ──
def get_client():
    api_key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "FEATHERLESS_API_KEY is not set.\n"
            "PowerShell:   $env:FEATHERLESS_API_KEY='your_key'\n"
            "CMD:          set FEATHERLESS_API_KEY=your_key\n"
            "Then re-run in the SAME terminal window."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.featherless.ai/v1",
    )


MODEL = os.environ.get("FEATHERLESS_MODEL", "deepseek-ai/DeepSeek-V3-0324")

# ── Templates ──────────────────────────────────────────────────────────────────

RULE_TEMPLATE = """## Learned Rule: [GIVE THE RULE A DESCRIPTIVE NAME]
**Observed behavior:** Model assigns X times higher perplexity when [describe the violation]
**Inferred constraint:** [step type A] must be preceded by [step type B] within [N] steps
**Confidence:** HIGH / MEDIUM / LOW
**Evidence:** valid_ppl=[X], invalid_ppl=[Y], ratio=[Z]x
**Rule type:** LOCAL (within a few steps) / GLOBAL (spans entire sequence)
"""

EXTRACTION_SYSTEM = """You are an expert in semiconductor manufacturing process sequences
and machine learning interpretability.
Your job is to analyze behavioral evidence from a trained sequence model
and infer what process rules the model has learned.
You will be given perplexity measurements. Higher perplexity = model more surprised.
High ratio (invalid/valid) = model learned that rule.
Do NOT make up rules not supported by the evidence. Be honest."""


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_extraction_prompt(probe_data: dict) -> str:
    model_info = f"""Model: N-gram (n={probe_data['n']})
Training sequences: {probe_data['train_sequences']}
Vocabulary size: {probe_data['vocab_size']}
Unique contexts: {probe_data['unique_contexts']}"""

    results_text = ""
    for r in probe_data["probe_results"]:
        results_text += f"""
---
Rule probe: {r['rule']}
Description: {r['description']}
Violation tested: {r['violation']}
Valid sequence: {' -> '.join(r['valid_seq'])}
Invalid sequence: {' -> '.join(r['invalid_seq'])}
Valid perplexity:   {r['valid_ppl']}
Invalid perplexity: {r['invalid_ppl']}
Perplexity ratio (invalid/valid): {r['ppl_ratio']}x
Assessment: {r['learned']}
"""

    return f"""I have trained an n-gram sequence model on semiconductor manufacturing
process sequences. I probed it with valid and rule-violated sequences.

MODEL INFORMATION:
{model_info}

PROBE RESULTS:
{results_text}

THRESHOLDS:
- ratio >= 2.0 = STRONG
- ratio 1.3-2.0 = PARTIAL
- ratio 1.05-1.3 = WEAK
- ratio ~1.0 = NOT LEARNED

Generate a learned_rules.md. For EACH rule probe use EXACTLY this template:

{RULE_TEMPLATE}

After all rules add these sections:

## Overall Assessment
[2-3 paragraphs on what the model learned, missed, and WHY n-gram structurally
fails certain rules]

## Rules Learned (summary list)

## Rules NOT Learned (list with reason)

## Key Insight: Local vs Global Rules
[Which rules needed local vs global context, how this maps to n-gram capability]

## What This Tells Us
[Does this model reproduce patterns or show genuine process logic understanding?]

Be specific and evidence-based."""


def build_comparison_prompt(learned_rules_md: str, ground_truth_md: str) -> str:
    return f"""Compare these two semiconductor process rule documents.

DOCUMENT 1 — LEARNED RULES (from n-gram behavioral probing, never saw ground truth):
{learned_rules_md}

DOCUMENT 2 — GROUND TRUTH RULES:
{ground_truth_md[:4000]}

Produce a comparison with:

1. Summary table:
| Rule ID | Ground Truth Says | Model Learned | Similarity (0-1) | Match? |

2. For each rule: ground truth spec, what model learned, gap analysis

3. Overall similarity score (0-1) with explanation

4. Final verdict: does the n-gram understand process LOGIC or just surface patterns?

Be direct and honest."""


# ── API call ───────────────────────────────────────────────────────────────────

def call_featherless(prompt: str, system: str = "", max_tokens: int = 4000) -> str:
    client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    print(f"  Calling Featherless model: {MODEL}")
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe_results",
                        default="./probe_results/ngram_probe_results.json")
    parser.add_argument("--ground_truth",
                        default="../training_data/generation_rules.md")
    parser.add_argument("--out_dir", default="./probe_results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Print key status for debugging
    key = os.environ.get("FEATHERLESS_API_KEY", "").strip()
    print(f"API key status: {'SET (length={})'.format(len(key)) if key else 'NOT SET'}")
    print(f"Model: {MODEL}")

    print(f"\nLoading: {args.probe_results}")
    with open(args.probe_results, encoding="utf-8") as f:
        probe_data = json.load(f)

    s = probe_data["summary"]
    print(f"n={probe_data['n']} | Strong:{s['strong']} Partial:{s['partial']} "
          f"NotLearned:{s['not_learned']} Score:{s['score_pct']}%")

    # ── Step 1: extract learned rules ─────────────────────────────────────────
    print("\nStep 1: Extracting learned rules...")
    print("(Model does NOT see generation_rules.md — purely behavioral evidence)")

    learned_md = call_featherless(
        build_extraction_prompt(probe_data),
        system=EXTRACTION_SYSTEM,
    )

    header = f"""# Learned Rules — N-gram Behavioral Probing

> **Model:** N-gram (n={probe_data['n']})
> **Training sequences:** {probe_data['train_sequences']}
> **Method:** Behavioral probing — perplexity on valid vs violated sequences
> **Note:** LLM never saw generation_rules.md — inferred purely from model behavior

---

"""
    learned_md = header + learned_md
    learned_path = os.path.join(args.out_dir, "learned_rules.md")
    with open(learned_path, "w", encoding="utf-8") as f:
        f.write(learned_md)
    print(f"Saved: {learned_path}")

    # ── Step 2: compare with ground truth ─────────────────────────────────────
    comparison_path = None
    if os.path.exists(args.ground_truth):
        print("\nStep 2: Comparing with ground truth...")
        with open(args.ground_truth, encoding="utf-8") as f:
            gt_md = f.read()
        comparison_md = call_featherless(
            build_comparison_prompt(learned_md, gt_md),
            system=EXTRACTION_SYSTEM,
        )
        full_comparison = f"""# Rules Comparison: Ground Truth vs N-gram Learned

> **Model:** N-gram (n={probe_data['n']})

---

{comparison_md}"""
        comparison_path = os.path.join(args.out_dir, "rules_comparison.md")
        with open(comparison_path, "w", encoding="utf-8") as f:
            f.write(full_comparison)
        print(f"Saved: {comparison_path}")
    else:
        print(f"[skip] Ground truth not found at {args.ground_truth}")

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        "model_type": "ngram", "n": probe_data["n"],
        "train_sequences": probe_data["train_sequences"],
        "summary": probe_data["summary"],
        "learned_rules_path": learned_path,
        "comparison_path": comparison_path,
    }
    with open(os.path.join(args.out_dir, "extraction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  learned_rules.md    -> {learned_path}")
    if comparison_path:
        print(f"  rules_comparison.md -> {comparison_path}")
    print(f"\nNext: python build_report.py "
          f"--probe_results {args.probe_results} "
          f"--learned_rules {learned_path}")


if __name__ == "__main__":
    main()