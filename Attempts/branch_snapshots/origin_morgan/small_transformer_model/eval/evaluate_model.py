"""
evaluate_model.py
=================
Tests a trained (fine-tuned) Qwen checkpoint on the actual hackathon tasks,
not just loss/perplexity. Mirrors the metric definitions in
training_data/generation_rules.md §5.2.

Tasks evaluated here (both supported by the valid full sequences we have):
  - Task 1  Next-step prediction : Top-1 / Top-3 / Top-5 / MRR   (via beam search)
  - Task 2  Sequence completion  : Exact Match / Token Acc /
                                   Normalized Edit Distance        (via greedy decode)

Reported per split (id_val, ood_test), per family, and at each cut fraction,
plus the ID -> OOD drop for each headline metric.

Anomaly detection (Task 3) is NOT evaluated here: the val/ood files contain only
valid sequences, so there is nothing to detect. Building an injected-violation
set is a separate step (training_data/generate_sequences.py --validate can label it).

The model is the one produced by train_model.py, which tokenizes the pipe-joined
`prompt` string with the Qwen subword tokenizer. We therefore predict steps by
generating text and splitting on "|", matching how the model was trained.

Quick smoke test (few examples, runs on CPU):
    python evaluate_model.py --checkpoint ../checkpoints/best_model --limit 8 --num_beams 3

Full run (on a GPU node, inside a Slurm job):
    python evaluate_model.py --checkpoint ../checkpoints/best_model
"""

import os
import json
import time
import argparse
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="../checkpoints/best_model",
                   help="Path to a fine-tuned checkpoint dir, or a HF model id.")
    p.add_argument("--data_dir", default="../data",
                   help="Dir holding dataset_id_val.jsonl and dataset_ood_test.jsonl.")
    p.add_argument("--splits", nargs="+", default=["id_val", "ood_test"],
                   help="Which split files to evaluate (without the 'dataset_' prefix).")
    p.add_argument("--cut_fractions", nargs="+", type=float, default=[0.6, 0.8],
                   help="Where to cut each sequence (matches organizer COMPLETION_FRACTION).")
    p.add_argument("--topk", type=int, default=5,
                   help="k for Top-k next-step prediction.")
    p.add_argument("--num_beams", type=int, default=5,
                   help="Beam width for next-step prediction (>= topk).")
    p.add_argument("--next_step_max_tokens", type=int, default=16,
                   help="Max new tokens to generate for a single next step.")
    p.add_argument("--completion_token_margin", type=float, default=1.6,
                   help="Generate this multiple of the gold-remainder token length.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N sequences per split (smoke test).")
    p.add_argument("--out", default="eval_results.json",
                   help="Where to write the JSON metrics report.")
    p.add_argument("--device", default=None,
                   help="cuda / cpu. Defaults to cuda if available.")
    return p.parse_args()


# ── Data ─────────────────────────────────────────────────────────────────────

def load_records(path, limit=None):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    return records


def build_prefix_text(family, prefix_steps):
    """Reproduce the exact prompt formatting used in train_model.py."""
    return f"<FAMILY>{family}</FAMILY> " + " | ".join(prefix_steps)


def steps_from_generated(text):
    """Split a generated chunk into clean step strings."""
    return [s.strip() for s in text.split("|") if s.strip()]


# ── Metrics ──────────────────────────────────────────────────────────────────

def levenshtein(a, b):
    """Edit distance over two lists of tokens (steps)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalized_edit_distance(pred, gold):
    if not pred and not gold:
        return 0.0
    return levenshtein(pred, gold) / max(len(pred), len(gold), 1)


def token_accuracy(pred, gold):
    """Position-wise step accuracy over the gold length (missing preds count as wrong)."""
    if not gold:
        return 1.0 if not pred else 0.0
    correct = sum(1 for i, g in enumerate(gold) if i < len(pred) and pred[i] == g)
    return correct / len(gold)


# ── Generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_step_topk(model, tokenizer, prefix_text, device, args):
    """Return up to topk ranked candidate next-step strings via beam search."""
    prompt = prefix_text + " | "
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=args.next_step_max_tokens,
        num_beams=max(args.num_beams, args.topk),
        num_return_sequences=args.topk,
        do_sample=False,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]  # only newly generated tokens
    ranked = []
    for row in gen:
        text = tokenizer.decode(row, skip_special_tokens=True)
        steps = steps_from_generated(text)
        cand = steps[0] if steps else ""
        if cand and cand not in ranked:
            ranked.append(cand)
    return ranked


@torch.no_grad()
def predict_completion(model, tokenizer, prefix_text, gold_rest, tokenizer_for_len, device, args):
    """Greedy-decode the remainder of the sequence; return predicted step list."""
    prompt = prefix_text + " | "
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    gold_text = " | ".join(gold_rest)
    gold_tok_len = len(tokenizer_for_len(gold_text)["input_ids"]) if gold_rest else 8
    max_new = max(16, int(gold_tok_len * args.completion_token_margin))
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, enc["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True)
    pred = steps_from_generated(text)
    # Cap to gold length so a runaway generation doesn't inflate edit distance.
    if gold_rest:
        pred = pred[: len(gold_rest)]
    return pred


# ── Eval loop ────────────────────────────────────────────────────────────────

def evaluate_split(model, tokenizer, records, device, args):
    """Accumulate metrics, keyed by (family, cut_fraction)."""
    acc = defaultdict(lambda: {
        "n": 0, "top1": 0, "top3": 0, "top5": 0, "mrr": 0.0,
        "exact": 0, "tok_acc": 0.0, "ned": 0.0,
    })

    for rec in records:
        family = rec.get("family", "UNKNOWN")
        steps = rec["steps"]
        for frac in args.cut_fractions:
            cut = max(1, int(round(len(steps) * frac)))
            if cut >= len(steps):
                continue  # nothing left to predict
            prefix_steps = steps[:cut]
            gold_next = steps[cut]
            gold_rest = steps[cut:]
            prefix_text = build_prefix_text(family, prefix_steps)
            bucket = acc[(family, frac)]
            bucket["n"] += 1

            # Task 1 — next step
            preds = predict_next_step_topk(model, tokenizer, prefix_text, device, args)
            if preds:
                if preds[0] == gold_next:
                    bucket["top1"] += 1
                if gold_next in preds[:3]:
                    bucket["top3"] += 1
                if gold_next in preds[:5]:
                    bucket["top5"] += 1
                for rank, p in enumerate(preds, 1):
                    if p == gold_next:
                        bucket["mrr"] += 1.0 / rank
                        break

            # Task 2 — completion
            pred_rest = predict_completion(
                model, tokenizer, prefix_text, gold_rest, tokenizer, device, args
            )
            if pred_rest == gold_rest:
                bucket["exact"] += 1
            bucket["tok_acc"] += token_accuracy(pred_rest, gold_rest)
            bucket["ned"] += normalized_edit_distance(pred_rest, gold_rest)

    return acc


def finalize(acc):
    """Turn raw counts into rates, plus an overall row."""
    out = {}
    overall = {"n": 0, "top1": 0, "top3": 0, "top5": 0, "mrr": 0.0,
               "exact": 0, "tok_acc": 0.0, "ned": 0.0}
    for (family, frac), b in acc.items():
        n = max(b["n"], 1)
        out[f"{family}@{frac}"] = {
            "n": b["n"],
            "top1": b["top1"] / n, "top3": b["top3"] / n,
            "top5": b["top5"] / n, "mrr": b["mrr"] / n,
            "exact_match": b["exact"] / n,
            "token_acc": b["tok_acc"] / n,
            "norm_edit_dist": b["ned"] / n,
        }
        for key in overall:
            overall[key] += b[key]
    n = max(overall["n"], 1)
    out["OVERALL"] = {
        "n": overall["n"],
        "top1": overall["top1"] / n, "top3": overall["top3"] / n,
        "top5": overall["top5"] / n, "mrr": overall["mrr"] / n,
        "exact_match": overall["exact"] / n,
        "token_acc": overall["tok_acc"] / n,
        "norm_edit_dist": overall["ned"] / n,
    }
    return out


def main():
    args = get_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    report = {}
    for split in args.splits:
        path = os.path.join(args.data_dir, f"dataset_{split}.jsonl")
        if not os.path.exists(path):
            print(f"  ! skipping {split}: {path} not found")
            continue
        records = load_records(path, limit=args.limit)
        print(f"\n=== {split}: {len(records)} sequences "
              f"({'limited' if args.limit else 'full'}) ===")
        t0 = time.time()
        acc = evaluate_split(model, tokenizer, records, device, args)
        report[split] = finalize(acc)
        print(f"  done in {time.time() - t0:.0f}s")
        ov = report[split]["OVERALL"]
        print(f"  OVERALL  top1={ov['top1']:.3f}  top5={ov['top5']:.3f}  "
              f"mrr={ov['mrr']:.3f}  exact={ov['exact_match']:.3f}  "
              f"tok_acc={ov['token_acc']:.3f}  ned={ov['norm_edit_dist']:.3f}")

    # ID -> OOD drop on headline metrics
    if "id_val" in report and "ood_test" in report:
        idov, oodov = report["id_val"]["OVERALL"], report["ood_test"]["OVERALL"]
        report["generalization_drop_id_to_ood"] = {
            m: idov[m] - oodov[m]
            for m in ["top1", "top3", "top5", "mrr", "token_acc"]
        } | {"norm_edit_dist_increase": oodov["norm_edit_dist"] - idov["norm_edit_dist"]}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote metrics to {args.out}")


if __name__ == "__main__":
    main()
