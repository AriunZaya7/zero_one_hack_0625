"""
finetune_llm.py
===============
Fine-tune a small HF causal LM (default Qwen2-0.5B) on text-rendered sequences
using LoRA. Sequences become "STEP -> STEP -> ... -> SHIP LOT".

Trains on IGBT + IC; MOSFET stays OOD. Saves a LoRA adapter that LLMModel loads
via `checkpoint_path`.

Run (GPU node):
    pixi run python -m solution.train.finetune_llm --model Qwen/Qwen2-0.5B \
        --out solution/checkpoints/llm_qwen --epochs 3

Smoke test (tiny model, CPU; downloads ~a few MB on a login node):
    pixi run python -m solution.train.finetune_llm --smoke
"""

from __future__ import annotations
import os
import argparse

from solution.data.loader import load_all_families, train_val_split

SEP = " -> "


def parse_families(values, available):
    if not values or values == ["all"]:
        return set(available)
    return {value.upper() for value in values}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="training_data")
    p.add_argument("--out", default="solution/checkpoints/llm_qwen")
    p.add_argument("--model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_families", nargs="+", default=["all"])
    p.add_argument("--wandb_project", default="zero-one-hack")
    p.add_argument("--wandb_run", default="llm_qwen_lora")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    if args.smoke:
        args.model = "sshleifer/tiny-gpt2"
        args.epochs, args.max_length = 1, 128

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              DataCollatorForLanguageModeling, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    fams = load_all_families(args.data_dir)
    train_families = parse_families(args.train_families, set(fams))
    train_texts, val_texts = [], []
    for fam, seqs in fams.items():
        if fam in train_families:
            tr, va = train_val_split(seqs, 0.1, args.seed)
            train_texts += [SEP.join(s) for s in tr]
            val_texts += [SEP.join(s) for s in va]
    if args.smoke:
        train_texts, val_texts = train_texts[:32], val_texts[:8]
    print(f"families={sorted(train_families)} train={len(train_texts)} val={len(val_texts)} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    ds_tr = Dataset.from_dict({"text": train_texts}).map(tokenize, batched=True, remove_columns=["text"])
    ds_va = Dataset.from_dict({"text": val_texts}).map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj"] if "gpt2" not in args.model else ["c_attn"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    report_to = []
    try:
        import wandb  # noqa
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        report_to = ["wandb"]
    except Exception:
        pass

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        warmup_steps=50, lr_scheduler_type="cosine", logging_steps=10,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        bf16=torch.cuda.is_available(), report_to=report_to, run_name=args.wandb_run,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"done. LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
