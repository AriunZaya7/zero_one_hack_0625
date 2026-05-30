"""
llm.py
======
Inference wrapper around a (optionally fine-tuned / LoRA) HuggingFace causal LM.
Sequences are rendered as text with " -> " between steps:

    "RECEIVE WAFER LOT -> LOT IDENTIFICATION -> ..."

Provides the shared task interface so it is swappable with NGramModel and the
ProcessTransformer in run_eval.py.

Smoke test (downloads a small model — run on a login node with internet):
    python -m solution.models.llm --model sshleifer/tiny-gpt2
"""

from __future__ import annotations
import argparse
import math

SEP = " -> "


class LLMModel:
    def __init__(self, model_name: str, checkpoint_path: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        if checkpoint_path:  # LoRA adapter
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)
        self.model.to(self.device).eval()
        self.step_candidates = self._load_step_candidates()

    def _fmt(self, steps: list[str]) -> str:
        return SEP.join(steps)

    @staticmethod
    def _parse(text: str) -> list[str]:
        return [s.strip() for s in text.split("->") if s.strip()]

    @staticmethod
    def _load_step_candidates() -> list[str]:
        try:
            from solution.data.loader import load_all_families

            families = load_all_families("training_data")
            return sorted({step for seqs in families.values() for seq in seqs for step in seq})
        except Exception:
            return []

    def top_k_next(self, partial_sequence: list[str], k: int = 5) -> list[tuple[str, float]]:
        if self.step_candidates:
            return self._score_candidate_steps(partial_sequence, k)
        return self._generate_top_k_next(partial_sequence, k)

    def _score_candidate_steps(self, partial_sequence: list[str], k: int = 5) -> list[tuple[str, float]]:
        torch = self.torch
        prompt = self._fmt(partial_sequence) + SEP
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        scores = []
        with torch.no_grad():
            for step in self.step_candidates:
                enc = self.tokenizer(prompt + step, return_tensors="pt").to(self.device)
                labels = enc["input_ids"].clone()
                labels[:, :prompt_len] = -100
                out = self.model(**enc, labels=labels)
                if math.isfinite(float(out.loss.item())):
                    scores.append((step, -float(out.loss.item())))
        ranked = sorted(scores, key=lambda item: item[1], reverse=True)[:k]
        if not ranked:
            return []
        max_score = max(score for _, score in ranked)
        weights = [math.exp(score - max_score) for _, score in ranked]
        denom = sum(weights) or 1.0
        return [(step, weight / denom) for (step, _), weight in zip(ranked, weights)]

    def _generate_top_k_next(self, partial_sequence: list[str], k: int = 5) -> list[tuple[str, float]]:
        torch = self.torch
        prompt = self._fmt(partial_sequence) + SEP
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=12, num_beams=max(k, 5),
                num_return_sequences=k, do_sample=False, early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        ranked: list[tuple[str, float]] = []
        for r, row in enumerate(gen):
            steps = self._parse(self.tokenizer.decode(row, skip_special_tokens=True))
            if steps and steps[0] not in [s for s, _ in ranked]:
                ranked.append((steps[0], 1.0 / (r + 1)))
        return ranked[:k]

    def complete_sequence(self, partial: list[str], max_new_steps: int = 200) -> list[str]:
        torch = self.torch
        prompt = self._fmt(partial) + SEP
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=max_new_steps * 8, do_sample=False, num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._parse(text)

    def sequence_log_prob(self, sequence: list[str]) -> float:
        torch = self.torch
        enc = self.tokenizer(self._fmt(sequence), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc, labels=enc["input_ids"])
        # HF returns mean NLL; negate to get mean log-prob (higher = more typical)
        return float(-out.loss.item())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2")
    args = ap.parse_args()
    m = LLMModel(args.model)
    demo = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "PRE CLEAN WAFER"]
    print("top-3 next:", m.top_k_next(demo, 3))
    print("logprob:", round(m.sequence_log_prob(demo), 3))
    print("llm OK")
