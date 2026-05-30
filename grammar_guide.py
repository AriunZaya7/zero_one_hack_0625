"""GrammarGuide: constrain next-step predictions to grammar-consistent choices.

Builds bigram + trigram successor sets from training sequences.
At inference, valid_next(prefix) returns ONLY the steps ever observed
to follow that context in training — drastically reducing the candidate
set from 198 to typically 2-10 at most positions.

Why this works:
  The process grammar has ~80% of step positions in the shared backbone
  (litho cycles, passivation, test suite) that is identical across all
  three families. The grammar guide exploits this to eliminate impossible
  predictions, boosting OOD accuracy without any additional training.

Integration:
    guide = GrammarGuide.build(train_seqs.values(), tok.step_to_id.keys())
    # GPT:
    ranking = gpt_model.next_step_ranking(prefix, k=5, guide=guide)
    # HF:
    ranking = hf_model.next_step_ranking(prefix, k=5, guide=guide)
"""
from __future__ import annotations
from collections import defaultdict


class GrammarGuide:

    def __init__(self, bigram: dict, trigram: dict, all_vocab: frozenset):
        self.bigram    = bigram    # str           → frozenset[str]
        self.trigram   = trigram   # (str, str)    → frozenset[str]
        self.all_vocab = all_vocab # fallback

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def build(cls, sequences, all_vocab, min_count: int = 1) -> "GrammarGuide":
        """Build from an iterable of step-lists (training split only).

        Args:
            sequences:  iterable of list[str]
            all_vocab:  full step vocabulary (for fallback)
            min_count:  minimum observed count to keep a successor
                        (1 = include everything seen at least once)
        """
        bg_counts  = defaultdict(lambda: defaultdict(int))
        tg_counts  = defaultdict(lambda: defaultdict(int))

        for seq in sequences:
            seq = list(seq)
            for i, step in enumerate(seq):
                if i > 0:
                    bg_counts[seq[i - 1]][step] += 1
                if i > 1:
                    tg_counts[(seq[i - 2], seq[i - 1])][step] += 1

        bigram  = {k: frozenset(v for v, c in vs.items() if c >= min_count)
                   for k, vs in bg_counts.items()}
        trigram = {k: frozenset(v for v, c in vs.items() if c >= min_count)
                   for k, vs in tg_counts.items()}

        return cls(bigram, trigram, frozenset(all_vocab))

    # ── core query ────────────────────────────────────────────────────────────

    def valid_next(self, prefix: list[str]) -> frozenset:
        """Constrained set for the next step.  Never returns empty.

        Fallback hierarchy:
          trigram(last 2 steps) → bigram(last step) → full vocab
        """
        if len(prefix) >= 2:
            vs = self.trigram.get((prefix[-2], prefix[-1]))
            if vs:
                return vs
        if len(prefix) >= 1:
            vs = self.bigram.get(prefix[-1])
            if vs:
                return vs
        return self.all_vocab

    # ── helpers used inside model classes ─────────────────────────────────────

    def mask_logits(self, logits, prefix: list[str], id_to_step: dict) -> None:
        """Set -inf for all steps NOT in valid_next(prefix).  In-place."""
        valid = self.valid_next(prefix)
        for step_id, step_str in id_to_step.items():
            if not step_str.startswith("<") and step_str not in valid:
                logits[step_id] = float("-inf")

    def mask_scores(self, scores, prefix: list[str], step_vocab: list[str]) -> None:
        """Set -inf for all candidate positions NOT in valid_next(prefix)."""
        valid = self.valid_next(prefix)
        for i, step in enumerate(step_vocab):
            if step not in valid:
                scores[i] = float("-inf")

    # ── diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        n_bg  = len(self.bigram)
        n_tg  = len(self.trigram)
        avg_bg  = sum(len(v) for v in self.bigram.values())  / max(n_bg, 1)
        avg_tg  = sum(len(v) for v in self.trigram.values()) / max(n_tg, 1)
        return {
            "bigram_contexts":           n_bg,
            "trigram_contexts":          n_tg,
            "avg_bigram_successors":     round(avg_bg,  2),
            "avg_trigram_successors":    round(avg_tg,  2),
            "vocab_size":                len(self.all_vocab),
        }


if __name__ == "__main__":
    from data import load_all, train_val_split
    from tokenizer import StepTokenizer

    all_seqs = load_all()
    tok      = StepTokenizer.build_from_sequences(all_seqs.values())
    train, _ = train_val_split(all_seqs, val_frac=0.1, seed=42)

    guide = GrammarGuide.build(train.values(), tok.step_to_id.keys())
    s = guide.stats()
    print("GrammarGuide stats:", s)

    prefix = ["RECEIVE WAFER LOT", "LOT IDENTIFICATION", "INITIAL WAFER INSPECTION",
              "MEASURE THICKNESS", "MEASURE SURFACE PARTICLES",
              "PRE CLEAN WAFER", "WET CLEAN RCA1", "WET CLEAN RCA2", "HF DIP"]
    valid = guide.valid_next(prefix)
    print(f"\nValid next steps after prefix ({len(valid)} candidates):")
    for s in sorted(valid):
        print(f"  {s}")
