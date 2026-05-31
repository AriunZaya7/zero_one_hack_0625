"""Step-level tokenizer for semiconductor process sequences.

Design choice: each process step STRING is exactly one token
(e.g. "RECEIVE WAFER LOT" -> 1 id). The vocabulary is BUILT FROM THE DATA
so it always stays in sync with generation_rules.md section 1 (~120 steps,
shared across families). Do NOT hardcode the vocab.

Special tokens:
  <pad>  padding
  <bos>  beginning of sequence  (note: real sequences also start with
         "RECEIVE WAFER LOT" -- we keep <bos> separate so the model can
         still represent "what comes first" as a learnable signal)
  <eos>  end of sequence        (real sequences end with "SHIP LOT")
  <unk>  unseen step (should be rare given shared vocab; useful for the
         hidden 4th family in case it introduces a new step)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class StepTokenizer:
    def __init__(self, step_to_id: dict[str, int]):
        self.step_to_id = dict(step_to_id)
        self.id_to_step = {i: s for s, i in self.step_to_id.items()}
        self.pad_id = self.step_to_id["<pad>"]
        self.bos_id = self.step_to_id["<bos>"]
        self.eos_id = self.step_to_id["<eos>"]
        self.unk_id = self.step_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.step_to_id)

    @classmethod
    def build_from_sequences(cls, sequences: Iterable[list[str]]) -> "StepTokenizer":
        """Build a deterministic vocab from an iterable of step-lists."""
        steps: set[str] = set()
        for seq in sequences:
            steps.update(seq)
        # sorted() -> deterministic ids -> reproducible runs
        ordered = SPECIAL_TOKENS + sorted(steps)
        step_to_id = {s: i for i, s in enumerate(ordered)}
        return cls(step_to_id)

    def encode(self, steps: list[str], add_special: bool = True) -> list[int]:
        ids = [self.step_to_id.get(s, self.unk_id) for s in steps]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        out: list[str] = []
        for i in ids:
            if skip_special and i in specials:
                continue
            out.append(self.id_to_step.get(int(i), "<unk>"))
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.step_to_id, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "StepTokenizer":
        return cls(json.loads(Path(path).read_text()))


if __name__ == "__main__":
    # quick smoke test
    seqs = [["RECEIVE WAFER LOT", "CLEAN", "SHIP LOT"],
            ["RECEIVE WAFER LOT", "OXIDATION", "CLEAN", "SHIP LOT"]]
    tok = StepTokenizer.build_from_sequences(seqs)
    ids = tok.encode(seqs[0])
    print("vocab_size:", tok.vocab_size)
    print("ids:", ids)
    print("decoded:", tok.decode(ids))
