"""
vocab.py
========
Integer vocabulary over process-step names for the from-scratch transformer.

Special tokens occupy the first four ids; real steps start at OFFSET=4 and are
assigned in sorted order so ids are stable across runs/machines.

Smoke test:
    python -m solution.data.vocab --data_dir training_data
"""

import json
import argparse


class Vocab:
    PAD = 0
    BOS = 1   # beginning of sequence
    EOS = 2   # end of sequence
    UNK = 3
    OFFSET = 4  # real steps start here

    SPECIAL = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self, sequences: list[list[str]] | None = None,
                 steps: list[str] | None = None):
        if steps is None:
            unique = set()
            for seq in (sequences or []):
                unique.update(seq)
            steps = sorted(unique)
        self.itos = list(self.SPECIAL) + list(steps)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, steps: list[str], add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(s, self.UNK) for s in steps]
        if add_special:
            return [self.BOS] + ids + [self.EOS]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> list[str]:
        out = []
        for i in ids:
            if strip_special and i < self.OFFSET:
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "<UNK>")
        return out

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v = cls(steps=[])
        v.itos = data["itos"]
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v


if __name__ == "__main__":
    from solution.data.loader import load_all_families
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="training_data")
    ap.add_argument("--out", default="solution/vocab.json")
    args = ap.parse_args()
    fams = load_all_families(args.data_dir)
    all_seqs = [s for seqs in fams.values() for s in seqs]
    v = Vocab(all_seqs)
    v.save(args.out)
    print(f"vocab size = {v.size} (incl. {len(Vocab.SPECIAL)} special) -> {args.out}")
    print("first real steps:", v.itos[Vocab.OFFSET:Vocab.OFFSET + 5])
    print("vocab OK")
