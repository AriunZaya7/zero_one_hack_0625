"""
transformer.py
==============
Small decoder-only (causal) transformer trained from scratch over step-level
tokens. Uses the integer Vocab from solution/data/vocab.py.

Smoke test (tiny, CPU):
    python -m solution.models.transformer
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from solution.data.vocab import Vocab


class ProcessTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 6, d_ff: int = 1024, max_seq_len: int = 256,
                 dropout: float = 0.1, pad_id: int = Vocab.PAD):
        super().__init__()
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        causal = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        key_padding = (input_ids == self.pad_id) if attention_mask is None else (attention_mask == 0)
        x = self.blocks(x, mask=causal, src_key_padding_mask=key_padding)
        return self.head(self.ln_f(x))  # (B, T, V)

    @torch.no_grad()
    def next_token_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.forward(input_ids)[:, -1, :]

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 200,
                 eos_id: int = Vocab.EOS, temperature: float = 1.0,
                 top_k: int | None = None, greedy: bool = True) -> torch.Tensor:
        self.eval()
        ids = prompt_ids
        for _ in range(max_new_tokens):
            ctx = ids[:, -self.max_seq_len:]
            logits = self.next_token_logits(ctx)
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return ids

    @torch.no_grad()
    def sequence_log_prob(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean per-token log-prob for each sequence in the batch. (B,)"""
        logits = self.forward(input_ids)
        logp = F.log_softmax(logits[:, :-1, :], dim=-1)
        tgt = input_ids[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        mask = (tgt != self.pad_id).float()
        return (tok_lp * mask).sum(1) / mask.sum(1).clamp(min=1)


if __name__ == "__main__":
    torch.manual_seed(42)
    V = 64
    m = ProcessTransformer(vocab_size=V, d_model=32, n_heads=2, n_layers=2,
                           d_ff=64, max_seq_len=32)
    x = torch.randint(4, V, (2, 10))
    out = m(x)
    assert out.shape == (2, 10, V), out.shape
    gen = m.generate(x[:, :3], max_new_tokens=5)
    lp = m.sequence_log_prob(x)
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"logits {tuple(out.shape)} | gen {tuple(gen.shape)} | logp {tuple(lp.shape)} | {n_params:.2f}M params")
    print("transformer OK")
