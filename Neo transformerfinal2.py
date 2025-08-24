# neoformer_x_fixed.py
# ---------------------------------
# Production-grade NeoFormer-X LM (patched)
# ---------------------------------
import argparse, math, os, sys
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from datasets import load_dataset

# -----------------------------
# 0. Tokenizer (tiktoken → fallback to HF GPT-2)
# -----------------------------
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def encode(txt:str):
        return _enc.encode(txt, allowed_special={"<|endoftext|>"})
    def decode(ids:List[int]):
        return _enc.decode(ids)
    VOCAB_SIZE = _enc.n_vocab
    print(f"✅ Using tiktoken cl100k_base (vocab={VOCAB_SIZE})")
except Exception:
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained("gpt2")
    _tok.pad_token = _tok.eos_token
    def encode(txt:str):
        return _tok.encode(txt)
    def decode(ids:List[int]):
        return _tok.decode(ids)
    VOCAB_SIZE = _tok.vocab_size
    print(f"⚠️  Falling back to HF tokenizer (gpt2, vocab={VOCAB_SIZE})")

# Optional: FlashAttention-2 (we still keep a safe SDPA fallback)
try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
    print("✅ FlashAttention-2 available")
except Exception:
    HAS_FLASH = False
    print("ℹ️  FlashAttention not available → using PyTorch 2.x SDPA")

# -----------------------------
# 1. RoPE helpers (correct shapes + device)
# -----------------------------

def precompute_rope(dim: int, max_len: int = 8192, base: float = 10000.0, device: Optional[torch.device] = None):
    device = device or torch.device("cpu")
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [L, D/2]
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin  # each [L, D/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """x: [B, H, L, D], cos/sin: [L, D/2]"""
    B, H, L, D = x.shape
    x_1 = x[..., ::2]
    x_2 = x[..., 1::2]
    cos = cos[:L].view(1, 1, L, -1)
    sin = sin[:L].view(1, 1, L, -1)
    out_1 = x_1 * cos - x_2 * sin
    out_2 = x_1 * sin + x_2 * cos
    return torch.stack((out_1, out_2), dim=-1).flatten(-2)

# -----------------------------
# 2. Causal Attention + KV cache (tuple-based)
# -----------------------------
class CausalAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, kv_heads: int, dropout: float = 0.1, rope_max_len: int = 8192):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead, self.kv_heads = nhead, kv_heads
        self.d_k = d_model // nhead
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_k * kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_k * kv_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Will be created on first forward with correct device
        self.register_buffer("rope_cos", None, persistent=False)
        self.register_buffer("rope_sin", None, persistent=False)
        self.rope_max_len = rope_max_len

    def _maybe_build_rope(self, device: torch.device):
        if self.rope_cos is None or self.rope_cos.device != device:
            cos, sin = precompute_rope(self.d_k, self.rope_max_len, device=device)
            self.rope_cos = cos
            self.rope_sin = sin

    def forward(self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        B, L, D = x.shape
        H, K, KV = self.nhead, self.d_k, self.kv_heads

        self._maybe_build_rope(x.device)

        q = self.q_proj(x).view(B, L, H, K).transpose(1, 2)        # [B, H, L, K]
        k = self.k_proj(x).view(B, L, KV, K).transpose(1, 2)        # [B, KV, L, K]
        v = self.v_proj(x).view(B, L, KV, K).transpose(1, 2)        # [B, KV, L, K]

        # GQA: repeat KV heads to match H
        if H != KV:
            rep = H // KV
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        # KV cache (tuple)
        if kv is not None:
            past_k, past_v = kv
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v) if use_cache else None

        # Attention (FlashAttn or SDPA)
        if HAS_FLASH:
            # flash_attn expects [B, L, H, K]
            out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                  dropout_p=self.dropout.p if self.training else 0.0, causal=True)
            out = out.transpose(1, 2)
        else:
            out = F.scaled_dot_product_attention(q, k, v,
                                                 dropout_p=self.dropout.p if self.training else 0.0,
                                                 is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), new_kv

# -----------------------------
# 3. Transformer block (pre-norm)
# -----------------------------
class Block(nn.Module):
    def __init__(self, d_model: int, nhead: int, kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalAttention(d_model, nhead, kv_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, use_cache: bool = False):
        a, new_kv = self.attn(self.norm1(x), kv=kv, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, new_kv

# -----------------------------
# 4. Recursive Memory Core
# -----------------------------
class RMC(nn.Module):
    def __init__(self, d_model: int, mem_size: int = 32):
        super().__init__()
        self.mem = nn.Parameter(torch.randn(mem_size, d_model) * 0.01)
        self.read = nn.Linear(d_model, mem_size)
        self.write = nn.Linear(d_model, mem_size)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor, update: bool = False):
        avg = x.mean(1)  # [B, D]
        weights = F.softmax(self.read(avg), -1)  # [B, M]
        w = weights @ self.mem  # [B, D]
        x = self.proj(torch.cat([x, w.unsqueeze(1).expand(-1, x.size(1), -1)], -1))
        delta = None
        if update:
            w_write = F.softmax(self.write(avg), -1)  # [B, M]
            delta = w_write.t() @ avg  # [M, D] aggregated over batch
        return x, delta

    def apply_update(self, delta: torch.Tensor, lr: float = 0.01):
        if delta is None:
            return
        with torch.no_grad():
            self.mem.add_(delta * lr)

# -----------------------------
# 5. Neural-Symbolic Bridge
# -----------------------------
class NSB(nn.Module):
    def __init__(self, d_model: int, rules: int = 32, rdim: int = 32):
        super().__init__()
        self.rules = nn.Embedding(rules, rdim)
        self.q = nn.Linear(d_model, rdim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model + rdim, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor):
        sel = F.softmax(self.q(x) @ self.rules.weight.t(), -1) @ self.rules.weight  # [B, L, rdim]
        return x + self.ffn(torch.cat([x, sel], -1))

# -----------------------------
# 6. Main model
# -----------------------------
class NeoFormerX(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 12, kv_heads: int = 3, layers: int = 12, mem_size: int = 32, max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([Block(d_model, nhead, kv_heads) for _ in range(layers)])
        self.rmc = RMC(d_model, mem_size)
        self.nsb = NSB(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, update: bool = False):
        B, L = x.shape
        h = self.embed(x)  # [B, L, D]

        # No additive pos-ids here (RoPE handles positions internally)
        h, delta = self.rmc(h, update)
        h = self.nsb(h)

        new_past = []
        if past is None:
            past = [None] * len(self.blocks)
        for blk, p in zip(self.blocks, past):
            h, new_p = blk(h, kv=p, use_cache=True)
            new_past.append(new_p)
        logits = self.lm_head(self.norm(h))
        return logits, new_past, delta

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new: int = 64, temp: float = 0.8, top_k: int = 50):
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)
        past = None
        for _ in range(max_new):
            if x.size(1) >= self.max_seq_len:
                # keep half context, trim caches accordingly
                keep = self.max_seq_len // 2
                x = x[:, -keep:]
                if past is not None:
                    trimmed = []
                    for kv in past:
                        if kv is None:
                            trimmed.append(None)
                        else:
                            k, v = kv
                            trimmed.append((k[:, :, -keep:, :], v[:, :, -keep:, :]))
                    past = trimmed
            logits, past, _ = self(x[:, -1:].contiguous() if past is not None else x, past=past, update=False)
            logits = logits[:, -1] / max(temp, 1e-8)
            if top_k and top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, tok], dim=1)
        return x

# -----------------------------
# 7. Training utilities
# -----------------------------
@dataclass
class TrainCfg:
    model: str = "350M"
    batch: int = 8
    lr: float = 3e-4
    epochs: int = 3
    compile: bool = False
    lora: bool = False
    seq: int = 256
    max_tokens: int = 100_000


def build_model(name: str) -> NeoFormerX:
    cfg = {"125M": (768, 12, 3, 12), "350M": (1024, 16, 4, 24), "760M": (1536, 16, 4, 32)}[name]
    return NeoFormerX(VOCAB_SIZE, *cfg)


def train_loop(model: NeoFormerX, device: torch.device, batch: int, lr: float, epochs: int, seq: int, max_tokens: int):
    print("📚 Loading dataset…")
    ds = load_dataset("openwebtext", split="train[:0.1%]", streaming=True)
    toks: List[int] = []
    for ex in ds:
        toks.extend(encode(ex["text"]))
        if len(toks) >= max_tokens:
            break
    data = torch.tensor(toks[:max_tokens], dtype=torch.long)

    X, Y = [], []
    for i in range(0, len(data) - seq - 1, seq):
        X.append(data[i:i + seq])
        Y.append(data[i + 1:i + seq + 1])
    X = torch.stack(X)
    Y = torch.stack(Y)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler() if device.type == "cuda" else None
    model.train()

    for epoch in range(epochs):
        idx = torch.randperm(len(X))
        total = 0.0
        steps = 0
        for i in tqdm(range(0, len(X), batch), desc=f"Epoch {epoch+1}"):
            sel = idx[i:i + batch]
            bx = X[sel].to(device)
            by = Y[sel].to(device)
            opt.zero_grad(set_to_none=True)
            ctx = autocast(device_type=device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with (ctx):
                logits, _, delta = model(bx, update=True)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), by.view(-1))
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            if delta is not None:
                model.rmc.apply_update(delta.mean(0), lr=0.01)
            total += loss.item(); steps += 1
        print(f"Epoch {epoch+1}: loss={total/steps:.4f}, ppl={math.exp(total/steps):.2f}")


# -----------------------------
# 8. CLI
# -----------------------------
from contextlib import nullcontext
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["125M", "350M", "760M"], default="350M")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--max_tokens", type=int, default=100_000)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)

    # Optional LoRA (PEFT) — safe no-op if not installed
    if args.lora:
        try:
            from peft import get_peft_model, LoraConfig
            lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj"], bias="none")
            model = get_peft_model(model, lora_cfg)
            print("✅ LoRA enabled")
        except Exception as e:
            print(f"⚠️  PEFT not available: {e}")

    if args.compile:
        try:
            model = torch.compile(model, dynamic=True)
            print("✅ torch.compile enabled")
        except Exception as e:
            print(f"⚠️  compile failed: {e}")

    if args.train:
        train_loop(model, device, args.batch, args.lr, args.epochs, args.seq, args.max_tokens)

    if args.eval:
        try:
            from lm_eval import simple_evaluate
            simple_evaluate(model, tasks=["hellaswag", "wikitext"], device=str(device))
        except Exception as e:
            print(f"⚠️  lm-eval not available/failed: {e}")

    # Quick demo
    prompt = torch.tensor([encode("The future of AI is ")], dtype=torch.long).to(device)
    out = model.generate(prompt, max_new=64)
    print("\n🎉 Sample:\n", decode(out[0].tolist()))


if __name__ == "__main__":
    main()
