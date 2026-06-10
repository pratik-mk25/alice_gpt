"""
generate.py — Load a trained checkpoint and generate text.

Usage:
    python generate.py
    python generate.py --prompt "The Queen said" --tokens 300 --temp 0.9 --top_k 50
"""

import os
import sys
import pickle
import argparse
import torch
from model import GPT, GPTConfig

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",   default="out/ckpt.pt",   help="Checkpoint path")
parser.add_argument("--prompt", default="Alice",          help="Prompt string")
parser.add_argument("--tokens", default=500,  type=int,  help="Tokens to generate")
parser.add_argument("--temp",   default=0.8,  type=float,help="Temperature (0.1–1.5)")
parser.add_argument("--top_k",  default=40,   type=int,  help="Top-k sampling")
parser.add_argument("--num",    default=1,    type=int,  help="Number of samples")
args = parser.parse_args()

# ── Load checkpoint ────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading checkpoint: {args.ckpt}  [device={device}]")

ckpt = torch.load(args.ckpt, map_location=device)
cfg  = ckpt["config"]
meta = ckpt["meta"]

config = GPTConfig(**cfg)
model  = GPT(config)
model.load_state_dict(ckpt["model"])
model.eval()
model.to(device)

print(f"Model loaded | step={ckpt['step']} | val_loss={ckpt['val_loss']:.4f}")
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

# ── Encode prompt ──────────────────────────────────────────────────────────
if meta["mode"] == "char":
    stoi = meta["stoi"]
    itos = meta["itos"]
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
else:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = enc.encode_ordinary
    decode = enc.decode

# ── Generate ───────────────────────────────────────────────────────────────
print(f"Prompt: {repr(args.prompt)}")
print(f"Generating {args.tokens} tokens × {args.num} sample(s)\n")

for i in range(args.num):
    start_ids = encode(args.prompt)
    ctx = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        generated = model.generate(ctx, args.tokens, args.temp, args.top_k)

    text = decode(generated[0].tolist())
    print(f"{'═'*60}")
    print(f"  Sample {i+1}")
    print(f"{'═'*60}")
    print(text)
    print()
