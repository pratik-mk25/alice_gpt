"""
prepare.py — Tokenize Alice in Wonderland for nanoGPT training.
Uses character-level tokenization (simple, no dependencies beyond Python).
Also supports GPT-2 BPE via tiktoken if you want sub-word tokens.
"""

import os
import pickle
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE  = "input.txt"
MODE        = "char"   # "char" or "bpe"
TRAIN_SPLIT = 0.9
# ────────────────────────────────────────────────────────────────────────────

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = f.read()

print(f"Dataset length: {len(data):,} characters")

if MODE == "char":
    chars = sorted(set(data))
    vocab_size = len(chars)
    print(f"Vocab size (char-level): {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)

    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi, "mode": "char"}

elif MODE == "bpe":
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size (GPT-2 BPE): {vocab_size}")
    encode = lambda s: enc.encode_ordinary(s)
    decode = lambda l: enc.decode(l)
    meta = {"vocab_size": vocab_size, "mode": "bpe"}

# Encode full dataset
ids = encode(data)
print(f"Total tokens: {len(ids):,}")

# Train / val split
n = int(TRAIN_SPLIT * len(ids))
train_ids = ids[:n]
val_ids   = ids[n:]
print(f"Train tokens: {len(train_ids):,} | Val tokens: {len(val_ids):,}")

# Save as uint16 arrays (sufficient for vocab ≤ 65535)
np.array(train_ids, dtype=np.uint16).tofile("train.bin")
np.array(val_ids,   dtype=np.uint16).tofile("val.bin")

with open("meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Saved: train.bin  val.bin  meta.pkl")
print(f"\nSample encode→decode round-trip:")
sample = data[:80]
print(f"  original : {repr(sample)}")
print(f"  decoded  : {repr(decode(encode(sample)))}")
