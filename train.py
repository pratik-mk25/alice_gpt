"""
train.py — Train a GPT on Alice in Wonderland (character-level).

Run:
    cd alice_gpt
    python prepare.py        # first time only
    python train.py

Checkpoints are saved to out/ every eval_interval steps.
Generation samples are printed during training so you can watch it learn.
"""

import os
import math
import time
import pickle
import numpy as np
import torch
from model import GPT, GPTConfig

# ═══════════════════════════════════════════════════════════════════════════════
#  Hyperparameters — tuned for RTX 3050 6GB, ~50K char dataset
# ═══════════════════════════════════════════════════════════════════════════════
# I/O
out_dir        = "out"
eval_interval  = 100       # evaluate every N steps
eval_iters     = 50        # iterations to average val loss over
log_interval   = 10        # print loss every N steps
always_save    = True      # save checkpoint whenever val loss improves

# Data
dataset_dir    = "."       # folder with train.bin, val.bin, meta.pkl
block_size     = 256       # context length

# Model — "nano" size, ~2M params, trains in ~5 min on CPU, <1 min on GPU
n_layer        = 6
n_head         = 6
n_embd         = 384
dropout        = 0.2

# Training
batch_size     = 64        # sequences per batch
max_iters      = 3000      # total gradient steps
grad_clip      = 1.0       # gradient clipping

# Learning rate schedule — cosine decay with warmup
learning_rate  = 3e-4
min_lr         = 3e-5
warmup_iters   = 100
lr_decay_iters = max_iters

# AdamW
weight_decay   = 1e-1
beta1          = 0.9
beta2          = 0.99

# System
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = "bfloat16" if torch.cuda.is_available() and \
         torch.cuda.is_bf16_supported() else "float16" \
         if torch.cuda.is_available() else "float32"
compile_model  = False     # torch.compile — set True for PyTorch ≥ 2.0 speedup

# Generation during training
gen_prompt     = "Alice"
gen_tokens     = 200
gen_temperature= 0.8
gen_top_k      = 40
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42)

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32,
           "bfloat16": torch.bfloat16,
           "float16": torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) \
      if device_type == "cuda" else torch.nullcontext()

# ── Load data ──────────────────────────────────────────────────────────────
def load_split(split):
    path = os.path.join(dataset_dir, f"{split}.bin")
    data = np.fromfile(path, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))

train_data = load_split("train")
val_data   = load_split("val")

with open(os.path.join(dataset_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

vocab_size = meta["vocab_size"]
print(f"Vocab size: {vocab_size} | Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

# Decode function for generation display
if meta["mode"] == "char":
    itos = meta["itos"]
    decode = lambda ids: "".join(itos[i] for i in ids)
else:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda ids: enc.decode(ids)

# ── Batch sampler ──────────────────────────────────────────────────────────
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i     : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ── Model ──────────────────────────────────────────────────────────────────
config = GPTConfig(
    block_size = block_size,
    vocab_size  = vocab_size,
    n_layer     = n_layer,
    n_head      = n_head,
    n_embd      = n_embd,
    dropout     = dropout,
)
model = GPT(config).to(device)

if compile_model:
    print("Compiling model with torch.compile…")
    model = torch.compile(model)

scaler = torch.amp.GradScaler("cuda", enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(
    lr=learning_rate, weight_decay=weight_decay,
    betas=(beta1, beta2), device_type=device_type
)

# ── LR schedule ───────────────────────────────────────────────────────────
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ── Eval ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ── Generation sample ─────────────────────────────────────────────────────
@torch.no_grad()
def sample():
    model.eval()
    if meta["mode"] == "char":
        stoi = meta["stoi"]
        start_ids = [stoi.get(c, 0) for c in gen_prompt]
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        start_ids = enc.encode_ordinary(gen_prompt)

    ctx_tensor = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = model.generate(ctx_tensor, gen_tokens, gen_temperature, gen_top_k)
    text = decode(generated[0].tolist())
    model.train()
    return text

# ── Training loop ──────────────────────────────────────────────────────────
best_val_loss = float("inf")
t0 = time.time()

print(f"\nDevice: {device} | dtype: {dtype}")
print(f"Training for {max_iters} steps | batch_size={batch_size} | block_size={block_size}\n")

X, Y = get_batch("train")

for step in range(max_iters + 1):
    # Set LR for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # ── Evaluate & checkpoint ──────────────────────────────────────────────
    if step % eval_interval == 0:
        losses = estimate_loss()
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        print(f"\n{'─'*60}")
        print(f"Step {step:>5} | train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | lr {lr:.2e} | {dt:.1f}s")

        if losses["val"] < best_val_loss or always_save:
            best_val_loss = losses["val"]
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.__dict__,
                "step": step,
                "val_loss": losses["val"].item(),
                "meta": meta,
            }
            torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))
            print(f"  ✓ Saved checkpoint (val_loss={best_val_loss:.4f})")

        # Print a generation sample every eval_interval
        print(f"\n  📖 Sample generation (prompt='{gen_prompt}'):")
        print("  " + "─"*50)
        sample_text = sample()
        for line in sample_text.split("\n"):
            print(f"  {line}")
        print()

    if step == max_iters:
        break

    # ── Forward + backward ─────────────────────────────────────────────────
    with ctx:
        logits, loss = model(X, Y)

    X, Y = get_batch("train")  # pre-fetch next batch

    scaler.scale(loss).backward()
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if step % log_interval == 0:
        print(f"  step {step:>5} | loss {loss.item():.4f} | lr {lr:.2e}", end="\r")

print("\n\nTraining complete!")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Checkpoint saved to: {out_dir}/ckpt.pt")
