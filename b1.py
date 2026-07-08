"""
b1.py — the data loader of gpt-roofline.  [YOUR BUILD-FROM-BLANK REP]

Goal: read the uint16 memmaps b0 wrote (train.bin / val.bin) WITHOUT pulling
them into RAM, and hand back one training batch (x, y) of random-offset slices.
Then take the project's FIRST real measurement: loader-only tokens/sec — prove
the data path is NOT the bottleneck before you ever blame the GPU (P1's whole
point, straight from the spec + the Outsider's north star).

Write every real line yourself. Skeleton = the steps + the names, not the logic.
Stuck on ONE line? Ask me — no answer key. You own the blank page.
"""

# ---- imports --------------------------------------------------------------
# You need: os, numpy as np, torch. (time for the throughput timer.)
import os
import numpy as np
import torch
import time


# ---- knobs (canonical names — every judgment call is a visible variable) ---
# DATA_DIR    -> where train.bin / val.bin live (same dir as this file)
# BLOCK_SIZE  -> context length T (how many tokens per sequence)
# BATCH_SIZE  -> B (how many sequences per batch)
# DEVICE      -> "cuda" if available else "cpu"
# DTYPE       -> np.uint16 (must match what b0 wrote — don't re-derive, mirror it)
BATCH_SIZE = 64
DTYPE = np.uint16
BLOCK_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
DATA_DIR = os.path.dirname(__file__)


# ---- _memmap(split) -------------------------------------------------------
# Given "train" or "val", open the matching .bin as an np.memmap, mode "r",
# dtype DTYPE. Reopen it FRESH each call (a leaked memmap across a fork/loop
# is a known nanoGPT footgun) — return the array.
def _memmap(split):
    filepath = os.path.join(DATA_DIR, f"{split}.bin")
    data = np.memmap(filepath, dtype=DTYPE, mode="r")
    return data




# ---- get_batch(split) -----------------------------------------------------
# 1. data = _memmap(split)
# 2. pick BATCH_SIZE random start offsets in [0, len(data) - BLOCK_SIZE).
# 3. x = stack of data[i : i+BLOCK_SIZE]        (the inputs)
#    y = stack of data[i+1 : i+1+BLOCK_SIZE]    (targets = inputs shifted by 1)
#    -> build as int64 torch tensors (token ids are indices, not uint16 math).
# 4. move x, y to DEVICE (pin + non_blocking if cuda — small optional win).
# 5. return x, y  with shape (BATCH_SIZE, BLOCK_SIZE).
def get_batch(split):
    data = _memmap(split)
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i : i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1 : i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y    
    


# ---- measure_loader_throughput(split, n_batches) --------------------------
# The FIRST measurement of the whole project. No model — just the data path.
# 1. warm up a few get_batch calls (don't time the cold first hit).
# 2. time n_batches of get_batch back-to-back.
# 3. tokens moved = n_batches * BATCH_SIZE * BLOCK_SIZE.
# 4. print tokens/sec and batches/sec. THIS is the number that says "the loader
#    is / isn't the bottleneck." Record it — every run logs speed, not just loss.
def measure_loader_throughput(split, n_batches):
    for _ in range(3):
        get_batch(split)
    x, y = get_batch(split)
    t0 = time.time()
    for _ in range(n_batches):
        get_batch(split)
    t1 = time.time()
    elapsed = t1 - t0
    tokens_moved = n_batches * BATCH_SIZE * BLOCK_SIZE
    print(f"tokens/sec: {tokens_moved / elapsed:,.0f}")
    print(f"batches/sec: {n_batches / elapsed:.1f}")


# ---- entrypoint -----------------------------------------------------------
# When run directly: sanity-check one batch (print x.shape, y.shape, x.dtype,
# confirm y is x shifted by one on a slice), THEN run the throughput measure.
if __name__ == "__main__":
    x, y = get_batch("train")
    print(x.shape, y.shape, x.dtype, y.dtype)
    print(torch.equal(y[:, :-1], x[:, 1:]))
    measure_loader_throughput("train", 100)

