"""
b1.py - the data loader of gpt-roofline.

Goal: read the uint16 memmaps b0 wrote (train.bin / val.bin) WITHOUT pulling
them into RAM, and hand back one training batch (x, y) of random-offset slices.
Then take the project's FIRST real measurement: loader-only tokens/sec - prove
the data path is NOT the bottleneck before blaming the GPU.
"""

# ---- imports --------------------------------------------------------------
import os
import numpy as np
import torch
import time


# ---- knobs (canonical names - every judgment call is a visible variable) ---
# DATA_DIR    -> where train.bin / val.bin live (same dir as this file)
# BLOCK_SIZE  -> context length T (how many tokens per sequence)
# BATCH_SIZE  -> B (how many sequences per batch)
# DEVICE      -> "cuda" if available else "cpu"
# DTYPE       -> np.uint16 (must match what b0 wrote)
BATCH_SIZE = 64
DTYPE = np.uint16
BLOCK_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
DATA_DIR = os.path.dirname(__file__)


# ---- _memmap(split) -------------------------------------------------------
def _memmap(split):
    filepath = os.path.join(DATA_DIR, f"{split}.bin")
    data = np.memmap(filepath, dtype=DTYPE, mode="r")
    return data


# ---- get_batch(split) -----------------------------------------------------
def get_batch(split):
    data = _memmap(split)
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i : i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1 : i+BLOCK_SIZE+1].astype(np.int64)) for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y    
    

# ---- measure_loader_throughput(split, n_batches) --------------------------
def measure_loader_throughput(split, n_batches):
    for _ in range(3):
        get_batch(split)
    t0 = time.time()
    for _ in range(n_batches):
        get_batch(split)
    t1 = time.time()
    elapsed = t1 - t0
    tokens_moved = n_batches * BATCH_SIZE * BLOCK_SIZE
    print(f"tokens/sec: {tokens_moved / elapsed:,.0f}")
    print(f"batches/sec: {n_batches / elapsed:.1f}")


# ---- entrypoint -----------------------------------------------------------
if __name__ == "__main__":
    x, y = get_batch("train")
    print(x.shape, y.shape, x.dtype, y.dtype)
    print(torch.equal(y[:, :-1], x[:, 1:]))
    measure_loader_throughput("train", 100)

