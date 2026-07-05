"""
b0.py — data prep of gpt-roofline.  [YOUR BUILD-FROM-BLANK REP]

Goal: tokenize OpenWebText ONCE (GPT-2 BPE), carve the val split BEFORE
tokenizing so no val token can leak into train, write train.bin / val.bin as
uint16 memmaps on disk.

Write every real line yourself. Skeleton gives you the steps + the names, not
the logic. Stuck on ONE line? Ask me — no answer key, you own the blank page.
"""

# ---- imports --------------------------------------------------------------
# You need: os, numpy as np, tiktoken, load_dataset from datasets, tqdm.
# (tqdm optional — only if you want a progress bar on the memmap write.)
import os
import json
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# ---- knobs (canonical names — every judgment call is a visible variable) ---
# DATA_SOURCE   -> the HF dataset id (the one we proved works in the probe)
# VAL_FRACTION  -> fraction of docs held out as val (spec says ~0.5%)
# NUM_PROC      -> tokenization worker processes (CPU-bound)
# OUT_DIR       -> where train.bin / val.bin land (the project dir)
# DTYPE         -> the uint type that fits GPT-2 vocab (50257 — what fits?)
# SEED          -> fixed int so the split is reproducible
DATA_SOURCE = "Skylion007/openwebtext"
VAL_FRACTION = 0.005
NUM_PROC = 2
OUT_DIR = os.path.dirname(__file__)
DTYPE = np.uint16
SEED = 1337



# Build the tiktoken encoder for "gpt2" here, and grab its end-of-text token id.
# (You'll append that token after each doc as a boundary marker.)
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


# ---- process(example) -----------------------------------------------------
# Take one dataset row -> tokenize example["text"] -> append the EOT id ->
# return a dict with the token list and its length.
# (Use encode_ordinary so special tokens in the text aren't treated specially.)
def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(EOT)
    return {"ids": ids, "len": len(ids)}


# ---- write_memmap(dset, filepath) -----------------------------------------
# 1. total length = sum of the per-doc "len" column.
# 2. open an np.memmap at filepath, mode "w+", shape (total_len,), your DTYPE.
# 3. loop over the split in shards (e.g. 1024) so you never hold it all in RAM:
#       - grab shard i, format as numpy
#       - concatenate that shard's token lists into one array
#       - write it into the memmap at the running index, advance the index
# 4. flush the memmap. return total_len so you can print it.
def write_memmap(dset, filepath, shards=1024):
    arr_len = int(np.sum(dset["len"], dtype=np.uint64))
    arr = np.memmap(filepath, dtype=DTYPE, mode="w+", shape=(arr_len,))
    idx = 0
    for i in tqdm(range(shards), desc=os.path.basename(filepath)):
        batch = dset.shard(num_shards=shards, index=i, contiguous=True).with_format("numpy")
        batch_ids = np.concatenate(batch["ids"])
        arr[idx : idx + len(batch_ids)] = batch_ids
        idx += len(batch_ids)
    arr.flush()
    return arr_len


# ---- main() ---------------------------------------------------------------
# 1. load_dataset(DATA_SOURCE) — the corpus arrives as a "train" split.
# 2. SPLIT FIRST: train_test_split on the raw docs (test_size=VAL_FRACTION,
#    seed=SEED, shuffle=True). Rename the "test" slice to "val".
#    >>> this MUST happen before any tokenizing — that's the whole point of P0.
# 3. tokenize: .map(process) over the split, drop the "text" column, num_proc.
# 4. for each split: build its filepath ("train.bin"/"val.bin"), write_memmap,
#    print the token count.
def main():
    dataset = load_dataset(DATA_SOURCE, num_proc=NUM_PROC)

    # THE INVARIANT: split raw docs BEFORE tokenizing so no val token leaks into train.
    split = dataset["train"].train_test_split(
        test_size=VAL_FRACTION, seed=SEED, shuffle=True
    )
    split["val"] = split.pop("test")

    tokenized = split.map(
        process,
        remove_columns=["text"],
        desc="tokenizing",
        num_proc=NUM_PROC,
    )

    token_counts = {}
    for name, dset in tokenized.items():
        filepath = os.path.join(OUT_DIR, f"{name}.bin")
        n = write_memmap(dset, filepath)
        token_counts[f"{name}_tokens"] = int(n)
        print(f"{name}: {n:,} tokens -> {filepath}")

    # Roofline hook: token count N is the seed of the MFU denominator
    # (achieved FLOPs = 6 * N * params). Dump it now so later bricks never re-derive it.
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(token_counts, f, indent=2)
    print(f"meta.json: {token_counts}")


# ---- entrypoint -----------------------------------------------------------
if __name__ == "__main__":
    main()