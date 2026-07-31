"""
b0.py - data prep of gpt-roofline.

Goal: tokenize OpenWebText ONCE (GPT-2 BPE), carve the val split BEFORE
tokenizing so no val token can leak into train, write train.bin / val.bin as
uint16 memmaps on disk.
"""

# ---- imports --------------------------------------------------------------
import os
import json
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


# ---- knobs (canonical names - every judgment call is a visible variable) ---
DATA_SOURCE = "Skylion007/openwebtext"
VAL_FRACTION = 0.005
NUM_PROC = 2
#Was originally 16 but it crashed my wsl virtual machine(OOM due to not enough allotted ram)
OUT_DIR = os.path.dirname(__file__)
DTYPE = np.uint16
SEED = 1337


enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token


# ---- process(example) -----------------------------------------------------
def process(example):
    ids = enc.encode_ordinary(example["text"])
    ids.append(EOT)
    return {"ids": ids, "len": len(ids)}


# ---- write_memmap(dset, filepath) -----------------------------------------
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
    # (achieved FLOPs = 6 * N * params).
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(token_counts, f, indent=2)
    print(f"meta.json: {token_counts}")


# ---- entrypoint -----------------------------------------------------------
if __name__ == "__main__":
    main()