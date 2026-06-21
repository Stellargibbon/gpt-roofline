"""
b0.py — data prep of gpt-roofline.  [YOUR BUILD-FROM-BLANK REP]

Goal: tokenize OpenWebText ONCE (GPT-2 BPE), carve the val split BEFORE
tokenizing so no val token can leak into train, write train.bin / val.bin as
uint16 memmaps on disk.

Write every real line yourself. Skeleton gives you the steps + the names, not
the logic. Stuck on ONE line? Ask me, or peek at b0_reference.py — then
fix the part you got wrong, don't copy the whole thing.
"""

# ---- imports --------------------------------------------------------------
# You need: os, numpy as np, tiktoken, load_dataset from datasets, tqdm.
# (tqdm optional — only if you want a progress bar on the memmap write.)


# ---- knobs (canonical names — every judgment call is a visible variable) ---
# DATA_SOURCE   -> the HF dataset id (the one we proved works in the probe)
# VAL_FRACTION  -> fraction of docs held out as val (spec says ~0.5%)
# NUM_PROC      -> tokenization worker processes (CPU-bound)
# OUT_DIR       -> where train.bin / val.bin land (the project dir)
# DTYPE         -> the uint type that fits GPT-2 vocab (50257 — what fits?)
# SEED          -> fixed int so the split is reproducible

# Build the tiktoken encoder for "gpt2" here, and grab its end-of-text token id.
# (You'll append that token after each doc as a boundary marker.)


# ---- process(example) -----------------------------------------------------
# Take one dataset row -> tokenize example["text"] -> append the EOT id ->
# return a dict with the token list and its length.
# (Use encode_ordinary so special tokens in the text aren't treated specially.)


# ---- write_memmap(dset, filepath) -----------------------------------------
# 1. total length = sum of the per-doc "len" column.
# 2. open an np.memmap at filepath, mode "w+", shape (total_len,), your DTYPE.
# 3. loop over the split in shards (e.g. 1024) so you never hold it all in RAM:
#       - grab shard i, format as numpy
#       - concatenate that shard's token lists into one array
#       - write it into the memmap at the running index, advance the index
# 4. flush the memmap. return total_len so you can print it.


# ---- main() ---------------------------------------------------------------
# 1. load_dataset(DATA_SOURCE) — the corpus arrives as a "train" split.
# 2. SPLIT FIRST: train_test_split on the raw docs (test_size=VAL_FRACTION,
#    seed=SEED, shuffle=True). Rename the "test" slice to "val".
#    >>> this MUST happen before any tokenizing — that's the whole point of P0.
# 3. tokenize: .map(process) over the split, drop the "text" column, num_proc.
# 4. for each split: build its filepath ("train.bin"/"val.bin"), write_memmap,
#    print the token count.


# ---- entrypoint -----------------------------------------------------------
# if __name__ == "__main__": call main()