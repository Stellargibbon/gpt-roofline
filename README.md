# gpt-roofline

A real GPT training pipeline on OpenWebText, instrumented for throughput and the
neural scaling law that falls out of the same runs.

Sequel to [`micro-gpt`](https://github.com/Stellargibbon/micro-gpt)
(from-scratch transformer + hyperparameter sweep). That project answered *"what
moves the loss?"* This one answers *"how fast and cheaply did I get there?"*

## Status

- P0 `b0.py` - tokenized OpenWebText -> `train.bin` (17G) / `val.bin` (86M) /
  `meta.json`.
- P1 `b1.py` - memmap loader + first measurement: loader-only throughput ~155K
  tok/s (worst-case floor; the "is data the bottleneck?" question is answered at
  P3).
- P2 `b2.py` (model half) - GPT model: init loss 10.8833 vs ln(50257) ~ 10.825
  on one real batch (the random-init sanity), plus GPT-2 residual sqrt-shrink
  init.
- P2 `b3.py` (loop half) - training loop: `estimate_loss()` (val, `no_grad`),
  AdamW, eval gate. First long run 2026-07-26: 12,500 iters, val loss 10.887 ->
  **4.458** (~205M tokens).
- P3 `b4.py` - systems instrumentation: CUDA-event timing harness, tokens/sec,
  MFU, peak VRAM, coarse step-time split, and an N-step measurement loop with
  warmup discard.

### Instrumented numbers (2026-07-26)

30M-param GPT, fp32 (`allow_tf32=False`), B*T = 16,384 tokens/step, n=105 steps
(5 warmup discarded), RTX PRO 6000 Blackwell Workstation Edition:

| metric | value |
|---|---|
| throughput | **192,574 tok/s** |
| MFU | **27.8%** (vs 125 TFLOP/s FP32 dense peak) |
| peak VRAM | 15.0 GB |
| step time | 85.12 ms |
| - data-wait | 4.21 ms (4.9%) |
| - fwd+bwd | 79.36 ms (93.2%) |
| - optimizer | 1.55 ms (1.8%) |

The loader in `b1.py` alone took **105.7** ms per **16,384 token** batch (where
**~155k** tok/sec came from) but the step time in `b4.py` with the loader took
only **85.12 ms**. The data phase in `b4.py` took only **4.21 ms** compared to
**105.7 ms (25x longer)**.

The loader is doing the same thing in both places. `np.memmap` doesn't actually
pull the data, it just makes a pointer/address and the amount of time it would
have taken is offloaded later when the data is called
(`data[i : i+BLOCK_SIZE]   # b1.py:41`). The CPU tries to read an address with
no actual memory in it and has to stop and hand off to the kernel. The kernel
has to find the file and read the 4 KB from disk (**~1.652 ms**[mostly from disk
wait] per slice with **64** slices since `BATCH_SIZE = 64`). The **25x**
discrepancy between the two is dependent on the fact that `b1.py` has to find a
page, 64 times per batch. But in `b4.py`, the data is likely already loaded in
RAM, so the **1.652 ms** of mainly disk wait time is essentially reduced to
none.

In P1, the question was "Is data the bottleneck?" The question is asking what
percentage of the step time is used to make a batch rather than compute
(including optimizer). The amount of time it took to make a batch is only
**4.9%** of the total step time (**4.21 ms**). The compute time took the largest
percentage of the time and it wasn't even close. The fwd+bwd alone took **~79 ms
(93%)** of the **85 ms**. In `b1.py`, the total slice time was **1.652 ms**. In
`b4.py`, I can derive each slice's time (**~0.066 ms**) from the data-wait time
(**4.21/64 = 0.066**). Assuming the **0.066 ms** is non-disk work and constant
between the two runs, **~96% (1.586/1.652)** of the time is disk-wait time
(**1.652-0.066**) in `b1.py`. That means the **155K tok/sec** rate is measuring
the storage, not the loader. If the rate is recalculated using **0.066** rather
than **1.652**:
```
b1:  256 / (1.652 / 1000)  =   154,963 tok/s
b4:  256 / (0.066 / 1000)  =  3,878,788 tok/s 
```
(this is the data phase throughput, NOT the full step)

**20** of these GPUs (3,878,788 / 192,574 = 20.1x) could run on the loader
before data becomes the limit.

A single run reported **0.53 ms** data-wait time and **31% MFU** vs the 100-step
averages of **4.21 ms** and **27.8%**, **8x** faster than the averaged data-wait
time. I thought the data in the single runs didn't seem right so I ran 105
steps, averaged it over 100 and discarded the first 5.

Earlier, I assumed that the data (4 KB) is already loaded into RAM for `b4.py`
and not `b1.py` without actually verifying my assumption. To test it, I'm going
to look for how many times each run had to go to disk with
`/usr/bin/time -v python b1.py` and `/usr/bin/time -v python b4.py`. A MAJOR
fault in the page cache means the page wasn't there and it went to disk; A MINOR
fault means the page was already in RAM.

|             what you would see             |             what it means            |
|---|---|
| b1 thousands of MAJOR faults, b4 near zero |   the page cache explains the 25x    |
|             both runs similar              | it does not, and something else does |

**RESULTS:**
Instead of measuring `b1.py` and `b4.py`, I measured `b1.py`'s loader with a cold pass (ran command `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`) then immediately a warm pass afterwards.

TEST RAN ON 8/3/2026 21:12 **100 COLD and 100 WARM passes**

Cache at start: **93828 kB**

===== COLD PASS =====
first batch:   586.73 ms
mean:   **130.67 ms**   median:  87.28   p90: 299.59   max:   586.73
majflt: **+2451**   minflt: **+34478**

===== WARM PASS =====
first batch:   172.31 ms
mean:    **15.92 ms**   median:  12.00   p90:  27.72   max:   172.31
majflt: **+483**   minflt: **+36413**

Cache at end: **18235524 kB**

The cold pass produced **2451** major faults, roughly 5x the amount in the warm pass, this strongly supports the page cache is behind the 25x difference. In the cold pass, the slice time of the mean (**~2.0 ms**) and median (**~1.4 ms**) straddles the slice time I measured in June (**1.652 ms**). Both of the passes use unseeded `randint` so the passes hit different random slices than each other. Warm being faster is NOT 'revisiting the same pages'. The passes only needed ~3 MB of data (6400 * 512 bytes) each, but cache grew from **~94 MB** at the beginning to **~18.2 GB** at the end.

## Convergence Rule

Training is considered "Done" when loss(val) improves by under **0.5%** in a row
**4** times. The measured noise floor (+- 0.2%) starts at ~9,750 steps and
onwards until run was ended at 12,500 steps. I chose 0.5% because it gave a
little headroom over the noise floor (+- 0.2%) and gave val more time to drop
(0.6% at 10,250 steps). I added "in a row **4** times" to ensure there were no
outliers that could have ended the run early before training could converge
(step 3250 val dipped to 0.5% that recovered to 0.8%). This data came from a
model with **30M** params, with OpenWebText data, ran on 2026-07-26, 12,500
steps. In this run, the loss(val) has not yet converged, val was still improving
0.6% at step 10,250 and ended at 4.458 and still descending. I measured a noise
floor, not the convergence.

## Run #1 launched 8/3/2026 at 14:58 -  18:57

`cd ~/workspace/gpt-roofline && rm -f logs/curve.tsv logs/*.pt && setsid nohup ./.venv/bin/python -u b3.py > logs/run.log 2>&1 &`

| | |
|---|---|
| launched | 2026-08-03 14:58 |
| finished | 18:57 · 3h 59m |
| steps | 100,000 |
| tokens | **1.638B** · 54.5 tok/param · **16.6% of corpus** |
| evals | 400 · every 250 steps · 100 iters |
| best val | **4.203 @ 90,250** (see correction) |
| final val | 4.2247 @ 100,000 |
| 7/26 run | 4.458 @ 12,500 |
| rule fired | **step 7,500** |
| rule on 7/26 curve | ~11,250 · 33% spread |
| val @ 7,500 | ~4.623 |
| overshoot | −0.420 · −9.1% (correction 0.4003 -8.7%) rel · 92.5% of budget |
| step time | 85.30 ms burst · +5-8% sustained @ 84-88 C |
| wall | 3h 59m actual vs ~3.0h planned |
| checkpoints | rolling best + 3 rotating full states / 5k |

This was the first full budget run and its purpose is to see where the rule would have fired, a benchmark for sustained load on the box, and used as reference for future runs. The best weights were saved using a rolling best that saved the weights each time val improves and 3 rotating full states that can be used to resume in case of a crash. The final val **4.2247** is worse than the rolling best **4.203** at **step 90,250**. This time, the convergence rule would have fired at step 7,500 vs 11,250 last time on 7/26. The rule isn't very stable and fires in a noise band rather than a point. This run is consistent with power-law diminishing returns, **13.3x** more compute gave me **9.1% (correction 8.7%)** more quality. (4.623 − 4.203) / 4.623 = ~9.1% (correction: (4.623 - 4.2227) / 4.623 = ~8.7%). One honest caveat is a static learning rate (`LR = 3e-4`) that flattens the tail. The convergence rule will temporarily not govern future runs  because it's still firing in a noise floor. I am going to keep testing and reworking it.

### Correction (2026-08-15): best val 4.203 -> 4.2227
After running 10 fresh draws at the same checkpoint, the mean returned at 4.2227 rather than the logged 4.203. Each eval is the mean of 100 random val batches (sd 0.0064), the 'best' is a running minimum of that noisy series. At the tail of run #1, the line is flat, which means each eval essentially gets the same loss + noise, a random draw. The running minimum over 100 evals picks the best noise, not a better model. Future runs will have eval set and seed frozen, the frozen set reads +0.005 (+0.8 sd) over fresh draws. The measurements for this correction live in `logs/bias_check_20260807_0237.txt` and predictions for these types of effects are at `PREDICTIONS_2026-08-13.md`, written before run #3. 


## Hardware / data

- NVIDIA RTX PRO 6000 Blackwell **Workstation Edition** (96 GB, GB202, 600W; 125
  TFLOP/s FP32 dense peak - the MFU denominator; the Max-Q variant is a
  different card at 110) - single GPU.
- OpenWebText (~9B tokens), GPT-2 BPE (token-level).
