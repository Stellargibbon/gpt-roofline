# gpt-roofline

A real GPT training pipeline on OpenWebText, instrumented for throughput — and the
neural scaling law that falls out of the same runs for free.

Sequel to [`micro-gpt`](../micro-gpt) (from-scratch transformer + honest
hyperparameter sweep). That project answered *"what moves the loss?"* This one
answers *"how fast and cheaply did I get there?"* — the ML-infra / Lane 5 question.

## The deliverable

> Built a real training pipeline on OpenWebText, measured my own roofline
> (MFU vs model size) across a scaling sweep, and made the worst-MFU config
> 40%→55% faster — here's the before/after flame graph. The neural scaling law
> fell out of the same runs.

Systems story in the headline (throughput, MFU, a flame graph and a number).
Scaling-law science in the body (loss vs size, log-log, the power-law exponent).
Both ship from one set of runs.

## Status

Direction locked via a 5-stance council panel (private planning doc).

**Instrumentation complete — P3 landed.** P0 (data prep), P1 (loader), P2 (model +
training loop), and P3 (instrumentation) all done. Next: the size sweep (P4).
- P0 `b0.py` — tokenized OpenWebText → `train.bin` (17G) / `val.bin` (86M) / `meta.json`.
- P1 `b1.py` — memmap loader + first measurement: loader-only throughput ~155K tok/s
  (worst-case floor; the "is data the bottleneck?" question is answered at P3 below).
- P2 `b2.py` (model half) — GPT model, landed: init loss 10.8833 vs ln(50257) ≈ 10.825 on one
  real batch (the random-init sanity), plus GPT-2 residual √-shrink init.
- P2 `b3.py` (loop half) — training loop, landed: `estimate_loss()` (val, `no_grad`), AdamW,
  eval gate. First long run 2026-07-26: 12,500 iters, val loss 10.887 → **4.458**
  (~205M tokens).
- P3 `b4.py` — systems instrumentation, landed: CUDA-event timing harness,
  tokens/sec, MFU, peak VRAM, coarse step-time split, and an N-step measurement
  loop with warmup discard.

### First instrumented numbers (2026-07-26)

30M-param GPT, fp32 (`allow_tf32=False`), B×T = 16,384 tokens/step, n=100 steps
(5 warmup discarded), RTX PRO 6000 Blackwell Workstation Edition:

| metric | value |
|---|---|
| throughput | **192,574 tok/s** |
| MFU | **27.8%** (vs 125 TFLOP/s FP32 dense peak) |
| peak VRAM | 15.0 GB |
| step time | 85.12 ms |
| ├ data-wait | 4.21 ms (4.9%) |
| ├ fwd+bwd | 79.36 ms (93.2%) |
| └ optimizer | 1.55 ms (1.8%) |

The step-time split answers P1's open question: **the data path is not the
bottleneck** (4.9%), and neither is the optimizer (1.8%) — 93% of the step is
compute, so the ~72% of peak left on the table is memory-bound compute rather
than starvation. That's the roofline P4 will map and P5 will attack.

Measurement note: single-shot runs reported 0.53 ms data-wait and 31% MFU, both
wrong. Step 0's `get_batch` hits data still warm in the OS page cache from the
warmup loop, so it undercounts the loader by ~8×. Only the N-step loop exposes
it — the numbers above are the honest ones.

## Hardware / data

- NVIDIA RTX PRO 6000 Blackwell **Workstation Edition** (96 GB, GB202, 600W;
  125 TFLOP/s FP32 dense peak — the MFU denominator above; the Max-Q variant is
  a different card at 110) — single GPU.
- OpenWebText (~9B tokens), GPT-2 BPE (token-level).

## Phase plan

Phases P0→P8 per the private spec. Current position: P3 landed
(instrumentation); next is P4 (the size sweep) — see Status above.
