# gpt-roofline

A real GPT training pipeline on OpenWebText, instrumented for throughput and the
neural scaling law that falls out of the same runs.

Sequel to [`micro-gpt`](https://github.com/Stellargibbon/micro-gpt) (from-scratch transformer +
hyperparameter sweep). That project answered *"what moves the loss?"* This one
answers *"how fast and cheaply did I get there?"*

## Status

- P0 `b0.py` - tokenized OpenWebText -> `train.bin` (17G) / `val.bin` (86M) / `meta.json`.
- P1 `b1.py` - memmap loader + first measurement: loader-only throughput ~155K tok/s
  (worst-case floor; the "is data the bottleneck?" question is answered at P3).
- P2 `b2.py` (model half) - GPT model: init loss 10.8833 vs ln(50257) ~ 10.825 on one
  real batch (the random-init sanity), plus GPT-2 residual sqrt-shrink init.
- P2 `b3.py` (loop half) - training loop: `estimate_loss()` (val, `no_grad`), AdamW,
  eval gate. First long run 2026-07-26: 12,500 iters, val loss 10.887 -> **4.458**
  (~205M tokens).
- P3 `b4.py` - systems instrumentation: CUDA-event timing harness,
  tokens/sec, MFU, peak VRAM, coarse step-time split, and an N-step measurement
  loop with warmup discard.

### Instrumented numbers (2026-07-26)

30M-param GPT, fp32 (`allow_tf32=False`), B*T = 16,384 tokens/step, n=100 steps
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

The step-time split: **the data path is not the
bottleneck** (4.9%), and neither is the optimizer (1.8%) - 93% of the step is
compute, so the ~72% of peak left on the table is memory-bound compute rather
than starvation.

Measurement note: single runs reported 0.53 ms data-wait and 31% MFU, both
wrong. Step 0's `get_batch` hits data still warm in the OS page cache from the
warmup loop, so it undercounts the loader by ~8x. Only the N-step loop exposes
it. The numbers above are the numbers averaged over 100 steps.

## Convergence Rule

Training is considered "Done" when loss(val) improves by under **0.5%** in a row **4** times.
The measured noise floor (+- 0.2%) starts at ~9,750 steps and onwards until run was ended at
12,500 steps. I chose 0.5% because it gave a little headroom over the noise floor (+- 0.2%)
and gave val more time to drop (0.6% at 10,250 steps). I added "in a row **4** times" to
ensure there were no outliers that could have ended the run early before training could
converge (step 3250 val dipped to 0.5% that recovered to 0.8%). This data came from a model
with **30M** params, with OpenWebText data, ran on 2026-07-26, 12,500 steps. In this
run, the loss(val) has not yet converged, val was still improving 0.6% at step 10,250 and
ended at 4.458 and still descending. I measured a noise floor, not the convergence.

## Hardware / data

- NVIDIA RTX PRO 6000 Blackwell **Workstation Edition** (96 GB, GB202, 600W;
  125 TFLOP/s FP32 dense peak - the MFU denominator; the Max-Q variant is
  a different card at 110) - single GPU.
- OpenWebText (~9B tokens), GPT-2 BPE (token-level).
