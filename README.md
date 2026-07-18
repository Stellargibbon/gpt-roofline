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

Direction locked via a 5-stance council panel (see `PROJECT_SPEC.md` for the full
plan, the panel's reasoning, and the guardrails).

**In progress — P3 (the training loop).** P0 (data prep), P1 (loader), and P2 (the model) done:
- P0 `b0.py` — tokenized OpenWebText → `train.bin` (17G) / `val.bin` (86M) / `meta.json`.
- P1 `b1.py` — memmap loader + first measurement: loader-only throughput ~155K tok/s
  (worst-case floor; the real "is data the bottleneck?" call comes at P3 vs GPU throughput).
- P2 `b2.py` — GPT model, landed: init loss 10.8833 vs ln(50257) ≈ 10.825 on one
  real batch (the random-init sanity), plus GPT-2 residual √-shrink init.
- P3 `b3.py` — training loop, in progress: model setup + AdamW optimizer done; eval
  helper, loop, and loss curve next.

Bricks build-from-blank; each `b<N>.py` lands (force-added + pushed) once it runs
end-to-end. Heavy runs happen on the Blackwell box, not the dev laptop.

## Hardware / data

- NVIDIA Blackwell 6000 (96GB, native fp8) — single GPU.
- OpenWebText (~9B tokens), GPT-2 BPE (token-level).

## Phase plan

Full phase plan P0→P8: see `PROJECT_SPEC.md`. Current position: P2 (the model) —
see Status above.
