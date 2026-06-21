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
plan, the panel's reasoning, and the guardrails). Not started — first move is P0.

## Hardware / data

- NVIDIA Blackwell 6000 (96GB, native fp8) — single GPU.
- OpenWebText (~9B tokens), GPT-2 BPE (token-level).

## The first move (P0)

`prepare.py` — tokenize OpenWebText once (`tiktoken` GPT-2 BPE), carve the val
split *before* tokenizing, write `train.bin` / `val.bin` as uint16 memmaps. Then
`dataset.py::get_batch(split)` over the memmap, and measure loader-only
tokens/sec before touching the model.

Full phase plan P0→P8: see `PROJECT_SPEC.md`.
