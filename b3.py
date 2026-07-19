"""
b3.py — the training loop of gpt-roofline.  [YOUR BUILD-FROM-BLANK REP]

Goal: train the GPT model from b2 on OpenWebText batches from b1's loader.
The mechanical loop is yours (Chain Rule -> Backprop -> GD -> the update
line -> SGD/Adam spine — [OWN] cold); this brick adds the optimizer choice
(AdamW), periodic train/val eval, and the deferred residual-init scaling
from b2's doorstep.

Write every real line yourself. Skeleton = the steps + the names, not the logic.
Stuck on ONE line? Ask me — no answer key. You own the blank page.
"""

# ---- imports --------------------------------------------------------------
# You need: torch, torch.optim (for AdamW), time (optional — for timing).
# Reuse b1's loader and b2's model: from b1 import get_batch, DEVICE; from b2 import GPT.
import torch
import torch.optim
import time
from b1 import get_batch, DEVICE
from b2 import GPT


# ---- knobs (canonical names — every judgment call is a visible variable) ---
# LR            -> learning rate (GPT-2 scale: ~3e-4 for small models; the main
#                  knob you'll tune in your sweep later — your call here)
# MAX_ITERS     -> total training iterations for a REAL run (thousands+)
# EVAL_INTERVAL -> print eval loss every N iterations
# EVAL_ITERS    -> batches to average for one eval estimate (one batch is noisy)
# SMOKE_ITERS   -> the entrypoint's smoke-scale loop count (tiny — prove the
#                  loop runs end-to-end, nothing more)
LR = 3e-4              # canonical GPT-2 small value; you own this knob — tune later
MAX_ITERS = 5000       # placeholder for the ARMED path; sweep target lives in your spec
EVAL_INTERVAL = 200
EVAL_ITERS = 200
SMOKE_ITERS = 10       # entrypoint only — never a real run


# ---- model setup ----------------------------------------------------------
# 1. model = GPT(config).to(DEVICE)
# 2. VERIFY the deferred residual-init scaling (1/sqrt(2*N_LAYER) per GPT-2) lives
#    in b2's GPT.__init__, right after self.apply(...) — it belongs INSIDE the
#    model (part of its identity), NOT here in the trainer. You're building it in
#    b2 now (the named_parameters + endswith touch-up). b3 just trusts it exists.
#    [CC review fix: original skeleton placed the scaling here — wrong altitude.]
model = GPT(None).to(DEVICE)


# ---- optimizer ------------------------------------------------------------
# AdamW on model.parameters() with lr=LR.
# (AdamW, not vanilla Adam — the decoupled weight decay is the GPT-2/nanoGPT
#  choice. If Adam vs AdamW is unclear, that's a fair question — ask.)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


# ---- eval helper ----------------------------------------------------------
# estimate_loss(): average loss over EVAL_ITERS batches for BOTH "train" and "val".
# Why eval mode: dropout is noise — eval turns it OFF so the loss read is stable.
# Why average over many batches: one batch's loss is noisy; the mean is the signal.
# Why no_grad: you're measuring, not training — don't build a graph you'll throw away.
#
# Steps:
# 1. turn off dropout (eval mode)
# 2. for each split ("train", "val"):
#      run EVAL_ITERS batches through the model, collect per-batch losses, average
# 3. turn dropout back on (train mode)
# 4. return the two averages
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}                            # labeled lockers — born ONCE (name it yourself)
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)   # numbered lockers — born fresh EACH split
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)                  # a batch from THIS split
            loss, logits = model(x, y)     # b2's return order — check YOUR forward
            losses[i] = loss               # this rep's loss into slot i
        out[split] = losses.mean()               # this split's finished average, filed by name
    model.train()
    return out


# ---- training loop (the [OWN] spine — you built this before, build it again) ---
# for it in range(n_iters):          (don't name the loop variable `iter` — that
#                                     shadows a Python builtin)
# 1. x, y = get_batch("train")
# 2. get the loss from the model — CHECK YOUR OWN b2 forward's return line: when
#    targets are given it hands back TWO things. Catch accordingly, or step 4
#    will throw "tuple has no attribute backward".
#    [CC review fix: original skeleton said `loss = model(x, y)` — wrong against
#     your b2 signature.]
# 3. optimizer.zero_grad()
# 4. loss.backward()
# 5. optimizer.step()
# 6. every EVAL_INTERVAL: print train/val loss via estimate_loss
def train_loop(n_iters):
    for i in range(n_iters):
        if i % EVAL_INTERVAL == 0:
            ev = estimate_loss()
            print(f"step {i}  train {ev['train']:.4f}  val {ev['val']:.4f}")
        x, y = get_batch("train")
        loss, logits = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# ---- entrypoint -----------------------------------------------------------
# HARD RULE: this entrypoint runs SMOKE_ITERS ONLY — a smoke test that proves
# the loop runs end-to-end and the loss goes DOWN from the b2 init (~10.8).
# Real/long training (MAX_ITERS) is ARMED territory: the author's `m arm` switch
# creates the ARMED marker file, and `m train` is the only path that fires a
# real run. This file must NEVER auto-fire a real run. The safety gate exists
# because training is expensive and one accidental fire wastes a GPU session.
#
# 1. Build model + optimizer (the setup block above)
# 2. Run the loop for SMOKE_ITERS iterations — just prove the loop runs
# 3. Print: final loss, and confirm it went DOWN from ~10.8
# 4. If loss did NOT descend, something in the loop is wrong — debug before arming.


if __name__ == "__main__":
    train_loop(SMOKE_ITERS)
    ev = estimate_loss()
    print(f"FINAL  train {ev['train']:.4f}  val {ev['val']:.4f}")