"""
b3.py - the training loop of gpt-roofline.

Goal: train the GPT model from b2 on OpenWebText batches from b1's loader.
This brick adds the optimizer choice (AdamW), periodic train/val eval, and
the deferred residual-init scaling from b2's doorstep.
"""

# ---- imports --------------------------------------------------------------
import torch
import torch.optim
import time
from b1 import get_batch, DEVICE
from b2 import GPT


# ---- knobs (canonical names - every judgment call is a visible variable) ---
LR = 3e-4              # canonical GPT-2 small value
MAX_ITERS = 12500       # placeholder for the ARMED path
EVAL_INTERVAL = 250
EVAL_ITERS = 100
SMOKE_ITERS = 10       # entrypoint only - never a real run


# ---- model setup ----------------------------------------------------------
model = GPT(None).to(DEVICE)


# ---- optimizer ------------------------------------------------------------
# AdamW, not vanilla Adam - decoupled weight decay is the GPT-2/nanoGPT choice.
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


# ---- eval helper ----------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(split)
            loss, logits = model(x, y)     # b2's return order
            losses[i] = loss
        out[split] = losses.mean()
    model.train()
    return out


# ---- training loop -------------------------------------------------------
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


if __name__ == "__main__":
    train_loop(SMOKE_ITERS)
    ev = estimate_loss()
    print(f"FINAL  train {ev['train']:.4f}  val {ev['val']:.4f}")