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
import math
from b1 import get_batch, DEVICE
from b2 import GPT


# ---- knobs (canonical names - every judgment call is a visible variable) ---
LR = 3e-4              # canonical GPT-2 small value
MIN_LR = 3e-5
MAX_ITERS = 100000       # placeholder for the ARMED path
EVAL_INTERVAL = 250
EVAL_ITERS = 100
SMOKE_ITERS = 400       # entrypoint only - never a real run
WARMUP_ITER = 300
EVAL_SEED = 1234
torch.manual_seed(EVAL_SEED)
RUN_SEED = 2
VAL_BATCHES = []
for _ in range(EVAL_ITERS):
    VAL_BATCHES.append(get_batch("val"))
torch.manual_seed(RUN_SEED)
# ---- model setup ----------------------------------------------------------
model = GPT(None).to(DEVICE)


# ---- optimizer ------------------------------------------------------------
# AdamW, not vanilla Adam - decoupled weight decay is the GPT-2/nanoGPT choice.
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

def get_lr(step, total):
    if step < WARMUP_ITER:
        return (step / WARMUP_ITER) * LR
    if step > total:
        return MIN_LR
    else: 
        decay_coeff = (math.cos((step - WARMUP_ITER) / (total - WARMUP_ITER) * math.pi) + 1) / 2
        return MIN_LR + decay_coeff * (LR - MIN_LR)

# ---- eval helper ----------------------------------------------------------
@torch.no_grad()
def estimate_val():
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for i, (x,y) in enumerate(VAL_BATCHES):
        loss, logits = model(x, y)     # b2's return order
        losses[i] = loss
    model.train()
    return losses.mean()

@torch.no_grad()
def estimate_train():
    model.eval()
    losses = torch.zeros(EVAL_ITERS)
    for i in range(EVAL_ITERS):
        x, y = get_batch('train')
        loss, logits = model(x, y)     # b2's return order
        losses[i] = loss
    model.train()
    return losses.mean()

# ---- training loop -------------------------------------------------------
def train_loop(n_iters):
    best_val = float('inf')
    for i in range(n_iters):
        lr = get_lr(i, n_iters)
        optimizer.param_groups[0]['lr'] = lr
        if i % EVAL_INTERVAL == 0:
            val_loss = float(estimate_val())
            print(f"step {i}  val {val_loss:.4f}")
            if best_val > val_loss:
                ckpt = {'model': model.state_dict(), 'step': i, 'val': val_loss}
                torch.save(ckpt, 'logs/best.pt')
                best_val = val_loss
            with open('logs/curve.tsv', "a") as f:
                f.write(f"{i}\t{val_loss:.4f}\t{lr:.4e}\n")
        if i % 5000 == 0:
            slot = i // 5000 % 3
            path = f'logs/ckpt_{slot}.pt'
            state = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': i, 'best_val': best_val}
            torch.save(state, path)
            train_loss = float(estimate_train())
            with open('logs/sanity.tsv', 'a') as a: 
                a.write(f"{i}\t{val_loss:.4f}\t{train_loss:.4f}\n")

        x, y = get_batch("train")
        loss, logits = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
# ---- entrypoint -----------------------------------------------------------


if __name__ == "__main__":
    train_loop(MAX_ITERS)
    ev = estimate_val()
    et = estimate_train()
    print(f"FINAL  train {et:.4f}  val {ev:.4f}")