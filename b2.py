"""
b2.py — the GPT model of gpt-roofline.  [YOUR BUILD-FROM-BLANK REP]

Goal: a GPT-style transformer that takes a batch (B, T) of token ids and returns
a SCALAR loss (next-token prediction, cross-entropy). No training loop here —
that's b3. This brick is just the architecture + a forward pass, sanity-checked
against one real batch from b1's loader.

Write every real line yourself. Skeleton = the steps + the names, not the logic.
Stuck on ONE line? Ask me — no answer key. You own the blank page.
"""

# ---- imports --------------------------------------------------------------
# You need: math, torch, torch.nn as nn, torch.nn.functional as F.
# Reuse b1's loader for the sanity check: from b1 import get_batch.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from b1 import get_batch, DEVICE


# ---- knobs (canonical names — every judgment call is a visible variable) ---
# VOCAB_SIZE  -> GPT-2 vocab (50257). Don't re-derive — it's in the tokenizer.
# BLOCK_SIZE  -> context length T (MUST match b1's BLOCK_SIZE — import or mirror).
# N_LAYER     -> number of transformer blocks (depth)
# N_HEAD      -> number of attention heads
# N_EMBD      -> embedding width (channels)
# DROPOUT     -> dropout rate (0 = off; keep small for a first run)
VOCAB_SIZE = 50257
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2


# ---- config helper --------------------------------------------------------
# A small dict or SimpleNamespace grouping the knobs so you can pass one object
# into the model instead of five args. Optional — if it stalls you, skip it and
# just read the module-level knobs directly inside the class.
# (If you keep it: name it `config` and have __init__ take `config=None`,
# falling back to the module globals when None.)



# ---- Block (one transformer block) ----------------------------------------
# The inner unit. Two sub-layers, each pre- or post-LN — nanoGPT uses post-LN
# (add then norm is WRONG for stability in practice; nanoGPT does norm-then-add).
# Actually nanoGPT style: x = x + attn(ln1(x)); x = x + mlp(ln2(x)). Pre-LN.
#
# Steps:
# 1. ln1 = LayerNorm(N_EMBD)
# 2. attn = causal self-attention (see the CausalAttention class below)
# 3. ln2 = LayerNorm(N_EMBD)
# 4. mlp = MLP (linear -> gelu -> linear -> dropout)
# 5. forward(x): x = x + attn(ln1(x)); x = x + mlp(ln2(x)); return x
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP(config)
        # ONE LINE PER OBJECT. ln1, attn, ln2, mlp. (the four module attrs)



    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x  # x = x + attn(ln1(x)); x = x + mlp(ln2(x)); return x


# ---- CausalSelfAttention ---------------------------------------------------
# The core. Steps:
# 1. qkv projection: ONE linear that outputs 3*N_EMBD channels, then chunk into q/k/v.
# 2. reshape to (B, T, N_HEAD, head_dim) and transpose to (B, N_HEAD, T, head_dim).
# 3. attention = softmax(q @ k^T / sqrt(head_dim) + causal_mask) @ v
#    (use F.scaled_dot_product_attention if available — it's fused + faster —
#     is_causal=True handles the mask. That IS the modern nanoGPT way.)
# 4. recombine: (B, N_HEAD, T, head_dim) -> (B, T, N_EMBD)
# 5. output projection: linear(N_EMBD -> N_EMBD), then dropout.
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3*N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.n_head = N_HEAD
        self.head_dim = N_EMBD // N_HEAD
        
        # key/query/value = ONE combined c_attn (3*N_EMBD), output = c_proj (N_EMBD)
        # dropout for attn + resid. head_dim = N_EMBD // N_HEAD.



    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attention = attention.transpose(1, 2).reshape(B, T, N_EMBD)
        attention = self.c_proj(attention)
        return self.resid_dropout(attention)
# ---- MLP ------------------------------------------------------------------
# The feed-forward. Linear(N_EMBD -> 4*N_EMBD) -> gelu -> Linear(4*N_EMBD -> N_EMBD) -> dropout.
# 4x is the canonical GPT expansion. Just four lines in __init__.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(N_EMBD, 4*N_EMBD)
        self.GELU = nn.GELU()
        self.proj = nn.Linear(4*N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    
        # fc, proj, dropout. (GELU via nn.GELU or F.gelu — pick one.)



    def forward(self, x):
        x = self.fc(x)
        x = self.GELU(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
        


# ---- GPT (the full model) -------------------------------------------------
# Token + positional embeddings -> drop -> blocks -> ln_f -> lm_head -> logits.
# Then compute next-token CE loss (shift by one internally — same invariant as
# b1's get_batch: predict token t+1 from token t). Return SCALAR loss.
#
# __init__ steps:
# 1. wte = token embedding  (VOCAB_SIZE, N_EMBD)
# 2. wpe = positional embedding  (BLOCK_SIZE, N_EMBD)
# 3. drop = Dropout
# 4. blocks = nn.ModuleList of N_LAYER Blocks
# 5. ln_f = LayerNorm(N_EMBD)
# 6. lm_head = Linear(N_EMBD, VOCAB_SIZE, bias=False)
#    (weight tying with wte is a nanoGPT OPTION — decide: tie or not. If you tie,
#     skip the separate head and use wte.weight as the output projection. Your call.)
# 7. init weights: apply _init_weights to every submodule, scale the residual
#    layers by 1/sqrt(2*N_LAYER) per GPT-2 (the std scaling for stability).
#
# # forward(idx, targets=None) steps:
# 1. B, T = idx.shape. assert T <= BLOCK_SIZE (can't attend beyond context).
# 2. tok = wte(idx), pos = wpe(torch.arange(T)) -> x = drop(tok + pos)
# 3. x = blocks(x); x = ln_f(x)
# 4. logits = lm_head(x)   -> (B, T, VOCAB_SIZE)
# 5. if targets is None: return logits (inference path)
# 6. else: loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1),
#        ignore_index=-1). return loss, logits

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.wpe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([ Block(config) for _ in range(N_LAYER) ])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith('proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02/math.sqrt(2*N_LAYER))
        




    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        # nn.Linear -> nn.init.normal_(w, mean=0, std=0.02); bias zeros.
        # nn.Embedding -> same. (GPT-2 init.)


    def forward(self, idx, targets=None):
        B, T = idx.shape 
        assert T <=BLOCK_SIZE
        tok = self.wte(idx)
        pos = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        for block in self.blocks:
            x = block(x) 
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None: return logits
        else: loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-1); return loss, logits



# ---- entrypoint -----------------------------------------------------------
# Sanity check: one real batch from b1's loader through the model.
# 1. x, y = get_batch("train")   (b1 already gives int64 on the right device)
# 2. model = GPT(); move to DEVICE (import DEVICE from b1, or recompute here)
# 3. loss, logits = model(x, y)  — this is a forward in eval; no grad needed.
#    Wrap in `with torch.no_grad():` so you don't build a graph for the sanity.
# 4. print: loss (scalar), logits.shape (expect (B, T, VOCAB_SIZE)).
# 5. A naive loss sanity: a random init over a 50257-way vocab gives
#    -ln(1/50257) ~ 10.8. If your loss is far from ~10.8 at init, something's
#    wrong with the init or the head. This is the smell test for P2's start.
if __name__ == "__main__":
    x, y = get_batch("train")
    model = GPT(None).to(DEVICE)
    with torch.no_grad():
        loss, logits = model(x, y)
    print(loss, logits.shape)
    print(sum(1 for n, _ in model.named_parameters() if n.endswith('proj.weight')))
    