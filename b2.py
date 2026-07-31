"""
b2.py - the GPT model of gpt-roofline.

Goal: a GPT-style transformer that takes a batch (B, T) of token ids and returns
a SCALAR loss (next-token prediction, cross-entropy). No training loop here -
that's b3. This brick is just the architecture + a forward pass, sanity-checked
against one real batch from b1's loader.
"""

# ---- imports --------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from b1 import get_batch, DEVICE


# ---- knobs (canonical names - every judgment call is a visible variable) ---
# VOCAB_SIZE  -> GPT-2 vocab (50257)
# BLOCK_SIZE  -> context length T (must match b1's BLOCK_SIZE)
# N_LAYER     -> number of transformer blocks (depth)
# N_HEAD      -> number of attention heads
# N_EMBD      -> embedding width (channels)
# DROPOUT     -> dropout rate (0 = off)
VOCAB_SIZE = 50257
BLOCK_SIZE = 256
N_LAYER = 6
N_HEAD = 6
N_EMBD = 384
DROPOUT = 0.2


# ---- config helper --------------------------------------------------------


# ---- Block (one transformer block) ----------------------------------------
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP(config)


    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---- CausalSelfAttention ---------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3*N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.n_head = N_HEAD
        self.head_dim = N_EMBD // N_HEAD
        


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
# 4x is the canonical GPT expansion.
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(N_EMBD, 4*N_EMBD)
        self.GELU = nn.GELU()
        self.proj = nn.Linear(4*N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    


    def forward(self, x):
        x = self.fc(x)
        x = self.GELU(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x
        

# ---- GPT (the full model) -------------------------------------------------

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
if __name__ == "__main__":
    x, y = get_batch("train")
    model = GPT(None).to(DEVICE)
    with torch.no_grad():
        loss, logits = model(x, y)
    print(loss, logits.shape)
    print(sum(1 for n, _ in model.named_parameters() if n.endswith('proj.weight')))
    