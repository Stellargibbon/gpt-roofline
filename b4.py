#30M-param GPT, fp32, RTX PRO 6000 Blackwell. 192,574 tok/s at 27.8% MFU, 15.0 GB peak VRAM. Step time 85.12 ms = 4.21 data-wait (4.9%) + 79.36 compute (93.2%) + 1.55 optimizer (1.8%). n=100, 5 warmup steps excluded.
#GPU USED: NVIDIA RTX PRO 6000 Blackwell Workstation Edition / GB202
#Peak = 125e12 FLOP/s (https://www.techpowerup.com/gpu-specs/)
#"FP32 dense, allow_tf32=False"
#Early N=1 estimate was ~0.31 but after n=100, MFU= ~0.278 
# tok/s = (B * T * accum) / (step_time_ms / 1000)
#       = tokens per optimizer step / seconds per step
#Early N=1 estimate was 211,000/s, but after n=100 ~ 192,500 tok/s
#n=100 measurements taken at 14:05 7/26/2026

import torch
import torch.nn as nn
from b3 import get_batch, DEVICE, model, train_loop, optimizer, EVAL_INTERVAL

loop = 105
warmup = 5
accum = 1
param_count = 0
peak = 125e12

mark_start = torch.cuda.Event(enable_timing=True)
mark_batch = torch.cuda.Event(enable_timing=True)
mark_compute = torch.cuda.Event(enable_timing=True)
mark_optimizer = torch.cuda.Event(enable_timing=True)

train_loop(warmup)

#==========TOKENS/SECOND==========#
def tok_s(x, accum, millisec):
    B, T = x.shape
    tok = B * T * accum
    sec = millisec/1000
    tok_sec = tok / sec
    return tok_sec

#==========MFU(Model FLOPs Utilization)==========#

for i in model.parameters(): param_count += i.numel()
def mfu(param_count, peak):
    achieved_FLOPS_sec = 6 * param_count * tok_s(x, accum, millisec)
    return achieved_FLOPS_sec / peak

stats = {"time": [], "batch": [], "compute": [], "optimizer": [], "tok_s": [], "MFU": [], "VRAM": []}
#==========LOOP==========#
for step in range(loop):
    print("Step:", step)
#==========TIMING HARNESS==========#
    torch.cuda.reset_peak_memory_stats()
    mark_start.record()
    x, y = get_batch("train")
    mark_batch.record()
    loss = model(x, y)[0]
    optimizer.zero_grad()
    loss.backward()
    mark_compute.record()
    optimizer.step()
    mark_optimizer.record()
    torch.cuda.synchronize()
    vram_bytes = torch.cuda.max_memory_allocated()
    millisec = mark_start.elapsed_time(mark_optimizer)
    stats["time"].append(millisec)
    batch_time = mark_start.elapsed_time(mark_batch)
    stats["batch"].append(batch_time)
    compute_time = mark_batch.elapsed_time(mark_compute)
    stats["compute"].append(compute_time)
    optimizer_time = mark_compute.elapsed_time(mark_optimizer)
    stats["optimizer"].append(optimizer_time)
    time_check = batch_time + compute_time + optimizer_time
    print(f"""data:      {batch_time:.4f}
compute:   {compute_time:.4f}
optimizer: {optimizer_time:.4f}
sanity check: {millisec - time_check:.6f}
time(millisec): {millisec:.4f}""")
#==========TOKENS/SECOND==========#
    stats["tok_s"].append(tok_s(x, accum, millisec))
    print('Tokens/Second: ', tok_s(x, accum, millisec))
#==========MFU(Model FLOPs Utilization)==========#
    stats["MFU"].append(mfu(param_count, peak))
    print('MFU: ', mfu(param_count, peak))
#==========PEAK VRAM==========#
    vram_gigabytes = vram_bytes / 1024**3
    stats["VRAM"].append(vram_gigabytes)
    print('peak VRAM (GB):', vram_gigabytes)

for name, values in stats.items():
    average = sum(values[5:]) / len(values[5:])
    print(f"{name}: {average:.4f}")
