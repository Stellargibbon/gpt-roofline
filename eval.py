import torch
from b1 import get_batch, DEVICE
from b2 import GPT
CHECKPOINTS = ["logs/checkpoints/run2A_seed2_best.pt", "logs/checkpoints/run2B_seed1_best.pt", "logs/checkpoints/run3A_seed3_best.pt", "logs/checkpoints/run3B_seed4_best.pt"]
for path in CHECKPOINTS:
    vals = torch.load(path)
    model = GPT(None).to(DEVICE)
    model.load_state_dict(vals["model"])
    model.eval()
    draw_means = []
    with torch.no_grad():
        for i in range(10):
            all_draw = 0
            for i in range(100):
                x, y = get_batch("val")
                out = model(x,y)
                all_draw += out[0]
            draw_means.append(all_draw / 100)
    print(draw_means)