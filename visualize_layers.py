import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os

def build_model(device):
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.SiLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.SiLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.SiLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.SiLU(),
        nn.MaxPool2d(2),

        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=0.5),
        nn.Linear(128, 1),
    )
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(device)

state_dict = torch.load("83_acc/final_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

os.makedirs("layer_filters", exist_ok=True)

conv_layers = []
for idx, m in enumerate(model.modules()):
    if isinstance(m, nn.Conv2d):
        conv_layers.append((idx, m))

print(f"Found {len(conv_layers)} conv layers.")

for idx, conv in conv_layers:
    w = conv.weight.data.clone() 
    if w.shape[1] > 1:  
        w = w.mean(dim=1, keepdim=True)   
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    grid = vutils.make_grid(w, nrow=8, padding=2)
    grid_cpu = grid.cpu().permute(1, 2, 0).squeeze()

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_cpu, cmap="gray")
    plt.axis("off")
    plt.title(f"Conv layer @ index {idx} filters")

    save_path = f"layer_filters/layer_{idx}_conv_filters.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved {save_path}")
