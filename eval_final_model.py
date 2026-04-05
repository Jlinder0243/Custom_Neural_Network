import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.amp import autocast

CHECKPOINT_DIR = "checkpoints"
TEST_DIR = "test"
IMG_SIZE = 256
BATCH_SIZE = 256
NUM_WORKERS = 6
PREFETCH = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# I needed to do this to make gpu optimizations work for CUDA 13.0
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
        nn.Dropout(0.5),
        nn.Linear(128, 1),
    )
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.to(device)

def load_state_or_none(path):
    if not os.path.exists(path):
        return None
    data = torch.load(path, map_location=DEVICE)
    if all(isinstance(v, torch.Tensor) for v in data.values()):
        return data

    if "model" in data:
        return data["model"]
    if "ema_model" in data:
        return data["ema_model"]

    return None

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Evaluating", ncols=120)
    for imgs, targets in pbar:

        if device.type == "cuda":
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
        else:
            imgs = imgs.to(device, non_blocking=True)

        targets = targets.float().unsqueeze(1).to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(torch.int32)

        total_correct += (preds == targets.to(torch.int32)).sum().item()
        total_loss += loss.item()
        total_samples += targets.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy, total_samples

def main():
    raw_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    ema_path = os.path.join(CHECKPOINT_DIR, "final_model_ema.pt")

    raw_state = load_state_or_none(raw_path)
    ema_state = load_state_or_none(ema_path)

    if raw_state is None and ema_state is None:
        print("No models found.")
        return
    if raw_state is not None:
        print(f"\nLoading RAW model: {raw_path}")
        model = build_model(DEVICE)
        model.load_state_dict(raw_state, strict=False)
        loss, acc, n = evaluate_model(model, test_loader, DEVICE)
        print(f"RAW loss={loss:.6f}, acc={acc:.4f}, samples={n}")
        del model
        torch.cuda.empty_cache()
    if ema_state is not None:
        print(f"\nLoading EMA model: {ema_path}")
        model = build_model(DEVICE)
        model.load_state_dict(ema_state, strict=False)
        loss, acc, n = evaluate_model(model, test_loader, DEVICE)
        print(f"EMA loss={loss:.6f}, acc={acc:.4f}, samples={n}")
        del model
        torch.cuda.empty_cache()

    print("\nDONE.")

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

test_dataset = datasets.ImageFolder(TEST_DIR, test_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=True,
    prefetch_factor=PREFETCH
)

if __name__ == "__main__":
    main()
