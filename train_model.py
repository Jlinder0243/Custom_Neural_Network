import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageFile
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import copy

# I needed to manually enable these for CUDA 13.0 support
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

IMG_SIZES = [200, 232, 256]
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 6  
PREFETCH = 4
BATCH_SIZE = 256

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def get_progressive_size(epoch, total_epochs, sizes):
    seg = total_epochs / (len(sizes) - 1)
    idx = int((epoch - 1) // seg)
    idx = min(idx, len(sizes) - 2)
    start_size = sizes[idx]
    end_size = sizes[idx + 1]
    pct = ((epoch - 1) % seg) / seg
    return int(start_size + pct * (end_size - start_size))

class RandomPixelDrop(nn.Module):
    def __init__(self, drop_prob=0.01):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.rand_like(x)
        x = torch.where(mask < self.drop_prob, torch.tensor(0.0, device=x.device), x)
        return x

def is_valid_file(file):
    try:
        with Image.open(file) as im:
            im.verify()
        return True
    except:
        return False

def pil_loader_safe(path):
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode == "P" and "transparency" in im.info:
                im = im.convert("RGBA")
            return im.convert("RGB")
    except:
        return Image.new("RGB", (256, 256), color=0)

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = copy.deepcopy(model)
        self.decay = decay
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(model_p, alpha=1 - self.decay)

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

def save_checkpoint(model, ema_model, optimizer, scaler, epoch, best_loss):
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "ema_model": ema_model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, path)
    print(f"Saved checkpoint at epoch {epoch} (raw + EMA)")

def load_latest_checkpoint(model, ema_model, optimizer, scaler, device):
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")]
    if not checkpoints:
        return 0, float("inf")
    checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
    latest = checkpoints[-1]
    path = os.path.join(CHECKPOINT_DIR, latest)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    ema_model.model.load_state_dict(checkpoint["ema_model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    print(f"Loaded checkpoint '{latest}' (epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint["best_loss"]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(device)
    ema_model = EMA(model)

    criterion = nn.BCEWithLogitsLoss()

    try:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    start_epoch, best_loss = load_latest_checkpoint(model, ema_model, optimizer, scaler, device)

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        img_size = get_progressive_size(epoch, EPOCHS, IMG_SIZES)
        train_transforms = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        ]
        if epoch > 5:
            train_transforms += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
            ]
        train_transforms += [
            transforms.ToTensor(),
            RandomPixelDrop(0.01),
            transforms.Normalize([0.5], [0.5]),
        ]
        train_transforms = transforms.Compose(train_transforms)

        test_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        train_dataset = datasets.ImageFolder(
            "train", transform=train_transforms,
            loader=pil_loader_safe, is_valid_file=is_valid_file
        )
        test_dataset = datasets.ImageFolder(
            "test", transform=test_transforms,
            loader=pil_loader_safe, is_valid_file=is_valid_file
        )

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=PREFETCH
        )
        test_loader = DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
            persistent_workers=(NUM_WORKERS > 0), prefetch_factor=PREFETCH
        )
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (size={img_size})", leave=True)

        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema_model.update(model)

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} loss: {epoch_loss:.6f}")
        if epoch % 2 == 0:
            save_checkpoint(model, ema_model, optimizer, scaler, epoch, best_loss)

        scheduler.step()
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pt"))
    torch.save(ema_model.model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model_ema.pt"))
    print("Saved final model and EMA model.")
