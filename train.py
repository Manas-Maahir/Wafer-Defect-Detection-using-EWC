import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import time

from tqdm import tqdm
from model import HybridWaferModel
from dataset_memmap import WaferDataset
from continual_learning import EWC, selective_ewc_loss

# ─────────────────────────── CONFIG ───────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 8
LR           = 1e-4
VAL_SPLIT    = 0.15          # 15 % of data used for validation
EWC_LAMBDA   = 500           # regularisation strength; tune as needed
GRAD_CLIP    = 1.0           # max gradient norm
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# ─────────────────────────── DATA ─────────────────────────────
full_dataset = WaferDataset("polar_strips.npy", "labels.csv")

val_size   = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Class weights computed from the FULL label array, then subset to training indices
all_labels = full_dataset.labels
train_indices = train_dataset.indices
train_labels  = all_labels[train_indices]
class_counts  = np.bincount(train_labels, minlength=9)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# pin_memory only helps when CUDA is available
_pin = DEVICE == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=_pin
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=_pin
)

# ─────────────────────── CHECKPOINTING ────────────────────────
def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_acc":             best_acc,
    }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    # weights_only=True avoids arbitrary code execution from untrusted checkpoints
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"], checkpoint["best_acc"]

# ─────────────────────── TRAIN EPOCH ──────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, epoch, total_epochs, ewc=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=_pin)
        labels = labels.to(DEVICE, non_blocking=_pin)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ── EWC penalty ──
        loss = selective_ewc_loss(model, loss, ewc, ewc_lambda=EWC_LAMBDA)

        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0

        pbar.set_postfix({
            "mem(GB)":  f"{mem:.2f}",
            "loss":     f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr":       f"{scheduler.get_last_lr()[0]:.2e}",
        })

    scheduler.step()
    return avg_loss


# ──────────────────────── EVALUATE ────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=_pin)
            labels = labels.to(DEVICE, non_blocking=_pin)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total if total > 0 else 0.0


# ──────────────────────────── MAIN ────────────────────────────
def main():
    print(f"Using device: {DEVICE}")
    print(f"Train samples: {train_size}  |  Val samples: {val_size}")

    model     = HybridWaferModel(num_classes=9).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # Cosine annealing — smoothly decays LR over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc    = 0.0
    start_epoch = 0

    # ── Resume from checkpoint ──
    if os.path.exists(LAST_PATH):
        print("Resuming from last checkpoint...")
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, LAST_PATH)
        print(f"Resumed from epoch {start_epoch + 1}  (best val acc so far: {best_acc:.4f})")

    # ── EWC: built from training data BEFORE starting training ──
    # This captures the Fisher Information of the initial model state.
    # If you are doing sequential task learning, re-register after each task.
    print("Computing EWC Fisher Information (one pass over training data)...")
    ewc = EWC(model, train_loader, device=DEVICE)
    ewc.register_prior_task()
    print("EWC ready.")

    for epoch in range(start_epoch, EPOCHS):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, EPOCHS, ewc=ewc)
        acc  = evaluate(model, val_loader)
        elapsed = time.time() - t0

        print(f"[Epoch {epoch+1:02d}/{EPOCHS}]  loss={loss:.4f}  val_acc={acc:.4f}  time={elapsed:.1f}s")

        # Save last checkpoint every epoch (for resuming)
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, LAST_PATH)

        # Save best checkpoint whenever val accuracy improves
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, BEST_PATH)
            print(f"  ✓ New best val acc: {best_acc:.4f}  → saved to {BEST_PATH}")

    print(f"\nTraining complete.  Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()