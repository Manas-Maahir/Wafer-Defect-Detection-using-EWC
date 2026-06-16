"""
baselines.py
------------
Trains comparison baselines on the SAME merged, stratified train/val/test split used by
evaluate.py, so their accuracy / macro-AUC / macro-F1 are directly comparable to
Embodiment B (pure Swin). Closes the "no baselines" gap in INVENTION_DISCLOSURE.md §2.4.

Models:
  cnn          ResNet18, 1-channel input, 9-class head (pure CNN baseline)
  swin_scratch swin_tiny_patch4_window7_224, NOT pretrained (Transformer-from-scratch)
  swin         swin_tiny_patch4_window7_224, pretrained == Embodiment B (reference)

Run from the repository root on a CUDA machine:
    python -m experiments.baselines --model cnn --epochs 8
"""

import argparse
import json
import os

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader

from dataset_memmap import WaferDataset, get_task_splits

DATA_PATH, LABEL_PATH = "polar_strips.npy", "labels.csv"
NUM_CLASSES = 9
CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
               "Loc", "Random", "Scratch", "Near-full", "none"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineModel(nn.Module):
    """Wraps a timm backbone; resizes the 64x360 polar strip to a square the backbone expects."""

    def __init__(self, name, pretrained):
        super().__init__()
        self.is_swin = name.startswith("swin")
        backbone = "resnet18" if name == "cnn" else "swin_tiny_patch4_window7_224"
        self.net = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=NUM_CLASSES)

    def forward(self, x):                              # x: (B, 1, 64, 360)
        if self.is_swin:                              # Swin needs 224x224
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.net(x)


def merged_loaders(seed, batch_size):
    tr, va, te = [], [], []
    for tid in range(1, 5):
        a, b, c = get_task_splits(DATA_PATH, LABEL_PATH, tid, seed=seed)
        tr.append(a); va.append(b); te.append(c)
    idx = {k: np.concatenate(v) for k, v in {"train": tr, "val": va, "test": te}.items()}
    loaders = {}
    for split, is_train in (("train", True), ("val", False), ("test", False)):
        ds = WaferDataset(DATA_PATH, LABEL_PATH, valid_indices=idx[split], is_train=is_train)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                                    num_workers=0, pin_memory=(DEVICE == "cuda"))
    return loaders


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps, probs = [], [], []
    use_amp = DEVICE == "cuda"
    for images, labels in loader:
        images = images.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
        p = F.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(p); ps.append(p.argmax(1)); ys.append(labels.numpy())
    y = np.concatenate(ys); pred = np.concatenate(ps); prob = np.concatenate(probs)
    acc = float((y == pred).mean())
    rep = classification_report(y, pred, target_names=CLASS_NAMES,
                                output_dict=True, zero_division=0)
    y1h = np.eye(NUM_CLASSES)[y]
    aucs = [roc_auc_score(y1h[:, i], prob[:, i]) for i in range(NUM_CLASSES)]
    return acc, float(np.mean(aucs)), float(rep["macro avg"]["f1-score"]), rep


def main():
    ap = argparse.ArgumentParser(description="Train comparison baselines")
    ap.add_argument("--model", choices=["cnn", "swin_scratch", "swin"], required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {DEVICE} | model={args.model}")

    loaders = merged_loaders(args.seed, args.batch_size)
    pretrained = args.model == "swin"
    model = BaselineModel(args.model, pretrained=pretrained).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == "cuda"))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 0.0
    for epoch in range(args.epochs):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                loss = crit(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        val_acc, _, _, _ = evaluate(model, loaders["val"])
        best_val = max(best_val, val_acc)
        print(f"  epoch {epoch+1}/{args.epochs}  val_acc={val_acc:.4f}")

    test_acc, macro_auc, macro_f1, rep = evaluate(model, loaders["test"])
    print(f"\n{args.model}: test_acc={test_acc:.4f}  macro_auc={macro_auc:.4f}  macro_f1={macro_f1:.4f}")

    os.makedirs("eval_outputs", exist_ok=True)
    out = os.path.join("eval_outputs", f"baseline_{args.model}.json")
    with open(out, "w") as f:
        json.dump({"model": args.model, "pretrained": pretrained, "epochs": args.epochs,
                   "seed": args.seed, "best_val_acc": best_val, "test_acc": test_acc,
                   "macro_auc": macro_auc, "macro_f1": macro_f1,
                   "classification_report": rep}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
