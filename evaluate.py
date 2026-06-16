"""
evaluate.py
-----------
Loads the best trained checkpoint and computes:
  - Classification Report (Accuracy, Precision, Recall, F1 per class)
  - Overall Macro / Weighted averages
  - Confusion Matrix (saved as confusion_matrix.png)
  - ROC-AUC per class (One-vs-Rest) + macro average (saved as roc_curves.png)
  - Loss curve from training (if training_log.csv exists)

Run after training:
    python evaluate.py
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # no display needed

from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from model import SwinWaferModel
from dataset_memmap import WaferDataset, get_task_splits

# ─────────────────────────── CONFIG ────────────────────────────
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pth")
DATA_PATH       = "polar_strips.npy"
LABEL_PATH      = "labels.csv"
BATCH_SIZE      = 64
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full", "none"
]
NUM_CLASSES = len(CLASS_NAMES)

OUTPUT_DIR = "eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── DATA ──────────────────────────────
def get_test_loader(seed=42):
    """Merge held-out test splits from all 4 tasks into a single evaluation loader."""
    all_test_idx = []
    for tid in range(1, 5):
        _, _, abs_test = get_task_splits(DATA_PATH, LABEL_PATH, tid, seed=seed)
        all_test_idx.append(abs_test)
    all_test_idx = np.concatenate(all_test_idx)

    test_ds = WaferDataset(DATA_PATH, LABEL_PATH, valid_indices=all_test_idx, is_train=False)
    return DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

# ─────────────────────── MODEL LOADING ─────────────────────────
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at '{CHECKPOINT_PATH}'. "
            "Run train.py first."
        )

    model = SwinWaferModel(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    epoch    = ckpt.get("epoch", "N/A")
    best_acc = ckpt.get("best_acc", None)
    
    if best_acc is not None:
        print(f"Loaded checkpoint  |  epoch={epoch}  |  best_val_acc={best_acc:.4f}")
    else:
        print(f"Loaded checkpoint  |  epoch={epoch}")
    return model

# ──────────────────────── INFERENCE ────────────────────────────
def run_inference(model, loader):
    all_labels  = []
    all_preds   = []
    all_probs   = []   # softmax probabilities for ROC-AUC

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs, axis=0),
    )

# ──────────────────── CONFUSION MATRIX ─────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(11, 9))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("Confusion Matrix — Wafer Defect Classifier", fontsize=14, pad=12)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return cm

# ───────────────────────── ROC CURVES ──────────────────────────
def plot_roc_curves(y_true, y_probs):
    # One-Hot encode ground truth for OvR ROC
    y_onehot = np.eye(NUM_CLASSES)[y_true]

    fig, ax = plt.subplots(figsize=(10, 7))

    aucs = []
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        auc_val     = roc_auc_score(y_onehot[:, i], y_probs[:, i])
        aucs.append(auc_val)
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={auc_val:.3f})")

    macro_auc = np.mean(aucs)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest)  |  Macro-AUC = {macro_auc:.4f}", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return aucs, macro_auc

# ─────────────────────── PRINT REPORT ──────────────────────────
def print_report(y_true, y_pred, cm, aucs, macro_auc):
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 65)
    print("  CLASSIFICATION REPORT")
    print("=" * 65)
    print(report)

    print("=" * 65)
    print("  ROC-AUC (One-vs-Rest)")
    print("=" * 65)
    for name, auc in zip(CLASS_NAMES, aucs):
        print(f"  {name:<12}  AUC = {auc:.4f}")
    print(f"  {'Macro Avg':<12}  AUC = {macro_auc:.4f}")

    # Per-class accuracy from confusion matrix diagonal
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\n" + "=" * 65)
    print("  PER-CLASS ACCURACY")
    print("=" * 65)
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        print(f"  {name:<12}  Acc = {acc:.4f}")

    overall_acc = cm.diagonal().sum() / cm.sum()
    print(f"\n  Overall Accuracy : {overall_acc:.4f}")
    print("=" * 65)

    # Save report to text file
    report_path = os.path.join(OUTPUT_DIR, "metrics_report.txt")
    with open(report_path, "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(report + "\n")
        f.write("ROC-AUC (One-vs-Rest)\n")
        f.write("=" * 65 + "\n")
        for name, auc in zip(CLASS_NAMES, aucs):
            f.write(f"  {name:<12}  AUC = {auc:.4f}\n")
        f.write(f"  {'Macro Avg':<12}  AUC = {macro_auc:.4f}\n")
        f.write("\nPER-CLASS ACCURACY\n")
        f.write("=" * 65 + "\n")
        for name, acc in zip(CLASS_NAMES, per_class_acc):
            f.write(f"  {name:<12}  Acc = {acc:.4f}\n")
        f.write(f"\n  Overall Accuracy : {overall_acc:.4f}\n")

    print(f"\nFull report saved: {report_path}")


# ───────────────────────────── MAIN ────────────────────────────
def main():
    print(f"Device : {DEVICE}")
    print(f"Loading held-out test data ...")
    val_loader = get_test_loader()
    print(f"Test samples : {len(val_loader.dataset)}")

    model = load_model()

    print("\nRunning inference ...")
    y_true, y_pred, y_probs = run_inference(model, val_loader)

    print("\nGenerating plots ...")
    cm              = plot_confusion_matrix(y_true, y_pred)
    aucs, macro_auc = plot_roc_curves(y_true, y_probs)

    print_report(y_true, y_pred, cm, aucs, macro_auc)
    print(f"\nAll outputs saved to: ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
