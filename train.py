import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import random
import numpy as np
import time
from tqdm import tqdm

from model import SwinWaferModel
from dataset_memmap import WaferDataset, get_task_splits
from continual_learning import EWC, selective_ewc_loss

# ─────────────────────────── CONFIG ───────────────────────────
BATCH_SIZE   = 32
EPOCHS       = 8
LR           = 1e-4
VAL_SPLIT    = 0.15
EWC_LAMBDA   = 500
GRAD_CLIP    = 1.0
LABEL_SMOOTH = 0.1
WARMUP_EPOCHS = 3
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────── DATA & SPLIT ───────────────────────
def get_task_dataloaders_and_criterion(device, replay_buffer, abs_tr_idx, abs_val_idx, use_replay=True):
    """
    Builds train/val loaders from pre-computed absolute indices (3-way split done upstream).
    When use_replay is True, also updates replay_buffer with 100 samples/class from abs_tr_idx
    and folds the accumulated buffer into the current task's training set.
    """
    # val_ds is created first so we can access .labels (the full label array) cheaply
    val_ds = WaferDataset("polar_strips.npy", "labels.csv", valid_indices=abs_val_idx, is_train=False)

    # Add current task's training samples to replay buffer for future tasks
    if use_replay:
        tr_labels_full = val_ds.labels[abs_tr_idx]
        for cls in np.unique(tr_labels_full):
            cls_idx = abs_tr_idx[tr_labels_full == cls]
            selected = np.random.choice(cls_idx, min(len(cls_idx), 100), replace=False)
            replay_buffer.extend(selected.tolist())

    # Combine current task's train indices with accumulated replay
    if len(replay_buffer) > 0:
        combined_tr_abs = np.concatenate([abs_tr_idx, replay_buffer]).astype(int)
    else:
        combined_tr_abs = abs_tr_idx.astype(int)

    combined_train_ds = WaferDataset("polar_strips.npy", "labels.csv", valid_indices=combined_tr_abs, is_train=True)

    tr_labels = combined_train_ds.labels[combined_train_ds.valid_indices]
    class_counts = np.bincount(tr_labels, minlength=9)

    # Sample weights for WeightedRandomSampler
    sample_weights_arr = np.zeros(len(tr_labels), dtype=np.float32)
    for i, lbl in enumerate(tr_labels):
        w = 1.0 / (class_counts[lbl] + 1e-6)
        if lbl in [1, 7]:   # Donut or Near-full
            w *= 5.0
        sample_weights_arr[i] = w

    sampler = WeightedRandomSampler(sample_weights_arr, len(sample_weights_arr))

    _pin = device == "cuda"
    train_loader = DataLoader(combined_train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=_pin)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin)

    # Class loss weights
    c_weights = 1.0 / (class_counts + 1e-6)
    c_weights = np.where(class_counts > 0, c_weights, 0)
    if class_counts[4] > 0: c_weights[4] *= 2.0
    if class_counts[6] > 0: c_weights[6] *= 2.0
    if class_counts[7] > 0: c_weights[7] *= 2.0

    c_weights_t = torch.tensor(c_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=c_weights_t, label_smoothing=LABEL_SMOOTH)

    return train_loader, val_loader, criterion

# ─────────────────────── CHECKPOINTING ────────────────────────
def save_checkpoint(model, path):
    torch.save({"model_state_dict": model.state_dict()}, path)

def get_scheduler(optimizer):
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(WARMUP_EPOCHS)
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ─────────────────────── TRAIN EPOCH ──────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, scheduler, criterion, epoch, total_epochs, ewc=None):
    model.train()
    running_loss = 0.0
    _pin = DEVICE == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(DEVICE, non_blocking=_pin)
        labels = labels.to(DEVICE, non_blocking=_pin)

        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss    = criterion(outputs, labels)
            # Apply EWC Penalty if prior tasks exist
            loss = selective_ewc_loss(model, loss, ewc, ewc_lambda=EWC_LAMBDA)

        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        current_lr = scheduler.get_last_lr()[0]

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

    scheduler.step()
    return avg_loss

def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    _pin = DEVICE == "cuda"

    with torch.no_grad(), torch.amp.autocast('cuda'):
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
    global EWC_LAMBDA, EPOCHS  # may be overridden by CLI below
    parser = argparse.ArgumentParser(description="Wafer defect continual learning trainer")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Max epochs per task (default: {EPOCHS}). Lower it to fit a time budget.")
    parser.add_argument("--no-ewc", action="store_true",
                        help="Disable EWC (ablation): never apply the Fisher penalty.")
    parser.add_argument("--no-replay", action="store_true",
                        help="Disable the experience-replay buffer (ablation).")
    parser.add_argument("--ewc-lambda", type=float, default=EWC_LAMBDA,
                        help=f"EWC regularisation strength (default: {EWC_LAMBDA}).")
    parser.add_argument("--tag", type=str, default=None,
                        help="Optional run tag; namespaces checkpoint/metrics filenames so "
                             "ablation regimes do not overwrite each other.")
    args = parser.parse_args()
    SEED = args.seed
    use_ewc    = not args.no_ewc
    use_replay = not args.no_replay

    # Allow CLI override of the EWC strength and epoch budget used during training.
    EWC_LAMBDA = args.ewc_lambda
    EPOCHS = args.epochs

    # Filename suffix: explicit tag, else derived from the active regime.
    if args.tag is not None:
        suffix = f"{args.tag}_seed{SEED}"
    else:
        regime = f"{'ewc' if use_ewc else 'noewc'}_{'replay' if use_replay else 'noreplay'}"
        # Preserve the historical default filename (cl_metrics_seed42.json) for the full config.
        suffix = f"seed{SEED}" if (use_ewc and use_replay) else f"{regime}_seed{SEED}"

    print(f"Regime: EWC={'on' if use_ewc else 'off'} (lambda={EWC_LAMBDA})  "
          f"replay={'on' if use_replay else 'off'}  -> outputs tagged '{suffix}'")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Seed/regime-namespaced output paths
    best_path = os.path.join(CHECKPOINT_DIR, f"best_model_{suffix}.pth")
    last_path = os.path.join(CHECKPOINT_DIR, f"last_model_{suffix}.pth")
    os.makedirs("eval_outputs", exist_ok=True)
    cl_metrics_path = os.path.join("eval_outputs", f"cl_metrics_{suffix}.json")

    print(f"Using device : {DEVICE}  |  seed={SEED}")
    model = SwinWaferModel(num_classes=9).to(DEVICE)

    # ── Pre-compute 3-way splits and test loaders for all tasks ──
    print("Pre-computing task splits ...")
    task_splits = {}      # task_id -> (abs_train, abs_val, abs_test)
    task_test_loaders = {}  # task_id -> DataLoader (held-out test, never seen during training)
    _pin = DEVICE == "cuda"
    for tid in range(1, 5):
        abs_tr, abs_val, abs_test = get_task_splits(
            "polar_strips.npy", "labels.csv", tid, seed=SEED)
        task_splits[tid] = (abs_tr, abs_val, abs_test)
        test_ds = WaferDataset("polar_strips.npy", "labels.csv", valid_indices=abs_test, is_train=False)
        task_test_loaders[tid] = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=_pin)
        print(f"  Task {tid}: train={len(abs_tr)}  val={len(abs_val)}  test={len(abs_test)}")

    # R[(i, j)] = accuracy on task j's test set after training task i
    R = {}
    best_acc = 0.0
    ewc = None
    replay_buffer = []

    # ── SEQUENTIAL CONTINUAL LEARNING LOOP ──
    for task_id in range(1, 5):
        print(f"\n{'='*40}")
        print(f"       STARTING TASK {task_id}")
        print(f"{'='*40}")

        abs_tr, abs_val, _ = task_splits[task_id]

        # FWT: zero-shot accuracy on this task BEFORE training (tasks 2-4 only)
        if task_id > 1:
            zs_acc = evaluate(model, task_test_loaders[task_id])
            R[(task_id - 1, task_id)] = zs_acc
            print(f"  [FWT] Zero-shot acc on Task {task_id} before training: {zs_acc:.4f}")

        train_loader, val_loader, criterion = get_task_dataloaders_and_criterion(
            DEVICE, replay_buffer, abs_tr, abs_val, use_replay=use_replay)

        # Reset optimizer, scheduler, and scaler for the new task
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = get_scheduler(optimizer)
        scaler = torch.amp.GradScaler('cuda')

        task_best_acc = 0.0
        epochs_no_improve = 0
        patience = 3

        for epoch in range(EPOCHS):
            t0   = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, criterion, epoch, EPOCHS, ewc=ewc)
            acc  = evaluate(model, val_loader)
            elapsed = time.time() - t0

            print(f"[Task {task_id} | Epoch {epoch+1:02d}/{EPOCHS}] loss={loss:.4f} val_acc={acc:.4f} time={elapsed:.1f}s")

            save_checkpoint(model, last_path)

            if acc > task_best_acc:
                task_best_acc = acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if acc > best_acc:
                best_acc = acc
                save_checkpoint(model, best_path)
                print(f"  >>> New overall best val acc: {best_acc:.4f} (Saved Best)")

            if epochs_no_improve >= patience:
                print(f"  >>> Early stopping triggered for Task {task_id} after {epoch+1} epochs!")
                break

        # Evaluate on all tasks seen so far (builds the R matrix for BWT/AA)
        print(f"  [CL Metrics] Evaluating on tasks 1..{task_id} test sets ...")
        for j in range(1, task_id + 1):
            R[(task_id, j)] = evaluate(model, task_test_loaders[j])
            print(f"    Task {j} test acc after training Task {task_id}: {R[(task_id, j)]:.4f}")

        if use_ewc:
            print(f"Finished Task {task_id}. Computing Fisher Information...")
            ewc = EWC(model, train_loader, device=DEVICE)
            ewc.register_prior_task()
            print(f"Fisher Information registered for Task {task_id}!")
        else:
            print(f"Finished Task {task_id}. EWC disabled (ablation) — no Fisher computed.")

    # ── Compute and save continual learning metrics ──
    T = 4
    AA  = float(np.mean([R[(T, j)] for j in range(1, T + 1)]))
    BWT = float(np.mean([R[(T, j)] - R[(j, j)] for j in range(1, T)]))
    FWT = float(np.mean([R[(j - 1, j)] for j in range(2, T + 1)]))

    accuracy_matrix = {str(k): float(v) for k, v in R.items()}
    cl_metrics = {
        "seed": SEED,
        "accuracy_matrix": accuracy_matrix,
        "average_accuracy": AA,
        "bwt": BWT,
        "fwt": FWT,
    }
    with open(cl_metrics_path, "w") as f:
        json.dump(cl_metrics, f, indent=2)

    print(f"\n{'='*40}")
    print(f"  CONTINUAL LEARNING METRICS  (seed={SEED})")
    print(f"{'='*40}")
    print(f"  Average Accuracy (AA) : {AA:.4f}")
    print(f"  Backward Transfer(BWT): {BWT:+.4f}  ({'forgetting' if BWT < 0 else 'improvement'})")
    print(f"  Forward Transfer (FWT): {FWT:+.4f}")
    print(f"  CL metrics saved : {cl_metrics_path}")
    print(f"  Best val acc     : {best_acc:.4f}")
    print(f"  Best checkpoint  : {best_path}")


if __name__ == "__main__":
    main()