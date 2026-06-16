# Architecture

## High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                         │
│                                                                 │
│  Raw LSWMD PKL files                                            │
│        │                                                        │
│        ▼                                                        │
│  convert_pkl.py ──► polar_strips.npy (3 GB, N×64×360 float32)  │
│                 ──► labels.csv (index, label)                   │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TRAINING PIPELINE                         │
│                                                                 │
│  WaferDataset (dataset_memmap.py)                               │
│    │ np.memmap on polar_strips.npy                              │
│    │ PolarTransform (roll, flip, erase)                         │
│    │ Normalize ÷2.0 → [0.0, 1.0]                               │
│    ▼                                                            │
│  WeightedRandomSampler ──► DataLoader (batch=32)                │
│                                │                                │
│                                ▼                                │
│  SwinWaferModel (model.py)                                      │
│    │ F.interpolate(x, 224×224)                                  │
│    │ swin_tiny_patch4_window7_224 (timm, pretrained)            │
│    │ 9-class linear head                                        │
│    ▼                                                            │
│  CrossEntropyLoss (label_smooth=0.1, class weights)             │
│  + EWC penalty (selective_ewc_loss)                             │
│                                │                                │
│                                ▼                                │
│  AdamW + GradScaler (AMP) + cosine LR + grad clip              │
│                                │                                │
│                                ▼                                │
│  checkpoints/best_model.pth                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴──────────────┐
                ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│     evaluate.py          │    │          app.py              │
│                          │    │                              │
│  Load best_model.pth     │    │  Upload wafer image          │
│  Inference on val set    │    │  image_to_wafer_map()        │
│  Confusion matrix PNG    │    │  preprocess_wafer()          │
│  ROC curves PNG          │    │  Normalize ÷2.0              │
│  metrics_report.txt      │    │  SwinWaferModel inference    │
│                          │    │  Saliency map overlay        │
└──────────────────────────┘    └──────────────────────────────┘
```

## Data Flow: Raw → Inference

1. **Raw PKL** (`WM811K.pkl`) — original LSWMD wafer maps as nested dicts
2. **`convert_pkl.py`** — extracts `waferMap` arrays, runs `preprocess_wafer()` on each, writes flat float32 to `polar_strips.npy` and class labels to `labels.csv`
3. **`polar_strips.npy`** — memory-mapped float32 array, shape `(N, 64, 360)`, raw pixel values {0.0, 1.0, 2.0}
4. **`WaferDataset.__getitem__`** — loads one row, adds channel dim, normalizes to [0,1] (÷2.0), applies random augmentations
5. **`SwinWaferModel.forward`** — bilinear interpolate to 224×224, Swin Tiny backbone + 9-class head
6. **Checkpoint** — `checkpoints/best_model.pth` dict: `{"model_state_dict": ...}`

## Preprocessing: Polar Transform (`preprocessing.py`)

```
Wafer Map (H×W, values 0/1/2)
    │
    ▼
detect_wafer(wafer_map)
    │  Moments-based centroid: cv2.moments()
    │  Radius from contour area: r = sqrt(area/π)
    │  Falls back to image center if detection fails
    ▼
cartesian_to_polar(wafer_map, cx, cy, radius)
    │  Angular grid: θ ∈ [0°, 360°), 360 columns
    │  Radial range: [0.85r, r], 64 rows
    │  cv2.remap() with linear interpolation
    ▼
Polar strip: float32, shape (64, 360)
    │  Row 0 = innermost ring (0.85r)
    │  Row 63 = outermost ring (r)
    │  Col 0–359 = θ=0° to 359°
```

## Model Architecture (`model.py`)

```
Input: (B, 1, H, W) — any spatial size, normalized [0,1]
    │
    ▼
F.interpolate → (B, 1, 224, 224)  [bilinear, no aspect-ratio preserve]
    │
    ▼
timm swin_tiny_patch4_window7_224
    │  patch_embed: 4×4 patches → (B, 56×56, 96)
    │  Stage 1: 2× Swin blocks, window=7, dim=96
    │  Stage 2: 2× Swin blocks, window=7, dim=192
    │  Stage 3: 6× Swin blocks, window=7, dim=384
    │  Stage 4: 2× Swin blocks, window=7, dim=768
    │  AdaptiveAvgPool → (B, 768)
    ▼
Linear(768 → 9)  [replaced by timm's num_classes param]
    │
    ▼
Output: (B, 9) logits
```

1-channel adaptation: timm sums pretrained 3-channel patch embedding weights along the channel axis to produce a 1-channel equivalent, preserving pretrained representations.

## EWC Pipeline (`continual_learning.py`)

```
After training Task k:
    │
    ▼
EWC(model, train_loader_k, device)
    │  _diag_fisher(): forward+backward over train_loader_k
    │  Accumulates squared gradients: F_i = Σ (∂L/∂θ_i)²
    │  Averages over all batches
    ▼
register_prior_task()
    │  Snapshots current parameters: θ*_i = θ_i.detach().clone()
    │  Stores Fisher diagonal: self.fisher = {name: F_i}
    │  self.prior_registered = True

During training Task k+1:
    │
    ▼
selective_ewc_loss(model, task_loss, ewc, λ=500)
    │  If prior_registered:
    │    penalty = Σ_i F_i · (θ_i - θ*_i)²
    │    return task_loss + λ · penalty
    │  Else: return task_loss unchanged
```

**Limitation**: `register_prior_task()` overwrites `self.fisher` each call. Only the most recent prior task is protected. See ROADMAP for accumulation approach.

## Training Strategy (`train.py`)

### 4-Task Sequential Schedule

| Task | Classes | Rationale |
|------|---------|-----------|
| 1 | none, Center | Most common; establish baseline |
| 2 | Edge-Ring, Edge-Loc | Edge patterns, angularly structured |
| 3 | Scratch, Loc, Random | Spatially diverse |
| 4 | Donut, Near-full | Rare, hardest; trained last with strongest EWC |

### Replay Buffer

After each task, 100 samples per class are stored in `replay_buffer` (absolute dataset indices). These are concatenated into the next task's training set, preventing forgetting of class distributions without storing full activation tensors.

### LR Schedule

```
Epoch 0–2 (warmup):  lr = base_lr × (epoch+1) / warmup_epochs
Epoch 3–7 (cosine):  lr = base_lr × 0.5 × (1 + cos(π × progress))
```

### Class Balancing

- **Sampler**: `WeightedRandomSampler` with inverse-frequency weights; ×5 boost for Donut (1) and Near-full (7)
- **Loss**: `CrossEntropyLoss` with inverse-frequency weights; ×2 boost for Loc (4), Scratch (6), Near-full (7)
