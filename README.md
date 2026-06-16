# Wafer Defect Classification with Continual Learning

## What is this project?

Semiconductor manufacturing is an extremely precise process. You start with a silicon wafer, you run it through hundreds of process steps, and at the end you hope that the tiny circuits etched on it actually work. When something goes wrong during manufacturing, it does not go wrong randomly. Defects on wafers tend to cluster in specific geometric patterns that reflect the specific process failure that caused them. An edge ring defect means something went wrong at the outer edge of the wafer during a deposition step. A scratch defect looks like a straight line and is usually caused by a mechanical handling issue. A center defect concentrates right in the middle and often points to an issue with process uniformity.

The goal of this project is to look at a wafer map, which is essentially a 2D grid showing which dies on the wafer passed or failed, and automatically classify which failure pattern is present. There are 9 classes in total: Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, and none (meaning no defect pattern detected).

## The Dataset

We use the LSWMD (Large-Scale Wafer Map Dataset), which is one of the most widely used public datasets for this problem. It contains tens of thousands of real wafer maps from actual fabrication processes. The raw data lives in a pickle file (`WM811K.pkl`), and we preprocess it into a memory-mapped numpy array (`polar_strips.npy`) alongside a `labels.csv`.

The memory-mapped format is important here. The full dataset is multiple gigabytes. Loading all of it into RAM at once would either crash your machine or leave you waiting a long time. Memory mapping lets you access the data as if it is in memory while actually reading it from disk on demand.

## Why Polar Coordinates?

Defect patterns on a wafer are fundamentally circular. Edge-Ring defects wrap around the entire perimeter. Center defects are radially symmetric. A standard CNN sees a rectangular grid and has no built-in awareness that the important structure is actually circular.

So we convert each wafer map from Cartesian coordinates to a polar coordinate representation. We extract the edge ring of the wafer (the annular region near the periphery) and unroll it into a rectangular strip. The horizontal axis becomes the angle around the wafer (0°–360°) and the vertical axis becomes the radial depth from the edge inward. After this transform, an Edge-Ring defect becomes a horizontal stripe across the full width. A localized Edge-Loc defect becomes a vertical blob at a specific angular position. The geometry that was hard to see in Cartesian coordinates becomes obvious.

This preprocessing step is done once offline and saved to disk as `polar_strips.npy` (shape: N × 64 × 360, float32). During training, the model only ever sees these polar strips.

## Model Architecture

The model is a **pure Swin Transformer** — specifically `swin_tiny_patch4_window7_224` from timm, pretrained on ImageNet. There is no CNN component.

```
Input (B, 1, 64, 360)  →  Bilinear resize to (B, 1, 224, 224)
  →  Swin Tiny backbone (pretrained, 1-channel adapted)
  →  Global avg pool  →  Linear(768 → 9)  →  Logits
```

The 1-channel adaptation is handled by timm, which sums the pretrained 3-channel patch embedding weights along the channel axis, preserving the pretrained representations. Swin Transformers are particularly good at capturing long-range dependencies — important for defects like Scratch that span large portions of the image.

Current best checkpoint performance: **83.19% validation accuracy, macro-AUC 0.9779**.

## Continual Learning with EWC

In the real world, you do not train a model once and deploy it forever. New defect types emerge, processes change, and new labeled data arrives. Naively fine-tuning a model on new data causes **catastrophic forgetting**: gradient updates for the new task overwrite the weights critical for old tasks.

We address this with **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al., 2017). The idea: not all parameters are equally important to past tasks. EWC identifies which weights are critical (via the diagonal Fisher Information Matrix — squared gradients averaged over the prior training data) and adds a quadratic penalty to the loss that discourages changing them.

Training is organized into 4 sequential tasks:

| Task | Classes |
|------|---------|
| 1 | none, Center |
| 2 | Edge-Ring, Edge-Loc |
| 3 | Scratch, Loc, Random |
| 4 | Donut, Near-full |

After each task, an **experience replay buffer** (100 samples/class) is also accumulated and mixed into subsequent tasks, providing dual protection against forgetting.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full EWC pipeline and [wiki/ai-models/ewc.md](wiki/ai-models/ewc.md) for theory.

## File Structure

```
Wafer/
├── train.py               Sequential continual learning training loop
├── model.py               SwinWaferModel (pure Swin Transformer, 9-class)
├── continual_learning.py  EWC: diagonal Fisher, register_prior_task(), penalty
├── dataset_memmap.py      Memory-mapped WaferDataset with polar augmentations
├── preprocessing.py       Cartesian → polar coordinate transform
├── evaluate.py            Post-training: confusion matrix, ROC curves, metrics report
├── app.py                 Streamlit inference dashboard with saliency overlay
├── visualize_attention.py Standalone gradient saliency map generator
├── convert_pkl.py         Converts raw LSWMD PKL → polar_strips.npy + labels.csv
├── polar_strips.npy       Preprocessed dataset (3 GB, N×64×360 float32, gitignored)
├── labels.csv             Sample index → class label mapping (gitignored)
├── checkpoints/           Saved model weights (best_model.pth, last_model.pth)
├── requirements.txt       CPU-compatible dependencies
├── ARCHITECTURE.md        Full system design and data flow
├── CONTRIBUTING.md        Setup, conventions, how to add new defect classes
├── ROADMAP.md             Planned improvements
└── wiki/                  Extended knowledge base
```

## Quickstart

```powershell
# 1. Activate environment (if execution policy blocks: Set-ExecutionPolicy RemoteSigned -Scope Process)
.\wafer_gpu_env\Scripts\Activate.ps1

# 2. Install dependencies (new environment)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Prepare data (place WM811K.pkl in project root first)
python convert_pkl.py

# 4. Train
python train.py

# 5. Evaluate
python evaluate.py

# 6. Run inference dashboard
streamlit run app.py
```

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| BATCH_SIZE | 32 | Samples per gradient update |
| EPOCHS | 8 | Max epochs per task (early stopping at patience=3) |
| LR | 1e-4 | Starting learning rate (cosine annealed after 3-epoch warmup) |
| VAL_SPLIT | 0.15 | Fraction held out for validation |
| EWC_LAMBDA | 500 | EWC penalty strength — increase to protect old tasks more |
| GRAD_CLIP | 1.0 | Gradient norm clamp |

## Known Limitations

- EWC stores Fisher for only the most recent prior task. See [ROADMAP.md](ROADMAP.md) for the accumulation fix.
- Bilinear resize from 64×360 to 224×224 distorts aspect ratio (~5.6× horizontal squash).
- Loc class recall is ~60% — spatially localized defects are harder to distinguish in polar space.
- `num_workers=0` in DataLoaders (Windows multiprocessing constraint with np.memmap).

For deeper documentation, see [ARCHITECTURE.md](ARCHITECTURE.md), [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md), and the [wiki/](wiki/).
