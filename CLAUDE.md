# CLAUDE.md — Wafer Defect Classifier

## Project Purpose

Classifies semiconductor wafer defects into 9 categories using a Swin Transformer trained with Elastic Weight Consolidation (EWC) for continual learning. Input: grayscale wafer maps. Output: defect class (Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, none).

## Key Files

| File | Role |
|------|------|
| `model.py` | `SwinWaferModel` — pure `swin_tiny_patch4_window7_224` via timm, 1-channel input, 9-class head |
| `train.py` | 4-task sequential continual learning loop with EWC + replay buffer |
| `dataset_memmap.py` | `WaferDataset` — memory-mapped loader for `polar_strips.npy`; normalizes to [0,1] by ÷2 |
| `preprocessing.py` | `preprocess_wafer()` — cartesian wafer map → 64×360 polar strip |
| `continual_learning.py` | `EWC` class — diagonal Fisher Information, `selective_ewc_loss()` |
| `evaluate.py` | Post-training metrics: classification report, confusion matrix, ROC curves |
| `app.py` | Streamlit inference dashboard with saliency map overlay |
| `visualize_attention.py` | Standalone gradient saliency map generator |
| `polar_strips.npy` | Preprocessed dataset — 3 GB memory-mapped float32, shape (N, 64, 360) |
| `labels.csv` | Two-column CSV: `index`, `label` |
| `checkpoints/best_model.pth` | Best validation checkpoint (dict with `model_state_dict`) |

## Running the Project

```powershell
# Training (GPU required for practical speed)
python train.py

# Evaluation (after training)
python evaluate.py

# Streamlit app
streamlit run app.py
```

## Common Gotchas

- **GPU required for training**: `train.py` assumes CUDA. CPU training is ~50x slower due to mixed-precision use and dataset size.
- **Data not in git**: `polar_strips.npy` (3 GB) and `labels.csv` are gitignored. Generate them with `convert_pkl.py` from the raw LSWMD PKL files.
- **Normalization contract**: All inputs entering the model must be in [0.0, 1.0] (÷2.0 from raw {0,1,2} pixel values). `WaferDataset.__getitem__` and `app.py` both apply this division — do not remove either.
- **EWC stores only last task**: `continual_learning.py` overwrites Fisher on each `register_prior_task()` call. It protects the immediately prior task but not all past tasks. This is a known limitation.
- **`requirements_gpu_locked.txt` is UTF-16**: Cannot be used directly with pip. Use `requirements.txt` instead.
- **Checkpoint format**: `torch.load(..., weights_only=False)` is intentional — the checkpoint dict wraps `model_state_dict`.

## Model Performance (best checkpoint)

| Metric | Value |
|--------|-------|
| Overall val accuracy | 83.19% |
| Macro-AUC (ROC, OvR) | 0.9779 |
| Weakest class | Loc (59.75% recall) |
| Strongest class | Edge-Ring (96%+ recall) |

## Known Limitations

1. EWC only prevents forgetting of the immediately prior task — not all historical tasks.
2. Input is bilinearly interpolated from 64×360 to 224×224, causing aspect ratio distortion.
3. Loc class recall is low (~60%) — localized spot defects are hard to disambiguate in polar space.
4. `num_workers=0` in DataLoader — no parallel data loading (Windows multiprocessing constraint).
