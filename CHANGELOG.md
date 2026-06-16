# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-05-20

### Added
- CNN-Swin hybrid architecture later refactored to pure Swin Transformer (`swin_tiny_patch4_window7_224`)
- Elastic Weight Consolidation (EWC) for 4-task sequential continual learning
- Experience replay buffer (100 samples/class/task)
- Polar coordinate preprocessing pipeline (`preprocessing.py`)
- Memory-mapped dataset loader for 3 GB `polar_strips.npy` (`dataset_memmap.py`)
- Weighted random sampling + inverse-frequency class loss weights
- 3-epoch linear warmup + cosine annealing LR schedule
- Mixed-precision training with `torch.amp.autocast`
- Streamlit inference dashboard with gradient saliency overlay (`app.py`)
- Post-training evaluation script with confusion matrix and ROC curves (`evaluate.py`)
- 83.19% validation accuracy, macro-AUC 0.9779 on LSWMD dataset

### Fixed
- Normalization mismatch: `WaferDataset.__getitem__` now divides by 2.0 to match `app.py` inference path
- `visualize_attention.py` updated from deleted `HybridWaferModel` to `SwinWaferModel`
- `torch.cuda.amp.GradScaler()` (deprecated) replaced with `torch.amp.GradScaler('cuda')`
- Global random seeds added to `train.py` for reproducibility
