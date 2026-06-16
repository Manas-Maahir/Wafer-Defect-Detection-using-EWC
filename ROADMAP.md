# Roadmap

## High Priority

### Add pytest suite
No automated tests exist. Untested edge cases include polar transform fallback path, EWC penalty sign correctness, and `__getitem__` output range after normalization. Test failures currently surface only at runtime during training.

### Fix EWC to accumulate Fisher across all tasks
`continual_learning.py` overwrites `self.fisher` on each `register_prior_task()` call. After Task 3, only Task 2 Fisher is retained — Task 1 knowledge is unprotected. Fix: accumulate with weighted sum `F_total_i = F_old_i + F_new_i` before overwriting.

### Add `requirements.txt` for CPU environments
Done in v0.1.0. Verify it stays current when dependencies are updated.

## Medium Priority

### Preserve aspect ratio in model input resize
`model.py` uses `F.interpolate(x, size=(224, 224))` on a 64×360 polar strip. This squashes the angular axis by ~5.6×. Alternative: pad the shorter dimension to square before resizing to preserve angular spacing.

### Per-channel normalization using dataset statistics
Training uses a fixed ÷2.0 normalization (maps {0,1,2} → {0, 0.5, 1.0}). Compute actual mean/std over `polar_strips.npy` and apply `(x - mean) / std` normalization for better gradient flow.

### Enable `num_workers > 0` in DataLoaders
Currently hard-coded to `num_workers=0` due to Windows multiprocessing constraints with np.memmap. Investigate `spawn` context or file-handle-per-worker approach to enable parallel data loading.

### Improve Loc class recall
Loc recall is ~60% — the lowest of all classes. Potential approaches: increase class-weight multiplier beyond current ×2, add Loc-specific augmentations, or investigate whether Loc samples cluster incorrectly in polar space.

## Low Priority

### Experiment tracking (MLflow or Weights & Biases)
Training currently prints to stdout only. Adding W&B or MLflow would enable loss/accuracy curves, hyperparameter comparison, and checkpoint artifact storage.

### Model quantization for edge deployment
The Swin Tiny model is ~28 MB. INT8 post-training quantization via `torch.quantization` could reduce inference latency and memory footprint for deployment on edge hardware (e.g., inspection stations without discrete GPUs).

### Multi-GPU support
`train.py` targets a single CUDA device. Wrapping `SwinWaferModel` with `nn.DataParallel` or `DistributedDataParallel` would enable multi-GPU training for faster iteration on larger datasets or additional defect classes.

### Configurable task groupings via YAML
Task groups are currently hard-coded in `dataset_memmap.py`. Moving them to a config file would allow experimenting with different continual learning orderings without code changes.
