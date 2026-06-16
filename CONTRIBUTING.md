# Contributing

## Environment Setup

### GPU Environment (recommended for training)

```powershell
# If execution policy blocks activation:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create and activate environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install GPU dependencies (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### CPU Environment (evaluation/app only)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: `requirements_gpu_locked.txt` is UTF-16 encoded and CUDA 12.1 specific — use `requirements.txt` for all new setups.

## Code Conventions

No formatter is configured in this project. Follow these conventions manually:

- **Indentation**: 4 spaces
- **Line length**: ~120 characters (not strictly enforced)
- **Naming**: `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for module-level constants
- **Comments**: Only when the WHY is non-obvious. No docstrings for simple wrapper functions.
- **Imports**: stdlib → third-party → local, each group separated by a blank line

## Adding a New Defect Class

To add a new class (e.g., `"Ring-Edge"`):

1. **`dataset_memmap.py`**: Add to `label_map` with the next integer index; add to the appropriate `task_groups` entry (or create a new task).
2. **`train.py`**: Update `minlength` argument in `np.bincount(tr_labels, minlength=9)` to the new class count; add a class-weight boost line if the class is rare.
3. **`evaluate.py`** and **`app.py`**: Add the new class name to `CLASS_NAMES` at the correct index position.
4. **`model.py`** / **`train.py`**: Update `num_classes=9` to the new count wherever it appears (2 call sites in train.py, 1 in evaluate.py, 1 in app.py).
5. **`preprocessing.py`**: No changes needed unless the new class requires a different polar transform region.
6. Regenerate `polar_strips.npy` and `labels.csv` with `convert_pkl.py` after updating the source data.

## Data Preparation

Raw data is not included in the repository. To regenerate the training artifacts:

```powershell
# Place WM811K.pkl in the project root, then:
python convert_pkl.py
# Produces: polar_strips.npy, labels.csv
```

## Testing

No test suite exists yet. Before submitting changes:

- **Import check**: `python -c "from model import SwinWaferModel; from dataset_memmap import WaferDataset; print('OK')"`
- **Saliency check**: `python test_saliency.py` (requires a checkpoint)
- **Evaluation**: `python evaluate.py` after training

A pytest suite is tracked as a high-priority roadmap item. Suggested structure:

```
tests/
  test_model.py          # forward pass shape checks
  test_preprocessing.py  # polar transform determinism, fallback path
  test_dataset.py        # __getitem__ output range, label correctness
  test_continual.py      # EWC Fisher computation, penalty sign
```

## Submitting Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes with clear, minimal commits.
3. Run the import and evaluation checks above.
4. Open a pull request describing what changed and why.
