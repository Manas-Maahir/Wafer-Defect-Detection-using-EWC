# Experiments — closing the evidence gaps for the invention disclosure

These scripts generate the evidence that [`../INVENTION_DISCLOSURE.md`](../INVENTION_DISCLOSURE.md)
flags as missing (§2.4, §4, §5). **Run them on a CUDA machine** — full retraining is
impractical on CPU (see `CLAUDE.md`). All write JSON/txt into `../eval_outputs/`.

Run from the repository root so `polar_strips.npy` and `labels.csv` resolve.

## 1. EWC / replay ablation  (substantiates §5 and §2.4 multi-seed)
`train.py` accepts `--no-ewc`, `--no-replay`, `--ewc-lambda`, `--tag`. The driver runs the
four regimes across seeds and prints aggregated AA / BWT / FWT.

```bash
python -m experiments.run_ablation --seeds 42 123 456
# or run one regime manually:
python train.py --no-ewc --no-replay --tag naive       --seed 42   # naive fine-tuning
python train.py            --no-replay --tag ewc_only  --seed 42   # EWC only
python train.py --no-ewc               --tag replay_only --seed 42 # replay only
python train.py                          --seed 42                 # EWC + replay (default)
```
Compare BWT across regimes: a **less-negative (or positive) BWT for EWC vs. naive** is the
forgetting-reduction evidence the patent needs. Only positive, reproducible deltas should be
filed.

## 2. Baselines  (substantiates §2.4 baseline gap; optional Cartesian variant for §4)
```bash
python -m experiments.baselines --model cnn         --epochs 8   # ResNet18, 1-ch, 9-class
python -m experiments.baselines --model swin_scratch --epochs 8  # Swin, no pretraining
python -m experiments.baselines --model swin         --epochs 8  # = Embodiment B (reference)
```
Reports overall accuracy, macro-AUC, macro-F1 on the same merged held-out test split used by
`evaluate.py`, written to `../eval_outputs/baseline_<model>.json`.

## 3. Latency / throughput  (substantiates §2.4 latency gap)
```bash
python -m experiments.benchmark_inference --batch-sizes 1 8 32 --iters 50
```
Reports parameter count, model size (MB), ms/sample and samples/sec for **both** embodiments
(pure Swin and the CNN–Swin hybrid) on CPU and, if available, CUDA. Written to
`../eval_outputs/latency.json`.
