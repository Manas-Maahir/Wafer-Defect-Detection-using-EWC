# Invention Disclosure — Geometry-Aware CNN/Transformer Wafer Defect Classifier with Continual Learning

> **Scope and integrity note.** Every quantitative figure below is traceable to a file in
> this repository (cited as `path` or `path:line`). Where the patent queries assume results
> that the repository does not yet contain (e.g., an EWC-vs-no-EWC ablation, baseline
> comparisons, latency numbers), this is stated explicitly and the experiment needed to
> substantiate the claim is provided as a runnable script under [`experiments/`](experiments/).
> No numbers in this document are invented.

---

## §0. Architecture provenance & metric legend

Two embodiments exist in the repository history. The disclosure covers **both**: the
CNN–Transformer hybrid as originally conceived, and the pure-Transformer variant as a
simplified embodiment.

| | **Embodiment A — CNN–Swin hybrid** | **Embodiment B — Pure Swin** |
|---|---|---|
| Source | git commit `a9c6dea` (`git show a9c6dea:model.py`) | current working tree, [model.py](model.py) |
| Class | `HybridWaferModel` / `CNNFeatureExtractor` | `SwinWaferModel` |
| Front end | ResNet18 stem (`conv1`→`bn1`→`relu`→`maxpool`→`layer1`→`layer2`), 1→128 ch | none (Swin takes 1-ch input directly, `in_chans=1`) |
| Adapter | `Conv2d(128, 3, kernel_size=1)` → 3-ch for Swin | none |
| Backbone | `swin_tiny_patch4_window7_224` (pretrained, `num_classes=0`) | `swin_tiny_patch4_window7_224` (pretrained, `in_chans=1`, `num_classes=9`) |
| Head | `nn.Linear(768, 9)` | timm built-in 9-class head |
| Resize | CNN features → bilinear → 224×224 | input → bilinear → 224×224 ([model.py:23](model.py#L23)) |

**Metric legend — every result in §2 carries this provenance:**

| Tag | Meaning | Produced by |
|---|---|---|
| **B/val** | Embodiment B, **validation** set (5,027 samples) | [eval_out.txt](eval_out.txt) (header: "Loading validation data") |
| **B/test** | Embodiment B, **held-out test** set (5,028 samples, merged across 4 tasks) | [eval_outputs/metrics_report.txt](eval_outputs/metrics_report.txt) via [evaluate.py](evaluate.py) |
| **B/CL** | Embodiment B, 4-task **continual-learning** run, seed 42 | [eval_outputs/cl_metrics_seed42.json](eval_outputs/cl_metrics_seed42.json) via [train.py](train.py) |

> **Important honesty point.** The headline **83.19%** figure is a **validation** number
> (`B/val`); it equals the checkpoint-selection metric `best_val_acc=0.8319`
> ([eval_out.txt:20](eval_out.txt)) and is therefore optimistic. The honest
> generalization figure on the **held-out test set** is **61.52%** (`B/test`). Both are
> reported below; for a patent claim, prefer `B/test`.

All current quantitative results were produced by **Embodiment B**. No metrics for
Embodiment A exist in the repository; to claim performance for the hybrid it must be
restored and re-evaluated (see [`experiments/`](experiments/)).

---

## §1. Technical problem and how the framework overcomes prior limitations (Query 1)

**Problem in the manufacturing/inspection environment.**
1. **Geometric variability.** Raw wafer maps are Cartesian rasters whose wafer center,
   radius, and rotation vary between tools and lots. A defect that is physically the same
   (e.g., an edge ring) appears at different pixel locations and orientations, so a
   Cartesian CNN must learn many rotated/translated variants of the same pattern.
2. **Circumferential / edge defects are spatially diffuse in Cartesian space.** An
   Edge-Ring defect is a thin annulus spread around the whole wafer; in Cartesian pixels it
   is a curved, low-contrast structure with no compact receptive-field footprint.
3. **Evolving defect taxonomy.** Production lines encounter new defect classes over time.
   Naively fine-tuning a model on new classes causes **catastrophic forgetting** of earlier
   classes.

**How the disclosed framework addresses each (grounded in code).**
1. **Wafer normalization via edge-ring polar transform.** [preprocessing.py](preprocessing.py)
   detects the wafer center and radius (`detect_wafer`, [preprocessing.py:11-28](preprocessing.py#L11-L28))
   and resamples the outer annular edge ring into a fixed `64×360` polar strip
   (`cartesian_to_polar`, [preprocessing.py:44-71](preprocessing.py#L44-L71)). This removes
   per-wafer center/scale variability — the representation is **registration-invariant**.
2. **Rotation handled as cyclic translation.** Because the angular axis is the strip's
   horizontal axis, wafer rotation becomes a horizontal shift. Training augments with a
   cyclic angular roll, `torch.roll(img_tensor, shifts=shift, dims=2)`
   ([dataset_memmap.py:21-23](dataset_memmap.py#L21-L23)), giving rotation invariance for
   free.
3. **Global attention over the unwrapped ring.** A circumferential defect that was diffuse
   in Cartesian space becomes a **contiguous horizontal band** in the polar strip, which the
   Swin Transformer's windowed self-attention can integrate globally
   ([model.py](model.py)).
4. **Continual learning.** Sequential tasks are trained with Fisher-information EWC plus an
   experience-replay buffer ([train.py](train.py), [continual_learning.py](continual_learning.py))
   so new defect classes can be added incrementally (§5).

**Honesty note (must be disclosed).** The angular periodicity is exploited only through
*augmentation*; the model itself contains **no circular padding and no polar-specific
convolutions** — the Swin backbone treats the `64×360` strip as an ordinary rectangular
image and bilinearly resizes it to `224×224` ([model.py:23](model.py#L23)), which distorts
the native ~5.6:1 aspect ratio. "Geometry-aware" therefore refers to the **representation
and augmentation**, not to a geometry-specialized network operator.

---

## §2. Quantitative results (Query 2)

### 2.1 Per-class metrics — Embodiment B, **validation** (`B/val`)
Source: [eval_out.txt:31-74](eval_out.txt). Overall accuracy **0.8319**, macro-AUC **0.9779**.

| Class | Precision | Recall | F1 | AUC (OvR) | Support |
|---|---|---|---|---|---|
| Center | 0.8072 | 0.7091 | 0.7550 | 0.9593 | 667 |
| Donut | 0.4551 | 0.9205 | 0.6090 | 0.9932 | 88 |
| Edge-Loc | 0.8906 | 0.9112 | 0.9008 | 0.9905 | 777 |
| Edge-Ring | 0.9838 | 0.9831 | 0.9834 | 0.9993 | 1480 |
| Loc | 0.7279 | 0.5975 | 0.6563 | 0.9416 | 564 |
| Random | 0.7875 | 0.9618 | 0.8660 | 0.9986 | 131 |
| Scratch | 0.7094 | 0.7826 | 0.7442 | 0.9814 | 184 |
| Near-full | 0.7778 | 1.0000 | 0.8750 | 0.9998 | 21 |
| none | 0.7368 | 0.7507 | 0.7437 | 0.9377 | 1115 |
| **macro avg** | 0.7640 | 0.8463 | 0.7926 | **0.9779** | 5027 |
| **weighted avg** | 0.8372 | 0.8319 | 0.8316 | — | 5027 |

### 2.2 Per-class metrics — Embodiment B, **held-out test** (`B/test`)
Source: [eval_outputs/metrics_report.txt](eval_outputs/metrics_report.txt). Overall accuracy
**0.6152**, macro-AUC **0.9102**. This is the honest generalization figure.

| Class | Precision | Recall | F1 | AUC (OvR) |
|---|---|---|---|---|
| Center | 0.7663 | 0.6568 | 0.7074 | 0.9244 |
| Donut | 0.1027 | 0.8193 | 0.1826 | 0.9282 |
| Edge-Loc | 0.8219 | 0.3851 | 0.5245 | 0.7980 |
| Edge-Ring | 0.9846 | 0.9229 | 0.9527 | 0.9902 |
| Loc | 0.3623 | 0.5492 | 0.4366 | 0.8466 |
| Random | 0.6474 | 0.7769 | 0.7063 | 0.9720 |
| Scratch | 0.1895 | 0.4637 | 0.2690 | 0.8784 |
| Near-full | 0.2056 | 1.0000 | 0.3411 | 0.9982 |
| none | 0.8070 | 0.3833 | 0.5198 | 0.8556 |
| **macro avg** | 0.5430 | 0.6619 | 0.5155 | **0.9102** |
| **weighted avg** | 0.7673 | 0.6152 | 0.6502 | — |

**Reading the gap.** The val→test drop (0.83→0.62) is consistent with checkpoint selection on
the validation set. Edge defects remain strong on test (Edge-Ring F1 0.95, AUC 0.99); the
representation's edge-ring focus (§3, §4) explains why interior-defect classes (Loc, Donut,
Scratch) degrade most.

### 2.3 Continual-learning metrics — Embodiment B (`B/CL`, seed 42)
Source: [eval_outputs/cl_metrics_seed42.json](eval_outputs/cl_metrics_seed42.json).
`R[i,j]` = accuracy on task *j*'s held-out test set after training task *i*.

| trained ↓ / task → | T1 (none+Center) | T2 (Edge-Ring+Edge-Loc) | T3 (Scratch+Loc+Random) | T4 (Donut+Near-full) |
|---|---|---|---|---|
| After T1 | **0.8141** | — | — | — |
| After T2 | 0.7030 | **0.9372** | — | — |
| After T3 | 0.5285 | 0.8906 | **0.5495** | — |
| After T4 | 0.4585 | 0.8754 | 0.5861 | **0.6981** |

- **Average Accuracy (AA) = 0.6545**
- **Backward Transfer (BWT) = −0.1270** → **net forgetting** (T1 falls 0.8141→0.4585)
- **Forward Transfer (FWT) = 0.0**

### 2.4 Evidence gaps (do not file as if these exist)
- **No baseline comparison** vs. a pure CNN, a Transformer-from-scratch, or published
  WM-811K/LSWMD methods is present in the repo. → run [`experiments/baselines.py`](experiments/baselines.py).
- **No inference latency / throughput** is measured anywhere. → run
  [`experiments/benchmark_inference.py`](experiments/benchmark_inference.py).
- **No multi-seed statistics**: only seed 42 exists for `B/CL`. → re-run [train.py](train.py)
  with `--seed` and aggregate via [cl_metrics.py](cl_metrics.py).
- **No FP/FN-rate table**: derivable from the confusion matrix
  ([eval_outputs/confusion_matrix.png](eval_outputs/confusion_matrix.png)); current
  [evaluate.py](evaluate.py) does not print explicit FPR/FNR.

**Methodology (for reproducibility).** [evaluate.py](evaluate.py) computes the
classification report (`sklearn.classification_report`, [evaluate.py:163](evaluate.py#L163)),
confusion matrix ([evaluate.py:117-130](evaluate.py#L117-L130)), and One-vs-Rest ROC-AUC by
one-hot encoding the labels and averaging per-class `roc_auc_score`
([evaluate.py:133-159](evaluate.py#L133-L159)); per-class accuracy is the confusion-matrix
diagonal ([evaluate.py:183](evaluate.py#L183)).

---

## §3. Preferred / best implementation (Query 3)

**Dataset.** Derived from LSWMD (`LSWMD.pkl`) by [convert_pkl.py](convert_pkl.py), which
balances classes (≤12,000 per defect class, ≤8,000 `none`; [convert_pkl.py:7-8](convert_pkl.py#L7-L8),
random_state 42) and writes a memory-mapped `polar_strips.npy` of shape `(N, 64, 360)`
float32 plus `labels.csv`. **9 classes** with the mapping in
[dataset_memmap.py:46-50](dataset_memmap.py#L46-L50): Center, Donut, Edge-Loc, Edge-Ring,
Loc, Random, Scratch, Near-full, none.

**Polar-coordinate transformation (as actually generated).**
[convert_pkl.py:77](convert_pkl.py#L77) calls `preprocess_wafer(wafer_map)` with the
**default `edge_depth=5`**. Therefore the stored representation is an **edge-ring-only**
strip:
- `detect_wafer` finds center (center-of-mass of all dies) and radius (max die distance)
  ([preprocessing.py:11-28](preprocessing.py#L11-L28)).
- `cartesian_to_polar` samples radii `np.linspace(radius, radius−5, 64)` over angles
  `np.linspace(0, 2π, 360)` and remaps with `cv2.remap(..., cv2.INTER_LINEAR)`
  ([preprocessing.py:55-69](preprocessing.py#L55-L69)). Rows = radial depth (row 0 ≈ outer
  edge, row 63 ≈ 5 px inward); columns = azimuth 0–360°.
- Pixel values {0,1,2} (background / normal die / defect) are normalized **÷2.0** → {0,0.5,1.0}
  at generation ([convert_pkl.py:80](convert_pkl.py#L80)) and again in the loader
  ([dataset_memmap.py:85](dataset_memmap.py#L85)).

> **Design consequence, disclosed.** Only the outermost ≈5-px annulus is retained, oversampled
> into 64 rows. This makes the system **edge/circumferential-defect-focused** by construction.
> It is consistent with the results: edge classes are strongest (Edge-Ring) and the two
> weakest classes are the *interior* defects (Loc 0.598, Center 0.709 recall in `B/val`).
> A broader claim covering interior defects would require increasing `edge_depth` or adding a
> full-disk channel; this is a candidate continuation.

**Network.** Both embodiments per §0. Preferred (validated) embodiment is **B**: timm
`swin_tiny_patch4_window7_224`, `in_chans=1`, `num_classes=9`, input bilinearly resized to
224×224 ([model.py:12-25](model.py#L12-L25)).

**Training configuration** ([train.py:17-26](train.py#L17-L26), [train.py:88-94](train.py#L88-L94),
[train.py:211-213](train.py#L211-L213)): optimizer AdamW (lr 1e-4, weight decay 1e-4),
batch size 32, ≤8 epochs/task, 3-epoch linear warmup + cosine decay, gradient clip 1.0,
label smoothing 0.1, mixed precision (AMP + GradScaler), early stopping patience 3. Class
imbalance handled with a `WeightedRandomSampler` and class-weighted cross-entropy, with extra
up-weighting for rare/hard classes (Donut, Near-full, Loc, Scratch)
([train.py:59-80](train.py#L59-L80)).

**Continual-learning configuration.** Four sequential tasks
([dataset_memmap.py:52-57](dataset_memmap.py#L52-L57)): T1 {none, Center}, T2 {Edge-Ring,
Edge-Loc}, T3 {Scratch, Loc, Random}, T4 {Donut, Near-full}. Per-task stratified 3-way split
(70/15/15) via `get_task_splits` ([dataset_memmap.py:94-120](dataset_memmap.py#L94-L120));
the 15% test split is never seen in training. Experience replay keeps 100 samples/class,
concatenated into later tasks ([train.py:40-51](train.py#L40-L51)). EWC λ = 500
([train.py:22](train.py#L22)).

---

## §4. Polar vs. Cartesian (Query 4)

**Mechanistic argument.**
- **Circumferential / Edge-Ring defects:** in Cartesian space a thin ring around the wafer;
  after the edge-ring polar transform it becomes a **single horizontal band spanning all 360
  columns**, i.e., a translation-stationary pattern ideal for windowed attention.
- **Edge-localized (Edge-Loc) defects:** a localized arc → a compact horizontal segment at a
  specific column range; its angular position is encoded directly by column index.
- **Radial defects (e.g., scratches reaching the edge):** appear as short vertical structures
  spanning the 64 radial rows at fixed columns.
- **Rotation invariance:** wafer rotation = horizontal cyclic shift, realized exactly by
  `torch.roll(..., dims=2)` augmentation ([dataset_memmap.py:21-23](dataset_memmap.py#L21-L23));
  no equivalent simple operator exists in Cartesian space.

**Supporting (not isolating) evidence.** Under this representation, edge classes are the
strongest in both `B/val` and `B/test` (Edge-Ring F1 0.983 / 0.953; Edge-Loc strong on val),
consistent with the polar-edge hypothesis.

**Honesty note.** The repository contains **no controlled Cartesian-vs-polar ablation**
(same network trained on Cartesian maps vs. polar strips). The above is the **design
rationale plus consistent observations**, not an isolated causal measurement. The ablation is
listed as a recommended experiment.

---

## §5. EWC implementation and forgetting evidence (Query 5)

**Diagonal Fisher information** ([continual_learning.py:33-63](continual_learning.py#L33-L63)).
After a task, gradients of the cross-entropy loss are accumulated as squared values over the
task's dataloader and averaged over batches: `F_i = mean_batch (∂L/∂θ_i)²`.

**Anchor (θ\*) and penalty.** `register_prior_task` snapshots current weights as θ\* and
computes the Fisher ([continual_learning.py:66-77](continual_learning.py#L66-L77)). The
penalty is the quadratic
`Σ_i F_i · (θ_i − θ*_i)²` ([continual_learning.py:79-89](continual_learning.py#L79-L89)).

**Selective loss.** `selective_ewc_loss` returns `task_loss + λ · penalty` with λ = 500
([continual_learning.py:93-115](continual_learning.py#L93-L115),
[train.py:114](train.py#L114)). "Selective" = the penalty is **weighted per-parameter by
Fisher importance**, so high-Fisher (task-critical) weights are strongly anchored while
low-Fisher weights stay plastic; parameters with no recorded Fisher are unconstrained
([continual_learning.py:86](continual_learning.py#L86)).

**Disclosed limitation — Fisher is overwritten each task.**
`self._precision_matrices = self._diag_fisher()` ([continual_learning.py:77](continual_learning.py#L77))
**replaces** rather than accumulates Fisher, and a fresh `EWC(...)` is constructed per task
([train.py:251-252](train.py#L251-L252)). Consequently EWC explicitly protects only the
**immediately prior task**; older tasks rely on the replay buffer alone.

**Experimental evidence — stated truthfully.** The single available CL run (`B/CL`, seed 42)
yields **BWT = −0.1270**, i.e., the current EWC+replay configuration **does not prevent
forgetting** on this task sequence (T1 accuracy falls 0.8141 → 0.4585 by T4; §2.3).

> **Therefore the Query-5 claim ("EWC reduces catastrophic forgetting vs. retraining without
> EWC") is NOT yet substantiated by this repository.** There is no EWC-vs-no-EWC comparison,
> and the one measured configuration shows net forgetting. To support the claim, run the
> ablation in [`experiments/`](experiments/) (four regimes: naive / EWC-only / replay-only /
> EWC+replay, multiple seeds) and report ΔBWT and ΔAA. Only positive, reproducible deltas
> should be put in a filing.

---

## §6. Recommended experiments to close the gaps (deliverables)

See [`experiments/README.md`](experiments/README.md). These run on a CUDA machine (the
authoring environment has no GPU; full retraining is impractical on CPU per CLAUDE.md).

1. **EWC / replay ablation** — [train.py](train.py) now accepts `--no-ewc`, `--no-replay`,
   `--ewc-lambda`, `--tag`. Driver: [`experiments/run_ablation.py`](experiments/run_ablation.py)
   runs the four regimes × seeds and aggregates AA/BWT/FWT via [cl_metrics.py](cl_metrics.py).
   *Closes §2.4 and §5.*
2. **Baselines** — [`experiments/baselines.py`](experiments/baselines.py) trains a pure-CNN
   (ResNet18, 1-ch, 9-class) and a Swin-from-scratch on the same merged split and reports
   accuracy / macro-AUC / macro-F1 against Embodiment B. *Closes §2.4 baseline gap and §4 (by
   adding a Cartesian-input variant).*
3. **Latency / throughput** — [`experiments/benchmark_inference.py`](experiments/benchmark_inference.py)
   reports params, model size, ms/sample, and samples/sec for both embodiments on CPU and
   CUDA. *Closes §2.4 latency gap.*

---

## Appendix A — Verbatim module definitions

**Embodiment B** ([model.py:6-26](model.py#L6-L26)):
```python
class SwinWaferModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True, in_chans=1, num_classes=num_classes)
    def forward(self, x):                       # x: (B, 1, 64, 360)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.swin(x)
```

**Embodiment A** (`git show a9c6dea:model.py`): `CNNFeatureExtractor` (ResNet18 conv1+bn1+
relu+maxpool+layer1+layer2 → 128 ch) → `F.interpolate` to 224×224 → `Conv2d(128,3,1)` →
pretrained Swin (`num_classes=0`) → `Linear(768, 9)`.
