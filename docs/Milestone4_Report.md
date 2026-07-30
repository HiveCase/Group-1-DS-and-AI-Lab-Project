
---

<div align="center">


<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">


<h1 style="font-size:26em;">Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 4: Model Training</h2>

<h3>Group 1</h3>

<br>

  ***Prepared by:***

  
| **Name** | **Email ID** | **GitHub Profile** |
| --- | --- | --- |
| SATYAJEET KUMAR | 23f1003132@ds.study.iitm.ac.in | [HiveCase](https://github.com/HiveCase) |
| ANUJ GAUTAM | 21f1002407@ds.study.iitm.ac.in | [anujgautam1](https://github.com/anujgautam1) |
| PRANAB KUMAR MANNA | 22f1000887@ds.study.iitm.ac.in | [pranab92](https://github.com/pranab92) |
| VENKATA SIVA KAMAL GUDDANTI | 22f2000094@ds.study.iitm.ac.in | [22f2000094](https://github.com/22f2000094) |
| HARSH PAL | 21f1002562@ds.study.iitm.ac.in | [HarshPalaps1](https://github.com/HarshPalaps1) |

</div>


---

# Table of Contents

- [1. Introduction](#1-introduction)
- [2. Training Dataset](#2-training-dataset)
- [3. Model Configuration](#3-model-configuration)
- [4. Training Environment](#4-training-environment)
- [5. Training Methodology](#5-training-methodology)
- [6. Hyperparameter Experiments](#6-hyperparameter-experiments)
- [7. Optimization Methods](#7-optimization-methods)
- [8. Regularization Techniques](#8-regularization-techniques)
- [9. Training Progress](#9-training-progress)
- [10. Model Selection](#10-model-selection)
- [11. Challenges Encountered](#11-challenges-encountered)
- [12. Summary and Next Steps](#12-summary-and-next-steps)

---

## 1. Introduction

### 1.1 Recap of Selected Models (Milestone 3)

Milestone 3 selected **YOLO11m** (plain bounding-box detection, not the `-seg` variant) as the primary architecture for the Damage Agent, on a tie-break against YOLOv8m after a 15-epoch probe found the two statistically indistinguishable (mAP@50 delta 0.0066, below the 0.02 tie-break threshold). The tie-break favoured YOLO11m's newer C3k2/C2PSA backbone blocks, reported to aid small-object detection — directly relevant given Milestone 2's EDA found a minimum normalised bounding-box area of 0.00002 in the training data.

### 1.2 Objectives of the Training Phase

- Fine-tune the Milestone 3-selected YOLO11m detector on the full VehiDE training set and reach a stable, converged checkpoint.
- Diagnose and correct the class-imbalance handling (`cls` loss weighting) that caused an early precision/recall collapse.
- Evaluate whether a CarDD-pretrained segmentation checkpoint, fine-tuned on the same VehiDE data, offers a viable supplementary or alternative path for the classes the detection model struggled with.
- Establish a robust, resumable multi-session training workflow given Kaggle's session time limits, after losing training progress to unplanned session terminations more than once during this milestone.
- Select the best available checkpoint(s) and document, honestly, which experiments reached a complete, evaluated result within this milestone's timeframe and which remain open going into Milestone 5.

---

## 2. Training Dataset

### 2.1 Final Datasets Used

Two label formats were used across the two experiment tracks, both derived from the same underlying VehiDE imagery and the same Milestone 2 stratified split, so results remain comparable across tracks:

| | Detection track (Damage Agent) | Segmentation track (CarDD contingency) |
| --- | --- | --- |
| Label format | YOLO bounding box (`class x_center y_center w h`) | YOLO instance segmentation polygon (`class x1 y1 x2 y2 ... xn yn`) |
| Source of labels | Milestone 2's bbox conversion of VehiDE VIA polygons | The same VehiDE VIA polygons, converted directly to YOLO-seg format (not re-annotated) |
| Image preprocessing | Letterboxed to 1280×1280 (Milestone 2) | Letterboxed to 1280×1280, using a corrected letterbox-aware coordinate transform (Section 11.4) |

### 2.2 Train / Validation / Test Split

Both tracks reuse the identical Milestone 2 stratified 70/15/15 split (by dominant class per image, seed 42), verified leakage-free by both filename-stem and MD5-hash checks:

| Split | Images |
| --- | --- |
| Train | 9,558 |
| Validation | 2,048 |
| Test | 2,049 |

Total retained instances (bounding-box track): 32,672. Segmentation-track instance counts per evaluation split (val / test), from actual evaluation runs:

| Class | Val instances | Test instances |
| --- | --- | --- |
| dent | 846 | 825 |
| scratch | 2,212 | 2,174 |
| crack | 856 | 765 |
| broken_lamp | 412 | 392 |
| shattered_glass | 321 | 325 |
| flat_tyre | 395 | 365 |
| **Total** | **5,042** | **4,846** |

### 2.3 Data Augmentation and Preprocessing

- **Detection track:** Ultralytics' default augmentation stack (mosaic, HSV jitter, flip, scale/translate/erasing), `close_mosaic=10` (mosaic disabled for the final 10 of 50 epochs).
- **Segmentation track, baseline runs:** default Ultralytics augmentation only.
- **Segmentation track, continuation run:** default augmentation plus deliberately added `copy_paste=0.3`, raised `hsv_v`/`hsv_s`, `degrees=10`, `shear=5`, and increased `scale`, targeted specifically at the thin, low-contrast, boundary-ambiguous classes (`dent`, `scratch`, `crack`) that underperformed in the baseline run.
- All preprocessing (letterboxing, PII blurring, deduplication) is inherited unchanged from Milestone 2; no new preprocessing was introduced in Milestone 4 beyond the segmentation-format polygon conversion described in Section 11.4.

---

## 3. Model Configuration

### 3.1 Final Architecture(s)

| Model | Track | Base checkpoint | Pretrained source |
| --- | --- | --- | --- |
| YOLO11m | Detection (primary Damage Agent) | `yolo11m.pt` | COCO/Objects365 (Ultralytics) |
| YOLOv8s-seg | Segmentation (CarDD contingency) | `abdullahg7/cardd-yolov8s` (v2.0) | CarDD dataset, via Hugging Face |
| YOLO11x-seg | Segmentation (explored, not completed — see Section 11) | `harpreetsahota/car-dd-segmentation-yolov11` | CarDD dataset, via Hugging Face |
| YOLO11s-seg | Segmentation (probed only, not trained to completion) | `yolo11s-seg.pt` | COCO (Ultralytics) |

### 3.2 Pretrained vs. Training from Scratch

All models were fine-tuned from pretrained checkpoints; none were trained from random initialisation. For the detection track, `yolo11m.pt`'s detection head was reinitialised for this project's 6 classes (COCO's 80-class head does not transfer). For the segmentation track, head transfer depended on class-name string matching between the checkpoint's own labels and this project's taxonomy:

- `cardd-yolov8s`: all 6 classes matched, full head transfer.
- `car-dd-segmentation-yolov11` (YOLO11x-seg): only 3 of 6 classes matched (`dent`, `scratch`, `crack`); `broken_lamp`, `shattered_glass`, `flat_tyre` heads were randomly reinitialised, confirmed directly in the training log (`"Remapped 3/6 cls head rows from pretrained weights by class name"`).
- `yolo11s-seg.pt`: COCO-pretrained, no class overlap with this project's taxonomy by design; all 6 heads reinitialised.

### 3.3 Model-Specific Configuration

| | YOLO11m (detection) | YOLOv8s-seg (final continuation) |
| --- | --- | --- |
| Input size | 1280×1280 | 1280×1280 |
| Task head | Detection only | Instance segmentation |
| Parameters | ~20.1M (measured) | ~11.8M (measured) |

---

## 4. Training Environment

### 4.1 Hardware

- Kaggle GPU sessions: **Tesla T4** (16 GB) used for all completed training runs.
- **P100** was attempted for one segmentation experiment and found **incompatible** with the installed PyTorch build (`CUDA error: no kernel image is available for execution on the device` — P100's Pascal architecture, compute capability 6.0, has no compiled kernels in the `torch-2.10.0+cu128` build Kaggle provisioned). Not fixable via configuration; resolved by switching back to T4.

### 4.2 Software

| Component | Version(s) observed |
| --- | --- |
| Ultralytics | 8.4.104 – 8.4.110 (varied slightly across sessions) |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.13 |

### 4.3 Resource Constraints

- Kaggle's free-tier GPU quota (weekly hours) was a binding constraint throughout this milestone and directly shaped which experiments were pursued to completion versus abandoned (Section 11).
- Kaggle interactive sessions do not reliably run long enough to complete a multi-hour training job in one sitting; this required building a custom multi-session checkpoint-relay mechanism (Section 11.2) partway through the milestone, after losing an entire ~25-epoch run to an unplanned session termination.
- `/kaggle/working/` does not persist across sessions; all cross-session checkpoint recovery relies on periodically publishing checkpoints to a separate Kaggle Dataset.

---

## 5. Training Methodology

### 5.1 Training Workflow

1. Mount the Milestone 2 (detection) or converted (segmentation) dataset as a Kaggle Dataset input.
2. Load the pretrained checkpoint.
3. Run a short probe (1 epoch on a fixed subsample, later extended to separately time training and validation — see Section 11.5) to confirm the configuration fits in VRAM and to estimate real wall-clock cost before committing to a full run.
4. Train with a per-epoch callback that periodically backs up `last.pt` and training state to a dedicated Kaggle Dataset, and optionally stops the session early at a configured epoch budget.
5. On a new session, check for an existing backup; if found, resume from it (`resume=True`); otherwise start fresh from the pretrained checkpoint.
6. Evaluate the final checkpoint against the held-out test split.

### 5.2 Batch Size

**4**, used consistently across all completed runs (both tracks), at 1280px input. This was set based on observed VRAM usage (~8.5–9.4 GB of 14.9 GB available at batch=4 for the detection model); larger batch sizes were not tested for the primary runs. Batch=4 at imgsz=1280/1024 also OOM'd for the much larger YOLO11x-seg model even after Ultralytics' automatic batch-size reduction to 1 (Section 11.1).

### 5.3 Number of Epochs

| Run | Epochs |
| --- | --- |
| Detection (YOLO11m), `cls=2.0` | Abandoned after confirming a precision/recall collapse pattern across the completed epochs (Section 6); not carried forward |
| Detection (YOLO11m), `cls=0.5` + `cos_lr` | Interrupted by unplanned session terminations on more than one occasion during this milestone (exact sequence not fully reconstructed here); furthest confirmed progress was **25 completed epochs**, measured mAP@50 = 0.0248 at that point; not resumed to completion within this milestone |
| Segmentation (YOLOv8s-seg), baseline | 30 |
| Segmentation (YOLOv8s-seg), continuation (augmentation) | 20 additional (50 total) |
| Segmentation (YOLOv8s-seg), DFL boundary-precision experiment | 30 (completed) |

### 5.4 Loss Function

Ultralytics' composite YOLO loss: CIoU box-regression loss, BCE classification loss (weighted by `cls`), DFL (Distribution Focal Loss, weighted by `dfl`) for box/mask boundary refinement, plus a segmentation mask loss (`seg_loss`) for the `-seg` models. All segmentation-track training logs also report a `sem_loss` term that remained at exactly 0 in every run observed; its origin was not identified during this milestone (see Section 11.6).

### 5.5 Learning Strategy

AdamW optimiser throughout. Learning rate strategy evolved across the milestone (Section 6): initial runs used a fixed low `lr0` with linear decay; later runs adopted cosine decay (`cos_lr=True`) with a lower final learning rate (`lrf`), and warmup epochs were extended for checkpoints requiring a larger adaptation (fresh COCO/CarDD transfer) versus shortened for checkpoints continuing from an already well-adapted state.

---

## 6. Hyperparameter Experiments

| Experiment | `cls` | `dfl` | `lr0` | LR schedule | Result | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Detection, attempt 1 | 2.0 | default (1.5) | 0.001 | linear | Precision collapsed to ~0.85–0.87 while recall collapsed toward 0 (model learned to predict almost nothing) | Rejected |
| Detection, `cls=0.5` attempt | 0.5 | default | 0.001 | cosine (`lrf=0.001`) | mAP@50 climbing across all 25 completed epochs, reaching 0.0248 at epoch 25 (still well below the ≥0.70 Milestone 1 target, but trending upward, not stalled) | Promising but incomplete — interrupted before reaching a comparable stage to the segmentation track |
| Segmentation baseline (YOLOv8s-seg) | 0.5 | default | 0.0005 | cosine | Converged; overall mask mAP50 = 0.36 (val, epoch 30) | Completed |
| Segmentation continuation (augmentation) | 0.5 | default | 0.0002 | cosine | Overall mask mAP50 (test) 0.348 → 0.3534; `dent`/`scratch`/`crack` essentially flat (+0.001 to +0.011); `shattered_glass` slightly regressed (−0.014) | Completed; hypothesis (augmentation fixes hard classes) not confirmed |
| Segmentation, DFL boundary-precision | 0.3 | 1.7 | 0.0002 | cosine | Overall test mask mAP50 0.3534 → 0.3549 (essentially flat); `dent`/`scratch` small gains (+0.005 to +0.014), `crack` flat (−0.001); `broken_lamp` gained (+0.041); `shattered_glass` regressed (−0.059) | Completed; same outcome pattern as the augmentation experiment — hard classes did not move |

**Justification for the segmentation track's final selected configuration** (Section 10): the continuation run (`cls=0.5`, augmentation added) is the only segmentation checkpoint with a complete, held-out test-set evaluation at the time of writing. It is selected as the current best available checkpoint on that basis, not because it resolved the underlying weak-class problem — it did not.

Two model-scale experiments were also run as probes, not full training:

| Model | Parameters | Measured/estimated cost | Outcome |
| --- | --- | --- | --- |
| YOLO11x-seg (CarDD-pretrained) | 62.1M | OOM at imgsz=1280 with `multi_scale=True`, even at batch=1 on a 14.9 GB T4 | Abandoned — architecture too large for available hardware at the attempted configuration |
| YOLO11s-seg (COCO-pretrained) | 10.08M | Probe indicated ~21 min/epoch after correcting a timing-measurement error (Section 11.5); full run not executed within this milestone | Not completed |

---

## 7. Optimization Methods

- **Optimizer:** AdamW throughout, for both tracks. No alternative optimiser (e.g. SGD) was tested during this milestone.
- **Learning-rate scheduler:** Cosine annealing (`cos_lr=True`) adopted after the initial linear-decay detection attempt; not formally A/B tested against linear decay once adopted.
- **Early stopping:** Ultralytics' `patience` parameter used (15–20 depending on run), monitoring validation fitness. Not observed to trigger in any completed run within this milestone — all completed runs reached their configured epoch count rather than stopping early.
- **Gradient clipping:** Not explicitly configured in any run (Ultralytics' default behaviour used, unmodified).
- **Mixed precision training:** `amp=True` used throughout (Ultralytics default); AMP compatibility checks passed in every training log observed.

---

## 8. Regularization Techniques

- **Dropout:** Not used in the detection track. Used at `0.1` in the segmentation continuation and DFL-boundary experiments, following the published recipe associated with the `harpreetsahota` checkpoint.
- **Weight decay:** `0.0005`, consistent across all runs, both tracks.
- **Data augmentation:** Ultralytics defaults throughout; additionally `copy_paste`, extended HSV jitter, and geometric jitter (`degrees`, `shear`, `scale`) in the segmentation continuation run (Section 2.3).
- **Label smoothing:** Not used in any run.
- **Class weighting:** A single global `cls` loss-gain scalar was used as a coarse imbalance-mitigation lever (varied across experiments, Section 6). Milestone 2's per-class inverse-frequency weight vector (designed in Section 8.2 of that milestone) was **not** implemented — Ultralytics' `cls_pw` parameter, which would carry true per-class weights, was left at its default (unused, confirmed `0.0` in every training log observed) in every run this milestone. This remains an open gap, not a completed piece of work.
- **Cross-validation:** Not applicable — a single fixed stratified split (Milestone 2) was used throughout, consistent with the project's leakage-safety design.

---

## 9. Training Progress

### 9.1 Evidence of Convergence

- **Segmentation baseline (30 epochs):** `box_loss`, `seg_loss`, `cls_loss`, and `dfl_loss` declined monotonically across all 30 epochs with no reversals; mask mAP50 rose from 0.157 (epoch 1) to 0.36 (epoch 30, val split), the majority of the gain concentrated in the final third of training.
- **Segmentation continuation (20 further epochs):** overall mask mAP50 held roughly flat in a narrow band (0.317–0.339) through most of the run, then rose sharply at the `close_mosaic` transition (epoch 16, mosaic disabled), reaching 0.36 by epoch 16 and continuing to a final value of 0.362 (val) by epoch 20.
- **Detection (`cls=0.5` attempt):** losses declined across all 25 completed epochs with no reversals observed; mAP@50 rose from near-zero to 0.0248 by epoch 25 — real, sustained progress, but well short of the Milestone 1 target and the run was interrupted before reaching a stage comparable to the segmentation track.

### 9.2 Underfitting Observations

The clearest and most consistent finding across this milestone: **`dent`, `scratch`, and `crack` underperform every other class, and this did not resolve with more epochs, augmentation, or a different `cls`/`dfl` weighting tried so far.** In the segmentation continuation run's final per-class test evaluation, `scratch` — the class with the *most* training instances of any class (2,174 in test alone) — still scored among the two worst-performing classes (mask mAP50 0.203), while `shattered_glass` — the class with the *fewest* instances — scored best (0.661). This rules out a simple data-volume explanation and points to intrinsic difficulty (thin, low-contrast, boundary-ambiguous shapes) as the dominant factor, consistent with the published CarDD paper's own finding that even its best benchmarked method underperforms on these same classes.

### 9.3 Overfitting Observations

No clear evidence of overfitting was observed in any completed run — validation metrics tracked training-loss improvements in the same direction throughout, with no divergence between training and validation performance recorded in the logs reviewed for this milestone.

---

## 10. Model Selection

### 10.1 Checkpoint Selected

Two segmentation checkpoints now have complete, held-out test-set evaluations and are essentially tied on overall performance:

| Checkpoint | Total epochs | Overall test mask mAP50 |
| --- | --- | --- |
| Continuation (augmentation) | 50 (30 baseline + 20) | 0.3534 |
| DFL boundary-precision | 80 (30 baseline + 20 augmentation + 30 DFL) | 0.3549 |

The **DFL boundary-precision checkpoint** is selected as the current best available result, on the basis of the marginally higher overall score and its additional gains on `dent`, `scratch`, and `broken_lamp`. This selection is a close call, not a clear win: the continuation checkpoint remains preferable specifically on `shattered_glass` (0.6607 vs. 0.6021), so the choice between them should be revisited once Milestone 5's evaluation weights per-class performance more explicitly rather than relying on a single overall figure.

### 10.2 Why It Was Selected

It is the most recent checkpoint in the segmentation track's fine-tuning chain, incorporates a genuinely different loss-weighting mechanism (`dfl`) than either of the two checkpoints before it, and has a complete test-set evaluation. The detection track's most promising run (`cls=0.5`, cosine schedule) was interrupted mid-training and has no comparably complete evaluation to select against.

### 10.3 Validation Metric Used

Overall **mask mAP@50** on the held-out test split was the primary comparison metric across all three segmentation runs (0.348 → 0.3534 → 0.3549). Per-class mask mAP@50 was used to assess whether each intervention (augmentation, then DFL reweighting) achieved its specific goal of improving `dent`/`scratch`/`crack` — **neither did, meaningfully**, across two independent attempts using two different mechanisms.

**This selection should be treated as provisional, not final.** It reflects the most complete result available within this milestone's time and compute budget, not a claim that the underlying weak-class problem has been solved — the evidence gathered this milestone points toward that problem being a genuine limit of this modelling approach on these classes, not a remaining configuration to find.

---

## 11. Challenges Encountered

### 11.1 GPU Memory Limitations

YOLO11x-seg (62.1M parameters, ~297 GFLOPs) OOM'd at `imgsz=1280` with `multi_scale=True`, even after Ultralytics' automatic batch-size reduction sequence (4 → 2 → 1) — confirming the issue was the model's base memory footprint at this image size, not simply an undersized batch. Resolved for smaller models by fixing `imgsz=1024`/`1280` (model-dependent) and disabling `multi_scale`; YOLO11x-seg itself was not successfully trained within this constraint.

### 11.2 Session Losses and the Multi-Session Checkpoint Relay

Training was interrupted by session termination on at least three separate occasions during this milestone, including one incident where ~25 epochs of detection training were lost entirely with no recoverable checkpoint (`/kaggle/working/` had been wiped by a fresh container allocation). This motivated building a checkpoint-relay system: periodic backup of `last.pt` to a dedicated Kaggle Dataset during training, with automatic detection and resumption from the latest backup at the start of each new session.

An early version of this system had a silent-failure bug (backup failures were not distinguished from successes in the printed output); this was identified and fixed to report backup failures explicitly. That fix then surfaced a further, more serious finding: in the DFL boundary-precision run (Section 6), **all six scheduled backup attempts failed** (epochs 5, 10, 15, 20, 25, 30), with the failure diagnostic itself printing no usable error text (`result.stderr` was empty on every failure). This run happened to complete in a single, uninterrupted session, so no training progress was actually lost — but the safety net was non-functional for its entire duration, and would have caused a full, unrecoverable loss had the session terminated early, exactly as happened earlier in this milestone. The diagnostic was extended to capture and print both `stdout` and `stderr` (some `kaggle` CLI failures route their error text through `stdout`), and the `-q` quiet flag was removed from the backup commands, since it may have been suppressing the relevant output. Whether this resolves the underlying failure has not yet been confirmed against a real run at the time of writing.

### 11.3 Hardware/Software Compatibility

P100 was found incompatible with the installed PyTorch build (Section 4.1) — a hardware/toolchain mismatch, not a configuration issue, discovered only after building a full training pipeline around it. A compute-capability check was added to subsequent notebooks to catch this class of failure immediately rather than after significant setup time.

### 11.4 Dataset/Label Quality

A coordinate-alignment error was identified in the letterbox-and-normalise logic used when converting VIA polygons to a new label format: bounding-box/polygon coordinates were being normalised against the *original* image dimensions but paired with the *letterboxed and padded* image, without adjusting for the padding offset. For a non-square image this silently misaligns labels against the image content (a 10% vertical offset was demonstrated on a representative test case). This was corrected for the segmentation-track label conversion. **Whether the same issue affects the Milestone 2 detection-track bounding-box labels currently in use for the Damage Agent has not been confirmed or ruled out within this milestone** and is flagged as an open item for Milestone 5.

### 11.5 Hyperparameter Sensitivity

`cls` loss weight had a large, non-obvious effect: `cls=2.0` produced a precision/recall collapse (Section 6); `cls=0.5` did not. A separate, unrelated measurement error was also found and corrected during this milestone: an early probe-based time/cost estimation method scaled a combined train+validation timing measurement by the training-set-size ratio, which incorrectly inflated validation's fixed cost as if it scaled with dataset size — one specific instance overstated a real per-epoch cost by roughly 8x (177 minutes estimated vs. an actual, corrected estimate of approximately 21 minutes). This was corrected by timing training and validation as two separate measurements.

### 11.6 Unresolved Observations

Every segmentation-track training log reported a `sem_loss` term at a constant value of 0. This does not appear in standard Ultralytics YOLO-seg loss reporting and may indicate an auxiliary loss head specific to one or more of the pretrained checkpoints used; its origin and whether it is functioning as intended was not investigated to a conclusion within this milestone.

### 11.7 Class Imbalance

Carried forward from Milestone 2 (6.68:1 imbalance ratio, `scratch` vs. `shattered_glass`). Only a uniform `cls` loss-gain scalar was applied as mitigation this milestone (Section 8); the per-class weighting designed in Milestone 2 was not implemented in any training run.

---

## 12. Summary and Next Steps

### 12.1 Best-Performing Training Configuration

Among completed, fully evaluated runs, the DFL boundary-precision configuration (YOLOv8s-seg, CarDD-pretrained, 80 total fine-tuning epochs, `cls=0.3`, `dfl=1.7` for its final 30 epochs) produced the highest overall result: test-set mask mAP@50 = 0.3549, marginally ahead of the continuation checkpoint's 0.3534. This is provisional, not final, and the margin between the two is small enough that either could reasonably be carried forward (Section 10.1).

### 12.2 Key Observations

- Loss-weighting choices (`cls`) have a large, easily-missed effect on whether a model learns to detect anything useful at all, independent of architecture or dataset.
- **Two independent interventions — targeted augmentation, then a boundary-precision loss reweighting (`dfl`) — both failed to meaningfully move the three hardest classes (`dent`, `scratch`, `crack`)** in this milestone's experiments. Across both, overall mAP moved by roughly +0.001 to +0.005 per intervention while individual non-target classes shifted considerably more (`broken_lamp` +0.04, `shattered_glass` −0.06 in the DFL run alone). Taken together, this is now fairly strong evidence that the limitation is intrinsic to these classes' visual characteristics (thin, low-contrast, boundary-ambiguous), not a solvable training-configuration problem, consistent with published literature on this same dataset (Section 9.2).
- Multi-session training infrastructure (checkpoint relay, explicit compatibility and timing checks) proved necessary, not optional, given Kaggle's session limits — and several of this milestone's most time-consuming problems were process/tooling failures (lost sessions, a flawed timing estimate, a hardware incompatibility, and a backup mechanism that failed silently and then failed completely) rather than modelling failures.

### 12.3 Readiness for Milestone 5

Readiness is **partial**, and this should be stated plainly rather than implied otherwise:

- The segmentation track has two checkpoints with complete, comparable test-set evaluations, ready for the more thorough error analysis Milestone 5 calls for — including the now well-evidenced open question of whether `dent`/`scratch`/`crack` should be addressed by a different approach entirely (e.g. the DCN+ architecture identified as this dataset's published state of the art) rather than further hyperparameter iteration on the current architecture family.
- The primary detection track (YOLO11m, the Milestone 3-selected Damage Agent) does **not** have a completed training run within this milestone — its most promising configuration was interrupted mid-training and has not yet been resumed to completion. This is the most significant open item going into Milestone 5, since it is the model actually selected for the deployed pipeline.
- The letterbox-alignment question raised in Section 11.4 for the detection-track labels remains open and should be resolved before treating any future detection-track evaluation as reliable.
- The backup-mechanism fix described in Section 11.2 has not yet been verified against a real training run; this should be confirmed working before relying on it for the detection track's remaining training.



---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar |  |  |
|Pranab Kumar Manna | | |
| Venkata Siva Kamal Guddanti |  |  |
| Anuj Gautam |  |  |
| Harsh Pal | | |  | |

---
