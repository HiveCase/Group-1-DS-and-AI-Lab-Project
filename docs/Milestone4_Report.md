
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
- [13. Policy Agent (RAG): Selection and Tuning](#13-policy-agent-rag-selection-and-tuning)

---

## 1. Introduction

### 1.1 Recap

Milestone 3 selected **YOLO11m-seg** (instance segmentation, `m`/medium scale, not plain bounding-box detection) as the primary architecture for the Damage Agent, so that the Severity Agent's area-ratio proxy could consume a pixel-precise mask rather than an overestimating bounding box. The YOLO11-vs-YOLOv8 generation choice was made via a fast box-only probe (mAP@50 delta 0.0066, statistically indistinguishable at that data scale), tie-broken in favour of YOLO11's newer C3k2/C2PSA backbone blocks, reported to aid small-object detection — directly relevant given Milestone 2's EDA found a minimum normalised bounding-box area of 0.00002 in the training data. That box-only probe was explicitly a low-cost proxy for the generation/backbone choice, not a claim that plain detection was the selected task.

### 1.2 Objectives of the Training Phase

- Fine-tune the Milestone 3-selected **YOLO11m-seg** (COCO-pretrained) on the full VehiDE segmentation training set and reach a stable, converged checkpoint — this is the milestone's **primary track**.
- Run domain-pretrained segmentation checkpoints (CarDD-pretrained YOLOv8s-seg, YOLO11x-seg) in parallel as a **comparative benchmark track**, to understand how much domain-specific pretraining is worth relative to the medium-scale YOLO11 architecture selected in Milestone 3, and to inform the trade-off discussion even though a different backbone generation/size than the Milestone 3 selection is not the production candidate.
- Diagnose and correct the class-imbalance handling (`cls` loss weighting) that caused an early precision/recall collapse in an initial sanity-check run.
- Establish a robust, resumable multi-session training workflow given GPU compute constraints.
- Select the checkpoint to carry forward, consistent with the Milestone 3 architecture selection, and document the experiments that reached completion.

---

## 2. Training Dataset

### 2.1 Final Datasets Used

Both this milestone's tracks — the **primary track** (YOLO11m-seg, COCO-pretrained) and the **comparative benchmark track** (CarDD-pretrained YOLOv8s-seg / YOLO11x-seg) — are instance-segmentation runs, consistent with the task Milestone 3 selected. An early **box-only sanity-check run** (Section 5.3) was also attempted at the start of this milestone using the box-only proxy configuration from the Milestone 3 architecture probe, before committing full compute to segmentation training; it used bounding-box labels and is retained here for completeness (Section 9.1), but is not a production track.

| | Primary / comparative tracks (segmentation) | Box-only sanity-check run |
| --- | --- | --- |
| Label format | YOLO instance segmentation polygon (`class x1 y1 x2 y2 ... xn yn`) | YOLO bounding box (`class x_center y_center w h`) |
| Source of labels | VehiDE VIA polygons, converted to YOLO-seg format | bbox conversion of the same VehiDE VIA polygons |
| Image preprocessing | Letterboxed to 1280×1280 | Letterboxed to 1280×1280 |

### 2.2 Train / Validation / Test Split

Both tracks reuse the identical stratified 70/15/15 split (seed 42):

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

- **Comparative — box-only sanity check:** Ultralytics' default augmentation stack (mosaic, HSV jitter, flip, scale/translate/erasing), `close_mosaic=10` (mosaic disabled for the final 10 of 50 epochs).
- **Comparative benchmark track (YOLOv8s-seg, CarDD), baseline runs:** default Ultralytics augmentation only.
- **Comparative benchmark track (YOLOv8s-seg, CarDD), continuation run:** default augmentation plus deliberately added `copy_paste=0.3`, raised `hsv_v`/`hsv_s`, `degrees=10`, `shear=5`, and increased `scale`, targeted specifically at the thin, low-contrast, boundary-ambiguous classes (`dent`, `scratch`, `crack`) that underperformed in the baseline run.
- **Primary track (YOLO11m-seg, COCO)**: augmentation and preprocessing configuration is documented in Milestone 5, Section 7 alongside the Optuna search.

---

## 3. Model Configuration

### 3.1 Final Architecture(s)

| Model | Track | Base checkpoint | Pretrained source |
| --- | --- | --- | --- |
| **YOLO11m-seg** | **Primary** (Milestone 3 selection: YOLO11 generation, `-seg` task, `m` scale) | `yolo11m-seg.pt` | COCO (Ultralytics) |
| YOLOv8s-seg | Comparative benchmark | `abdullahg7/cardd-yolov8s` (v2.0) | CarDD dataset, via Hugging Face |
| YOLO11x-seg | Comparative benchmark | `harpreetsahota/car-dd-segmentation-yolov11` | CarDD dataset, via Hugging Face |
| YOLO11s-seg | Comparative benchmark (probed only, not trained to completion) | `yolo11s-seg.pt` | COCO (Ultralytics) |
| YOLO11m (box-only) | Box-only sanity-check run, not a production track | `yolo11m.pt` | COCO/Objects365 (Ultralytics) |

The comparative benchmark checkpoints (YOLOv8s-seg, YOLO11x-seg, YOLO11s-seg) differ from the Milestone 3 selection either in backbone generation (YOLOv8 rather than YOLO11) or in scale (`x`/`s` rather than `m`); they are trained and evaluated here to measure how much domain-specific (CarDD) pretraining is worth, not as candidates to replace the Milestone 3 architecture decision.

### 3.2 Pretrained vs. Training from Scratch

All models were fine-tuned from pretrained checkpoints; none were trained from random initialisation. For the **primary track**, `yolo11m-seg.pt`'s detection and mask heads were reinitialised for this project's 6 classes (standard Ultralytics transfer-learning behaviour when `nc` changes). For the box-only sanity-check run, `yolo11m.pt`'s detection head was likewise reinitialised. For the comparative-benchmark segmentation checkpoints, head transfer depended on class-name string matching between the checkpoint's own labels and this project's taxonomy:

- `cardd-yolov8s`: all 6 classes matched, full head transfer.
- `car-dd-segmentation-yolov11` (YOLO11x-seg): only 3 of 6 classes matched (`dent`, `scratch`, `crack`); `broken_lamp`, `shattered_glass`, `flat_tyre` heads were randomly reinitialised, confirmed directly in the training log (`"Remapped 3/6 cls head rows from pretrained weights by class name"`).
- `yolo11s-seg.pt`: COCO-pretrained, no class overlap with this project's taxonomy by design; all 6 heads reinitialised.

### 3.3 Model-Specific Configuration

| | YOLO11m-seg (primary, tuned) | YOLOv8s-seg (comparative, final continuation) |
| --- | --- | --- |
| Input size | 640×640 | 1280×1280 |
| Task head | Instance segmentation | Instance segmentation |
| Parameters | ~22.3M (measured) | ~11.8M (measured) |

The primary track was trained at 640px (Ultralytics' standard segmentation input size, on Google Colab) rather than the 1280px used elsewhere in this project; this resolution difference, alongside the different training environment, is one reason the primary and comparative tracks' numbers are not directly comparable (Milestone 5, Section 2).

---

## 4. Training Environment

### 4.1 Hardware

- Kaggle GPU sessions: **Tesla T4** (16 GB) used for the comparative-benchmark track and the box-only sanity-check run.
- **P100** was attempted for one comparative-benchmark segmentation experiment and found **incompatible** with the installed PyTorch build (`CUDA error: no kernel image is available for execution on the device` — P100's Pascal architecture, compute capability 6.0, has no compiled kernels in the `torch-2.10.0+cu128` build Kaggle provisioned). Not fixable via configuration; resolved by switching back to T4.
- The **primary track** (YOLO11m-seg, COCO-pretrained) was run separately on **Google Colab**, also on a T4-class GPU, using its own dataset copy and its own hyperparameter-search process (Optuna-based; Section 6, detailed further in Milestone 5, Section 7). It is documented in this report as the milestone's primary track and evaluated in full in Milestone 5.

### 4.2 Software

| Component | Version(s) observed |
| --- | --- |
| Ultralytics | 8.4.104 – 8.4.110 (varied slightly across sessions) |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12.13 |

### 4.3 Resource Constraints

- Kaggle's free-tier GPU quota (weekly hours) was a binding constraint throughout this milestone and directly shaped which experiments were pursued to completion versus abandoned.
- Kaggle interactive sessions do not reliably run long enough to complete a multi-hour training job in one sitting; this required building a custom multi-session checkpoint-relay mechanism partway through the milestone, after losing an entire ~25-epoch run to an unplanned session termination.
- `/kaggle/working/` does not persist across sessions, all cross-session checkpoint recovery relies on periodically publishing checkpoints to a separate Kaggle Dataset.

---

## 5. Training Methodology

### 5.1 Training Workflow

1. Mount the segmentation dataset as a Kaggle Dataset input.
2. Load the pretrained checkpoint.
3. Run a short probe (1 epoch on a fixed subsample, later extended to separately time training and validation) to confirm the configuration fits in VRAM and to estimate real wall-clock cost before committing to a full run.
4. Train with a per-epoch callback that periodically backs up `last.pt` and training state to a dedicated Kaggle Dataset, and optionally stops the session early at a configured epoch budget.
5. On a new session, check for an existing backup; if found, resume from it (`resume=True`), otherwise start fresh from the pretrained checkpoint.
6. Evaluate the final checkpoint against the held-out test split.

### 5.2 Batch Size

**4**, used consistently across all completed runs, at 1280px input. This was set based on observed VRAM usage, larger batch sizes were found to `OOM Error`. Batch=4 at imgsz=1280/1024 also OOM'd for the much larger YOLO11x-seg model even after Ultralytics' automatic batch-size reduction to.

### 5.3 Number of Epochs

| Run | Epochs |
| --- | --- |
| **Primary — YOLO11m-seg (COCO), baseline** | 40, `optimizer="auto"` (later found to silently fix `lr0=0.001`, Milestone 5 Section 7.2) |
| **Primary — YOLO11m-seg (COCO), Optuna-tuned** | 40, corrected `optimizer="AdamW"`, `lr0≈0.000105` (full search and results in Milestone 5, Section 7) |
| Comparative — box-only sanity check (YOLO11m), `cls=2.0` | Abandoned after confirming a precision/recall collapse pattern across the completed epochs, not carried forward |
| Comparative — box-only sanity check (YOLO11m), `cls=0.5` + `cos_lr` | Interrupted by unplanned session terminations on more than one occasion, furthest progress was **25 completed epochs**, measured mAP@50 = 0.0248 at that point, well below what the box-only probe in Milestone 3 already suggested was achievable relatively, and not resumed to completion — this run is not part of any production track |
| Comparative benchmark — YOLOv8s-seg (CarDD), baseline | 30 |
| Comparative benchmark — YOLOv8s-seg (CarDD), continuation (augmentation) | 20 additional (50 total) |
| Comparative benchmark — YOLOv8s-seg (CarDD), DFL boundary-precision experiment | 30 (completed) |

The Primary track's baseline → tuned structure differs from the CarDD comparative track's baseline → continuation → refinement pattern in its specific intervention (an Optuna-based hyperparameter search rather than augmentation/DFL-reweighting), since the two were run independently. Full primary-track results are in Milestone 5.

### 5.4 Loss Function

Ultralytics' composite YOLO loss: CIoU box-regression loss, BCE classification loss (weighted by `cls`), DFL (Distribution Focal Loss, weighted by `dfl`) for box/mask boundary refinement, plus a segmentation mask loss (`seg_loss`) for the `-seg` models. All segmentation-track training logs also report a `sem_loss` term that remained at exactly 0 in every run observed.

### 5.5 Learning Strategy

`AdamW` optimiser was used throughout. Learning rate strategy evolved across the milestone - initial runs used a fixed low `lr0` with linear decay, later runs adopted cosine decay (`cos_lr=True`) with a lower final learning rate (`lrf`), and warmup epochs were extended for checkpoints requiring a larger adaptation versus shortened for checkpoints continuing from an already well-adapted state.

---

## 6. Hyperparameter Experiments

| Experiment | `cls` | `dfl` | `lr0` | LR schedule | Result | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| **Primary — YOLO11m-seg (COCO), baseline** | 0.5 | default | 0.001 (unintentionally fixed by `optimizer="auto"`) | linear | Val mask mAP50 ≈ 0.401 at 40 epochs | Completed; superseded by the tuned run below |
| **Primary — YOLO11m-seg (COCO), Optuna-tuned** | 0.5 | default | ≈0.000105 (`AdamW`, found via a 12-trial Optuna search) | linear | Val mask mAP50 = 0.449 at 40 epochs, improving on every reported metric over the baseline with no regression | Completed and selected as the primary-track checkpoint (Section 10); full search and per-class results in Milestone 5, Sections 5-7 |
| Comparative — box-only sanity check, attempt 1 | 2.0 | default (1.5) | 0.001 | linear | Precision collapsed to ~0.85–0.87 while recall collapsed toward 0 (model learned to predict almost nothing) | Rejected |
| Comparative — box-only sanity check, `cls=0.5` attempt | 0.5 | default | 0.001 | cosine (`lrf=0.001`) | mAP@50 climbing across all 25 completed epochs, reaching 0.0248 at epoch 25 (still well below the target) | Not promising and hence interrupted before completion |
| Comparative benchmark — YOLOv8s-seg (CarDD) baseline | 0.5 | default | 0.0005 | cosine | Converged; overall mask mAP50 = 0.36 (val, epoch 30) | Completed |
| Comparative benchmark — YOLOv8s-seg (CarDD) continuation (augmentation) | 0.5 | default | 0.0002 | cosine | Overall test mask mAP50 improved from 0.348 to 0.3534. `dent`/`scratch`/`crack` essentially flat (+0.001 to +0.011). `shattered_glass` slightly regressed (−0.014) | Completed |
| Comparative benchmark — YOLOv8s-seg (CarDD), DFL boundary-precision | 0.3 | 1.7 | 0.0002 | cosine | Overall test mask mAP50 0.3534 → 0.3549 (essentially flat). `dent`/`scratch` small gains (+0.005 to +0.014), `crack` flat (−0.001), `broken_lamp` gained (+0.041), `shattered_glass` regressed (−0.059) | Completed |

**Justification for the primary track's selected configuration**: the Optuna-tuned run improves on every reported validation metric over the baseline with no regression (Milestone 5, Section 6) — this isolates the gain to the hyperparameter correction itself. It is selected as the checkpoint carried forward (Section 10), consistent with the Milestone 3 architecture choice. Within the comparative benchmark track, the DFL boundary-precision configuration is the strongest of the CarDD-pretrained checkpoints, and remains a useful point of comparison even though it is not the production candidate.

Two model-scale experiments were also run as probes, not full training:

| Model | Parameters | Measured/estimated cost | Outcome |
| --- | --- | --- | --- |
| YOLO11x-seg | 62.1M | OOM at imgsz=1280 with `multi_scale=True`, even at batch=1 on a 14.9 GB T4 | Abandoned as the architecture was too large for available hardware at the attempted configuration |
| YOLO11s-seg | 10.08M | Probe indicated ~21 min/epoch, but the full training run was not pursued due to poor preliminary results. | Not completed |

---

## 7. Optimization Methods

- **Optimizer:** AdamW was used throughout the experiments. No alternative optimiser was tested.
- **Learning-rate scheduler:** Cosine annealing (`cos_lr=True`) was adopted after the initial linear-decay detection attempt in an effort to reduce optimization stagnation and improve convergence.
- **Early stopping:** Ultralytics' `patience` parameter used (15–20 depending on run), monitoring validation fitness. Not observed to trigger in any completed run within this milestone — all completed runs reached their configured epoch count rather than stopping early.
- **Gradient clipping:** Not explicitly configured in any run (Ultralytics' default behaviour used, unmodified).
- **Mixed precision training:** `amp=True` used throughout (Ultralytics default); AMP compatibility checks passed in every training log observed.

---

## 8. Regularization Techniques

- **Dropout:** Not used in the detection track. Used at `0.1` in the segmentation continuation and DFL-boundary experiments, following the published recipe associated with the `harpreetsahota` checkpoint.
- **Weight decay:** `0.0005`, consistent across all runs, both tracks.
- **Data augmentation:** Ultralytics defaults throughout; additionally `copy_paste`, extended HSV jitter, and geometric jitter (`degrees`, `shear`, `scale`) in the segmentation continuation run (Section 2.3).
- **Label smoothing:** Not used in any run.
- **Class weighting:** A single global `cls` loss-gain scalar was used as a coarse imbalance-mitigation lever (varied across experiments). Ultralytics' `cls_pw` parameter, which would carry true per-class weights, was left at its default in every run this milestone.
- **Cross-validation:** Not applicable - a single fixed stratified split was used throughout, consistent with the project's leakage-safety design.

---

## 9. Training Progress

### 9.1 Evidence of Convergence

- **Primary — YOLO11m-seg (COCO), baseline → Optuna-tuned (40 epochs each):** training losses (`box_loss`, `seg_loss`, `cls_loss`, `dfl_loss`) declined steadily with no reversals in both runs; validation mask mAP50 climbed with no plateau through the full 40-epoch budget in each case, `patience=15` did not trigger in either run. The tuned run improved on every reported metric over the baseline (Milestone 5, Sections 5-6).
- **Comparative benchmark — YOLOv8s-seg (CarDD) baseline (30 epochs):** `box_loss`, `seg_loss`, `cls_loss`, and `dfl_loss` declined monotonically across all 30 epochs with no reversals, mask mAP50 rose from 0.157 (epoch 1) to 0.36 (epoch 30, val split), the majority of the gain concentrated in the final third of training.
- **Comparative benchmark — YOLOv8s-seg (CarDD) continuation (20 further epochs):** overall mask mAP50 held roughly flat in a narrow band (0.317–0.339) through most of the run, then rose sharply at the `close_mosaic` transition (epoch 16, mosaic disabled), reaching 0.36 by epoch 16 and continuing to a final value of 0.362 (val) by epoch 20.
- **Comparative — box-only sanity check (`cls=0.5` attempt):** losses declined across all 25 completed epochs with no reversals observed; mAP@50 rose from near-zero to 0.0248 by epoch 25, well short of the target, and with each epoch requiring ~30 minutes, the run was discontinued due to poor preliminary results. This run is a box-only sanity check, not part of any production track.

### 9.2 Underfitting Observations

The clearest and most consistent finding across this milestone: **`dent`, `scratch`, and `crack` underperform every other class, and this did not resolve with more epochs, augmentation, or a different `cls`/`dfl` weighting tried so far.** In the segmentation continuation run's final per-class test evaluation, `scratch` the class with the *most* training instances of any class (2,174 in test alone) still scored among the two worst-performing classes (mask mAP50 0.203), while `shattered_glass` the class with the *fewest* instances scored best (0.661). This rules out a simple data-volume explanation and points to intrinsic difficulty (thin, low-contrast, boundary-ambiguous shapes) as the dominant factor, consistent with the published CarDD paper's own finding that even its best benchmarked method underperforms on these same classes.

### 9.3 Overfitting Observations

No clear evidence of overfitting was observed in any completed run as the validation metrics tracked training-loss improvements in the same direction throughout, with no divergence between training and validation performance recorded in the logs reviewed for this milestone.

---

## 10. Model Selection

### 10.1 Checkpoint Selected

Two checkpoint families were evaluated to completion this milestone:

| Checkpoint | Track | Total epochs | Overall evaluation |
| --- | --- | --- | --- |
| **YOLO11m-seg (COCO), Optuna-tuned** | **Primary** | 40 | Val mask mAP50 = 0.449, val mask mAP50-95 = 0.241 (full breakdown in Milestone 5) |
| YOLOv8s-seg (CarDD), DFL boundary-precision | Comparative benchmark | 80 (30 baseline + 20 augmentation + 30 DFL) | Test mask mAP50 = 0.3549 |
| YOLOv8s-seg (CarDD), continuation (augmentation) | Comparative benchmark | 50 (30 baseline + 20) | Test mask mAP50 = 0.3534 |

**The YOLO11m-seg (COCO-pretrained, Optuna-tuned) checkpoint is selected as the Damage Agent checkpoint carried forward**, consistent with the architecture Milestone 3 selected (YOLO11 generation, `-seg` task, `m` scale). This is a **generation/scale consistency decision**, not a claim that this checkpoint's numbers are directly comparable to or better than the CarDD comparative benchmark's: the two tracks were evaluated on different splits (validation vs. held-out test) and different pretraining sources (COCO vs. CarDD, a domain-specific vehicle-damage dataset), so a apples-to-apples ranking between them is not yet available (Milestone 5, Section 2 discusses this gap and recommends a reconciliation experiment).

The CarDD-pretrained comparative checkpoints remain valuable: Section 12.2's finding that domain-specific (CarDD) pretraining outperforms general-purpose COCO pretraining, at matched architecture, is a genuine result and is carried forward as a recommendation for future work (Milestone 5, Section 11) — specifically, applying CarDD-style domain pretraining to the YOLO11m-seg architecture the project has actually selected, rather than switching architectures to chase the CarDD checkpoints' benchmark numbers.

The box-only sanity-check run's most promising configuration was not trained to completion due to poor results and was never a production track; it is not considered for checkpoint selection.

### 10.2 Why It Was Selected

The **YOLO11m-seg (COCO), Optuna-tuned** checkpoint was selected primarily because it is the direct fine-tuning product of the architecture Milestone 3 selected (same generation, same task, same scale), and because — within its own track — it improved on every reported validation metric over its own untuned baseline with no regression, by a larger margin than 20 further epochs of unmodified training achieved (Milestone 5, Section 6). This isolates the improvement to the hyperparameter correction itself rather than to additional training time.

The **YOLOv8s-seg (CarDD) DFL boundary-precision checkpoint** achieved the highest raw test-set mask mAP@50 among all completed experiments this milestone, and is retained as the strongest comparative benchmark result — but it uses a different backbone generation (YOLOv8, not YOLO11) and a smaller scale (`s`, not `m`) than the Milestone 3 selection, so it is not carried forward as the production candidate. Its strong performance, attributable substantially to CarDD domain pretraining (Section 12.2), motivates future work applying the same domain-pretraining strategy to the selected YOLO11m-seg architecture.

### 10.3 Validation Metric Used

**Mask mAP@50** was used as the primary metric for comparing experiments within each track. For the primary track, this was measured on the **validation** split (Milestone 5, Section 3.3 flags the pending test-split run as provisional); for the comparative benchmark track, on the held-out **test** split, improving incrementally from **0.348** to **0.3534** and finally **0.3549** across the CarDD-pretrained runs.

To evaluate the effectiveness of each training intervention, **per-class mask mAP@50** was also examined, with particular emphasis on the underperforming classes `dent`, `scratch`, and `crack`. On the comparative benchmark track, two targeted interventions - enhanced data augmentation followed by DFL loss reweighting - produced only small overall gains and no meaningful, consistent improvement for these three classes. The primary track's Optuna-tuned run improved these same three classes' mask mAP50 as well, without closing the gap to `shattered_glass`/`broken_lamp`/`flat_tyre` (Milestone 5, Section 8) — the same qualitative pattern reproduces across both tracks, independently of pretraining source or architecture generation, strengthening the case that this is an intrinsic property of the damage types rather than an artifact of one track's setup.

Consequently, the selected checkpoint should be regarded as the **best-available, architecture-consistent model within the scope of the completed experiments**, rather than a definitive solution to the remaining per-class performance limitations.

---

## 11. Challenges Encountered

### 11.1 GPU Memory Limitations

Training large segmentation models was constrained by the available GPU memory on Kaggle's Tesla T4 (16 GB).

- **YOLO11x-seg** (62.1M parameters, ~297 GFLOPs) consistently encountered **out-of-memory (OOM)** errors when trained at `imgsz=1280` with `multi_scale=True`.
- Ultralytics automatically reduced the batch size from **4 → 2 → 1**, but OOM errors persisted, indicating that the limitation was the model's memory footprint rather than the batch size.
- For smaller models, memory usage was successfully managed by:
  - Disabling `multi_scale` training.
  - Using an input resolution of **1024×1024** or **1280×1280**, depending on the model.
- Despite these adjustments, **YOLO11x-seg could not be trained successfully** within the available hardware constraints and was therefore excluded from further experimentation.

### 11.2 Session Losses and the Multi-Session Checkpoint Relay

Training was interrupted multiple times due to Kaggle session terminations, including one instance where approximately **25 epochs** of detection training were lost because no recoverable checkpoint was available. Since the `/kaggle/working/` directory is reset whenever a new container is allocated, all locally stored training progress was erased.

To mitigate this issue, a **multi-session checkpoint relay system** was developed. During training, the latest `last.pt` checkpoint was periodically backed up to a dedicated Kaggle Dataset. At the start of each new session, the training script automatically checked for the most recent backup and resumed training from that checkpoint, enabling long-running experiments to continue across multiple Kaggle sessions.

### 11.3 Hardware/Software Compatibility

The Kaggle **P100** GPU was found to be incompatible with the installed PyTorch build, resulting in a `CUDA error: no kernel image is available for execution on the device`. This was a hardware/toolchain compatibility issue rather than a notebook configuration error. The problem was resolved by switching to a **Tesla T4**, which is compatible with the installed software stack.

### 11.4 Dataset/Label Quality

A coordinate-alignment error was identified during the conversion of VIA polygon annotations to the YOLO segmentation format. Bounding-box and polygon coordinates were normalized using the **original image dimensions** but paired with **letterboxed and padded images** without accounting for the padding offsets. As a result, annotations became misaligned with the image content, particularly for non-square images, with a representative test case showing approximately a **10% vertical offset**.

The conversion pipeline was corrected by applying the appropriate letterbox transformation before normalization, ensuring that the generated segmentation labels accurately aligned with the preprocessed images used for training.

### 11.5 Hyperparameter Sensitivity

The classification loss weight (`cls`) was found to have a significant impact on model behaviour. Using `cls=2.0` caused a severe precision–recall collapse, whereas reducing it to `cls=0.5` resulted in stable training and consistent improvements in mAP, highlighting the importance of careful hyperparameter tuning.

---

## 12. Summary and Next Steps

### 12.1 Selected Training Configuration

The checkpoint carried forward as the Damage Agent, consistent with the Milestone 3 architecture selection, is **YOLO11m-seg (COCO-pretrained)**, fine-tuned for 40 epochs with Optuna-selected hyperparameters (`optimizer="AdamW"`, `lr0≈0.000105`, `weight_decay≈0.00029`, `degrees≈5.5`). This run improved on every reported validation metric over its own untuned 40-epoch baseline with no regression (Milestone 5, Sections 6-7).

Among the comparative-benchmark experiments, the **DFL boundary-precision** configuration achieved the highest overall performance within that track. This model, based on **YOLOv8s-seg** pretrained on **CarDD**, was fine-tuned for a total of **80 epochs**, with the final 30 epochs using `cls=0.3` and `dfl=1.7`. It achieved a **test-set mask mAP@50 of 0.3549**, marginally outperforming the continuation checkpoint, which achieved **0.3534**. This is a strong result, and is discussed further in Section 12.2, but it uses a different backbone generation and scale than the Milestone 3 selection, so it is retained as a benchmark rather than the production candidate (Section 10).

### 12.2 Key Observations

- Within the comparative-benchmark track, fine-tuning models pretrained on a **domain-specific dataset (CarDD)** consistently produced better raw scores than the general-purpose **COCO-pretrained YOLO11s-seg** probe. This highlights the value of domain-specific pretraining for vehicle damage assessment, where pretrained features are more closely aligned with the target task — though the comparison is confounded by architecture/scale differences (YOLOv8s vs. YOLO11s) and is not a controlled, single-variable test of pretraining source alone. It motivates future work applying CarDD-style domain pretraining to the project's selected YOLO11m-seg architecture specifically (Section 10.1), rather than a change of architecture.

- Two targeted interventions—enhanced data augmentation followed by **DFL loss reweighting**—produced only marginal improvements in overall performance (mask mAP@50 increased from **0.348 → 0.3534 → 0.3549**). Neither approach yielded a meaningful or consistent improvement for the most challenging classes (`dent`, `scratch`, and `crack`), suggesting that these categories remain intrinsically difficult for the current model architecture.

- Practical engineering challenges had a significant impact on the training process. Kaggle session limits, GPU memory constraints, runtime estimation errors, and checkpoint management required the development of a multi-session training workflow with automated checkpoint recovery. Addressing these infrastructure issues proved as important as model optimization for successfully completing long-running experiments.

### 12.3 Readiness for Milestone 5

Substantial groundwork has been completed in preparation for **Milestone 5**. The **primary track's YOLO11m-seg (COCO-pretrained, Optuna-tuned) checkpoint** — the one carried forward per Section 10 — is ready for detailed error analysis, robustness testing, and the pending test-split evaluation; Milestone 5 covers this in full. The comparative-benchmark track additionally provides three fully trained and evaluated CarDD-pretrained checkpoints representing distinct fine-tuning strategies, together with a validated multi-session training workflow capable of recovering from Kaggle session interruptions and an external published benchmark (CarDD) against which results can be assessed, rather than relying solely on comparisons between internal experiments.

According to the original **CarDD** paper (Wang, Li, & Wu, *IEEE Transactions on Intelligent Transportation Systems*, 2023), per-class Average Precision (AP) values are reported only for the `dent`, `scratch`, and `crack` categories. A comparison with the best-performing checkpoints from this project's **comparative-benchmark track** (CarDD-pretrained YOLOv8s-seg) is shown below; the primary track (YOLO11m-seg, COCO-pretrained) is not included here since it is evaluated separately in Milestone 5 and is not directly comparable given the different pretraining source and split (Section 10.1).

| Class | CarDD Baseline | CarDD DCN+ (SOTA) | This Project Baseline (30 epochs) | This Project Continuation (50 epochs) | This Project DFL Boundary (80 epochs) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `dent` | 32.0% | **40.5%** | 22.6% | 22.9% | **24.3%** |
| `scratch` | 24.0% | **34.3%** | 20.2% | 20.3% | **20.9%** |
| `crack` | 9.8% | 16.6% | 19.8% | **20.9%** | **20.9%** |

**Observations**

- The DFL boundary-precision experiment achieved the best results within this project for `dent` and `scratch`; however, both classes remain below the AP reported by the CarDD DCN+ model.
- For `crack`, all three project checkpoints outperform both the CarDD baseline and the published DCN+ result, indicating that this class is well learned by the current training pipeline.
- Overall, the comparison confirms that `dent` and `scratch` remain the primary performance bottlenecks, while `crack` has already reached or exceeded the performance reported in the original CarDD benchmark.

**Prepared for Milestone 5**

- The **YOLO11m-seg (COCO-pretrained, Optuna-tuned) primary-track checkpoint** is ready for detailed error analysis and qualitative evaluation — this is the checkpoint Milestone 5 evaluates in full, consistent with the Milestone 3 architecture selection.
- The comparative-benchmark track's two best-performing CarDD-pretrained checkpoints remain available as a reference point, and the persistent gap between the CarDD comparative track and the published CarDD DCN+ benchmark for `dent` and `scratch` provides a clear direction for future work — most usefully, applying CarDD-style domain pretraining to the selected YOLO11m-seg architecture rather than adopting a different backbone.
- With the training infrastructure, evaluation pipeline, and external benchmark now established, Milestone 5 can focus on targeted model improvements rather than further experimentation with training infrastructure.

---

## 13. Policy Agent (RAG): Selection and Tuning

Sections 1-12 cover the Damage Agent (YOLO). This section covers the equivalent work on the
Policy Agent's retrieval stack.

**The RAG stack contains no trained weights.** MiniLM is used frozen, the Report Agent's LLMs
are accessed by prompting only, and YOLO is the sole trained model in the project (Milestone 3,
Section 9). "Fine-tuning" for this agent therefore means **model selection and parameter
tuning**, which is what follows.

### 13.1 Model and Infrastructure Selection

Each choice was decided by a head-to-head benchmark on the actual 185-chunk corpus, not by
default.

| Decision | Alternatives compared | Result | Chosen because |
| :--- | :--- | :--- | :--- |
| Embedding model | MiniLM-L6-v2 (22.7M) vs. BGE-small-en-v1.5 (33.4M) | P@3 **1.00** vs. **0.94** on the 6-query smoke test (BGE lagged on `crack`, 0.67) | Higher score at two-thirds the parameters; both embed the corpus in under a second |
| Larger encoders | MPNet-base, E5-base (~110M) | Not run | 3-5x cost for marginal published gain; the demo target is a CPU-only Hugging Face Space |
| Vector store | ChromaDB vs. FAISS `IndexFlatIP` | Identical top-1 on 6/6 queries; FAISS <0.01ms vs. Chroma ~0.5ms (50-60x) | Latency gap not operationally meaningful at 185 chunks; Chroma provides metadata filtering and persistence out of the box |
| Report LLM | `llama-3.3-70b-versatile` vs. `openai/gpt-oss-20b` | Both reached faithfulness composite **1.00** | Llama selected: free tier, ~0.7-0.9s per report, OpenAI-compatible API |

An important qualification: the embedding model was **not** what fixed retrieval. On the
pre-fix 179-chunk corpus the two models scored 0.94 and 0.89; the PDF-extraction fix and
heading-breadcrumb change moved MiniLM to 1.00. **Corpus quality dominated model choice.**

### 13.2 Retrieval Parameter Tuning

Milestone 2 swept one retrieval parameter (the dense:sparse weight ratio) and settled the rest
by inspection or by leaving library defaults in place. That gap has since been closed: **84
configurations across 6 sweeps**, evaluated on the same 50-incident set. The harness was
validated first by reproducing all four published weight-ratio data points exactly.

**A. Dense:sparse weight ratio** - production **3:1**

| Ratio | P@3 | MRR@5 | zero-hit |
| :--- | ---: | ---: | ---: |
| 100:0 (dense-only) | 0.8933 | 0.9800 | 0 |
| 50:50 | 0.9067 | 0.9717 | 1 |
| 66:33 (2:1) | 0.9133 | 0.9767 | 0 |
| **75:25 (3:1)** | **0.9133** | **0.9767** | **0** |
| 80:20 (4:1) | 0.9067 | 0.9767 | 0 |
| 0:100 (sparse-only) | 0.7400 | 0.8950 | 1 |

**B. RRF_K** - 13 values from k=1 to k=1000 give a P@3 range of only **0.8933-0.9133**, with
zero zero-hit incidents throughout. Production k=60 sits on the top plateau it shares with
k=5, 45, 80 and 100. The parameter is effectively inert at this corpus size.

**C. CANDIDATE_POOL** - 11 values from 3 to 100 give a range of **0.8933-0.9200**. Only
pool=3 is clearly too shallow. Production pool=20 scores 0.9133.

**D. RRF_K x CANDIDATE_POOL** - the full 5x5 grid spans **0.8933-0.9200**, with no ridge and
no interaction effect, so the two one-dimensional results above hold jointly.

**E. MIN_CLAUSE_SCORE = 0.01 - inactive.** The lowest score any candidate surviving fusion can
carry is `min(dense_w, sparse_w) / (RRF_K + CANDIDATE_POOL)` = 1.0 / 80 = **0.0125**, already
above the configured floor, so no clause can ever be filtered by it. Empirically the margin is
wider still: the lowest fused score observed was 0.0400, and the lowest score on a clause
actually returned to the LLM was 0.0435. Across all 12 clause queries the floor removes **0 of
55** returned clauses. It would have to rise to 0.05 to remove anything.

**F. Chunk size** - production **300**

| chunk_size | chunks | random P@3 | P@3 | lift x | mixed coverage+exclusion |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 150 | 364 | 0.125 | 0.9067 | **7.24** | 9.9% |
| **300** | **185** | **0.163** | **0.9133** | **5.59** | **15.1%** |
| 500 | 131 | 0.185 | 0.9467 | **5.13** | 21.4% |
| 1000 | 93 | 0.203 | 0.9333 | **4.61** | 26.9% |

Raw P@3 rises with chunk size, but that is an artifact rather than an improvement: longer
chunks match more keyword families, so more of them count as relevant and the random baseline
rises in step. Measured as lift over random, larger chunks are **worse**. Separately, chunks
containing both a coverage grant and its qualifying exclusion nearly double (15.1% to 26.9%);
because `clause_type` is single-valued, one half of such a chunk becomes unreachable from its
bucket - the exact failure the two-query coverage/exclusion split exists to prevent. 300 is
retained on this evidence.

**Chunk overlap** - values 0 through 120 yield 184-221 chunks and no meaningful score change;
40 retained. **DEDUP_THRESHOLD = 0.90 - inactive**: every threshold from 0.80 to 1.00 yields
an identical 185 chunks, and across all 8 chunk sizes deduplication removes **0 chunks in
total**. No chunk pair in this corpus reaches even 0.80 word-trigram Jaccard similarity.

Sweeps are reproducible via `scripts/sweep_rag_params.py --with-chunking`.

### 13.3 Architecture Change: The User Uploads Their Own Policy

**The fixed 5-policy catalog described in Milestone 3 (Sections 3.5, 5.3, 9) has been replaced
by per-user policy upload.**

| | Before (catalog) | After (per-user) |
| :--- | :--- | :--- |
| Policy source | Fixed 5-policy catalog | User uploads their own PDF |
| Index | One shared `policy_clauses` collection | One collection per user, `user_{user_id}` |
| Scoping | `doc_id` metadata filter | Structural - a separate collection per user |
| TF-IDF vocabulary | Fit across all 5 policies | Fit on that user's chunks only |
| Policy identification | Inferred from the damage profile | Not required - only one candidate exists |

**Why it changed.** An exhaustive 315-case census (63 damage-class subsets x 5 documents)
established that the damage profile alone cannot identify which policy applies:

| Metric | Value |
| :--- | ---: |
| top-1 accuracy | **0.20** |
| top-2 accuracy | 0.40 |
| MRR | 0.457 |
| confusions | 252 |

At 0.20 top-1 accuracy the selector was wrong four times in five. `scripts/policy_catalog.py`
and `scripts/policy_selector.py` were deleted. Per-user upload dissolves the problem rather
than solving it, and eliminates cross-policy clause leakage by construction.

**Caveat.** The per-user path has **no retrieval metrics of its own**. Every figure in Section
13.2 comes from the shared 5-policy corpus, which was never a simulation of the per-user flow.
Under per-user indexing the TF-IDF vocabulary is fit on a single document, a materially
different sparse signal from the one that was tuned. Measuring the per-user path is the
highest-value outstanding work on this agent.

A first pass at measuring the per-user path (10 claims, run through the real per-user pipeline
instead of the shared corpus) found and fixed a real retrieval bug along the way — some damage
types were getting no useful policy clause at all, which a closer evaluation of report quality
caught. Details and results are in Milestone 5, Section 13.4. This is still a small, one-time
check, not the fuller per-user evaluation this caveat is really asking for.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | 30-07-2026 | S.K. |
|Pranab Kumar Manna | | |
| Venkata Siva Kamal Guddanti |  |  |
| Anuj Gautam | 30-07-2026 | Anuj |
| Harsh Pal | 31-07-2026| harshpal |  | |

---
