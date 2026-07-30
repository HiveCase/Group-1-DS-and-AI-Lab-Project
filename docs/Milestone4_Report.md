
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

### 1.1 Recap

Milestone 3 selected **YOLO11m** (plain bounding-box detection, not the `-seg` variant) as the primary architecture for the Damage Agent, on a tie-break against YOLOv8m after a 15-epoch probe found the two statistically indistinguishable (mAP@50 delta 0.0066, below the 0.02 tie-break threshold). The tie-break favoured YOLO11m's newer C3k2/C2PSA backbone blocks, reported to aid small-object detection — directly relevant given Milestone 2's EDA found a minimum normalised bounding-box area of 0.00002 in the training data.

### 1.2 Objectives of the Training Phase

- Fine-tune the Milestone 3-selected YOLO11m detector on the full VehiDE training set and reach a stable, converged checkpoint.
- Diagnose and correct the class-imbalance handling (`cls` loss weighting) that caused an early precision/recall collapse.
- Evaluate whether a pretrained segmentation checkpoint, fine-tuned on the same VehiDE data, offers a viable supplementary or alternative path for the classes the detection model struggled with.
- Establish a robust, resumable multi-session training workflow given GPU compute constraints.
- Select the best available checkpoint(s) and document the experiments which reached the completion.

---

## 2. Training Dataset

### 2.1 Final Datasets Used

Two label formats were used across the two experiment tracks, both derived from the same underlying VehiDE image dataset with stratified split, so results remain comparable across tracks:

| | Detection track | Segmentation track |
| --- | --- | --- |
| Label format | YOLO bounding box (`class x_center y_center w h`) | YOLO instance segmentation polygon (`class x1 y1 x2 y2 ... xn yn`) |
| Source of labels | bbox conversion of VehiDE VIA polygons | The same VehiDE VIA polygons, converted to YOLO-seg format |
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

- **Detection track:** Ultralytics' default augmentation stack (mosaic, HSV jitter, flip, scale/translate/erasing), `close_mosaic=10` (mosaic disabled for the final 10 of 50 epochs).
- **Segmentation track, baseline runs:** default Ultralytics augmentation only.
- **Segmentation track, continuation run:** default augmentation plus deliberately added `copy_paste=0.3`, raised `hsv_v`/`hsv_s`, `degrees=10`, `shear=5`, and increased `scale`, targeted specifically at the thin, low-contrast, boundary-ambiguous classes (`dent`, `scratch`, `crack`) that underperformed in the baseline run.

---

## 3. Model Configuration

### 3.1 Final Architecture(s)

| Model | Track | Base checkpoint | Pretrained source |
| --- | --- | --- | --- |
| YOLO11m | Detection | `yolo11m.pt` | COCO/Objects365 (Ultralytics) |
| YOLOv8s-seg | Segmentation | `abdullahg7/cardd-yolov8s` (v2.0) | CarDD dataset, via Hugging Face |
| YOLO11x-seg | Segmentation | `harpreetsahota/car-dd-segmentation-yolov11` | CarDD dataset, via Hugging Face |
| YOLO11s-seg | Segmentation (probed only, not trained to completion) | `yolo11s-seg.pt` | COCO (Ultralytics) |

### 3.2 Pretrained vs. Training from Scratch

All models were fine-tuned from pretrained checkpoints; none were trained from random initialisation. For the detection track, `yolo11m.pt`'s detection head was reinitialised for this project's 6 classes. For the segmentation track, head transfer depended on class-name string matching between the checkpoint's own labels and this project's taxonomy:

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
| Detection (YOLO11m), `cls=2.0` | Abandoned after confirming a precision/recall collapse pattern across the completed epochs, not carried forward |
| Detection (YOLO11m), `cls=0.5` + `cos_lr` | Interrupted by unplanned session terminations on more than one occasion, furthest progress was **25 completed epochs**, measured mAP@50 = 0.0248 at that point, which was far below the target value and hence was not resumed to completion |
| Segmentation (YOLOv8s-seg), baseline | 30 |
| Segmentation (YOLOv8s-seg), continuation (augmentation) | 20 additional (50 total) |
| Segmentation (YOLOv8s-seg), DFL boundary-precision experiment | 30 (completed) |

### 5.4 Loss Function

Ultralytics' composite YOLO loss: CIoU box-regression loss, BCE classification loss (weighted by `cls`), DFL (Distribution Focal Loss, weighted by `dfl`) for box/mask boundary refinement, plus a segmentation mask loss (`seg_loss`) for the `-seg` models. All segmentation-track training logs also report a `sem_loss` term that remained at exactly 0 in every run observed.

### 5.5 Learning Strategy

`AdamW` optimiser was used throughout. Learning rate strategy evolved across the milestone - initial runs used a fixed low `lr0` with linear decay, later runs adopted cosine decay (`cos_lr=True`) with a lower final learning rate (`lrf`), and warmup epochs were extended for checkpoints requiring a larger adaptation versus shortened for checkpoints continuing from an already well-adapted state.

---

## 6. Hyperparameter Experiments

| Experiment | `cls` | `dfl` | `lr0` | LR schedule | Result | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| Detection, attempt 1 | 2.0 | default (1.5) | 0.001 | linear | Precision collapsed to ~0.85–0.87 while recall collapsed toward 0 (model learned to predict almost nothing) | Rejected |
| Detection, `cls=0.5` attempt | 0.5 | default | 0.001 | cosine (`lrf=0.001`) | mAP@50 climbing across all 25 completed epochs, reaching 0.0248 at epoch 25 (still well below the target) | Not promising and hence interrupted before completion |
| Segmentation baseline (YOLOv8s-seg) | 0.5 | default | 0.0005 | cosine | Converged; overall mask mAP50 = 0.36 (val, epoch 30) | Completed |
| Segmentation continuation (augmentation) | 0.5 | default | 0.0002 | cosine | Overall test mask mAP50 improved from 0.348 to 0.3534. `dent`/`scratch`/`crack` essentially flat (+0.001 to +0.011). `shattered_glass` slightly regressed (−0.014) | Completed |
| Segmentation, DFL boundary-precision | 0.3 | 1.7 | 0.0002 | cosine | Overall test mask mAP50 0.3534 → 0.3549 (essentially flat). `dent`/`scratch` small gains (+0.005 to +0.014), `crack` flat (−0.001), `broken_lamp` gained (+0.041), `shattered_glass` regressed (−0.059) | Completed |

**Justification for the segmentation track's final selected configuration**: the continuation run is the only segmentation checkpoint with a complete, held-out test-set evaluation at the time of writing. It is selected as the current best available checkpoint on that basis.

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

- **Segmentation baseline (30 epochs):** `box_loss`, `seg_loss`, `cls_loss`, and `dfl_loss` declined monotonically across all 30 epochs with no reversals, mask mAP50 rose from 0.157 (epoch 1) to 0.36 (epoch 30, val split), the majority of the gain concentrated in the final third of training.
- **Segmentation continuation (20 further epochs):** overall mask mAP50 held roughly flat in a narrow band (0.317–0.339) through most of the run, then rose sharply at the `close_mosaic` transition (epoch 16, mosaic disabled), reaching 0.36 by epoch 16 and continuing to a final value of 0.362 (val) by epoch 20.
- **Detection (`cls=0.5` attempt):** losses declined across all 25 completed epochs with no reversals observed; mAP@50 rose from near-zero to 0.0248 by epoch 25, well short of the target, and with each epoch requiring ~30 minutes, the run was discontinued due to poor preliminary results.

### 9.2 Underfitting Observations

The clearest and most consistent finding across this milestone: **`dent`, `scratch`, and `crack` underperform every other class, and this did not resolve with more epochs, augmentation, or a different `cls`/`dfl` weighting tried so far.** In the segmentation continuation run's final per-class test evaluation, `scratch` the class with the *most* training instances of any class (2,174 in test alone) still scored among the two worst-performing classes (mask mAP50 0.203), while `shattered_glass` the class with the *fewest* instances scored best (0.661). This rules out a simple data-volume explanation and points to intrinsic difficulty (thin, low-contrast, boundary-ambiguous shapes) as the dominant factor, consistent with the published CarDD paper's own finding that even its best benchmarked method underperforms on these same classes.

### 9.3 Overfitting Observations

No clear evidence of overfitting was observed in any completed run as the validation metrics tracked training-loss improvements in the same direction throughout, with no divergence between training and validation performance recorded in the logs reviewed for this milestone.

---

## 10. Model Selection

### 10.1 Checkpoint Selected

Two segmentation checkpoints now have complete, held-out test-set evaluations and are essentially tied on overall performance:

| Checkpoint | Total epochs | Overall test mask mAP50 |
| --- | --- | --- |
| Continuation (augmentation) | 50 (30 baseline + 20) | 0.3534 |
| DFL boundary-precision | 80 (30 baseline + 20 augmentation + 30 DFL) | 0.3549 |

The **DFL boundary-precision checkpoint** is selected as the current best available result, on the basis of the marginally higher overall score and its additional gains on `dent`, `scratch`, and `broken_lamp`. This selection is a close call, not a clear win: the continuation checkpoint remains preferable specifically on `shattered_glass` (0.6607 vs. 0.6021). Consequently, this selection should be regarded as provisional and may be revised following further experimentation and evaluation.

### 10.2 Why It Was Selected

The **DFL boundary-precision checkpoint** was selected because it achieved the highest overall test-set mask mAP@50 among all completed experiments while maintaining competitive performance across most damage classes. It also incorporates an additional optimization stage through DFL loss reweighting, providing modest improvements on `dent`, `scratch`, and `broken_lamp` compared with the previous checkpoint.

Although the performance gain over the continuation checkpoint is marginal—and the continuation checkpoint performs better on `shattered_glass` (0.6607 vs. 0.6021)—the DFL checkpoint represents the strongest overall result obtained in this milestone. Accordingly, it is selected as the current best checkpoint, with the understanding that this decision remains provisional and may be revised based on future experiments.

The detection track's most promising configuration was not trained to completion due to poor results hence, it was not be considered for final checkpoint selection.


### 10.3 Validation Metric Used

Overall **mask mAP@50** on the held-out test split was used as the primary metric for comparing the completed segmentation experiments, improving incrementally from **0.348** to **0.3534** and finally **0.3549**. This metric served as the basis for selecting the best-performing checkpoint.

To evaluate the effectiveness of each training intervention, **per-class mask mAP@50** was also examined, with particular emphasis on the underperforming classes `dent`, `scratch`, and `crack`. Two targeted interventions - enhanced data augmentation followed by DFL loss reweighting were introduced to improve performance on these classes. While both interventions produced small gains in overall mAP@50, neither resulted in a meaningful or consistent improvement for the targeted classes.

Consequently, the selected checkpoint should be regarded as the **best-performing model within the scope of the completed experiments**, rather than a definitive solution to the remaining performance limitations.

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
