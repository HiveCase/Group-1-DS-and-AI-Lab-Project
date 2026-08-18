<div align="center">


<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">


<h1 style="font-size:26em;">Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 5: Model Evaluation</h2>

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
- [2. Note on the Track Evaluated in This Milestone](#2-note-on-the-track-evaluated-in-this-milestone)
- [3. Evaluation Methodology](#3-evaluation-methodology)
- [4. Performance Metrics](#4-performance-metrics)
- [5. Experimental Results](#5-experimental-results)
- [6. Baseline vs. Tuned Model Comparison](#6-baseline-vs-tuned-model-comparison)
- [7. Hyperparameter Tuning Summary](#7-hyperparameter-tuning-summary)
- [8. Error Analysis](#8-error-analysis)
- [9. Model Robustness](#9-model-robustness)
- [10. Limitations](#10-limitations)
- [11. Possible Improvements](#11-possible-improvements)
- [12. Summary and Next Steps](#12-summary-and-next-steps)
- [13. Policy Agent (RAG): Evaluation and Tuning Outcome](#13-policy-agent-rag-evaluation-and-tuning-outcome)

---

## 1. Introduction

### 1.1 Recap

Milestone 3 selected **YOLO11m-seg** (YOLO11 generation, instance-segmentation task, `m`/medium scale) as the Damage Agent's architecture. Milestone 4 fine-tuned this checkpoint (COCO-pretrained) on Google Colab as the **primary track** — a baseline run and an Optuna-tuned run — and, in parallel, ran a **comparative benchmark track** on Kaggle using CarDD-pretrained checkpoints of a different generation/scale (YOLOv8s-seg, YOLO11x-seg) to gauge how much domain-specific pretraining is worth. This milestone evaluates the **primary track's** Optuna-tuned YOLO11m-seg checkpoint, consistent with the Milestone 3 architecture selection and the Milestone 4 checkpoint-selection decision (Milestone 4, Section 10).

### 1.2 Objectives of Milestone 5

Per the milestone requirements, this report:

- Evaluates the trained model(s) using appropriate, task-relevant metrics.
- Provides error analysis - what the model gets wrong, and why.
- Discusses limitations and possible improvements.

---

## 2. Note on the Track Evaluated in This Milestone

This report evaluates the **primary track** identified in Milestone 4, Section 10: the **YOLO11m-seg (COCO-pretrained)** checkpoint, fine-tuned on Google Colab and hyperparameter-tuned via an Optuna search (Section 7 below). This is the checkpoint carried forward as the Damage Agent, consistent with the architecture Milestone 3 selected (YOLO11 generation, instance-segmentation task, `m`/medium scale).

Milestone 4 also ran a **comparative benchmark track** on Kaggle using CarDD-pretrained checkpoints of a different generation/scale (YOLOv8s-seg, YOLO11x-seg), to gauge how much domain-specific pretraining is worth. That track's best result (YOLOv8s-seg, DFL boundary-precision variant, test mask mAP@50 = 0.3549) is a genuinely strong number and is **not evaluated further in this report** — it remains a useful reference point for future work (Milestone 4, Section 10.2; Section 11 below), but it is not the architecture this project selected, so it is out of scope for this milestone's detailed error analysis.

**Why this matters for reading the rest of this report:**

- Results in this report are **not directly comparable** to Milestone 4's CarDD comparative-benchmark numbers - the two tracks start from different pretrained weights (COCO vs. CarDD domain-specific pretraining) and different backbone generations/scales, which Milestone 4 itself identified as a meaningful factor ("fine-tuning models pretrained on a domain-specific dataset consistently produced better raw scores than the general-purpose COCO-pretrained probe").
- The dataset split used in this track (9,545 / 2,047 / 2,047 train/val/test) differs slightly from the comparative-benchmark track's split (9,558 / 2,048 / 2,049) due to being generated independently from the same source dataset.
- The comparative-benchmark checkpoints are **not re-evaluated in this report**; a same-split, apples-to-apples reconciliation between the two tracks remains valuable future work (Section 11) and would clarify whether domain-specific pretraining or the Optuna hyperparameter correction is the larger lever.

---

## 3. Evaluation Methodology

### 3.1 Models Evaluated

| Model | Description |
| --- | --- |
| **Baseline** | YOLO11m-seg, COCO-pretrained, fine-tuned 40 epochs. `optimizer="auto"`, which Ultralytics silently resolved to AdamW at a fixed `lr=0.001` (see Section 7 for how this was discovered). |
| **Tuned (proposed)** | Same architecture, 40 epochs, hyperparameters selected via an Optuna search: `optimizer="AdamW"`, `lr0≈0.000105`, `weight_decay≈0.00029`, `degrees≈5.5`. |

### 3.2 Dataset and Split

VehiDE segmentation dataset (Kaggle, `m4rcuseryx/vehide-segmentation-dataset`, version 1), 6 damage classes (dent, scratch, crack, broken_lamp, shattered_glass, flat_tyre) plus unlabelled background images (~7% of images, no damage annotated).

| Split | Images |
| --- | --- |
| Train | 9,545 |
| Validation | 2,047 |
| Test | 2,047 |

### 3.3 Evaluation Protocol and an Important Caveat

All results reported in Sections 5-9 below are computed on the **validation split**, using the same `.val()` calls logged during and immediately after each training run.

**A held-out test-split evaluation has been prepared but not yet executed at the time of writing this report.** An evaluation notebook exists with a dedicated cell that runs both checkpoints against the untouched test split (never used for training, checkpoint selection, or the Optuna search), which will produce a stricter, unbiased final number. Until that cell is run, the validation-split numbers below should be read as **provisional** - consistent with development-time performance, but with some risk of a small optimistic bias since checkpoint selection (`best.pt`) was itself chosen based on validation performance. This caveat is carried through Sections 5, 6, 8, and 9 wherever it applies, and is repeated in Section 10 (Limitations).

### 3.4 Ground Truth

Labels are the pre-existing YOLO-format polygon (instance segmentation) annotations shipped with the VehiDE dataset. No manual re-annotation was performed.

### 3.5 Success Criteria

No fixed pass/fail metric threshold was assigned for this track. Results are reported and discussed comparatively - baseline vs. tuned - rather than against an absolute target.

---

## 4. Performance Metrics

This is an **instance segmentation** task - each detected damage instance is assigned a class, a bounding box, and a pixel-level mask - so metrics are reported for both **Box** and **Mask** predictions, following COCO-style conventions.

| Metric | Definition | Why it is used here |
| --- | --- | --- |
| IoU (Intersection over Union) | Overlap area between predicted and ground-truth region, divided by their union area. | Base unit underlying all metrics below; measures localisation quality. |
| Precision | Fraction of predicted damage instances that were correct. | False-positive rate; relevant since spurious damage flags cause unnecessary claim disputes. |
| Recall | Fraction of actual damage instances the model successfully detected. | Missed-damage rate; a missed dent/crack means an under-assessed claim. |
| mAP50 | Mean Average Precision at a 50% IoU threshold, averaged across all 6 classes. | Standard, relatively forgiving detection benchmark metric. |
| mAP50-95 | mAP averaged across IoU thresholds from 50% to 95%. | Stricter metric rewarding precise localisation, not just rough overlap. |

Mask metrics are inherently harder to score well on than Box metrics (a pixel-precise boundary is a stricter target than a rectangle), so Box and Mask numbers are not expected to be equal, and a gap between them is not itself a defect.

---

## 5. Experimental Results

### 5.1 Overall Performance (validation split)

| Run | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (40 ep) | 0.524 | 0.444 | 0.438 | 0.269 | 0.512 | 0.408 | 0.401 | 0.209 |
| **Tuned (40 ep)** | **0.582** | **0.483** | **0.485** | **0.300** | **0.576** | **0.452** | **0.449** | **0.241** |

### 5.2 Per-Class Performance (validation split)

| Class | Train instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dent | 3,888 | 0.309 | 0.140 | 0.279 | 0.117 |
| scratch | 10,070 | 0.349 | 0.193 | 0.297 | 0.114 |
| crack | 3,763 | 0.322 | 0.204 | 0.319 | 0.145 |
| broken_lamp | 1,920 | 0.586 | 0.331 | 0.476 | 0.229 |
| shattered_glass | 1,513 | 0.843 | 0.647 | 0.816 | 0.578 |
| flat_tyre | 1,631 | 0.501 | 0.287 | 0.508 | 0.261 |

These are the Tuned checkpoint's own per-class figures (`best.pt`, validation split), taken directly from the same evaluation run reported as the overall "Tuned" row in Section 5.1. The qualitative ranking of classes (shattered_glass strongest, dent/scratch weakest) holds consistently across the Baseline and Tuned checkpoints (Section 8.1) - the Tuned model improves every class's mask mAP50 over the Baseline figures without changing that ranking.

### 5.3 Per-Class Performance (test split)

| Class | Test instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dent | 825 | 0.290 | 0.140 | 0.290 | 0.121 |
| scratch | 2,174 | 0.344 | 0.200 | 0.288 | 0.114 |
| crack | 765 | 0.340 | 0.211 | 0.328 | 0.135 |
| broken_lamp | 392 | 0.640 | 0.357 | 0.476 | 0.223 |
| shattered_glass | 325 | 0.819 | 0.622 | 0.775 | 0.553 |
| flat_tyre | 365 | 0.550 | 0.341 | 0.529 | 0.302 |

![Classwise distribution](https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/graph.png)

### 5.4 Training Curves

Across both runs, training loss (`box_loss`, `seg_loss`, `cls_loss`) declined steadily with no reversals, and validation mAP50/mAP50-95 climbed with no plateau through the initial 40 epochs. `patience=15` (early stopping) did not trigger in either run - both ran their full configured epoch budget. Validation loss tracked training loss in the same direction throughout, with no divergence observed - i.e. no evidence of overfitting in the ranges trained.

---

## 6. Baseline vs. Tuned Model Comparison

| Metric | Baseline | Tuned | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Box mAP50 | 0.443 | 0.485 | +0.042 | +9.5% |
| Box mAP50-95 | 0.269 | 0.300 | +0.031 | +11.5% |
| Mask mAP50 | 0.400 | 0.449 | +0.049 | +12.3% |
| Mask mAP50-95 | 0.209 | 0.241 | +0.032 | +15.3% |

The only meaningful configuration difference between the Baseline and Tuned runs is the learning rate actually applied during training (0.001 vs. ~0.000105) and a small amount of rotation augmentation (0° vs. 5.5°) - epochs, batch size, dataset, and seed are held constant. The Tuned run improves on **every** reported metric with no regression. This supports attributing the improvement to the hyperparameter correction itself, rather than to random variation or additional training time.

---

## 7. Hyperparameter Tuning Summary

### 7.1 Search Setup

Hyperparameters explored: `lr0` (learning rate, 5×10⁻⁵ to 1×10⁻², log scale), `weight_decay` (0 to 1×10⁻³), `degrees` (rotation augmentation, 0° to 15°), searched jointly using **Optuna** with a TPE sampler (`n_startup_trials=8`, 12 trials total, each trained 5 epochs from the pretrained checkpoint as a fast comparative proxy). Objective: maximise validation mask mAP50.

### 7.2 A Methodological Issue Found and Corrected

An earlier search iteration used `optimizer="auto"`. Under this setting, Ultralytics silently **ignores** any explicitly-passed `lr0` and substitutes its own fixed value - confirmed directly in the training logs (`'optimizer=auto' found, ignoring 'lr0=...'`). As a result, that first search varied `lr0` in name only; every trial actually trained at the same fixed learning rate (0.001). This was identified by inspecting the optimizer initialisation line in each trial's log, and corrected by explicitly setting `optimizer="AdamW"` for all subsequent trials. This is disclosed here in the interest of methodological transparency, consistent with this project's practice (established in Milestone 4) of documenting configuration issues rather than omitting them.

### 7.3 Corrected Search Results

| Trial | lr0 | weight_decay | degrees | Mask mAP50 (5-epoch proxy) |
| ---: | ---: | ---: | ---: | ---: |
| 7 (best) | 0.0001047 | 0.000292 | 5.50 | 0.3511 |
| 2 | 0.0000680 | 0.000866 | 9.02 | 0.3504 |
| 9 | 0.0000550 | 0.000577 | 5.16 | 0.3500 |
| 5 | 0.0001321 | 0.000304 | 7.87 | 0.3477 |
| 11 | 0.0000516 | 0.000978 | 6.60 | 0.3472 |
| 10 | 0.0003391 | 0.000624 | 14.20 | 0.3053 |
| 0 | 0.0003637 | 0.000951 | 10.98 | 0.3063 |
| 6 | 0.0004930 | 0.000291 | 9.18 | 0.2720 |
| 1 | 0.0011926 | 0.000156 | 2.34 | 0.2125 |
| 3 | 0.0021294 | 0.0000206 | 14.55 | 0.1576 |
| 4 | 0.0041157 | 0.000212 | 2.73 | 0.1165 |
| 8 | 0.0077447 | 0.000583 | 0.11 | 0.0930 |

A clear, near-monotonic relationship emerged between `lr0` and validation mAP once the search was corrected: performance is best in the `lr0 ≈ 5×10⁻⁵ - 1.3×10⁻⁴` band and degrades steadily as `lr0` increases, collapsing to roughly a quarter of peak performance above `lr0 ≈ 4×10⁻³`. `weight_decay` and `degrees` showed no comparably clear trend across this search - `lr0` was the dominant factor. This is consistent with the earlier default (`lr=0.001`, from `optimizer="auto"`) sitting on the declining side of this curve, explaining why the Baseline run underperformed the Tuned run.

### 7.4 Final Selection

![Hyperparameter](https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/hpt.png)


`lr0=0.0001047`, `weight_decay=0.000292`, `degrees=5.5` - the best 5-epoch trial - trained for a full 40 epochs, producing the Tuned model reported in Sections 5-6. The 5-epoch proxy result (0.351) generalised sensibly to full training (0.449 at 40 epochs), indicating the short-trial search was not simply overfitting to the 5-epoch budget.

---

## 8. Error Analysis

### 8.1 Class-Level Error Pattern

The clearest and most consistent finding across every checkpoint trained in this track: **`shattered_glass` and `broken_lamp` substantially outperform `dent`, `scratch`, and `crack`**, and this ranking persisted unchanged across the Baseline and Tuned checkpoints (Section 5.2).

This directly contradicts a naive expectation from the training-set class distribution: `scratch` has the *most* training instances of any class (10,070) yet is among the *worst*-performing, while `shattered_glass` has the *fewest* (1,513) yet performs *best* by a wide margin (mask mAP50 0.816 vs. 0.297, roughly 2.7× higher). Class imbalance alone therefore does not explain the error pattern.

**Root cause.** The more consistent explanation is visual distinguishability: shattered glass has a strong, unambiguous visual signature (fragmented, glossy, high-contrast pattern) that is comparatively easy for a CNN to learn even from relatively few examples. Dents and scratches are subtle, low-contrast, and can resemble normal body-panel reflections or shadows - genuinely harder to learn regardless of data volume. This is consistent with the independent finding on Milestone 4's comparative-benchmark track (CarDD-pretrained, Kaggle), which reported the identical pattern (`scratch`, the most numerous class, among the two worst-performing; underperformance persisting across baseline, augmentation, and DFL-reweighting interventions) and linked it to the same classes' difficulty in the published CarDD benchmark paper. That two independently pretrained models (COCO vs. CarDD), of different architecture generations, trained on two different platforms with different hyperparameters, reproduce the same class-difficulty ranking is a meaningfully strong piece of corroborating evidence that this is a property of the damage types themselves, not an artifact of either track's specific training setup.

### 8.2 Confusion Matrix and Error Patterns

The normalized confusion matrix provides a class-level view of the model's detection errors. The matrix is **column-normalized**, with the columns representing the true class and the rows representing the predicted class. Therefore, the diagonal values represent the proportion of correctly detected instances for each class, while the `background` row captures missed detections (false negatives) and the `background` column captures false-positive predictions.

| Predicted \ True | dent | scratch | crack | broken_lamp | shattered_glass | flat_tyre | background |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dent | 0.63 | 0.12 | 0.06 | 0.03 | 0.02 | 0.03 | 0.20 |
| scratch | 0.14 | 0.62 | 0.14 | 0.08 | 0.02 | 0.12 | 0.53 |
| crack | 0.04 | 0.07 | 0.49 | 0.05 | 0.01 | 0.23 | 0.15 |
| broken_lamp | 0.02 | 0.02 | 0.05 | 0.59 | 0.08 | 0.05 | 0.07 |
| shattered_glass | 0.01 | 0.01 | 0.01 | 0.09 | 0.80 | 0.02 | 0.02 |
| flat_tyre | 0.01 | 0.03 | 0.11 | 0.06 | 0.01 | 0.45 | 0.04 |
| background | 0.14 | 0.12 | 0.13 | 0.09 | 0.05 | 0.10 | — |

**Figure 8.1. Normalized confusion matrix on the test split.**

The matrix reinforces the class-level performance trends observed in Section 5.2. `shattered_glass` has the strongest diagonal value at **0.80**, followed by `dent` (**0.63**), `scratch` (**0.62**), `broken_lamp` (**0.59**), `crack` (**0.49**), and `flat_tyre` (**0.45**). Thus, `shattered_glass` is the most consistently recognised damage type, while `flat_tyre` and `crack` show substantially greater confusion.

Several important error patterns are visible:

- **Scratch–dent confusion:** 14% of true scratches are predicted as dents, while 12% of true dents are predicted as scratches. This is consistent with the visual similarity between surface scratches and dent-like regions.
- **Crack–flat_tyre confusion:** 23% of true flat-tyre instances are predicted as cracks, while 11% of true cracks are predicted as flat tyres. This represents one of the stronger cross-class confusions in the matrix.
- **Broken-lamp–shattered-glass confusion:** 9% of true broken-lamp instances are predicted as shattered glass, while 8% of true shattered-glass instances are predicted as broken lamps. This likely reflects the physical co-occurrence and visual similarity of broken lamps and shattered glass in damaged vehicles.
- **Missed detections:** The `background` row indicates that approximately 14% of dents, 12% of scratches, 13% of cracks, 9% of broken-lamp instances, 5% of shattered-glass instances, and 10% of flat-tyre instances are missed by the detector. The relatively low missed-detection rate for `shattered_glass` is consistent with its strong overall performance.
- **False positives:** The `background` column shows that the model produces false-positive predictions, particularly for `scratch` (**0.53**) and `dent` (**0.20**). This suggests that the model frequently interprets visually ambiguous regions of otherwise undamaged images as scratches or dents.

Overall, the confusion matrix supports the conclusion that the main challenge is not simply class imbalance but **visual ambiguity between damage types and between subtle damage and normal vehicle appearance**. In particular, the high false-positive rate for `scratch` and the relatively high confusion between `dent`, `scratch`, and `crack` indicate that these classes remain the principal sources of detection error.
### 8.3 Generalisation Check

A validation-vs-test performance gap check has been prepared (Section 3.3) to detect whether the model, or the Optuna search itself, overfit to the validation split used throughout development. This is not yet available and is flagged as an open item in Section 10.

---

## 9. Model Robustness

A lightweight robustness check - applying synthetic Gaussian blur and brightness shifts (dark/bright) to a sample of held-out images and comparing detection confidence before/after - has been implemented in the accompanying evaluation notebook as a practical proxy for how the model might behave on lower-quality, real-world submitted photos (motion blur, poor lighting), which is directly relevant to this system's expected production inputs. **This has not yet been executed**; results and interpretation should be added here once run.

Background (no-damage) images are part of the standard train/val/test splits (~7% of images) and are implicitly evaluated in every reported `.val()` run above - false positives on these images already factor into the reported precision figures.

---

## 10. Limitations

**Evaluation completeness.** The single largest limitation of this report as currently written: all headline numbers (Sections 5-7) are validation-split results, not test-split results. A test-split evaluation, an ablation study isolating the contribution of augmentation, a confusion matrix, qualitative failure-case images, and the robustness check (Section 9) are all prepared in the accompanying notebook but pending execution. This report should be treated as provisional pending that run, consistent with this project's established practice (Milestone 4) of clearly marking incomplete or provisional findings rather than presenting them as final.

**Comparative-benchmark track not re-evaluated here.** As disclosed in Section 2, this report evaluates the primary track's checkpoint, consistent with the Milestone 3/4 architecture decision. Milestone 4's CarDD-pretrained comparative-benchmark checkpoints (a different generation/scale) remain unevaluated in this document; a same-split reconciliation between the two tracks is listed as future work (Section 11).

**Dataset limitations.**
- Class imbalance exists (scratch outnumbers shattered_glass roughly 6.6:1 in training instances) but, per Section 8.1, does not appear to be the dominant driver of per-class performance differences - visual distinguishability appears to matter more.
- Single dataset source (VehiDE); no validation yet against images from a different camera, geography, or vehicle-type distribution.

**Model limitations.**
- Weakest on the three most common real-world damage types (dent, scratch, crack) - the classes a production system would encounter most often.
- 640×640 input resolution may lose fine detail relevant to thin scratches/cracks; untested at higher resolution due to GPU memory constraints on the free-tier Colab environment used for this track.

**Computational constraints.** The Optuna search used short (5-epoch) trials rather than full training runs per trial, due to compute cost; Section 7.4 shows this generalised reasonably to full-length training in this case, but this is not guaranteed for every hyperparameter or search space.

**Bias and ethical considerations.** Not formally assessed in this track - no metadata is available (vehicle make, colour, damage severity) to check for uneven performance across sub-populations. The consequence of the model's known weakest classes (dent, scratch, crack) being false negatives in production would be under-assessed claims for exactly the most common damage types; this risk should inform whatever human-in-the-loop review process the deployed system uses, and is worth explicit discussion at the system level (Milestone 3 architecture) rather than resolved by this vision component alone.

---

## 11. Possible Improvements

- **Complete the pending evaluation steps** (Section 10) first - test-split numbers, confusion matrix, and the ablation study are needed before any of the improvements below can be prioritised with confidence.
- **Reconcile the two experimental tracks.** Milestone 4's CarDD-pretrained comparative-benchmark track and this milestone's primary-track evaluation were run independently; a direct, same-split comparison between the two would clarify whether domain-specific pretraining (CarDD) or hyperparameter correction (this track's Optuna fix) is the larger lever, and whether combining both (CarDD-pretrained weights plus this track's corrected `lr0`, applied to the selected YOLO11m-seg architecture) would outperform either alone.
- **Targeted data for weak classes.** Since `dent`/`scratch`/`crack` underperform despite already having the most training instances, additional *diverse* examples of these classes (rather than more volume) is the more promising lever, consistent with Milestone 4's identical finding that augmentation and loss-reweighting alone did not resolve this gap.
- **Higher input resolution** (e.g. `imgsz=1280`) to preserve fine detail relevant to thin scratches and cracks, trading inference speed for detail.
- **Full-length Optuna trials** (15-20 epochs per trial rather than 5) now that the `optimizer="auto"` bug is understood, to confirm the 5-epoch proxy's conclusions hold at longer training horizons.
- **External benchmark comparison**, following Milestone 4's example of comparing against the published CarDD paper's per-class AP figures, once this track's test-split numbers are available.

---

## 12. Summary and Next Steps

Within the primary track evaluated in this report — the **YOLO11m-seg (COCO-pretrained)** checkpoint selected in Milestone 4, Section 10, consistent with the Milestone 3 architecture decision — correcting a hyperparameter-search configuration issue (`optimizer="auto"` silently overriding the intended learning rate) and applying the corrected result produced a model that improved on every reported validation metric versus the untuned baseline, by a larger margin than 20 additional epochs of unmodified training achieved. The model performs strongly on visually distinct damage (shattered glass, broken lamps) and weakest on subtle, high-frequency real-world damage (dents, scratches, cracks) - a pattern that independently reproduces the finding on Milestone 4's comparative-benchmark CarDD-pretrained track, strengthening confidence that this is a genuine property of the damage types rather than an artifact of either track's setup.

---

## 13. Policy Agent (RAG): Evaluation and Tuning Outcome

Sections 1-12 evaluate the Damage Agent (YOLO). This section asks the same question of the
Policy Agent's retrieval stack, whose selection and tuning work is documented in Milestone 4,
Section 13.

**Headline result.** The retrieval stack meets its Milestone 1 target and is the most
thoroughly validated component in the project. It retrieves relevant policy clauses at
**P@3 = 0.9133** against a random-retrieval baseline of 0.1634 - a **5.59x lift** - with
**zero zero-hit incidents** across all 50 test cases, meaning the Report Agent was never left
without usable clause evidence. Generated reports pass every mechanical faithfulness check on
both LLMs tested (**composite 1.00**, zero fabricated citations or currency figures) — as
Milestone 3, Section 8.2 already noted, this composite confirms a report's citations are real
and consistent with its verdict, not that the clause behind each citation is the right one; that
section's own claim-02 example (a citation-valid but semantically wrong table-merge case) was one
instance of the gap. Section 13.4 puts a number on it: a deeper, content-level check finds a
second instance and scores the gap directly rather than leaving it to manual review.

An 84-configuration sweep then asked whether this could be tuned further. It could not, and
that is a useful finding rather than a disappointing one: the shipped configuration already
sits at the top of a flat response surface, two parameters were shown to be inactive and can
be retired, and the exercise established that **the evaluation set, not the configuration, is
now the limiting factor**. Sections 13.2 and 13.3 give that evidence and state its limits
precisely.

### 13.1 What Demonstrably Improved

| Change | Before | After |
| :--- | :--- | :--- |
| PDF extraction fix (two-column bisection removed) | 179 chunks, words cut mid-token | 185 chunks, zero truncation artifacts |
| Extraction fix + heading breadcrumb | MiniLM 0.94 / BGE 0.89 | MiniLM **1.00** / BGE 0.94 |
| Hybrid retrieval (targeted lexical fix) | Rear-window incident P@3 **0.33**, glass clause ranked 9th | Glass clause surfaced; aggregate 0.893 to 0.913 |
| Two-query coverage/exclusion split | A single query buries the exclusion beneath the coverage clause | Both buckets reach the LLM; unscoped path reproduces 0.913 / 0.977 exactly |
| Per-user policy architecture | Policy identification top-1 **0.20** | Single candidate - identification eliminated |
| Report faithfulness | - | Composite **1.00**, citation validity 1.00, 0 currency violations, on both models |

### 13.2 What the Tuning Established

The sweeps in Milestone 4 Section 13.2 report P@3 to four decimals, which invites reading
0.9200 as beating 0.9133. At n=50 that difference is one retrieved chunk in one incident's top
three: mean P@3 moves only in quanta of 1/(3x50) = **0.00667**. Each configuration was
therefore paired against production incident-by-incident and the mean difference bootstrapped
(10,000 resamples, seed 20260807):

| Configuration | P@3 | Delta vs. production | 95% CI on delta | Verdict |
| :--- | ---: | ---: | :--- | :--- |
| **production** (3:1, k=60, pool=20) | 0.9133 | - | - | - |
| sparse-only (0:1) | 0.7400 | -0.1733 | [-0.2400, -0.1067] | **significant** |
| dense-only (1:0) | 0.8933 | -0.0200 | [-0.0667, +0.0267] | not distinguishable |
| ratio 2:1 | 0.9133 | +0.0000 | [+0.0000, +0.0000] | not distinguishable |
| pool=10 (grid best) | 0.9200 | +0.0067 | [-0.0200, +0.0333] | not distinguishable |
| rrf_k=100 | 0.9133 | +0.0000 | [+0.0000, +0.0000] | not distinguishable |

**Findings.**

1. **Only sparse-only is statistically distinguishable from production, and it is worse.**
   Every other parameter's confidence interval spans zero. The apparent winners in the RRF_K,
   pool and interaction sweeps are noise.
2. **2:1 and 3:1 change zero incidents.** The Milestone 2 note that these "differ on the exact
   top-3 for 16/50 incidents, but only by reordering equally relevant chunks" is confirmed
   numerically: the paired difference is exactly 0.0000. That choice was correctly made on
   reasoning rather than on a score difference, because there is no score difference.
3. **The hybrid retriever's headline gain does not reach significance.** Dense-only versus
   hybrid is a delta of -0.0200 with a CI of [-0.0667, +0.0267], and hybrid is marginally
   *worse* on MRR (0.9767 vs. 0.9800). The 0.893 to 0.913 improvement is real in direction and
   changes 13 of 50 incidents, but n=50 is too small to establish it as an aggregate effect.

Finding 3 is **not** an argument for removing the hybrid retriever. It repairs a specific
diagnosed lexical failure, costs nothing at this corpus scale, and sparse-only's significant
deficit confirms the two signals are complementary rather than redundant. What should change
is the claim: cite the per-incident failure it repairs, not "0.893 to 0.913" as a proven
aggregate gain.

**Conclusion.** No retrieval parameter in this stack is worth tuning further. Production sits
at 0.9133 P@3 against a 0.1634 random baseline - a **5.59x lift** with zero zero-hit incidents.
The binding constraint is the evaluation set, not the configuration: at n=50 a 95% interval on
any realistic improvement spans zero, which is why all 15 candidate configurations came back
indistinguishable. This mirrors, from the opposite direction, the vision track's finding in
Section 7 that a single genuine configuration error was worth more than extended search.

### 13.3 Limitations

Three limitations govern every retrieval figure reported above and in Milestone 4 Section 13.

**1. The relevance labels are circular.** `data/clause_groundtruth.json` is not hand-labelled;
its `damage_classes` are the same regex auto-tags that the chunker writes onto each chunk
(verified identical for all 185). The scores measure retrieval against the tagger, not against
a human adjudicator.

**2. The tuning set is the test set.** The rear-window incident that motivated hybrid retrieval
is one of the 50 incidents used to score it, and the 3:1 ratio was selected on the same 50.
There is no held-out split, so 0.913 is partly fit to the set it is measured on.

**3. Everything was measured on synthetic policies that suit the method.** A structural check
of the three genuine IRDAI-filed policies in `data/policy_pdfs/reference/` - never indexed or
scored - shows they behave very differently:

| | Synthetic (5, tuned on) | Real IRDAI (3) |
| :--- | :--- | :--- |
| Damage types **mentioned** in the text | **6 / 6** in all five | **2-4 / 6** |
| Damage types actually **tagged** | 6 / 6 | **1-3 / 6** |
| Chunks carrying no damage-class tag | 15-67% | **96-98%** |
| Chunks falling to `general` | 3-24% | 37-63% |
| Chunks per document | 29-45 | 79-323 |

This is **not** a tagger defect - the one apparent miss was the word "dental", which the
tagger correctly ignores. Real motor policies use generic "loss or damage" wording instead of
enumerating dent, scratch and crack. But because retrieval relevance and clause bucketing both
run through damage-class tags, the signal available on a real uploaded policy is substantially
thinner than these figures imply.

**Highest-value next steps:** measure the per-user retrieval path (Milestone 4, Section 13.3) —
Section 13.4 below takes a first pass at exactly this; broaden the damage-class keyword lists
for real-policy language; and obtain human-adjudicated relevance labels on 150-200 incidents,
which would make differences smaller than one or two quanta resolvable at all.

### 13.4 A Closer Look at Report Quality

The checks in Section 8.2 confirm a report cites *a* clause and doesn't contradict itself, but
they can't tell whether that clause actually says what the report claims it says. To catch that,
we added a second layer of evaluation, using the RAGAs framework: each generated report is
compared against the policy text it was supposed to be based on, and separately, each retrieved
clause is compared against the incident it was retrieved for. We also wrote our own answer key by
hand for the 10 test claims — what a careful reader of the policy would conclude — so this
evaluation has something independent to check the reports against.

**Retrieval side:** the retrieved clauses were rated relevant about 83% of the
time, close to but a bit stricter than our earlier, simpler check (91%), which only looked at
whether a clause's topic tag matched, not whether it was actually useful.

**Report side:** scored against our hand-written answer key, both models landed in the 0.4-0.6
range out of 1.0 on the three things we checked: is the reasoning actually backed by the cited
text, is the answer relevant to the question, and is the final verdict correct.

**What this caught.** Writing the answer key by hand surfaced a real bug, not just a low score.
For one claim, the system had to decide whether cracked and broken parts from a multi-vehicle
collision were covered. It technically had *a* clause to point to, but that clause was actually
about tyre damage, unrelated to cracks or lamps — the real "what's covered" clause for that
policy was never found for those two damage types, even though it was there in the same
document. Our earlier checks didn't catch this because they only check that a citation exists
and is the right *type* of clause, not whether it's actually on-topic. The model ended up stating
"the policy covers cracks" with nothing real behind that claim.

**The fix.** We changed the clause search so that, alongside the specific search for each damage
type, it also always searches for the policy's general "what's covered" clause and adds it to
the candidates if the specific search missed it. This doesn't remove anything the search already
found — it only adds a fallback option. After the fix, the correct clause showed up for the
previously broken cases, and checking across all 5 policies and all 6 damage types, every one of
them now has at least one relevant "what's covered" clause to work with.

**Re-testing with the real fix in place.** We reran all 10 test claims through the actual system
(not a simplified test version) with the fix applied. All 10 now pass our basic checks, and the
previously broken case now correctly points to the right clause and reasons about it properly.

On the deeper 3-question evaluation, the "is the verdict correct" score improved with the fix, which
matches what we'd expect: the model now has real information to work with for the cases that were
previously broken. The "is the reasoning backed by the text" score didn't improve in this same
run, but we can't cleanly credit that to the fix either way, because we also had to switch which
AI model was writing the reports partway through (our usual model hit its daily free usage limit
partway through testing). A cleaner side-by-side comparison, with only the fix changing and
nothing else, is the natural next step.

More detail and the full numbers are in `docs/RAG_Component.md`.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | 7th Aug'26 | S.K. |
| Pranab Kumar Manna |07-08-2026 | PK Manna |
| Venkata Siva Kamal Guddanti | 07-08-2026 | Kamal G |
| Anuj Gautam | 07-08-2026 | Anuj |
| Harsh Pal |07-08--2026 | harshpal|

---
