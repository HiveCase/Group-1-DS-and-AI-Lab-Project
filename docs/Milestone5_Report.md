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
- [2. Note on the Experimental Track Evaluated in This Milestone](#2-note-on-the-experimental-track-evaluated-in-this-milestone)
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

Milestone 3 selected YOLO-family instance segmentation as the modelling approach for the Damage Agent. Milestone 4 pursued two parallel tracks: a Kaggle-based track fine-tuning CarDD-pretrained checkpoints (YOLOv8s-seg, YOLO11x-seg), and a separate, individually-run track (documented here) fine-tuning a **COCO-pretrained YOLO11m-seg** checkpoint on Google Colab. This milestone evaluates the second track.

### 1.2 Objectives of Milestone 5

Per the milestone requirements, this report:

- Evaluates the trained model(s) using appropriate, task-relevant metrics.
- Provides error analysis - what the model gets wrong, and why.
- Discusses limitations and possible improvements.

---

## 2. Note on the Experimental Track Evaluated in This Milestone

This is an important disclosure and should be read before the results that follow.

Milestone 4's own "Prepared for Milestone 5" section names the **Kaggle-trained, CarDD-pretrained YOLOv8s-seg checkpoint** (DFL boundary-precision variant, test mask mAP@50 = 0.3549) as the checkpoint intended for detailed error analysis in this milestone.

**This report instead evaluates a separate, individually-run experimental track**: a **COCO-pretrained YOLO11m-seg** model, fine-tuned on Google Colab, using the same underlying VehiDE dataset but a different pretrained starting point, training environment, and hyperparameter search process (Optuna-based, described in Section 7). This track was not part of the Milestone 4 report and is documented here for the first time.

**Why this matters for reading the rest of this report:**

- Results in this report are **not directly comparable** to Milestone 4's CarDD-track numbers - the two tracks start from different pretrained weights (COCO vs. CarDD domain-specific pretraining), which Milestone 4 itself identified as a meaningful factor ("fine-tuning models pretrained on a domain-specific dataset consistently produced better results than fine-tuning a general-purpose COCO-pretrained model").
- The dataset split used in this track (9,545 / 2,047 / 2,047 train/val/test) differs slightly from Milestone 4's split (9,558 / 2,048 / 2,049) due to being generated independently from the same source dataset.
- The CarDD-track checkpoints Milestone 4 flagged as ready for evaluation are **not evaluated in this report**. If the group's final submission needs both tracks covered, that CarDD-track evaluation should be prepared as a companion analysis using the Kaggle-side checkpoints and logs, which are not available in this track's working environment.

---

## 3. Evaluation Methodology

### 3.1 Models Evaluated

| Model | Description |
| --- | --- |
| **Baseline** | YOLO11m-seg, COCO-pretrained, fine-tuned 40 epochs. `optimizer="auto"`, which Ultralytics silently resolved to AdamW at a fixed `lr=0.001` (see Section 7 for how this was discovered). |
| **Extended** | Same baseline, continued for 20 further epochs (60 total), no hyperparameter changes - tests whether more training alone closes the gap. |
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

**A held-out test-split evaluation has been prepared but not yet executed at the time of writing this report.** An evaluation notebook exists with a dedicated cell that runs all three checkpoints against the untouched test split (never used for training, checkpoint selection, or the Optuna search), which will produce a stricter, unbiased final number. Until that cell is run, the validation-split numbers below should be read as **provisional** - consistent with development-time performance, but with some risk of a small optimistic bias since checkpoint selection (`best.pt`) was itself chosen based on validation performance. This caveat is carried through Sections 5, 6, 8, and 9 wherever it applies, and is repeated in Section 10 (Limitations).

### 3.4 Ground Truth

Labels are the pre-existing YOLO-format polygon (instance segmentation) annotations shipped with the VehiDE dataset. No manual re-annotation was performed.

### 3.5 Success Criteria

No fixed pass/fail metric threshold was assigned for this track. Results are reported and discussed comparatively - baseline vs. extended vs. tuned - rather than against an absolute target.

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
| Baseline (40 ep) | - | - | ~0.443 | ~0.269 | - | - | ~0.400 | ~0.209 |
| Extended (60 ep) | 0.537 | 0.443 | 0.443 | 0.270 | 0.528 | 0.413 | 0.400 | 0.209 |
| **Tuned (40 ep)** | **-** | **-** | **0.485** | **0.300** | **-** | **-** | **0.449** | **0.241** |

The Extended run's per-epoch metrics were effectively identical to the Baseline run's final epoch, indicating the additional 20 epochs of unmodified training produced negligible further gains once the initial 40-epoch mark was reached.

### 5.2 Per-Class Performance (Tuned model, validation split)

| Class | Train instances | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dent | 3,888 | 0.268 | - | 0.247 | 0.097 |
| scratch | 10,070 | 0.309 | - | 0.247 | 0.094 |
| crack | 3,763 | 0.283 | - | 0.280 | 0.121 |
| broken_lamp | 1,920 | 0.548 | - | 0.417 | 0.189 |
| shattered_glass | 1,513 | 0.820 | - | 0.767 | 0.537 |
| flat_tyre | 1,631 | 0.429 | - | 0.444 | 0.216 |

The values above are the Extended-run per-class figures, prior to the final hyperparameter-tuned run; the qualitative ranking of classes (shattered_glass strongest, dent/scratch weakest) held consistently across the Baseline, Extended, and Tuned checkpoints, with the Tuned model improving every class's mask mAP50 by roughly 0.03-0.06 over these figures without changing the ranking.

### 5.3 Training Curves

Across all three runs, training loss (`box_loss`, `seg_loss`, `cls_loss`) declined steadily with no reversals, and validation mAP50/mAP50-95 climbed with no plateau through the initial 40 epochs. `patience=15` (early stopping) did not trigger in any run - all three ran their full configured epoch budget. Validation loss tracked training loss in the same direction throughout, with no divergence observed - i.e. no evidence of overfitting in the ranges trained.

---

## 6. Baseline vs. Tuned Model Comparison

| Metric | Baseline | Tuned | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Box mAP50 | 0.443 | 0.485 | +0.042 | +9.5% |
| Box mAP50-95 | 0.269 | 0.300 | +0.031 | +11.5% |
| Mask mAP50 | 0.400 | 0.449 | +0.049 | +12.3% |
| Mask mAP50-95 | 0.209 | 0.241 | +0.032 | +15.3% |

The only meaningful configuration difference between the Baseline and Tuned runs is the learning rate actually applied during training (0.001 vs. ~0.000105) and a small amount of rotation augmentation (0° vs. 5.5°) - epochs, batch size, dataset, and seed are held constant. The Tuned run improves on **every** reported metric with no regression, and by a larger margin than the Extended run's 20 additional epochs achieved. This supports attributing the improvement to the hyperparameter correction itself, rather than to random variation or additional training time.

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

`lr0=0.0001047`, `weight_decay=0.000292`, `degrees=5.5` - the best 5-epoch trial - trained for a full 40 epochs, producing the Tuned model reported in Sections 5-6. The 5-epoch proxy result (0.351) generalised sensibly to full training (0.449 at 40 epochs), indicating the short-trial search was not simply overfitting to the 5-epoch budget.

---

## 8. Error Analysis

### 8.1 Class-Level Error Pattern

The clearest and most consistent finding across every checkpoint trained in this track: **`shattered_glass` and `broken_lamp` substantially outperform `dent`, `scratch`, and `crack`**, and this ranking persisted unchanged across the Baseline, Extended, and Tuned checkpoints (Section 5.2).

This directly contradicts a naive expectation from the training-set class distribution: `scratch` has the *most* training instances of any class (10,070) yet is among the *worst*-performing, while `shattered_glass` has the *fewest* (1,513) yet performs *best* by a wide margin (mask mAP50 0.767 vs. 0.247, roughly 3× higher). Class imbalance alone therefore does not explain the error pattern.

**Root cause.** The more consistent explanation is visual distinguishability: shattered glass has a strong, unambiguous visual signature (fragmented, glossy, high-contrast pattern) that is comparatively easy for a CNN to learn even from relatively few examples. Dents and scratches are subtle, low-contrast, and can resemble normal body-panel reflections or shadows - genuinely harder to learn regardless of data volume. This is consistent with the independent, Kaggle-track finding in Milestone 4, which reported the identical pattern (`scratch`, the most numerous class, among the two worst-performing; underperformance persisting across baseline, augmentation, and DFL-reweighting interventions) and linked it to the same classes' difficulty in the published CarDD benchmark paper. That two independently pretrained models (COCO vs. CarDD), trained on two different platforms with different hyperparameters, reproduce the same class-difficulty ranking is a meaningfully strong piece of corroborating evidence that this is a property of the damage types themselves, not an artifact of either track's specific training setup.

### 8.2 False Positives and False Negatives

- **False positives** (predicted damage with no matching ground truth): expected to be concentrated in the classes with lower Box precision - `dent`, `scratch`, `crack` - and plausibly include cases where shadows or reflections on a vehicle surface are mistaken for a subtle dent or scratch.
- **False negatives** (missed ground-truth damage): expected to concentrate on smaller or thinner damage instances - fine cracks, small scratches - given the model's demonstrated weaker performance on exactly these classes, and given the dataset's own minimum annotated damage area is very small relative to full image size.
- A full confusion matrix and a qualitative sample of flagged failure cases (predicted-instance-count vs. ground-truth-count mismatches) have been prepared in the accompanying evaluation notebook but not yet executed; this section should be updated with specific visual examples and confusion-matrix values once that step is run.

### 8.3 Generalisation Check

A validation-vs-test performance gap check has been prepared (Section 3.3) to detect whether the model, or the Optuna search itself, overfit to the validation split used throughout development. This is not yet available and is flagged as an open item in Section 10.

---

## 9. Model Robustness

A lightweight robustness check - applying synthetic Gaussian blur and brightness shifts (dark/bright) to a sample of held-out images and comparing detection confidence before/after - has been implemented in the accompanying evaluation notebook as a practical proxy for how the model might behave on lower-quality, real-world submitted photos (motion blur, poor lighting), which is directly relevant to this system's expected production inputs. **This has not yet been executed**; results and interpretation should be added here once run.

Background (no-damage) images are part of the standard train/val/test splits (~7% of images) and are implicitly evaluated in every reported `.val()` run above - false positives on these images already factor into the reported precision figures.

---

## 10. Limitations

**Evaluation completeness.** The single largest limitation of this report as currently written: all headline numbers (Sections 5-7) are validation-split results, not test-split results. A test-split evaluation, an ablation study isolating the contribution of augmentation, a confusion matrix, qualitative failure-case images, and the robustness check (Section 9) are all prepared in the accompanying notebook but pending execution. This report should be treated as provisional pending that run, consistent with this project's established practice (Milestone 4) of clearly marking incomplete or provisional findings rather than presenting them as final.

**Track fragmentation.** As disclosed in Section 2, this report evaluates a track separate from the one Milestone 4 flagged for Milestone 5 evaluation. The group's CarDD-pretrained checkpoints remain unevaluated in this document.

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
- **Reconcile the two experimental tracks.** Milestone 4's CarDD-pretrained track and this Colab COCO-pretrained track were run independently; a direct, same-split comparison between the two would clarify whether domain-specific pretraining (CarDD) or hyperparameter correction (this track's Optuna fix) is the larger lever, and whether combining both (CarDD-pretrained weights plus this track's corrected `lr0`) would outperform either alone.
- **Targeted data for weak classes.** Since `dent`/`scratch`/`crack` underperform despite already having the most training instances, additional *diverse* examples of these classes (rather than more volume) is the more promising lever, consistent with Milestone 4's identical finding that augmentation and loss-reweighting alone did not resolve this gap.
- **Higher input resolution** (e.g. `imgsz=1280`) to preserve fine detail relevant to thin scratches and cracks, trading inference speed for detail.
- **Full-length Optuna trials** (15-20 epochs per trial rather than 5) now that the `optimizer="auto"` bug is understood, to confirm the 5-epoch proxy's conclusions hold at longer training horizons.
- **External benchmark comparison**, following Milestone 4's example of comparing against the published CarDD paper's per-class AP figures, once this track's test-split numbers are available.

---

## 12. Summary and Next Steps

Within the track evaluated in this report, correcting a hyperparameter-search configuration issue (`optimizer="auto"` silently overriding the intended learning rate) and applying the corrected result produced a model that improved on every reported validation metric versus the untuned baseline, by a larger margin than 20 additional epochs of unmodified training achieved. The model performs strongly on visually distinct damage (shattered glass, broken lamps) and weakest on subtle, high-frequency real-world damage (dents, scratches, cracks) - a pattern that independently reproduces Milestone 4's finding on the separate CarDD-pretrained track, strengthening confidence that this is a genuine property of the damage types rather than an artifact of either track's setup.

---

## 13. Policy Agent (RAG): Evaluation and Tuning Outcome

Sections 1-12 evaluate the Damage Agent (YOLO). This section asks the same question of the
Policy Agent's retrieval stack, whose selection and tuning work is documented in Milestone 4,
Section 13.

**Headline result.** The retrieval stack meets its Milestone 1 target and is the most
thoroughly validated component in the project. It retrieves relevant policy clauses at
**P@3 = 0.9133** against a random-retrieval baseline of 0.1634 - a **5.59x lift** - with
**zero zero-hit incidents** across all 50 test cases, meaning the Report Agent was never left
without usable clause evidence. Generated reports pass every faithfulness check on both LLMs
tested (**composite 1.00**, zero fabricated citations or currency figures).

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

**Highest-value next steps:** measure the per-user retrieval path (Milestone 4, Section 13.3);
broaden the damage-class keyword lists for real-policy language; and obtain human-adjudicated
relevance labels on 150-200 incidents, which would make differences smaller than one or two
quanta resolvable at all.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | 7th Aug'26 | S.K. |
| Pranab Kumar Manna |07-08-2026 | PK Manna |
| Venkata Siva Kamal Guddanti | | |
| Anuj Gautam | 07-08-2026 | Anuj |
| Harsh Pal |07-08--2026 | harshpal|

---
