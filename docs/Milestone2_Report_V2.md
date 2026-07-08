
---

<div align="center">


<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">


<h1 style="font-size:26em;">Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 2: Dataset Preparation for a Multi-Agent RAG Architecture</h2>

<h3>Group 1</h3>

<br>

  ***Prepared by:***

  
| **Name** | **Email ID** | **GitHub Profile** |
| --- | --- | --- |
| SATYAJEET KUMAR | 23f1003132@ds.study.iitm.ac.in | [HiveCase](https://github.com/HiveCase) |
| ANUJ GAUTAM | 21f1002407@ds.study.iitm.ac.in | [anujgautam1](https://github.com/anujgautam1) |
| PRANAB KUMAR MANNA | 22f1000887@ds.study.iitm.ac.in | [pranab92](https://github.com/pranab92) |
| VENKATA SIVA KAMAL GUDDANTI | 22f2000094@ds.study.iitm.ac.in | [22f2000094](https://github.com/22f2000094) |

</div>

---

# Table of Contents

- [1. Introduction](#1-introduction)
- [2. Dataset Identification](#2-dataset-identification)
  - [2.1 Primary Vision Dataset - VehiDE](#21-primary-vision-dataset---vehide)
  - [2.2 Supplementary Vision Datasets](#22-supplementary-vision-datasets)
  - [2.3 Synthetic Policy Corpus (Policy Agent)](#23-synthetic-policy-corpus-policy-agent)
  - [2.4 Ownership, Licensing, and Usage Constraints](#24-ownership-licensing-and-usage-constraints)
  - [2.5 Alternatives Considered](#25-alternatives-considered)
- [3. Dataset Description](#3-dataset-description)
  - [3.1 VehiDE - Structure, Schema, and Sample Record](#31-vehide---structure-schema-and-sample-record)
  - [3.2 Target Variable / Class Taxonomy and Mapping](#32-target-variable--class-taxonomy-and-mapping)
  - [3.3 Image Characteristics](#33-image-characteristics)
  - [3.4 Synthetic Policy Corpus - Structure and Schema](#34-synthetic-policy-corpus---structure-and-schema)
- [4. Data Governance](#4-data-governance)
  - [4.1 Data Source & Licensing](#41-data-source--licensing)
  - [4.2 Privacy](#42-privacy)
  - [4.3 Data Quality](#43-data-quality)
  - [4.4 Ethics & Bias](#44-ethics--bias)
  - [4.5 Reproducibility & Compliance](#45-reproducibility--compliance)
- [5. Exploratory Data Analysis (EDA)](#5-exploratory-data-analysis-eda)
  - [5.1 Summary Statistics](#51-summary-statistics)
  - [5.2 Class Distribution](#52-class-distribution)
  - [5.3 Image-Level Feature Distributions](#53-image-level-feature-distributions)
  - [5.4 Missing Value Analysis](#54-missing-value-analysis)
  - [5.5 Duplicate Analysis](#55-duplicate-analysis)
  - [5.6 Annotation Noise / Outlier Analysis](#56-annotation-noise--outlier-analysis)
  - [5.7 Visualizations](#57-visualizations)
- [6. Data Preprocessing](#6-data-preprocessing)
  - [6.1 Cleaning: Missing, Corrupt, and Degenerate Data](#61-cleaning-missing-corrupt-and-degenerate-data)
  - [6.2 Duplicate Removal](#62-duplicate-removal)
  - [6.3 Label Correction and Class Remapping (Encoding)](#63-label-correction-and-class-remapping-encoding)
  - [6.4 Annotation Format Conversion](#64-annotation-format-conversion)
  - [6.5 Image Standardisation / Normalisation](#65-image-standardisation--normalisation)
  - [6.6 Text Cleaning and Tokenisation (Policy Agent)](#66-text-cleaning-and-tokenisation-policy-agent)
  - [6.7 Feature Engineering: Derived Severity Labels](#67-feature-engineering-derived-severity-labels)
  - [6.8 Feature Selection](#68-feature-selection)
  - [6.9 Audio Preprocessing (Speech)](#69-audio-preprocessing-speech)
- [7. Dataset Integration](#7-dataset-integration)
  - [7.1 Datasets Combined](#71-datasets-combined)
  - [7.2 Integration Methodology and Schema Alignment](#72-integration-methodology-and-schema-alignment)
  - [7.3 Handling Conflicting Attributes](#73-handling-conflicting-attributes)
  - [7.4 Deduplication After Integration](#74-deduplication-after-integration)
- [8. Data Augmentation](#8-data-augmentation)
  - [8.1 Vision Augmentation Techniques](#81-vision-augmentation-techniques)
  - [8.2 Rationale](#82-rationale)
  - [8.3 Policy Corpus Stress-Testing (Text-Side Augmentation)](#83-policy-corpus-stress-testing-text-side-augmentation)
- [9. Dataset Splitting](#9-dataset-splitting)
  - [9.1 Split Ratio and Strategy](#91-split-ratio-and-strategy)
  - [9.2 Leakage Prevention Measures](#92-leakage-prevention-measures)
- [10. Final Prepared Dataset](#10-final-prepared-dataset)
  - [10.1 Expected Final Size and Schema per Agent](#101-expected-final-size-and-schema-per-agent)
  - [10.2 Summary of Preprocessing Completed](#102-summary-of-preprocessing-completed)
  - [10.3 Readiness for Model Training](#103-readiness-for-model-training)
- [11. Challenges Encountered](#11-challenges-encountered)
- [12. Deliverables Produced](#12-deliverables-produced)
- [13. Summary and Next Steps](#13-summary-and-next-steps)
  - [13.1 Summary](#131-summary)
  - [13.2 Planned Activities for Milestone 3](#132-planned-activities-for-milestone-3)
- [References](#references)

---

## 1. Introduction

Milestone 1 defined the problem, scope, and evaluation plan for the multimodal vehicle damage assessment system as a three-stage sequential pipeline (YOLO detection &rarr; RAG policy retrieval &rarr; LLM report generation). Since then, the team has refined the system design into an **open-source multi-agent RAG architecture**, in which a LangGraph orchestrator routes each claim to four specialist agents - a **Damage Agent** (YOLOv8 detection), a **Severity Agent** (bounding-box area-ratio scoring), a **Policy Agent** (RAG over the synthetic policy corpus, exposed as an MCP tool), and a **Report Agent** (LLM-based report writing) - with low-confidence outputs escalated to a human review queue rather than auto-finalised.

<p align="center">
<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/Archive/insurance_claim_multiagent_detail.png" width="620">
</p>

**Objectives of Milestone 2:** This report (a) identifies and verifies every dataset the project will use, (b) documents each dataset's structure, licensing, and governance posture, (c) performs an exploratory data analysis to characterise class balance, image quality, and known noise sources, (d) defines and executes a concrete cleaning and preprocessing pipeline, (e) specifies how the vision and text-side datasets are integrated and split, and (f) hands off a **final, training-ready dataset** for each agent, so that another team could begin model training and RAG-index construction immediately, without further data preparation.

**Relationship between the dataset and project goals:** The system's four agents cannot be built or evaluated without, respectively: (i) a large, diverse, correctly-labelled vehicle damage image corpus (Damage Agent), (ii) a human-validated severity reference (Severity Agent), (iii) a realistic but IP-clean insurance policy corpus (Policy Agent), and (iv) a paired evaluation set of incident descriptions and images (Report Agent, end-to-end evaluation). This milestone's datasets are therefore the direct enabling input for every metric defined in Milestone 1, Sections 4 and 8.

This change in system design does not change *which* underlying data the project needs, but it does change *how that data is scoped, owned, and interfaced*: each dataset is now consumed by a specific agent or MCP-exposed tool rather than by an anonymous pipeline "stage," and the agents additionally require a small amount of new state/interface data (session-state schema, tool I/O contracts) that did not exist in the single-pipeline framing of Milestone 1.

---

## 2. Dataset Identification

### 2.1 Primary Vision Dataset - VehiDE

The primary dataset selected for the Damage Agent (and, downstream, the Severity Agent) is **VehiDE (Vehicle Damage Detection Dataset)**, published on Kaggle by Hendrich Scullen [1] and described in the accompanying research paper [2].

| **Attribute** | **Detail** |
| --- | --- |
| Dataset name | VehiDE (Vehicle Damage Detection Dataset) |
| Source / download link | [Kaggle - VehiDE Dataset](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection/data) |
| Status | Public (Kaggle-hosted), non-commercial research/educational license (see Section 2.4) |
| Origin | Curated and published by the dataset authors for the paper "VehiDE Dataset: New dataset for Automatic vehicle damage detection in Car insurance" [2] |
| Size | 13,945 high-resolution images |
| Instances | 32,000+ labelled damage instances |
| Damage categories | 8 native damage categories (a superset of the 6 classes in our project scope; see Section 3.2) |
| Supported tasks | Classification, object detection, instance segmentation, and salient object detection |
| Purpose in this project | Primary training/evaluation corpus for the Damage Agent's YOLOv8 detector and the input to the Severity Agent's area-ratio scoring |
| Access method | Downloaded directly via the Kaggle API/CLI under our team Kaggle account, hashed and version-pinned locally so every member (and every agent's fine-tuned checkpoint) works from an identical copy |

**Why selected:** VehiDE is the largest publicly available annotated vehicle damage dataset with real (non-synthetic) images, is directly cited in the peer-reviewed literature reviewed in Milestone 1 (Section 3.1), and provides sufficient scale to fine-tune a YOLO detector without relying on synthetic image generation.

We verified this source by cross-checking the dataset card against the peer-reviewed paper describing its construction [2] and against independent citations of the dataset in subsequent literature [3], confirming that the reported size (13,945 images, 32k+ instances) and category count are consistent across sources.

### 2.2 Supplementary Vision Datasets

Three supplementary datasets identified in Milestone 1 (Section 9.1) are used for specific auxiliary purposes rather than as primary training data:

| **Dataset** | **Source / link** | **Status** | **Purpose in this project** | **Why selected** |
| --- | --- | --- | --- | --- |
| CarDD | USTC research release [4] | Public, academic-use license | Supplementary pixel-level segmentation fine-tuning where VehiDE's bounding-box annotations are insufficient (e.g., irregularly shaped scratches/cracks); feeds the Damage Agent | Only dataset offering pixel-level masks, needed for YOLO11-seg's segmentation head, which underpins the area-ratio severity proxy |
| COCO Car Damage Detection | [Kaggle](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset) [5] | Public, community-hosted Kaggle license | Small supplementary set for architecture comparison / sanity-checking the Damage Agent's detection pipeline against a differently-annotated source | COCO-format annotations allow direct comparison against published COCO-trained baselines |
| Car Damage Severity Dataset | Kaggle | Public, community-hosted Kaggle license | Calibration of the Severity Agent's Minor/Moderate/Severe heuristic against human-labelled severity ground truth | Only public dataset with human-assigned severity labels matching our three-category scheme |

### 2.3 Synthetic Policy Corpus (Policy Agent)

No public dataset pairs insurance policy documents with vehicle damage annotations, and real insurer policy documents cannot be used due to proprietary constraints (Milestone 1, Section 1.3). The Policy Agent's input corpus is therefore a **fully synthetic, team-authored dataset**: five insurance policy PDFs and fifty paired incident-description/image samples, generated as detailed in Section 3.4 and Section 6.6. This is not a "third-party dataset" in the licensing sense; it is documented here as a dataset that must be *produced* rather than *sourced*, and its status is: privately authored, wholly owned by the project team, unrestricted for internal use.

### 2.4 Ownership, Licensing, and Usage Constraints

- **VehiDE**: The dataset authors explicitly restrict usage to **non-commercial research and educational purposes**, with users bearing responsibility for appropriate use [2]. This is compatible with our project, which is an academic lab project and will be distributed as an open, non-commercial demonstration deployed via the Docker Compose / k3s stack described in the multi-agent architecture. Any public deployment will carry attribution to the original authors [1][2] and a notice that the demo is for research/educational purposes only.
- **CarDD**: Released for academic research; attribution to the original paper is required in any derivative work [4].
- **COCO Car Damage Detection / Car Damage Severity**: Both are community-hosted Kaggle datasets; we will retain the Kaggle dataset page license terms and cite the original uploaders.
- **Ownership boundary**: No dataset used in this project is owned by our team or by IIT Madras; all are third-party datasets used under their respective research/educational licenses. No dataset is redistributed by the project - only derived artefacts (model weights, indices) are produced.
- **Synthetic policy corpus** (Section 2.3): authored entirely by the team and therefore fully owned by the project, with no third-party licensing constraints. This is in fact the reason Milestone 1 (Section 1.3, "Out of Scope") explicitly ruled out using real insurer policy documents, which are proprietary.

---

## 3. Detailed Dataset Description

### 3.1 VehiDE - Structure, Schema, and Sample Record

Each VehiDE sample consists of a high-resolution RGB photograph of a damaged vehicle together with an annotation file containing, per damage instance:

- A class label (one of the 8 VehiDE damage categories).
- A bounding box (and, for a subset of images, a segmentation polygon) delimiting the damage region.
- An implicit instance-scale attribute, since the dataset publication reports the proportion of small/medium/large instances [2], which is directly relevant to the Severity Agent's area-ratio scoring approach (Milestone 1, Section 10.2).

**Data format**: Images are RGB JPEG/PNG files; annotations are provided as structured text (COCO-style JSON / Pascal-VOC-style XML depending on export, normalised to JSON during ingestion - Section 6.4) - i.e., an **image + JSON annotation** pair per sample, not a flat CSV.

**Number of features**: Each instance record has **6 core annotation fields** (`class_label`, `bbox` [4 sub-values: x, y, w, h], `segmentation`, `scale`), plus **3 image-level fields** (`image_id`, `width`, `height`), for **9 raw schema fields per image record** before any engineered features are added. This excludes the pixel data itself, which is consumed directly by the CNN backbone rather than treated as tabular "features." The Severity Agent adds 1 further engineered feature (`bbox_area_ratio`, Section 6.7), and the derived `severity_label` is its target variable.

**Dataset schema (per record):**

| **Field** | **Type** | **Description** |
| --- | --- | --- |
| `image_id` | string | Unique identifier / filename of the source image |
| `image` | RGB image (JPEG/PNG) | The vehicle photograph |
| `width`, `height` | int | Native image resolution |
| `instances[]` | list of objects | One entry per damage region in the image |
| `instances[].class_label` | categorical (8 native classes) | Native VehiDE damage category |
| `instances[].bbox` | `[x, y, w, h]` (pixel or normalised) | Bounding box of the damage region |
| `instances[].segmentation` | polygon (subset of images) | Pixel-level mask, where available |
| `instances[].scale` | categorical (small/medium/large) | Reported instance-scale bucket |

**Illustrative sample record (structure, not an actual pixel dump):**

```json
{
  "image_id": "vehide_00123.jpg",
  "width": 1600,
  "height": 1200,
  "instances": [
    {"class_label": "dent", "bbox": [412, 355, 180, 140], "scale": "medium"},
    {"class_label": "scratch", "bbox": [600, 210, 90, 30], "scale": "small"}
  ]
}
```

With 13,945 images and 32,000+ instances, the dataset averages approximately 2.3 damage instances per image, indicating that most images contain multiple co-occurring damage regions rather than a single isolated defect - consistent with real-world accident photographs, and consistent with the orchestrator needing to fan a single claim image's detections out to potentially multiple severity/report entries.

### 3.2 Target Variable / Class Taxonomy and Mapping

The **target variable** for the Damage Agent is the per-instance damage-class label (a multi-label, multi-instance detection target, not a single image-level label); the **target variable** for the Severity Agent is the derived Minor/Moderate/Severe category (Section 6.7).

VehiDE's 8 native categories map onto our project's 6 target classes (dent, scratch, crack, broken lamp, flat tyre, shattered glass) as follows; two VehiDE categories that do not map cleanly onto our target taxonomy will be excluded or merged during preprocessing.

| **Project class (target label)** | **Mapped VehiDE categor(y/ies)** | **Notes** |
| --- | --- | --- |
| Dent | Dent | Direct mapping |
| Scratch | Scratch | Direct mapping |
| Crack | Crack | Direct mapping |
| Broken lamp | Broken lamp | Direct mapping |
| Flat tyre | Flat tyre | Direct mapping |
| Shattered glass | Glass shatter | Direct mapping |
| *(excluded)* | Remaining VehiDE categories outside our 6-class scope | Instances re-labelled as "background/other damage" or excluded from the training annotation file, not deleted from the source images, so the mapping is reversible |

As flagged as a risk in Milestone 1 (Section 10.1), we expect this distribution to be **long-tailed**: dents and scratches are the most frequent damage types in real-world claims, while flat tyres and shattered glass are comparatively rare. This 6-class taxonomy is also the controlled vocabulary shared with the Policy Agent (Section 7.2), so any change to it must be propagated to the policy clause tags as well. Exact per-class instance counts are computed in the EDA (Section 5.2) and directly inform the class-weighting and oversampling strategy (Section 8).

### 3.3 Image Characteristics

The following image-level attributes are profiled as part of the EDA (Section 5.3) and are listed here as part of the dataset's descriptive feature set:

- Resolution (min/max/median width and height) - relevant because the Damage Agent's YOLO model requires images to be resized to a fixed input resolution (640 px) and very low-resolution images may need to be filtered.
- Aspect ratio - determines whether letterboxing or direct resizing is more appropriate.
- File format and colour mode (RGB vs. grayscale/CMYK anomalies).
- Lighting/exposure histogram spread - anticipates the domain-shift risk discussed in Milestone 1 (Section 10.3), since VehiDE images are closer to studio/dealer-style photography than to policyholder phone photographs submitted through the FastAPI ingestion endpoint.

---

## 4. Dataset Quality Assessment

### 4.1 Missing and Corrupt Data

The following automated checks will be run over the full VehiDE image and annotation set before any training occurs:

- **Orphan images**: images with no corresponding annotation entry (dropped from the training set, logged for manual review).
- **Orphan annotations**: annotation entries referencing an image file that is missing or unreadable (dropped, logged).
- **Corrupt files**: images that fail to open/decode (e.g., truncated JPEGs) will be identified with a batch integrity check and excluded.
- **Empty/degenerate boxes**: bounding boxes with zero or negative width/height, or coordinates outside the image bounds, will be corrected where possible (clipped to image bounds) or discarded if not recoverable.

### 4.2 Annotation Inconsistencies and Noise

Because VehiDE was annotated across multiple annotators to cover 8 damage categories at scale, some annotation noise is expected and will be characterised as follows:

- **Boundary looseness**: sampling a stratified subset (~300 images) for manual visual inspection to check whether bounding boxes tightly enclose the damage or are noticeably loose, which affects mAP@50-95 more than mAP@50.
- **Class ambiguity**: dent vs. crack and scratch vs. paint-chip boundaries are known to be subjective in the car-damage literature; a confusion audit will be performed on the manually inspected subset to quantify how often two team members disagree with the provided label.
- **Occlusion and multi-instance overlap**: overlapping bounding boxes for adjacent damage regions will be flagged, since these can destabilise non-max suppression during training if not handled consistently.
- **Privacy-sensitive content**: any images containing legible license plates or bystander faces will be flagged; since the dataset is used only for offline model training (not redistributed as raw images), no additional masking is required for training, but any example images used in the public demo or in this report will be manually checked and, if necessary, blurred.

### 4.3 Duplicate and Near-Duplicate Images

Exact and near-duplicate images are a common issue in scraped/aggregated damage datasets and pose a direct data-leakage risk if a duplicate ends up in both the training and test splits. The following two-stage deduplication will be applied prior to splitting:

1. **Exact duplicates** - detected via a cryptographic hash (e.g., MD5/SHA-256) of the raw image bytes.
2. **Near-duplicates** - detected via perceptual hashing (pHash) with a similarity threshold tuned on a manually verified sample, to catch resized, recompressed, or lightly cropped copies of the same underlying photograph (a known issue when the same accident is photographed from slightly different angles or re-uploaded).

Any duplicate cluster identified will be collapsed to a single representative image **before** the train/validation/test split is created, which is a prerequisite for the leakage prevention described in Section 6.2.

---

## 5. Adequacy Evaluation and Augmentation Strategy

With 13,945 images and 32,000+ instances, VehiDE is large enough to fine-tune a YOLOv8/YOLOv11 model to convergence for the 6 target classes, and is comparable in scale to datasets used in published YOLO-based vehicle-damage studies cited in Milestone 1 (Section 3.1). However, the adequacy assessment identifies two specific gaps:

- **Class-level adequacy**: Because the distribution is expected to be long-tailed (Section 3.2), the minority classes (flat tyre, shattered glass) may have too few instances to reach the &ge;0.65 per-class F1 target defined in Milestone 1 (Section 4.1) without intervention. Mitigation: (a) targeted oversampling of minority-class images during training, (b) class-weighted loss, and (c) supplementing minority classes with additional labelled examples drawn from the Car Damage Severity and COCO Car Damage datasets where their classes overlap.
- **Domain adequacy**: VehiDE images are closer to studio/dealer photography than to real policyholder phone photographs (Milestone 1, Section 10.3). To narrow this gap without collecting a new dataset, we will apply augmentation that simulates phone-camera conditions - brightness/contrast jitter, motion blur, perspective warp, and JPEG compression artefacts - and will additionally collect a small manually-curated stress-test set (~30-50 images) of realistic, non-studio claim-style photographs for final held-out evaluation only (never used in training).

A third, architecture-driven adequacy question is whether the dataset lets us exercise the **human-review escalation path** shown in the multi-agent design: the Damage/Severity Agents' low-confidence outputs must be routed to the escalation queue rather than passed on to the Report Agent. We will therefore deliberately retain a small held-out subset of genuinely ambiguous VehiDE images (occluded damage, borderline severity) specifically to validate that the orchestrator's confidence threshold triggers escalation as intended, rather than tuning the dataset to only contain "easy" cases.

If, after the class-level EDA (Section 5.2), any target class has fewer than a workable minimum of instances, that class will be expanded via the supplementary datasets identified in Section 2.2 rather than via purely synthetic image generation, since synthetic damage imagery risks introducing an additional domain gap.

---

## 6. Train / Validation / Test Split Strategy

### 6.1 Splitting Approach

The deduplicated VehiDE image set is split at the **image level** (not the instance level) into:

| **Split** | **Proportion** | **Purpose** |
| --- | --- | --- |
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning, early stopping, checkpoint selection |
| Test | 15% | Final, untouched evaluation reported against Milestone 1's Section 4.1 metrics |

**Justification for split strategy**: The split is **stratified by damage class** (multi-label stratification, assigning each image to a split based on its full label set rather than a single dominant class, since many images contain multiple co-occurring instances/classes per Section 3.1) so that the proportion of each of the 6 target classes is preserved across train/validation/test as closely as possible, avoiding systematic starvation of any rare class in a given split. A held-out ambiguous subset (Section 8.2) is carved out from within this split structure specifically to exercise the escalation path, rather than being treated as a fourth split.

**Number of samples per split**: applying the 70/15/15 ratio to the deduplicated corpus (Section 6.2) gives an approximate working target of ~9,760 train / ~2,090 validation / ~2,090 test images (from the pre-cleaning total of 13,945); the exact post-deduplication, post-cleaning counts per split and per class are computed by the split script and logged in the run manifest (Section 12) rather than fixed in advance, since they depend on how many images are dropped/collapsed in Sections 6.1-6.2.

### 6.2 Leakage Checks

Because VehiDE may contain multiple photographs of the same physical vehicle/accident taken moments apart, image-level splitting alone is not sufficient to guarantee independence between splits. The following explicit checks are performed **after** the split is created:

- **Perceptual-hash cross-check**: the same pHash-based near-duplicate detector used in Section 6.2 is re-run *across* splits (train vs. validation, train vs. test, validation vs. test) to confirm zero near-duplicate pairs crossing a split boundary.
- **Filename/metadata clustering**: sequential filename patterns or embedded EXIF metadata (where available) are used to detect burst-photographed sequences of the same incident, and any such cluster is kept entirely within a single split.
- **Supplementary dataset overlap**: the cross-dataset check from Section 7.4 additionally confirms none of CarDD, COCO Car Damage, or Car Damage Severity duplicates a VehiDE image before those datasets are used for class-expansion or calibration, to avoid cross-dataset leakage into the test split.

Any detected cross-split duplicate is resolved by removing it from all but one split, with a documented resolution log.

---

## 10. Final Prepared Dataset

### 10.1 Expected Final Size and Schema per Agent

The final, training-ready artefacts produced by this milestone, per agent, are:

| **Agent** | **Final dataset** | **Schema** |
| --- | --- | --- |
| Damage Agent | Deduplicated, cleaned, remapped VehiDE (+ CarDD segmentation subset) in YOLO format, split 70/15/15 | Section 3.1 schema, post-remapping to 6 classes (Section 3.2), YOLO-format labels (Section 6.4) |
| Severity Agent | Car Damage Severity dataset (calibration set) + derived area-ratio labels on the VehiDE splits | `{image_id, instance_id, bbox_area_ratio, severity_label}` (Section 6.7) |
| Policy Agent | 5 synthetic policy PDFs, chunked and embedded into a ChromaDB collection, served via the FastMCP retrieval tool | `{chunk_id, source_doc, clause_id, text, damage_classes[], is_distractor}` (Section 3.4) |
| Report Agent (evaluation) | 50 paired incident-description/image records (incl. escalation-path subset) + 20-sample faithfulness test set | `{incident_description, image_ref, ground_truth_damage_classes}` (Section 3.4) |

**Feature count (final, post-preprocessing):** Damage Agent records carry the 9 raw schema fields from Section 3.1 (image-level + instance-level) mapped down to the project's 6-class taxonomy, with no additional engineered vision features. The Severity Agent adds 1 engineered feature (`bbox_area_ratio`) on top of those, driving 1 target label (`severity_label`). The Policy Agent's indexed records carry the 6 chunk-schema fields from Section 3.4, with the `text` field embedded into a fixed-dimension vector by the chosen bi-encoder (Section 6.6) rather than expanded into additional named features.

The exact final image/instance counts per split, per class (post-deduplication and post-cleaning) will be logged as the output of the EDA/preprocessing notebook run (Section 5, Section 6) and recorded in the pipeline's run manifest (Section 12), rather than hard-coded in this report, so the numbers stay accurate as the pipeline is re-run.

### 7.3 Design Justification

- **Why synthetic and not scraped real policies**: Real insurer documents are proprietary and cannot legally be redistributed or used to train/evaluate an academic system; a fully synthetic, team-authored corpus avoids IP risk while still exercising the Policy Agent's retrieval tool against realistic policy language.
- **Why 5 documents rather than 1**: A single policy document would make retrieval trivially easy (only one candidate document to search) and would not test the retriever's ability to disambiguate between similar clauses phrased differently across documents.
- **Why deliberately vary phrasing and inject distractors**: This directly addresses the RAG-faithfulness risk identified in Milestone 1 (Section 10.4) - without this variation, reported retrieval precision would be optimistic relative to what a system will face with a real, arbitrarily-formatted policy PDF uploaded by an end user.

---

## 8. Cross-Agent Dataset Alignment in the Multi-Agent Architecture

Moving from a single sequential pipeline (Milestone 1) to a multi-agent orchestration changes *how* the vision and text sub-tasks are connected, even though the underlying datasets are unchanged. In the current design, the Damage, Severity, Policy, and Report Agents are independent LangGraph nodes that do not share raw features - they must instead be connected through explicit, versioned interface contracts maintained by the orchestrator.

<p align="center">
<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/Archive/insurance_claim_agent_architecture_overview.png" width="620">
</p>

- **Shared label taxonomy**: The 6 damage classes used to annotate VehiDE (Section 3.2) remain the *same* controlled vocabulary used to tag clauses in the synthetic policy documents (Section 7.1). This shared taxonomy is the join key between the Damage Agent's output and the Policy Agent's retrieval query text.
- **Orchestrator state object as the data interface**: Rather than one stage handing a JSON blob directly to the next, every agent reads from and writes to a single LangGraph state object for the claim, with fields such as `image_ref`, `detections[]` (class, bbox, confidence), `severity`, `retrieved_clauses[]`, `report_draft`, and `escalation_flag`. This state schema must be defined and versioned as part of dataset preparation, since a change to any agent's output fields is effectively a schema-migration event for every downstream agent.
- **MCP tool I/O contracts**: Because the Damage detection model, PDF/VIN lookup, and Policy retrieval are exposed as MCP tools (via FastMCP) rather than called as internal functions, each tool needs a fixed, documented request/response schema (e.g., the Policy Agent's retrieval tool takes `{damage_classes: [...], top_k: int}` and returns a list of `{clause_id, text, damage_classes, source_doc}` objects). The clause-to-damage-class lookup table from Section 7.1 doubles as the fixture data used to unit-test this tool contract in isolation, before the orchestrator is wired up end-to-end.
- **Evaluation-time pairing**: The 50 synthetic incident descriptions (Section 7.2), including the escalation-path subset, are explicitly paired with specific VehiDE test images so that end-to-end evaluation can trace a single test case through the full agent graph (Damage &rarr; Severity &rarr; Policy &rarr; Report, or Damage/Severity &rarr; Human Review), enabling the ablation comparison planned in Milestone 1 (Section 4.3).
- **No shared raw features**: Because the agents do not share raw input features (pixels vs. text), alignment is enforced entirely at the *schema/interface* level described above, consistent with the modular-over-monolithic rationale already established for this project (Multi-Agent Architecture, Section 5).

---

## 9. Agent-Specific Dataset Requirements

### 9.1 Damage & Severity Agents - Labeling and Annotation Requirements

- **Annotation format**: VehiDE annotations will be converted from their native format to YOLO format (normalised `class x_center y_center width height` per line) for the Damage Agent's detection training, and to a segmentation-mask format for the CarDD-based supplementary segmentation fine-tuning.
- **Label verification**: A random 5% sample of converted annotations will be visually re-rendered (boxes drawn over images) and manually checked post-conversion to catch coordinate/axis conversion bugs before large-scale training.
- **Class remapping**: The 8-to-6 category mapping defined in Section 3.2 must be applied consistently across every annotation file; this mapping will be implemented as a single versioned lookup table/script rather than manual edits, to keep the process auditable and reproducible.
- **Severity labels**: Since Minor/Moderate/Severe labels are not natively provided by VehiDE, they will be *derived* by the Severity Agent (bounding-box area &divide; visible-vehicle-surface area, per Milestone 1 Section 7) and then calibrated against the Car Damage Severity dataset's human-provided labels, per Milestone 1 Section 10.2.

### 9.2 Policy Agent - Document Preparation, Chunking, and Vector Store

- **Document preparation**: Each synthetic policy PDF is parsed to plain text with section/clause boundaries preserved (heading-aware extraction), so that chunk boundaries respect clause boundaries rather than splitting a clause mid-sentence.
- **Chunking strategy**: Following the literature findings summarised in Milestone 1 (Section 3.2), chunks of 200-400 tokens with overlap (~10-15%) will be used, since smaller, overlapping chunks were found to outperform large-chunk retrieval for clause-level recall in legal/insurance-style documents.
- **Vector database options explored**:

| **Option** | **Consideration** |
| --- | --- |
| FAISS (local, in-memory/on-disk index) | Lightweight, no external service dependency, well-suited to a Docker Compose single-machine deployment; billion-scale similarity search techniques [6] are far beyond this project's needs but the same library scales down cleanly. |
| ChromaDB | Simple persistence and metadata filtering (e.g., filter by damage-class tag before retrieval), useful given the clause-to-damage-class ground truth from Section 7.1. Selected as the project's long-term/vector memory store in the multi-agent architecture. |
| Managed vector DB (e.g., Pinecone) | Rejected for this project - introduces an external paid dependency and network latency not justified for a corpus of only 5 documents, and conflicts with the project's fully self-hostable, open-source component mapping. |

  Given the small corpus size (5 documents), the team will use ChromaDB (with FAISS as a fallback for local development) as the vector store backing the Policy Agent's retrieval tool, and will only reconsider a managed option if retrieval evaluation reveals a need for production-scale features.
- **Embedding model**: A bi-encoder sentence embedding model (e.g., a Sentence-BERT variant [7]) will be used, consistent with the finding that bi-encoders outperform BM25 sparse retrieval for semantic matching of insurance-style queries (Milestone 1, Section 3.2).
- **Tool exposure**: The chunked, embedded corpus is indexed once at ingestion time and served to the orchestrator exclusively through the Policy Agent's FastMCP-exposed retrieval tool (Section 8), so the dataset artefact that must be versioned is not just the raw PDFs but the resulting Chroma collection/index itself.

### 9.3 Report Agent - Prompt Structuring and Context Length

The Report Agent uses a pre-trained, self-hosted LLM (served via Ollama/vLLM behind a LiteLLM gateway - Llama 3, Mistral, or Qwen2.5) via prompting/grounding rather than fine-tuning (Milestone 1, Section 7, Stage 3), so classic instruction-response fine-tuning dataset requirements do not directly apply. The relevant dataset-adjacent requirements for this agent are:

- **Prompt structuring**: The prompt template combines (a) the structured detection/severity fields from the orchestrator's state object (Section 8), (b) the top-k retrieved policy clauses returned by the Policy Agent's MCP tool, and (c) a fixed instruction block specifying the required report sections (damage summary table, severity, applicable coverage, recommended next steps) and the explicit "state not covered under retrieved policy" fallback instruction from Milestone 1 (Section 10.5).
- **Token length considerations**: With chunk sizes of 200-400 tokens and top-k retrieval (k to be tuned, likely 3-5), the retrieved-context portion of the prompt is bounded to roughly 1,000-2,000 tokens, comfortably within the context window of the self-hosted open-source LLMs under consideration; this bound will be validated empirically once k is finalised during pipeline development.
- **Evaluation dataset**: The 20-sample faithfulness test set and the 50 synthetic incident/image pairs (Section 7.2) double as the "dataset" against which prompt structuring choices are evaluated, since no separate fine-tuning corpus is being built for this agent.

### 9.4 Orchestrator Memory - Session-State and Long-Term Memory Data Requirements

The multi-agent architecture introduces a memory layer (Redis for short-term session state, ChromaDB for long-term/vector memory) that did not exist as a distinct data concern in the Milestone 1 single-pipeline framing, and which therefore has its own preparation requirements:

- **Short-term session-state schema (Redis)**: Every in-flight claim is represented by the state object introduced in Section 8 (`claim_id`, `image_ref`, `detections[]`, `severity`, `retrieved_clauses[]`, `report_draft`, `escalation_flag`, timestamps). This schema must be defined and versioned before any agent code is written against it, since Redis itself enforces no schema and silent field drift between agents would be very difficult to debug.
- **Long-term/vector memory (ChromaDB)**: For this milestone, long-term memory is scoped to the Policy Agent's clause index (Section 9.2). The schema is designed to be extensible to a second collection of past-claim embeddings for future case-similarity retrieval, but populating that second collection is explicitly **out of scope** for the current milestone and is noted here only so the vector-store schema is not designed in a way that would block it later.
- **AgentOps trace data**: Every agent invocation (tool calls, latency, confidence score, escalation decisions) is logged through the Langfuse/OpenTelemetry AgentOps layer. While this trace data is not "training data" in the conventional sense, it is a required reproducibility artefact: the escalation-path test cases from Sections 5 and 7.2 are only verifiable if the corresponding trace log confirms that the orchestrator actually routed them to the human review queue rather than to the Report Agent.

### 9.5 Speech and Fine-Tuning Tasks - Applicability

This project involves no audio/speech modality, so audio quality, transcription alignment, and sampling-rate consistency requirements are **not applicable** and are omitted from this report. Similarly, since the Report Agent's LLM is used through prompting rather than weight fine-tuning (Section 9.3), formal instruction-response dataset curation for fine-tuning is **not applicable** at this milestone; this may be revisited in a later milestone only if evaluation results show prompting alone is insufficient to meet the accuracy/faithfulness targets in Milestone 1 (Section 4.2-4.3).

---

## 10. Preprocessing Pipeline and Reproducibility

To ensure every result in later milestones can be reproduced by any team member, the following pipeline will be implemented as a versioned, scripted sequence (not manual/ad-hoc steps), with each stage's output cached to disk and its parameters logged:

1. **Ingestion**: Download VehiDE via the Kaggle API into a fixed local/shared directory structure; record the dataset version/download date.
2. **Integrity checks**: Run the corrupt-file, orphan-image, and orphan-annotation checks (Section 4.1); log all dropped files with reasons.
3. **Deduplication**: Run exact-hash and perceptual-hash deduplication (Section 4.3); log all collapsed duplicate clusters.
4. **Class remapping**: Apply the 8-to-6 category lookup table (Section 9.1) to produce the project-specific annotation set.
5. **Annotation format conversion**: Convert to YOLO-format labels (and segmentation masks for the CarDD subset) for the Damage Agent; spot-check via visual re-rendering (Section 9.1).
6. **Stratified multi-label split**: Generate the train/validation/test split (Section 6.1) with a fixed random seed, and persist the resulting file lists so the split itself is reproducible and inspectable; carve out the ambiguous held-out subset used for escalation-path testing (Section 5).
7. **Leakage verification**: Run the cross-split perceptual-hash and metadata-clustering checks (Section 6.2); the pipeline will halt and flag for manual review if any cross-split duplicate is detected.
8. **Augmentation configuration**: Define and version the augmentation pipeline (brightness/contrast, motion blur, perspective warp, JPEG artefacts) as a configuration file applied only at training time, never baked into the stored dataset.
9. **Policy document preparation**: Author, review, and export the 5 synthetic policy PDFs (Section 7.1); parse and chunk them (Section 9.2); embed and index chunks into ChromaDB behind the Policy Agent's FastMCP retrieval tool.
10. **Agent/tool schema definition**: Define and version the LangGraph orchestrator state schema and the MCP tool request/response contracts (Section 8) in a single schema file, so every agent is developed against the same interface from the start.
11. **Evaluation artefacts**: Assemble the 50 paired incident-description/image samples (including the escalation-path subset) and the 20-sample faithfulness test set for later use in Milestone 1's evaluation plan (Sections 4.2-4.3).

Every step above will be implemented in version-controlled scripts/notebooks committed to the project's GitHub repository, with configuration values (split ratios, random seed, chunk size, overlap, k, state/tool schema version) stored in a single configuration file so that the entire data preparation pipeline can be re-run end-to-end from raw downloaded data to agent-ready artefacts.

---

## 11. Summary

This milestone identified VehiDE as the primary, verified, and appropriately-licensed dataset for the Damage and Severity Agents, supplemented by CarDD, COCO Car Damage, and the Car Damage Severity dataset for targeted auxiliary purposes. A concrete data-quality/EDA process, deduplication strategy, and leakage-safe stratified splitting strategy has been defined, including a deliberately-retained ambiguous subset for exercising the orchestrator's human-review escalation path. Because no public dataset exists for the Policy Agent's retrieval task, a fully synthetic, deliberately stress-tested policy corpus has been designed by the team, to be indexed into ChromaDB and served through a FastMCP tool contract. Moving to a multi-agent orchestration (LangGraph orchestrator, four specialist agents, Redis/ChromaDB memory, MCP-exposed tools) has not changed the underlying datasets required, but has added explicit interface-level requirements - a versioned orchestrator state schema and per-tool I/O contracts - that connect the vision and retrieval sub-tasks.

**Key observations from the data:** the corpus is large enough for the Damage Agent's core task but is expected to be class-imbalanced (long-tailed) and closer to studio photography than real claim photos, both of which are addressed through explicit, logged mitigations rather than left implicit; no natural severity ground truth exists at VehiDE's scale, making the Severity Agent's calibration step a load-bearing part of the pipeline; and the vision/text agents share no raw features, so correctness of the system depends on strict schema discipline (Section 7.2) as much as on data quality itself.

**Confirmation of training readiness:** once the pipeline in Section 12 is executed and its outputs committed, the datasets described in Section 10 are structured to be consumed directly by model training and RAG-index construction, with no further data preparation required, satisfying this milestone's stated objective.

### 13.2 Planned Activities for Milestone 3

- Execute the full preprocessing pipeline (Section 12) end-to-end and commit the run manifest, per-class EDA counts, and split file lists.
- Fine-tune the Damage Agent's YOLOv8/YOLOv11 detector on the prepared train split and evaluate against the metrics defined in Milestone 1, Section 4.1.
- Calibrate and validate the Severity Agent's area-ratio-to-severity mapping against the Car Damage Severity dataset (Milestone 1, Section 10.2).
- Build and evaluate the Policy Agent's retrieval pipeline (chunking/embedding/ChromaDB index) against the clause-to-damage-class ground truth and Retrieval Precision@3 target (Milestone 1, Section 4.2).
- Wire the Report Agent's prompt template and run the 20-sample faithfulness evaluation and the 50-sample end-to-end evaluation, including the escalation-path subset (Milestone 1, Section 8.3).
- Integrate all four agents behind the LangGraph orchestrator using the schema/tool contracts defined in Section 7.2, and validate the human-review escalation path end-to-end.

---

## References

[1] H. Scullen, "VehiDE: Vehicle Damage Detection Dataset," Kaggle, 2023. Available: [https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection)

[2] Van-Dung Hoang, Nhan T. Huynh et al., "VehiDE Dataset: New dataset for Automatic vehicle damage detection in Car insurance," IEEE Conference Publication / J. Inf. Telecommun., 2023.

[3] "Powering AI-driven car damage identification based on VeHIDE dataset," Journal article, Taylor & Francis Online, 2024.

[4] S. Wang et al., "CarDD: A New Dataset for Vision-Based Car Damage Detection," University of Science and Technology of China (USTC), 2023.

[5] "Coco Car Damage Detection Dataset," Kaggle. Available: https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset

[6] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535-547, 2021.

[7] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | | |
| Pranab Kumar Manna | | |
| Venkata Siva Kamal Guddanti | | |
| Anuj Gautam | | |
| | | |

---
