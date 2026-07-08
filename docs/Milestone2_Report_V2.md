
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

### 2.5 Alternatives Considered

As documented in Milestone 1 (Section 9.1), the vision-dataset landscape for vehicle damage was surveyed before selecting VehiDE as primary. Smaller or narrower alternatives - a standalone COCO-format set (~500 images) and a severity-only set (~2,300 images) - were considered but rejected as *primary* training sources because their scale is insufficient to fine-tune a YOLO detector to convergence on 6 classes without severe overfitting; both are retained instead as supplementary/calibration sources (Section 2.2). For the Policy Agent, the alternative of scraping or licensing real insurer policy documents was considered and rejected outright due to proprietary/IP constraints (Section 2.3), in favour of a fully synthetic corpus.

---

## 3. Dataset Description

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

### 3.4 Synthetic Policy Corpus - Structure and Schema

| **Attribute** | **Detail** |
| --- | --- |
| Format | Markdown/Word (authoring) exported to PDF (ingestion format) |
| Number of documents | 5 policies, ~8-12 pages each |
| Structure per document | Sections for collision coverage, comprehensive coverage, deductibles, exclusions, claim limits, third-party liability |
| Chunked unit (schema) | `{chunk_id, source_doc, clause_id, text, damage_classes[], is_distractor}` |
| Target/label | `damage_classes[]` per clause - the ground-truth retrieval label used to compute Retrieval Precision@3 (Milestone 1, Section 4.2) |
| Paired evaluation records | 50 `{incident_description, image_ref, ground_truth_damage_classes}` records (Section 6.6), of which ~5-8 are drawn from the ambiguous escalation-path subset (Section 8) |

**Data format**: Source documents are authored as Markdown, then exported to **PDF** (the production ingestion format); the indexed/queryable artefact is a **JSON-like chunk record** stored in ChromaDB (not a flat CSV/PDF at query time).

**Number of features**: Each indexed chunk has **6 fields** (`chunk_id`, `source_doc`, `clause_id`, `text`, `damage_classes[]`, `is_distractor`), of which `damage_classes[]` is the retrieval target/label. Each paired evaluation record has **3 fields** (`incident_description`, `image_ref`, `ground_truth_damage_classes`).

**Illustrative sample record (chunked policy clause, structure only - not reproduced from any real insurer document):**

```json
{
  "chunk_id": "policy_03_clause_014",
  "source_doc": "policy_03.pdf",
  "clause_id": "4.2.1",
  "text": "Damage to the windscreen or other glass components caused by an insured collision event is covered under comprehensive coverage, subject to the applicable deductible stated in Section 2.",
  "damage_classes": ["shattered_glass"],
  "is_distractor": false
}
```

A companion **distractor** clause illustrates the stress-testing described in Section 8.3: *"Flat tyre damage is covered only if resulting from an insured collision event, not from normal wear and tear"* - semantically related to the `flat_tyre` class but conditional, designed to test whether the retriever (and the Report Agent's faithfulness) correctly distinguishes conditional coverage from unconditional coverage.

---

## 4. Data Governance

### 4.1 Data Source & Licensing

All third-party datasets (VehiDE, CarDD, COCO Car Damage, Car Damage Severity) are owned by their respective original authors/publishers, not by the project team or IIT Madras, and are used strictly under their published research/educational or Kaggle community license terms (full detail in Section 2.4). No dataset is redistributed in raw form by this project; only derived artefacts (model weights, ChromaDB indices, evaluation reports) are produced and shared. The synthetic policy corpus is wholly owned by the project team with no external licensing constraint.

### 4.2 Privacy

Vehicle damage photographs may incidentally contain PII such as legible license plates or bystander faces. Any such images identified during annotation review (Section 5.6) will be flagged; since the datasets are used only for offline model training and are not redistributed as raw images, no additional masking is required for training itself, but any example images reproduced in this report, the public demo, or any presentation will be manually checked and blurred where necessary. At runtime, the production system does not retain user-submitted claim photographs or policy documents beyond the active session (Milestone 1, Section 11.2), and any internal test data containing PII will be anonymised via PII masking (Microsoft Presidio) or attribute shuffling before use, consistent with the two strategies defined in Milestone 1, Section 11.2. In a production deployment, data handling would need to comply with India's Digital Personal Data Protection (DPDP) Act and, where relevant, the GDPR.

### 4.3 Data Quality

Automated and manual validation is performed prior to any training use, covering corrupt/orphan files, degenerate bounding boxes, annotation looseness, class-label ambiguity, and duplicate/near-duplicate images. The concrete checks and their results are detailed in the EDA (Section 5) and the corresponding cleaning actions in Data Preprocessing (Section 6).

### 4.4 Ethics & Bias

If the training datasets (VehiDE, CarDD) are not representative of the full diversity of vehicle types, damage patterns, or photographic conditions encountered by real policyholders, the model may systematically underperform for certain groups - for example, if certain vehicle colours, body types, or damage patterns are underrepresented, detection rates may differ across policyholders, introducing indirect algorithmic bias into claim handling (Milestone 1, Section 11.1). As mitigation, a stratified error analysis across damage classes and, where metadata is available, across vehicle types will be performed in later milestones, and all generated reports will explicitly state that they are AI-generated preliminary assessments subject to human review. The known long-tail class imbalance (Section 5.2) is itself a bias risk and is addressed via class-weighting/oversampling (Section 8) rather than left unaddressed.

### 4.5 Reproducibility & Compliance

Every dataset used is version-pinned (download date and, where applicable, dataset version/hash recorded at ingestion - Section 6.1), every preprocessing step is implemented as a versioned, scripted pipeline rather than manual/ad-hoc edits (Section 6, Section 12), and every configuration value (split ratios, random seed, chunk size, overlap, top-k, schema version) is stored in a single configuration file so the entire pipeline can be re-run end-to-end from raw downloaded data to agent-ready artefacts. Licensing and copyright requirements for each source dataset are tracked in Section 2.4 and re-verified before any public release of the demo.

---

## 5. Exploratory Data Analysis (EDA)

The EDA is implemented as a versioned notebook (Section 12) run over the ingested, pre-cleaning VehiDE corpus, so that data-quality decisions in Section 6 are evidence-based rather than assumed. The specific analyses to be produced are:

### 5.1 Summary Statistics

Total image count, total instance count, instances-per-image distribution (mean/median, expected ≈2.3 per image based on the published dataset statistics - Section 3.1), and per-category instance counts across all 8 native VehiDE categories before remapping.

### 5.2 Class Distribution

Per-class (post-remapping, 6-class) instance and image counts, presented as a bar chart, to quantify the long-tail imbalance anticipated in Milestone 1 (Section 10.1) and Section 3.2 above. This distribution directly determines the class-weighting/oversampling parameters used in Section 8, and the workable-minimum-instance check described in Section 11.

### 5.3 Image-Level Feature Distributions

Histograms of resolution (width/height), aspect ratio, file format/colour mode, and exposure/brightness, per Section 3.3, used to decide the resizing strategy (Section 6.5) and to quantify the domain gap between VehiDE's studio-style photography and expected phone-camera claim photos (Milestone 1, Section 10.3).

### 5.4 Missing Value Analysis

Counts of **orphan images** (no corresponding annotation) and **orphan annotations** (referencing a missing/unreadable image file), reported as a percentage of the corpus, informing the drop/repair decisions in Section 6.1.

### 5.5 Duplicate Analysis

Exact-duplicate counts (cryptographic hash of raw image bytes) and near-duplicate counts (perceptual hash / pHash, catching resized, recompressed, or lightly cropped copies of the same underlying photograph), reported as the number and size of duplicate clusters found, informing the deduplication step in Section 6.2 and the leakage checks in Section 9.2.

### 5.6 Annotation Noise / Outlier Analysis

- **Degenerate boxes**: bounding boxes with zero/negative width or height, or coordinates outside image bounds, counted and flagged as outliers.
- **Boundary looseness**: a stratified manual sample (~300 images) is visually inspected to assess whether bounding boxes tightly enclose the damage, since loose boxes disproportionately affect mAP@50-95.
- **Class ambiguity**: a confusion audit on the same manually inspected subset quantifies how often two team members disagree with the provided label for the known-subjective dent/crack and scratch/paint-chip boundaries.
- **Occlusion / overlap outliers**: overlapping bounding boxes for adjacent damage regions are flagged, since these can destabilise non-max suppression during training if not handled consistently.

No formal correlation analysis is applicable to VehiDE's categorical/spatial annotation schema; the closest analogue - co-occurrence of damage classes within the same image - is captured instead by the multi-label statistics in Section 5.1/5.2 and used directly by the multi-label stratified split in Section 9.1.

### 5.7 Visualizations

The EDA notebook (Section 12) will produce the following charts, each tied to a specific decision made elsewhere in this report:

| **Visualization** | **What it shows** | **Feeds into** |
| --- | --- | --- |
| Bar chart | Per-class instance/image counts (6 project classes, pre- and post-remapping) | Class-weighting strategy (Section 8.2) |
| Histogram | Image resolution (width/height) and instances-per-image distribution | Resize/letterbox decision (Section 6.5) |
| Histogram | Brightness/exposure distribution | Domain-shift assessment (Section 3.3) |
| Bar chart | Duplicate-cluster size distribution (exact vs. near-duplicate) | Deduplication scope (Section 6.2) |
| Heatmap | Damage-class co-occurrence matrix (which classes appear together in the same image) | Multi-label stratified split design (Section 9.1) |
| Scatter plot | Bounding-box area ratio vs. human-assigned severity label (Car Damage Severity dataset) | Severity calibration (Section 6.7) |
| Bar chart | Class-level confusion counts from the manual annotation audit (Section 5.6) | Label-correction scope (Section 6.3) |

---

## 6. Data Preprocessing

### 6.1 Cleaning: Missing, Corrupt, and Degenerate Data

The following automated checks are run over the full VehiDE image and annotation set, using the findings from Section 5.4/5.6, before any training occurs:

- **Orphan images** (no corresponding annotation entry) are dropped from the training set and logged for manual review.
- **Orphan annotations** (referencing a missing/unreadable image file) are dropped and logged.
- **Corrupt files** (e.g., truncated JPEGs that fail to open/decode) are identified with a batch integrity check and excluded.
- **Empty/degenerate boxes** (zero/negative width-height, or out-of-bounds coordinates) are corrected where possible (clipped to image bounds) or discarded if not recoverable.

### 6.2 Duplicate Removal

Exact and near-duplicate images identified in Section 5.5 pose a direct data-leakage risk if a duplicate ends up in both the training and test splits. Any duplicate cluster identified is collapsed to a single representative image **before** the train/validation/test split is created (Section 9), which is a prerequisite for leakage prevention.

### 6.3 Label Correction and Class Remapping (Encoding)

The 8-to-6 category mapping defined in Section 3.2 is applied consistently across every annotation file, implemented as a single versioned lookup table/script (not manual edits), so the process is auditable and reproducible. Instances outside the 6-class scope are re-labelled as "background/other damage" rather than deleted, so the mapping is reversible. Corrections identified during the annotation-noise audit (Section 5.6) - e.g., genuinely mislabelled dent/crack instances agreed on by both reviewing team members - are applied as a documented correction log, not silent edits.

### 6.4 Annotation Format Conversion

VehiDE annotations are converted from their native format to YOLO format (normalised `class x_center y_center width height` per line) for the Damage Agent's detection training, and to a segmentation-mask format for the CarDD-based supplementary segmentation fine-tuning. A random 5% sample of converted annotations is visually re-rendered (boxes drawn over images) and manually checked post-conversion to catch coordinate/axis conversion bugs before large-scale training.

### 6.5 Image Standardisation / Normalisation

Based on the resolution/aspect-ratio profiling in Section 5.3, images are resized to the YOLO input resolution (640 px), using letterboxing where aspect-ratio variance is high to avoid distorting damage geometry; pixel values are normalised to the range expected by the YOLOv8/YOLOv11 backbone. Very low-resolution outlier images identified in the EDA are filtered out rather than upscaled.

### 6.6 Text Cleaning and Tokenisation (Policy Agent)

- **Text cleaning**: PDF-to-text extraction artefacts - page headers/footers, page numbers, hyphenation breaks across line wraps, and non-semantic whitespace/control characters - are stripped before chunking; clause numbering and section headings are normalised to a consistent format so they don't fragment embeddings.
- **Document preparation**: Each synthetic policy PDF is parsed to plain text with section/clause boundaries preserved (heading-aware extraction), so chunk boundaries respect clause boundaries rather than splitting a clause mid-sentence.
- **Chunking/tokenisation strategy**: Following the literature findings summarised in Milestone 1 (Section 3.2), chunks of 200-400 tokens with overlap (~10-15%) are used, since smaller, overlapping chunks were found to outperform large-chunk retrieval for clause-level recall in legal/insurance-style documents.
- **Embedding**: A bi-encoder sentence embedding model (e.g., a Sentence-BERT variant [7]) is used, consistent with the finding that bi-encoders outperform BM25 sparse retrieval for semantic matching of insurance-style queries (Milestone 1, Section 3.2).
- **Incident-description text**: The 50 synthetic incident descriptions (Section 3.4) receive the same whitespace/formatting cleaning as the policy text, but are not chunked (each is short enough to be used whole as a query string).

### 6.7 Feature Engineering: Derived Severity Labels

Minor/Moderate/Severe severity labels are not natively provided by VehiDE and are therefore an **engineered feature**: derived by the Severity Agent as bounding-box area &divide; visible-vehicle-surface area (Milestone 1, Section 7), then calibrated against the Car Damage Severity dataset's human-provided labels (Milestone 1, Section 10.2). This calibration is itself a preprocessing artefact (a fitted mapping/threshold) that must be produced and versioned before the Severity Agent can be evaluated.

### 6.8 Feature Selection

No explicit feature-selection step is applied to the vision data: the Damage Agent's YOLO backbone consumes raw pixels end-to-end rather than a hand-engineered tabular feature set, so there is no candidate feature set to prune. Feature selection *does* apply narrowly in two places: (a) the `scale` (small/medium/large) attribute reported by VehiDE (Section 3.1) is retained only as a diagnostic/EDA field and is deliberately **excluded** from the Severity Agent's input, since it would leak information correlated with the target severity label; and (b) for the Policy Agent, retrieval uses the full clause `text` embedding rather than hand-picked keyword features, so no feature selection is performed on the text side either - chunk size (Section 6.6) is the only tunable "selection" knob, and it is treated as a hyperparameter, not a feature-selection step.

### 6.9 Audio Preprocessing (Speech)

**Not applicable.** This project involves no audio/speech modality (Milestone 1, Section 1.3 scope; system design, Section 1) - all inputs are images (Damage/Severity Agents) or text/PDF (Policy/Report Agents). No sampling-rate, transcription-alignment, or audio-quality preprocessing is required at this or any future milestone unless the project scope changes.

---

## 7. Dataset Integration

### 7.1 Datasets Combined

Two integration problems exist in this project: (a) combining the *vision-side* datasets (VehiDE as primary, CarDD/COCO Car Damage/Car Damage Severity as supplementary, per Section 2) into a single consistent training/evaluation resource for the Damage and Severity Agents, and (b) aligning the *vision-side* output with the *text-side* Policy Agent corpus so the agents can be composed in the orchestrator graph.

### 7.2 Integration Methodology and Schema Alignment

- **Shared label taxonomy**: The 6 damage classes used to annotate VehiDE (Section 3.2) are the *same* controlled vocabulary used to tag clauses in the synthetic policy documents (Section 3.4) and to select supplementary-dataset instances used for class expansion (Section 8). This shared taxonomy is the join key between the Damage Agent's output and the Policy Agent's retrieval query text.
- **Annotation-format alignment**: CarDD's pixel-level segmentation masks and VehiDE's bounding boxes are aligned to a common per-instance record (Section 3.1 schema) with an optional `segmentation` field, so a single training loader can consume either annotation type for the corresponding YOLO task head.
- **Orchestrator state object as the data interface**: Rather than one stage handing a JSON blob directly to the next, every agent reads from and writes to a single LangGraph state object for the claim, with fields such as `image_ref`, `detections[]` (class, bbox, confidence), `severity`, `retrieved_clauses[]`, `report_draft`, and `escalation_flag`. This state schema is defined and versioned as part of dataset preparation, since a change to any agent's output fields is effectively a schema-migration event for every downstream agent.
- **MCP tool I/O contracts**: Because the Damage detection model and Policy retrieval are exposed as MCP tools (via FastMCP) rather than called as internal functions, each tool needs a fixed, documented request/response schema (e.g., the Policy Agent's retrieval tool takes `{damage_classes: [...], top_k: int}` and returns a list of `{clause_id, text, damage_classes, source_doc}` objects). The clause-to-damage-class lookup table from Section 3.4 doubles as fixture data used to unit-test this tool contract in isolation.

### 7.3 Handling Conflicting Attributes

Where the same damage class is represented differently across sources (e.g., VehiDE's "glass shatter" vs. a differently-worded label in COCO Car Damage or the Car Damage Severity dataset), the project's 6-class taxonomy (Section 3.2) is treated as the canonical schema, and every supplementary dataset is remapped into it - never the reverse - so the Damage Agent always trains against one consistent label set regardless of source.

### 7.4 Deduplication After Merging

Since CarDD, COCO Car Damage, and the Car Damage Severity dataset are drawn from different sources than VehiDE, cross-dataset image overlap is checked using the same hashing approach as Section 6.2 before using any of them for the class-expansion or calibration purposes described in Section 8, to avoid cross-dataset leakage in addition to the within-VehiDE leakage checks in Section 9.2.

---

## 8. Data Augmentation

### 8.1 Vision Augmentation Techniques

Brightness/contrast jitter, motion blur, perspective warp, and JPEG compression artefacts are applied to simulate phone-camera conditions. Class-weighted loss and targeted oversampling of minority-class images (flat tyre, shattered glass - per the long-tail distribution quantified in Section 5.2) are used at training time to address per-class F1 risk (Milestone 1, Section 4.1).

**Example**: a single VehiDE training image of a dented rear bumper, photographed under even studio lighting, is expanded at training time into multiple augmented views - e.g., one variant with brightness reduced ~30% and mild motion blur applied (simulating a hastily taken evening phone photo), and a second variant with a small perspective warp and JPEG re-compression artefacts (simulating an off-angle, re-uploaded phone photo) - while the underlying bounding-box/class annotation is transformed consistently with the image so the label remains correct.

### 8.2 Rationale

Two adequacy gaps motivate this augmentation strategy:

- **Class-level adequacy**: minority classes may have too few instances to reach the &ge;0.65 per-class F1 target (Milestone 1, Section 4.1) without intervention; addressed via oversampling, class-weighted loss, and supplementing minority classes with additional labelled examples drawn from the Car Damage Severity and COCO Car Damage datasets where their classes overlap (Section 7).
- **Domain adequacy**: VehiDE images are closer to studio/dealer photography than to real policyholder phone photographs (Milestone 1, Section 10.3); the vision augmentations above narrow this gap without collecting a new dataset. A small manually-curated stress-test set (~30-50 images) of realistic, non-studio claim-style photographs is additionally collected for final held-out evaluation only (never used in training, and therefore not part of the augmented training set).

All augmentation is defined and versioned as a **configuration file applied only at training time**, never baked into the stored dataset, so the exact number of augmented samples seen by the model is a function of the training schedule (epochs &times; augmentation probability) rather than a fixed pre-generated count; this configuration will be finalised and logged once training begins in Milestone 3.

A third adequacy question is whether the dataset lets us exercise the **human-review escalation path** shown in the multi-agent design. Rather than augmenting this away, a small held-out subset of genuinely ambiguous VehiDE images (occluded damage, borderline severity) is deliberately retained (not "fixed" via augmentation) specifically to validate that the orchestrator's confidence threshold triggers escalation as intended.

If, after the class-level EDA (Section 5.2), any target class has fewer than a workable minimum of instances, that class will be expanded via the supplementary datasets identified in Section 2.2 rather than via purely synthetic image generation, since synthetic damage imagery risks introducing an additional domain gap.

### 8.3 Policy Corpus Stress-Testing (Text-Side Augmentation)

For the Policy Agent, the analogue of augmentation is deliberate linguistic variation rather than image transforms:

- Clause phrasing is deliberately varied across the 5 policies (synonyms, different sentence structures, negations) to stress-test the embedding model's ability to retrieve semantically equivalent clauses under surface variation.
- Each policy includes at least five distractor clauses per damage class - clauses that are semantically related but do not grant coverage (exclusions, sub-limits, conditional clauses) - so retrieval precision is not artificially inflated by an overly easy corpus.
- **Rationale**: without this variation, reported retrieval precision would be optimistic relative to what the system will face with a real, arbitrarily-formatted policy PDF uploaded by an end user (Milestone 1, Section 10.4).

**Example**: the shattered-glass coverage clause shown in Section 3.4 (*"Damage to the windscreen or other glass components caused by an insured collision event is covered..."*) is deliberately paired, within the same policy, with the conditional flat-tyre distractor clause also shown in Section 3.4 (*"...covered only if resulting from an insured collision event, not from normal wear and tear"*). Across the 5 policies, the same underlying coverage concept is additionally re-phrased with different sentence structure and clause numbering, so the retriever cannot rely on surface-level phrase matching alone.

---

## 9. Dataset Splitting

### 9.1 Split Ratio and Strategy

The deduplicated VehiDE image set is split at the **image level** (not the instance level) into:

| **Split** | **Proportion** | **Purpose** |
| --- | --- | --- |
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning, early stopping, checkpoint selection |
| Test | 15% | Final, untouched evaluation reported against Milestone 1's Section 4.1 metrics |

**Justification for split strategy**: The split is **stratified by damage class** (multi-label stratification, assigning each image to a split based on its full label set rather than a single dominant class, since many images contain multiple co-occurring instances/classes per Section 3.1) so that the proportion of each of the 6 target classes is preserved across train/validation/test as closely as possible, avoiding systematic starvation of any rare class in a given split. A held-out ambiguous subset (Section 8.2) is carved out from within this split structure specifically to exercise the escalation path, rather than being treated as a fourth split.

**Number of samples per split**: applying the 70/15/15 ratio to the deduplicated corpus (Section 6.2) gives an approximate working target of ~9,760 train / ~2,090 validation / ~2,090 test images (from the pre-cleaning total of 13,945); the exact post-deduplication, post-cleaning counts per split and per class are computed by the split script and logged in the run manifest (Section 12) rather than fixed in advance, since they depend on how many images are dropped/collapsed in Sections 6.1-6.2.

### 9.2 Leakage Prevention Measures

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

### 10.2 Summary of Preprocessing Completed

By the end of this milestone, the following will have been executed and version-logged for every dataset in scope: integrity/orphan/corrupt-file checks (Section 6.1), exact + near-duplicate collapsing (Section 6.2), 8-to-6 class remapping and any manual label corrections (Section 6.3), annotation format conversion to YOLO/segmentation formats with 5% manual re-verification (Section 6.4), image resizing/normalisation to the YOLO input contract (Section 6.5), policy-document parsing, chunking, and embedding (Section 6.6), and derivation of the severity feature with calibration (Section 6.7).

### 10.3 Readiness for Model Training

Once the steps above are executed and their logs committed to the repository, another team will be able to (a) point the Damage Agent's training script at the versioned train/validation/test file lists (Section 9.1) and begin YOLO fine-tuning immediately, (b) load the calibrated severity mapping directly into the Severity Agent, (c) query the Policy Agent's FastMCP tool against the already-indexed ChromaDB collection without re-running ingestion, and (d) run end-to-end evaluation against the paired incident/image and faithfulness test sets - all without any additional data preparation, satisfying the stated objective of this milestone.

---

## 11. Challenges Encountered

- **Data availability**: No public dataset pairs insurance policy text with vehicle damage annotations, and no single public vision dataset provides both bounding boxes and pixel-level masks at scale - requiring one primary dataset (VehiDE) plus three narrowly-scoped supplementary datasets (Section 2.2) instead of one comprehensive source.
- **Data quality problems**: VehiDE was annotated across multiple annotators at scale, so annotation looseness, class-boundary ambiguity (dent vs. crack, scratch vs. paint-chip), and occluded/overlapping boxes are expected and must be characterised (Section 5.6) rather than assumed away.
- **Privacy concerns**: incidental PII (license plates, bystander faces) may appear in vehicle photographs and must be identified and, for any externally-shared example, blurred (Section 4.2).
- **Licensing constraints**: VehiDE's non-commercial research license shapes what can and cannot be done with any public demo deployment (Section 2.4); real insurer policy documents could not be used at all, forcing the fully synthetic Policy Agent corpus (Section 2.3).
- **Class imbalance**: the expected long-tailed distribution (dents/scratches over-represented, flat tyres/shattered glass under-represented) directly threatens the per-class F1 target and requires explicit mitigation (Section 8.2) rather than being left to the model to learn unaided.
- **Missing labels**: severity (Minor/Moderate/Severe) is not natively present in VehiDE and must be derived and calibrated against a separate, much smaller dataset (Section 6.7), introducing an additional calibration dependency.
- **Integration challenges**: the vision (image/bbox) and text (policy clause) datasets share no raw features, so alignment has to be enforced entirely at the schema/interface level (shared taxonomy, orchestrator state object, MCP tool contracts - Section 7), which is more design overhead than a single-pipeline framing would have required.
- **Limitations that remain**: domain shift between VehiDE's studio-style photography and real policyholder phone photos is only partially mitigated by augmentation (Section 8.1) and will need to be re-assessed once real or realistic stress-test images are evaluated; the synthetic policy corpus, despite deliberate stress-testing (Section 8.3), remains an approximation of real insurer documents and retrieval performance reported in later milestones should be read with that caveat.

---

## 12. Deliverables Produced

- **Cleaned dataset**: VehiDE (and supplementary datasets) after integrity checks, deduplication, and label correction (Section 6.1-6.3).
- **Processed dataset**: Class-remapped, YOLO-format (and segmentation-format) annotation files, resized/normalised images, ready for direct ingestion by the training pipeline (Section 6.4-6.5).
- **Train/Validation/Test datasets**: Stratified, leakage-checked 70/15/15 split file lists, with the ambiguous escalation-path subset separately identified (Section 9).
- **Derived severity dataset**: Calibrated bounding-box-area-ratio-to-severity mapping and labelled outputs for the Severity Agent (Section 6.7).
- **Synthetic policy corpus**: 5 authored policy PDFs, their chunked/embedded ChromaDB collection, and the clause-to-damage-class ground-truth lookup table (Section 3.4, Section 6.6).
- **Evaluation fixtures**: 50 paired incident-description/image records (including the escalation-path subset) and the 20-sample faithfulness test set (Section 3.4).
- **Preprocessing scripts/notebooks**: The full versioned pipeline described in the run manifest below (ingestion, integrity checks, deduplication, remapping, format conversion, splitting, leakage verification, augmentation config, policy preparation, schema definition), committed to the project's GitHub repository.
- **Interface/schema documentation**: The LangGraph orchestrator state schema and MCP tool request/response contracts (Section 7.2), versioned in a single schema file.
- **Documentation**: This report, together with the data dictionary below.
- **Data dictionary**: The per-record schema tables in Section 3.1, Section 3.4, and Section 10.1.

**Preprocessing pipeline (reproducibility record):** implemented as a versioned, scripted sequence, with each stage's output cached to disk and its parameters logged: (1) ingestion of VehiDE via the Kaggle API with recorded dataset version/download date; (2) integrity checks (Section 6.1); (3) deduplication (Section 6.2); (4) class remapping (Section 6.3); (5) annotation format conversion (Section 6.4); (6) stratified multi-label split with a fixed random seed (Section 9.1); (7) leakage verification (Section 9.2); (8) augmentation configuration (Section 8.1); (9) policy document preparation, chunking, and indexing (Section 6.6); (10) agent/tool schema definition (Section 7.2); (11) assembly of evaluation artefacts (Section 3.4). Every configuration value (split ratios, random seed, chunk size, overlap, top-k, schema version) is stored in a single configuration file so the pipeline can be re-run end-to-end.

---

## 13. Summary and Next Steps

### 13.1 Summary

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
