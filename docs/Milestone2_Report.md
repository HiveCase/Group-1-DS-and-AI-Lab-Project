
---

<div align="center">


<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">


<h1 style="font-size:26em;">Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 2: Dataset Preparation</h2>

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
- [2. Data Sources: Identification and Verification](#2-data-sources-identification-and-verification)
  - [2.1 Primary Vision Dataset - VehiDE](#21-primary-vision-dataset---vehide)
  - [2.2 Supplementary Vision Datasets](#22-supplementary-vision-datasets)
  - [2.3 Ownership, Licensing, and Usage Constraints](#23-ownership-licensing-and-usage-constraints)
- [3. Detailed Dataset Description](#3-detailed-dataset-description)
  - [3.1 VehiDE - Structure and Feature Distribution](#31-vehide---structure-and-feature-distribution)
  - [3.2 Class Distribution and Imbalance](#32-class-distribution-and-imbalance)
  - [3.3 Image Characteristics](#33-image-characteristics)
- [4. Dataset Quality Assessment](#4-dataset-quality-assessment)
  - [4.1 Missing and Corrupt Data](#41-missing-and-corrupt-data)
  - [4.2 Annotation Inconsistencies and Noise](#42-annotation-inconsistencies-and-noise)
  - [4.3 Duplicate and Near-Duplicate Images](#43-duplicate-and-near-duplicate-images)
- [5. Adequacy Evaluation and Augmentation Strategy](#5-adequacy-evaluation-and-augmentation-strategy)
- [6. Train / Validation / Test Split Strategy](#6-train--validation--test-split-strategy)
  - [6.1 Splitting Approach](#61-splitting-approach)
  - [6.2 Leakage Checks](#62-leakage-checks)
- [7. Synthetic Dataset Generation](#7-synthetic-dataset-generation)
  - [7.1 Synthetic Insurance Policy Documents](#71-synthetic-insurance-policy-documents)
  - [7.2 Synthetic Incident Descriptions](#72-synthetic-incident-descriptions)
  - [7.3 Design Justification](#73-design-justification)
- [8. Cross-Task Dataset Alignment](#8-cross-task-dataset-alignment)
- [9. Task-Specific Dataset Requirements](#9-task-specific-dataset-requirements)
  - [9.1 Vision Task - Labeling and Annotation Requirements](#91-vision-task---labeling-and-annotation-requirements)
  - [9.2 RAG Task - Document Preparation, Chunking, and Vector Store](#92-rag-task---document-preparation-chunking-and-vector-store)
  - [9.3 LLM Report Generation - Prompt Structuring and Context Length](#93-llm-report-generation---prompt-structuring-and-context-length)
  - [9.4 Speech and Fine-Tuning Tasks - Applicability](#94-speech-and-fine-tuning-tasks---applicability)
- [10. Preprocessing Pipeline and Reproducibility](#10-preprocessing-pipeline-and-reproducibility)
- [11. Summary](#11-summary)
- [12. References](#12-references)

---

## 1. Introduction

Milestone 1 defined the problem, scope, and evaluation plan for the multimodal vehicle damage assessment system. This milestone focuses on identifying and preparing the data required by the three pipeline stages described in Section 7 of Milestone 1: the YOLO damage detection module, the RAG-based policy retrieval module, and the LLM report generator. This report documents the datasets selected, verifies their provenance and licensing, assesses their quality and adequacy, defines the train/validation/test split strategy, and describes the synthetic data that must be generated where no public dataset exists (insurance policy documents). It also specifies the concrete preprocessing steps required before ingestion into the model pipeline, so that the work is reproducible by any team member.

---

## 2. Data Sources: Identification and Verification

### 2.1 Primary Vision Dataset - VehiDE

The primary dataset selected for the damage detection module is **VehiDE (Vehicle Damage Detection Dataset)**, published on Kaggle by Hendrich Scullen [1] and described in the accompanying research paper [2].

| **Attribute** | **Detail** |
| --- | --- |
| Source | [Kaggle - VehiDE Dataset](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection/data) |
| Origin | Curated and published by the dataset authors for the paper "VehiDE Dataset: New dataset for Automatic vehicle damage detection in Car insurance" [2] |
| Size | 13,945 high-resolution images |
| Instances | 32,000+ labelled damage instances |
| Damage categories | 8 damage categories (a superset of the 6 classes in our project scope; see Section 3.2) |
| Supported tasks | Classification, object detection, instance segmentation, and salient object detection |
| Access method | Downloaded directly via the Kaggle API/CLI under our team Kaggle account, hashed and version-pinned locally so every member works from an identical copy |

We verified this source by cross-checking the dataset card against the peer-reviewed paper describing its construction [2] and against independent citations of the dataset in subsequent literature [3], confirming that the reported size (13,945 images, 32k+ instances) and category count are consistent across sources.

### 2.2 Supplementary Vision Datasets

Three supplementary datasets identified in Milestone 1 (Section 9.1) will be used for specific auxiliary purposes rather than as primary training data:

| **Dataset** | **Source** | **Role in this project** | **Verification status** |
| --- | --- | --- | --- |
| CarDD | USTC research release [4] | Supplementary pixel-level segmentation fine-tuning where VehiDE's bounding-box annotations are insufficient (e.g., irregularly shaped scratches/cracks) | Verified against the original CarDD paper; academic-use dataset |
| COCO Car Damage Detection | Kaggle (COCO-format) [5] | Small supplementary set for architecture comparison / sanity-checking the detection pipeline against a differently-annotated source | Verified via Kaggle dataset card; noted as a small (~500 image) set, used only for comparison, not primary training |
| Car Damage Severity Dataset | Kaggle | Calibration of the Minor/Moderate/Severe severity heuristic against human-labelled severity ground truth | Verified via Kaggle dataset card |

### 2.3 Ownership, Licensing, and Usage Constraints

- **VehiDE**: The dataset authors explicitly restrict usage to **non-commercial research and educational purposes**, with users bearing responsibility for appropriate use [2]. This is compatible with our project, which is an academic lab project and will be distributed as an open, non-commercial demonstration on Hugging Face Spaces. Any public deployment will carry attribution to the original authors [1][2] and a notice that the demo is for research/educational purposes only.
- **CarDD**: Released for academic research; attribution to the original paper is required in any derivative work [4].
- **COCO Car Damage Detection / Car Damage Severity**: Both are community-hosted Kaggle datasets; we will retain the Kaggle dataset page license terms and cite the original uploaders.
- **Ownership boundary**: No dataset used in this project is owned by our team or by IIT Madras; all are third-party datasets used under their respective research/educational licenses. No dataset contains personally identifiable information (license plates and faces, where visible, are treated as sensitive - see Section 4.2) that would create additional privacy obligations beyond standard research use.
- **Synthetic policy documents** (Section 7) are authored entirely by the team and are therefore fully owned by the project, with no third-party licensing constraints. This was in fact the reason Milestone 1 (Section 1.3, "Out of Scope") explicitly ruled out using real insurer policy documents, which are proprietary.

---

## 3. Detailed Dataset Description

### 3.1 VehiDE - Structure and Feature Distribution

Each VehiDE sample consists of a high-resolution RGB photograph of a damaged vehicle together with an annotation file containing, per damage instance:

- A class label (one of the 8 VehiDE damage categories).
- A bounding box (and, for a subset of images, a segmentation polygon) delimiting the damage region.
- An implicit instance-scale attribute, since the dataset publication reports the proportion of small/medium/large instances [2], which is directly relevant to our severity-estimation approach (Milestone 1, Section 10.2).

With 13,945 images and 32,000+ instances, the dataset averages approximately 2.3 damage instances per image, indicating that most images contain multiple co-occurring damage regions rather than a single isolated defect - consistent with real-world accident photographs.

### 3.2 Class Distribution and Imbalance

VehiDE's 8 native categories map onto our project's 6 target classes (dent, scratch, crack, broken lamp, flat tyre, shattered glass) as follows; two VehiDE categories that do not map cleanly onto our target taxonomy will be excluded or merged during preprocessing.

| **Project class** | **Mapped VehiDE categor(y/ies)** | **Notes** |
| --- | --- | --- |
| Dent | Dent | Direct mapping |
| Scratch | Scratch | Direct mapping |
| Crack | Crack | Direct mapping |
| Broken lamp | Broken lamp | Direct mapping |
| Flat tyre | Flat tyre | Direct mapping |
| Shattered glass | Glass shatter | Direct mapping |
| *(excluded)* | Remaining VehiDE categories outside our 6-class scope | Instances re-labelled as "background/other damage" or excluded from the training annotation file, not deleted from the source images, so the mapping is reversible |

As flagged as a risk in Milestone 1 (Section 10.1), we expect this distribution to be **long-tailed**: dents and scratches are the most frequent damage types in real-world claims, while flat tyres and shattered glass are comparatively rare. An exact per-class instance count will be computed as the first step of the exploratory data analysis (EDA) notebook (Section 10) and will directly inform the class-weighting and oversampling strategy.

### 3.3 Image Characteristics

Prior to preprocessing, the following image-level attributes will be profiled in the EDA notebook:

- Resolution distribution (min/max/median width and height), since YOLO requires images to be resized to a fixed input resolution (640 px) and very low-resolution images may need to be filtered.
- Aspect ratio distribution, to decide whether letterboxing or direct resizing is more appropriate.
- File format and colour mode (RGB vs. grayscale/CMYK anomalies).
- Lighting/exposure histogram spread, to anticipate the domain-shift risk discussed in Milestone 1 (Section 10.3), since VehiDE images are closer to studio/dealer-style photography than to policyholder phone photographs.

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

- **Class-level adequacy**: Because the distribution is expected to be long-tailed (Section 3.2), the minority classes (flat tyre, shattered glass) may have too few instances to reach the ≥0.65 per-class F1 target defined in Milestone 1 (Section 4.1) without intervention. Mitigation: (a) targeted oversampling of minority-class images during training, (b) class-weighted loss, and (c) supplementing minority classes with additional labelled examples drawn from the Car Damage Severity and COCO Car Damage datasets where their classes overlap.
- **Domain adequacy**: VehiDE images are closer to studio/dealer photography than to real policyholder phone photographs (Milestone 1, Section 10.3). To narrow this gap without collecting a new dataset, we will apply augmentation that simulates phone-camera conditions - brightness/contrast jitter, motion blur, perspective warp, and JPEG compression artefacts - and will additionally collect a small manually-curated stress-test set (~30-50 images) of realistic, non-studio claim-style photographs for final held-out evaluation only (never used in training).

If, after the class-level EDA, any target class has fewer than a workable minimum of instances (to be confirmed once exact counts are available), we will explore expanding that class via the supplementary datasets identified in Section 2.2 rather than via purely synthetic image generation, since synthetic damage imagery risks introducing an additional domain gap.

---

## 6. Train / Validation / Test Split Strategy

### 6.1 Splitting Approach

The deduplicated VehiDE image set will be split at the **image level** (not the instance level) into:

| **Split** | **Proportion** | **Purpose** |
| --- | --- | --- |
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning, early stopping, checkpoint selection |
| Test | 15% | Final, untouched evaluation reported in Milestone 1's Section 4.1 metrics |

The split will be **stratified by damage class** so that the proportion of each of the 6 target classes is preserved across train/validation/test as closely as possible, given that many images contain multiple instances/classes simultaneously. Where an image contains multiple classes, a multi-label stratification approach (assigning the image to a split based on its full label set rather than a single dominant class) will be used to avoid systematically starving any split of a rare class.

### 6.2 Leakage Checks

Because VehiDE may contain multiple photographs of the same physical vehicle/accident taken moments apart, image-level splitting alone is not sufficient to guarantee independence between splits. The following explicit leakage checks will be performed after the split is created:

- **Perceptual-hash cross-check**: the same pHash-based near-duplicate detector used in Section 4.3 will be re-run *across* splits (train vs. validation, train vs. test, validation vs. test) to confirm zero near-duplicate pairs crossing a split boundary.
- **Filename/metadata clustering**: where available, sequential filename patterns or embedded EXIF metadata will be used to detect burst-photographed sequences of the same incident, and any such cluster will be kept entirely within a single split.
- **Supplementary dataset overlap**: since CarDD, COCO Car Damage, and the Car Damage Severity dataset are drawn from different sources, we will additionally verify (via the same hashing approach) that none of their images duplicate a VehiDE image before using them for the class-expansion or calibration purposes described in Section 5, to avoid cross-dataset leakage.

Any detected cross-split duplicate will be resolved by removing it from all but one split, with a documented resolution log.

---

## 7. Synthetic Dataset Generation

As identified in Milestone 1 (Section 9.2), no public dataset pairs insurance policy documents with vehicle damage annotations, and real insurer policy documents cannot be used due to proprietary constraints (Milestone 1, Section 1.3). This makes the RAG pipeline's input corpus a fully synthetic dataset that the team must generate and document.

### 7.1 Synthetic Insurance Policy Documents

- **Volume**: 5 synthetic insurance policy PDFs, each approximately 8-12 pages.
- **Coverage**: Each document will cover collision coverage, comprehensive coverage, deductibles, exclusions, claim limits, and third-party liability.
- **Generation process**:
  1. Draft a common policy template structure (sections, clause numbering) based on the general structure of publicly available *sample* motor insurance policy summaries (not any single insurer's proprietary wording).
  2. Use an LLM to draft clause text for each section, explicitly varying phrasing, clause ordering, and section headings across the 5 documents so they are not near-duplicates of each other.
  3. Manually review and edit every generated document for internal consistency (e.g., a deductible referenced in one section must match the value used elsewhere in the same document).
  4. Deliberately inject **distractor clauses**, negations, and exceptions (e.g., "flat tyre damage is covered only if resulting from an insured collision event, not from normal wear") to stress-test retrieval faithfulness, as planned in Milestone 1 (Section 10.4).
  5. Map every clause to one or more of the 6 damage classes, producing a ground-truth clause-to-damage-class lookup table that will be used to compute Retrieval Precision@3 (Milestone 1, Section 4.2).
- **Format**: Each document is authored as structured Markdown/Word first (for ease of editing and version control) and then exported to PDF, since the production RAG pipeline ingests PDF policy uploads.

### 7.2 Synthetic Incident Descriptions

- **Volume**: 50 synthetic incident descriptions, each paired with a test image drawn from the VehiDE test split (Section 6.1).
- **Generation process**: Each description is a short first-person narrative of how the damage occurred (e.g., "reversed into a bollard, denting the rear bumper"), written to be consistent with the ground-truth damage class(es) present in the paired image, and used only for end-to-end report-quality evaluation (Milestone 1, Section 8.3), not for training.

### 7.3 Design Justification

- **Why synthetic and not scraped real policies**: Real insurer documents are proprietary and cannot legally be redistributed or used to train/evaluate an academic system; a fully synthetic, team-authored corpus avoids IP risk while still exercising the RAG pipeline against realistic policy language.
- **Why 5 documents rather than 1**: A single policy document would make retrieval trivially easy (only one candidate document to search) and would not test the retriever's ability to disambiguate between similar clauses phrased differently across documents.
- **Why deliberately vary phrasing and inject distractors**: This directly addresses the RAG-faithfulness risk identified in Milestone 1 (Section 10.4) - without this variation, reported retrieval precision would be optimistic relative to what a system will face with a real, arbitrarily-formatted policy PDF uploaded by an end user.

---

## 8. Cross-Task Dataset Alignment

This project has two sub-tasks that consume structurally different data - image-based damage detection and text-based policy retrieval - which must be connected at inference time even though they are trained/evaluated on separate datasets. Alignment is established as follows:

- **Shared label taxonomy**: The 6 damage classes used to annotate VehiDE (Section 3.2) are the *same* controlled vocabulary used to tag clauses in the synthetic policy documents (Section 7.1). This shared taxonomy is the join key between the vision output and the RAG query: a detected damage class becomes (part of) the retrieval query text.
- **Evaluation-time pairing**: The 50 synthetic incident descriptions (Section 7.2) are explicitly paired with specific VehiDE test images so that end-to-end evaluation can trace a single test case through all three pipeline stages (detection → retrieval → report), enabling the ablation comparison planned in Milestone 1 (Section 4.3).
- **No shared raw features**: Because the two sub-tasks do not share raw input features (pixels vs. text), alignment is enforced at the *label/schema* level rather than at the feature level; the structured JSON output of the detection stage (class, bounding box, confidence, severity) is the sole interface passed to the retrieval and generation stages, as described architecturally in Milestone 1 (Section 7).

---

## 9. Task-Specific Dataset Requirements

### 9.1 Vision Task - Labeling and Annotation Requirements

- **Annotation format**: VehiDE annotations will be converted from their native format to YOLO format (normalised `class x_center y_center width height` per line) for detection training, and to a segmentation-mask format for the CarDD-based supplementary segmentation fine-tuning.
- **Label verification**: A random 5% sample of converted annotations will be visually re-rendered (boxes drawn over images) and manually checked post-conversion to catch coordinate/axis conversion bugs before large-scale training.
- **Class remapping**: The 8-to-6 category mapping defined in Section 3.2 must be applied consistently across every annotation file; this mapping will be implemented as a single versioned lookup table/script rather than manual edits, to keep the process auditable and reproducible.
- **Severity labels**: Since Minor/Moderate/Severe labels are not natively provided by VehiDE, they will be *derived* (bounding-box area ÷ visible-vehicle-surface area, per Milestone 1 Section 7) and then calibrated against the Car Damage Severity dataset's human-provided labels, per Milestone 1 Section 10.2.

### 9.2 RAG Task - Document Preparation, Chunking, and Vector Store

- **Document preparation**: Each synthetic policy PDF is parsed to plain text with section/clause boundaries preserved (heading-aware extraction), so that chunk boundaries respect clause boundaries rather than splitting a clause mid-sentence.
- **Chunking strategy**: Following the literature findings summarised in Milestone 1 (Section 3.2), chunks of 200-400 tokens with overlap (~10-15%) will be used, since smaller, overlapping chunks were found to outperform large-chunk retrieval for clause-level recall in legal/insurance-style documents.
- **Vector database options explored**:

| **Option** | **Consideration** |
| --- | --- |
| FAISS (local, in-memory/on-disk index) | Lightweight, no external service dependency, well-suited to Hugging Face Spaces deployment constraints (Milestone 1, Section 3.4); billion-scale similarity search techniques [6] are far beyond this project's needs but the same library scales down cleanly. |
| Chroma | Simple persistence and metadata filtering (e.g., filter by damage-class tag before retrieval), useful given the clause-to-damage-class ground truth from Section 7.1. |
| Managed vector DB (e.g., Pinecone) | Rejected for this project - introduces an external paid dependency and network latency not justified for a corpus of only 5 documents. |

  Given the small corpus size (5 documents), the team will start with FAISS or Chroma for local development and only reconsider a managed option if retrieval evaluation reveals a need for production-scale features.
- **Embedding model**: A bi-encoder sentence embedding model (e.g., a Sentence-BERT variant [7]) will be used, consistent with the finding that bi-encoders outperform BM25 sparse retrieval for semantic matching of insurance-style queries (Milestone 1, Section 3.2).

### 9.3 LLM Report Generation - Prompt Structuring and Context Length

The project uses a pre-trained LLM via prompting/grounding rather than fine-tuning (Milestone 1, Section 7, Stage 3), so classic instruction-response fine-tuning dataset requirements do not directly apply. The relevant dataset-adjacent requirements for this stage are:

- **Prompt structuring**: The prompt template combines (a) the structured JSON detection output from Stage 1, (b) the top-k retrieved policy clauses from Stage 2, and (c) a fixed instruction block specifying the required report sections (damage summary table, severity, applicable coverage, recommended next steps) and the explicit "state not covered under retrieved policy" fallback instruction from Milestone 1 (Section 10.5).
- **Token length considerations**: With chunk sizes of 200-400 tokens and top-k retrieval (k to be tuned, likely 3-5), the retrieved-context portion of the prompt is bounded to roughly 1,000-2,000 tokens, comfortably within the context window of the LLMs under consideration; this bound will be validated empirically once k is finalised during pipeline development.
- **Evaluation dataset**: The 20-sample faithfulness test set and the 50 synthetic incident/image pairs (Section 7.2) double as the "dataset" against which prompt structuring choices are evaluated, since no separate fine-tuning corpus is being built for this stage.

### 9.4 Speech and Fine-Tuning Tasks - Applicability

This project involves no audio/speech modality, so audio quality, transcription alignment, and sampling-rate consistency requirements are **not applicable** and are omitted from this report. Similarly, since the LLM report generator is used through prompting rather than weight fine-tuning (Section 9.3), formal instruction-response dataset curation for fine-tuning is **not applicable** at this milestone; this may be revisited in a later milestone only if evaluation results show prompting alone is insufficient to meet the accuracy/faithfulness targets in Milestone 1 (Section 4.2-4.3).

---

## 10. Preprocessing Pipeline and Reproducibility

To ensure every result in later milestones can be reproduced by any team member, the following pipeline will be implemented as a versioned, scripted sequence (not manual/ad-hoc steps), with each stage's output cached to disk and its parameters logged:

1. **Ingestion**: Download VehiDE via the Kaggle API into a fixed local/shared directory structure; record the dataset version/download date.
2. **Integrity checks**: Run the corrupt-file, orphan-image, and orphan-annotation checks (Section 4.1); log all dropped files with reasons.
3. **Deduplication**: Run exact-hash and perceptual-hash deduplication (Section 4.3); log all collapsed duplicate clusters.
4. **Class remapping**: Apply the 8-to-6 category lookup table (Section 9.1) to produce the project-specific annotation set.
5. **Annotation format conversion**: Convert to YOLO-format labels (and segmentation masks for the CarDD subset); spot-check via visual re-rendering (Section 9.1).
6. **Stratified multi-label split**: Generate the train/validation/test split (Section 6.1) with a fixed random seed, and persist the resulting file lists so the split itself is reproducible and inspectable.
7. **Leakage verification**: Run the cross-split perceptual-hash and metadata-clustering checks (Section 6.2); the pipeline will halt and flag for manual review if any cross-split duplicate is detected.
8. **Augmentation configuration**: Define and version the augmentation pipeline (brightness/contrast, motion blur, perspective warp, JPEG artefacts) as a configuration file applied only at training time, never baked into the stored dataset.
9. **Policy document preparation**: Author, review, and export the 5 synthetic policy PDFs (Section 7.1); parse and chunk them (Section 9.2); embed and index chunks into the chosen vector store.
10. **Evaluation artefacts**: Assemble the 50 paired incident-description/image samples and the 20-sample faithfulness test set for later use in Milestone 1's evaluation plan (Sections 4.2-4.3).

Every step above will be implemented in version-controlled scripts/notebooks committed to the project's GitHub repository, with configuration values (split ratios, random seed, chunk size, overlap, k) stored in a single configuration file so that the entire data preparation pipeline can be re-run end-to-end from raw downloaded data to model-ready artefacts.

---

## 11. Summary

This milestone identified VehiDE as the primary, verified, and appropriately-licensed dataset for the vision detection task, supplemented by CarDD, COCO Car Damage, and the Car Damage Severity dataset for targeted auxiliary purposes. A concrete data-quality assessment, deduplication, and leakage-safe stratified splitting strategy has been defined. Because no public dataset exists for the RAG policy-retrieval task, a fully synthetic, deliberately stress-tested policy corpus will be authored by the team, with an explicit shared-taxonomy mechanism connecting the vision and retrieval sub-tasks. Task-specific requirements for annotation, chunking/vector-store selection, and prompt/context-length design have been specified, and a fully scripted, versioned preprocessing pipeline has been laid out to guarantee reproducibility ahead of model training in the next milestone.

---

## 12. References

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
