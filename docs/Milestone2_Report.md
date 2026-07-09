
---

<div align="center">

<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">

<h1>Multimodal Damage Assessment for Insurance Claims</h1>

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
- [2. Dataset Identification](#2-dataset-identification)
  - [2.1 Vision Datasets](#21-vision-datasets)
  - [2.2 Policy and Text Datasets](#22-policy-and-text-datasets)
  - [2.3 Ownership, Licensing, and Usage Constraints](#23-ownership-licensing-and-usage-constraints)
- [3. Dataset Description](#3-dataset-description)
  - [3.1 VehiDE: Structure, Schema, and Sample Records](#31-vehide-structure-schema-and-sample-records)
  - [3.2 Supplementary Vision Datasets](#32-supplementary-vision-datasets)
  - [3.3 Policy Document Corpus](#33-policy-document-corpus)
- [4. Data Governance](#4-data-governance)
  - [4.1 Data Source and Licensing](#41-data-source-and-licensing)
  - [4.2 Privacy](#42-privacy)
  - [4.3 Data Quality Validation](#43-data-quality-validation)
  - [4.4 Ethics and Bias](#44-ethics-and-bias)
  - [4.5 Reproducibility and Compliance](#45-reproducibility-and-compliance)
- [5. Exploratory Data Analysis](#5-exploratory-data-analysis)
  - [5.1 Dataset Summary Statistics](#51-dataset-summary-statistics)
  - [5.2 Class Distribution and Imbalance](#52-class-distribution-and-imbalance)
  - [5.3 Bounding Box Area Distribution](#53-bounding-box-area-distribution)
  - [5.4 Instances per Image](#54-instances-per-image)
  - [5.5 Image Resolution Analysis](#55-image-resolution-analysis)
  - [5.6 Missing Value and Orphan Analysis](#56-missing-value-and-orphan-analysis)
  - [5.7 Duplicate Analysis](#57-duplicate-analysis)
- [6. Data Preprocessing](#6-data-preprocessing)
  - [6.1 Vision Data Preprocessing](#61-vision-data-preprocessing)
  - [6.2 Policy Document Preprocessing](#62-policy-document-preprocessing)
- [7. Dataset Integration](#7-dataset-integration)
    - [7.1 Training Corpus - VehiDE](#71-Training-Corpus---VehiDE)
    - [7.2 Contingency Datasets](#72-Contingency-Datasets)
    - [7.3 Planned Integration Approach](#73-Planned-Integration-Approach)
- [8. Data Augmentation](#8-data-augmentation)
    - [8.1 Augmentation Configuration](#81-Augmentation-Configuration)
    - [8.2 Class-Targeted Oversampling](#82-Class-Targeted-Oversampling)
- [9. Dataset Splitting](#9-dataset-splitting)
    - [9.1 Splitting Approach](#91-Splitting-Approach)
    - [9.2 Split Sizes](#92-Split-Sizes)
    - [9.3 Class Distribution per Split](#93-Class-Distribution-per-Split)
    - [9.4 Leakage Prevention](#94-Leakage-Prevention)
    - [9.5 Escalation-Path Subset](#95-Escalation-Path-Subset)
- [10. Final Prepared Dataset](#10-final-prepared-dataset)
    - [10.1 Vision Dataset Summary](#101-Vision-Dataset-Summary)
    - [10.2 Policy Corpus Summary](#102-Policy-Corpus-Summary)
    - [10.3 Readiness Confirmation](#103-Readiness-Confirmation)
- [11. Challenges Encountered](#11-challenges-encountered)
- [12. Deliverables Produced](#12-deliverables-produced)
- [13. Summary and Next Steps](#13-summary-and-next-steps)
    - [13.1 Summary of Work Completed](#131-Summary-of-Work-Completed)
    - [13.2 Key Observations from the Data](#132-Key-Observations-from-the-Data)
    - [13.3 Confirmation of Training Readiness](#133-Confirmation-of-Training-Readiness)
    - [13.4 Planned Activities for Milestone 3](#134-Planned-Activities-for-Milestone-3)
- [14. References](#14-references)


---

## 1. Introduction

### 1.1 Project Recap

Milestone 1 defined the problem, scope, and evaluation plan for a multimodal vehicle damage assessment system that accepts vehicle damage photographs and insurance policy documents as input, detects and classifies visible damage using a fine-tuned YOLO-based model, retrieves relevant policy clauses using a Retrieval-Augmented Generation (RAG) pipeline, and generates a structured preliminary claim assessment report using an LLM. Since Milestone 1, the system design has been refined into a multi-agent RAG architecture in which a LangGraph orchestrator routes each claim to four specialist agents: a Damage Agent (YOLOv8 detection), a Severity Agent (bounding-box area-ratio scoring), a Policy Agent (RAG over the synthetic policy corpus, exposed as an MCP tool), and a Report Agent (LLM-based report writing), with low-confidence outputs escalated to a human review queue. 

![High-Level Architecture Diagram](multiagent_architecture_staged.svg)

This refinement extends the modular separation-of-concerns argument already established in Milestone 1 (Section 3.4) by adding a routing layer that can act conditionally per claim, e.g. skipping the Policy Agent when no PDF is supplied, or escalating low-confidence detections to human review rather than always executing a fixed three-stage sequence.

### 1.2 Objectives of Milestone 2

The three primary objectives of this milestone are:

1. **Dataset verification and download**: Identify, verify provenance, confirm licensing, and download all datasets required by each agent.
2. **Vision data preparation**: Assess quality, preprocess, augment, and split the VehiDE dataset and supplementary vision datasets into training-ready artefacts for the Damage and Severity Agents.
3. **Policy corpus construction**: Author, review, chunk, embed, and index the synthetic insurance policy document corpus for the Policy Agent, using publicly available IRDAI-registered policy wordings as structural references.

By the end of this milestone, any team member with repository access should be able to begin training without performing any additional data preparation.

### 1.3 Relationship Between Datasets and Project Goals

| **Dataset** | **Agent it feeds** | **What it enables** |
| --- | --- | --- |
| VehiDE (primary) | Damage Agent + Severity Agent | YOLO fine-tuning for 6-class damage detection and severity calibration |
| CarDD | Damage Agent (supplementary) | Pixel-level segmentation masks for irregularly shaped damage |
| Car Damage Severity Dataset | Severity Agent (calibration) | Human-labelled Minor/Moderate/Severe ground truth |
| COCO Car Damage | Damage Agent (comparison) | Architecture sanity-check against a differently-annotated source |
| Synthetic policy PDFs | Policy Agent | RAG retrieval corpus for claim coverage determination |
| Synthetic incident descriptions | Report Agent + Evaluation | Ground-truth incident/report pairs prepared for later end-to-end evaluation |

---

## 2. Dataset Identification

### 2.1 Vision Datasets

#### Primary: VehiDE

| **Attribute** | **Detail** |
| --- | --- |
| Dataset name | VehiDE: Vehicle Damage Detection Dataset |
| Source | Kaggle: [hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection) |
| Download method | `kaggle datasets download hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection` |
| License | **Apache-2.0** |
| Purpose in this project | Primary training and evaluation dataset for the Damage Agent |
| Why selected | Largest publicly available annotated vehicle damage dataset (13,945 images, 36,081 instances); peer-reviewed construction paper; covers all 7 damage classes; supports detection, segmentation, and salient object detection tasks |

#### Alternative Vision Datasets

| **Dataset** | **Download Link** | **License** | **Purpose** | **Why selected**|
| --- | --- | --- | --- | --- |
| CarDD | [cardd-ustc.github.io](https://cardd-ustc.github.io/) | Academic research use | Pixel-level segmentation masks for irregularly shaped damage (scratches, cracks) not well-represented by bounding boxes alone | Only public dataset with pixel-level damage segmentation across 6 damage categories with a peer-reviewed benchmark |
| COCO Car Damage | [kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset) | Community (Kaggle) | Architecture comparison and pipeline sanity-checking | COCO format allows direct benchmark comparison against published COCO-trained baselines |
| Car Damage Severity | [kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset) | Community (Kaggle) | Calibrating the Severity Agent\'s bounding-box area-ratio proxy against human-labelled severity | Only public dataset with human-assigned Minor/Moderate/Severe labels matching our three-category scheme |

### 2.2 Policy and Text Datasets

No public dataset of insurance policy documents paired with vehicle damage annotations exists. The policy corpus used by the Policy Agent is therefore a combination of two publicly available IRDAI-registered reference documents, used only for structural guidance, and five synthetic policy PDFs authored entirely by the project team.

**Reference documents (structural reference only, not indexed or used for training):**

| **Document** | **Insurer** | **UIN** | **Pages** | **Role in this project** |
| --- | --- | --- | --- | --- |
| Motor Private Car 3 Years Policy Wordings | Universal Sompo General Insurance Co. Ltd | IRDAN134RP0003V01201819 | 23 | Primary structural reference; clause vocabulary, section hierarchy, and depreciation schedule design |
| Private Car Standalone Own Damage Policy | United India Insurance Company Limited | IRDAN545RP0001V01201920 | 4 | Secondary structural reference; alternative phrasing of similar coverage and exclusion clauses |

These documents are publicly available IRDAI-registered policy wordings, not proprietary schedules or individual policyholder documents. No clause text from either document is reproduced verbatim in the synthetic corpus.

**Synthetic policy corpus (team-authored, fully indexed into ChromaDB):**

| **Document** | **Insurer (fictitious)** | **Pages** | **Style** | **Key design feature** |
| --- | --- | --- | --- | --- |
| policy_1_bharat_suraksha.pdf | Bharat Suraksha Motor Insurance Co. Ltd | 5 | Formal traditional IRDAI language | All 6 damage classes covered under one umbrella accidental-external-means clause; 11 numbered exclusions |
| policy_2_safedrive_assurance.pdf | SafeDrive Assurance Corporation | 4 | Modern plain language with dedicated subsections per damage type | Explicit nil-depreciation glass clause; separate tyre sub-limit section; 5 exclusions per damage type |
| policy_3_quickclaim_general.pdf | QuickClaim General Insurance Ltd | 4 | Dense legal with named exclusion schedules | Hardest document for RAG retrieval; 6 named exclusion schedules (A-F) with 5 clauses each, producing 30 distractor clauses across all damage classes |
| policy_4_autoguard_premium.pdf | AutoGuard Premium Insurance Services Ltd | 4 | Consumer-facing with a coverage summary table | Coverage presented as a structured table; 6 exclusion subsections; includes nil-depreciation, return-to-invoice, and consumables add-on covers |
| policy_5_valuemotor.pdf | ValueMotor Comprehensive Insurance Ltd | 4 | Budget insurer, concise with conditional sub-limits | Most conditional language; 7 exclusion subsections; tyre cover explicitly conditional on concurrent vehicle body damage |

The five documents were designed to vary along three dimensions to stress-test the RAG retrieval pipeline. First, phrasing: the same coverage concept is expressed differently across documents (for example, dent coverage appears as "accidental external means", "bodily damage from impact", "accidental collision" or "impact", and "physical impact damage" across the five policies). Second, distractor density: each policy contains at least five distractor clauses per damage class exclusion clauses, sub-limit clauses, and conditional coverage clauses that are semantically related to a damage class but do not grant coverage. Third, structural format: coverage appears as numbered lists in some documents and as tables in others, preventing the retriever from relying on positional cues.

### 2.3 Ownership, Licensing, and Usage Constraints

| **Dataset** | **Ownership** | **Permitted use** | **Restrictions** |
| --- | --- | --- | --- |
| VehiDE | Dataset authors (Scullen et al.) | **Apache-2.0** | Commercial use permitted under Apache-2.0; standard attribution |
| CarDD | USTC research group | Academic research | Attribution to original paper required |
| COCO Car Damage | Kaggle community uploader | Community use | Cite dataset page |
| Car Damage Severity | Kaggle community uploader | Community use | Cite dataset page |
| Universal Sompo policy | Universal Sompo General Insurance Co. Ltd | Publicly available IRDAI filing | Used as structural reference only; no verbatim reproduction |
| United India Insurance policy | United India Insurance Co. Ltd | Publicly available IRDAI filing | Used as structural reference only; no verbatim reproduction |
| Synthetic policy PDFs | This project team | Fully team-owned | No restrictions |


---

## 3. Dataset Description

### 3.1 VehiDE: Structure, Schema, and Sample Records

**Dataset annotation schema:**

```json
{
  "<filename>.jpg": {
    "regions": [
      {"all_x": [...], "all_y": [...], "class": "rach"},
      ...
    ]
  },
  ...
}
```

Bounding boxes used were derived from each region's polygon extent (min/max of `all_x`/`all_y`), producing an absolute-pixel `[x, y, width, height]`. 
The traget classes were present in Vietnamese which had to be mapped to english translation.

**Native class vocabulary (Vietnamese → English):**

| **Vietnamese (raw)** | **English (mapped)** |
| --- | --- |
| `tray_son` | paint_scratches |
| `mop_lom` | dents |
| `rach` | torn_body |
| `mat_bo_phan` | lost_parts |
| `be_den` | broken_lamp |
| `thung` | puncture |
| `vo_kinh` | broken_glass |




**Top-level dataset statistics:**

| **Attribute** | **Value** |
| --- | --- |
| Total images | 13,945 |
| Total annotated instances | 36,081 |
| Average instances per image | 2.59 |
| Max instances in a single image | 20 |
| Image format | JPEG (.jpg) |
| Annotation format | 2 VIA-format JSON files (VGG Image Annotator), Vietnamese class labels, polygon regions |
| Native damage classes | 7 |
| Annotation types supported | Bounding box, instance segmentation polygon |

**Class distribution:**

| **Class (English)** | **% of instances** |
| --- | --- |
| paint_scratches | 40.6% |
| dents | 15.8% |
| torn_body | 15.3% |
| lost_parts | 7.8% |
| broken_lamp | 7.7% |
| puncture | 6.7% |
| broken_glass | 6.2% |

Imbalance ratio (largest/smallest): **6.59:1** (`paint_scratches` vs `broken_glass`)

### 3.2 Supplementary Vision Datasets

| **Dataset** | **Images** | **Instances / Labels** | **Format** | **Target variable** |
| --- | --- | --- | --- | --- |
| CarDD | 4,000  | Pixel-level segmentation masks | COCO-format JSON + PNG masks | Damage class (6 categories matching project taxonomy) |
| COCO Car Damage | 70 | 379 annotated instances (6 categories: `damage`, `door`, `front_bumper`, `headlamp`, `hood`, `rear_bumper`) | COCO JSON | Not "damage present/type." The 6 categories are 5 parts and one generic damage class. This does not match the project's requirements |
| Car Damage Severity | 1631 | Image-level severity label per image | Folder-organised JPEG | Severity: Minor / Moderate / Severe |

### 3.3 Policy Document Corpus

**Reference documents (structural reference only, not training data):**

The Universal Sompo document (23 pages, IRDAN134RP0003V01201819) contains the following sections relevant to damage claim assessment:

| **Policy section** | **Content** | **Relevant damage classes** |
| --- | --- | --- |
| Section I, clauses 1-10 | Covered perils (fire, theft, riot, flood, accidental external means, malicious act, etc.) | All 5 project classes; accidental external means covers dents, scratches, broken lamps |
| Section I, exclusions (a-d) | Consequential loss, tyre damage at 50% only, accessories theft exclusion, intoxication | Distractor clauses for flat tyre and shattered glass |
| Depreciation schedule | Age-based depreciation rates (Nil to 50%) | Relevant to all classes for partial loss claims |
| Add-on 2: Depreciation Waiver | Nil depreciation for replaced parts (Plans a/b/c) | Crack, dent, scratch |
| Add-on 12: Hydrostatic Lock | Engine water ingression cover | Exclusion distractor |
| Add-on 15: Engine Protector | Consequential damage from oil leakage / flood | Exclusion distractor |

**Synthetic policy corpus (team-authored):**

| **Attribute** | **Value** |
| --- | --- |
| Number of documents | 5 synthetic PDFs |
| Total chunks produced (after splitting and indexing) | 179 chunks |
| Chunk size | 300 tokens with 40-token overlap |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (collection `policy_clauses`) |
| Ground-truth clause mappings | 1 JSON file mapping each chunk ID to damage classes it addresses |
| Distractor clauses per damage class | At least 5 per class (exclusions, sub-limit clauses, conditional coverage) |

**Chunks per damage class:**

| **Damage class** | **Chunks** |
| --- | --- |
| dent | 53 |
| flat_tyre | 32 |
| shattered_glass | 29 |
| scratch | 28 |
| broken_lamp | 27 |
| crack | 21 |

A chunk can be tagged with more than one damage class (e.g. a general coverage clause applying to all classes), so the per-class counts above sum to more than the total chunk count.

**Clause type distribution:**

| **Clause type** | **Chunks** |
| --- | --- |
| general | 113 |
| coverage | 28 |
| sub_limit | 19 |
| exclusion | 13 |
| condition | 4 |
| definition | 2 |
| **Total** | **179** |
---

## 4. Data Governance

### 4.1 Data Source and Licensing

All datasets are used strictly within their permitted scope. VehiDE is used for non-commercial academic research under the terms stated by the dataset authors. CarDD is used under academic research terms with attribution. The two IRDAI policy wording documents are publicly available regulatory filings used only as structural references: no clause text is reproduced verbatim in any project output. The synthetic policy PDFs are entirely team-authored and carry no third-party licensing constraints.

No dataset used in this project is owned by the team or by IIT Madras. All are third-party datasets used under their respective research or educational licenses.

### 4.2 Privacy

**VehiDE images:** Vehicle damage photographs may incidentally capture visible license plates and faces. An automated face/license-plate detector was run across the retained 13,655 images as a precaution; it flagged 0 images containing visible faces or license plates, so no blurring was required for this dataset in its current form. The detection step remains part of the pipeline (`scripts/preprocess_vehide.py`) so that any future dataset additions are still checked automatically. The original downloaded dataset is retained unmodified with a SHA-256 hash recorded for reproducibility.

**Policy documents:** The two reference policy documents (Universal Sompo, United India) contain no personal policyholder data. They are regulatory-approved template wordings. The synthetic policy PDFs contain no real names, vehicle registration numbers, or financial data.

**Demo deployment:** The Gradio demo on Hugging Face Spaces will display a notice that no uploaded images or documents are stored or logged beyond the current session.

### 4.3 Data Quality Validation

The following automated checks were run as part of the preprocessing pipeline (`scripts/preprocess_vehide.py`):

| **Check** | **Method** | **Result** |
| --- | --- | --- |
| Corrupt / unreadable images | PIL `Image.verify()` on every file | 0 corrupt images found |
| Orphan images (no annotation file) | Set difference of image stems vs annotation stems | 0 orphan images found |
| Orphan annotations (no image file) | Set difference of annotation stems vs image stems | 0 orphan annotations found |
| Malformed annotation regions | Check each polygon region has valid `all_x`/`all_y` coordinates | 1 malformed annotation region found and corrected |
| Exact-hash duplicates | MD5 hash of raw image bytes | 18 exact duplicate images found and removed |
| Near-duplicate images | Perceptual hash (pHash) | 272 near-duplicate images found and removed |
| Excluded instances (class-based) | `lost_parts` (`mat_bo_phan`) instances excluded, as this class does not correspond to a visible damage type | 2,703 instances excluded |

**After preprocessing (`scripts/preprocess_vehide.py`):**

| **Metric** | **Before** | **After** |
| --- | --- | --- |
| Total images | 13,945 | 13,655 |
| Total instances | 36,081 | 32,672 (retained, after excluding `lost_parts`) |
| Corrupt / unreadable files | 0 | 0 |
| Orphan images | 0 | 0 |
| Exact duplicates | 18 | 0 |
| Near duplicates removed | 272 | 0 |
| Faces blurred | — | 0 |
| License plates blurred | — | 0 |

### 4.4 Ethics and Bias

**Geographic representation:** VehiDE was constructed primarily from vehicle images collected in Vietnam and Southeast Asia. This introduces a potential geographic bias: vehicle types prevalent in India (compact sedans, two-wheelers, autorickshaws), damage patterns common in India (monsoon-related surface oxidation, potholes causing tyre and underbody damage), and camera conditions typical of Indian mobile claim submissions (low light, dust-covered lenses) may be underrepresented in the training distribution. This is explicitly acknowledged as a domain shift risk (Milestone 1, Section 10.3) and is mitigated through augmentation (Section 8) and domain-shift stress testing.

**Class imbalance:** After remapping, `scratch` alone accounts for roughly 61% of all retained instances (Section 5.2), far outnumbering `shattered_glass`, the smallest class. Without mitigation, a model trained on the raw distribution will exhibit higher precision on `scratch` and `dent` and poor recall on the rarer classes. Mitigation strategies are described in Section 8.

**Severity proxy bias:** The bounding-box area-ratio severity proxy may systematically underestimate severity for small but deep damage (cracks, punctures) and overestimate it for large but superficial damage (surface scratches spanning a door panel). This is a known limitation discussed in Milestone 1 (Section 10.2).

**Synthetic policy corpus:** The synthetic policies are modelled on Indian IRDAI-regulated policy structures (Universal Sompo, United India). This is appropriate for the target deployment context but means the RAG pipeline has not been validated against policy formats from other jurisdictions.

### 4.5 Reproducibility and Compliance

- Dataset version and download date are recorded in `configs/dataset_versions.yaml`.
- All preprocessing steps are implemented as version-controlled Python scripts committed to the project GitHub repository.
- A single configuration file (`configs/pipeline_config.yaml`) stores all tunable parameters (split ratios, random seed, chunk size, overlap, embedding model name) so the entire data preparation pipeline can be re-run end-to-end from raw downloaded data to agent-ready artefacts.
- SHA-256 hashes of all raw downloaded dataset archives are recorded in `data/checksums.txt` to verify dataset integrity at any future point.

---

## 5. Exploratory Data Analysis

All exploratory data analysis (EDA) notebooks were created in `./notebooks/EDA` and committed to the project repository.

### 5.1 Dataset Summary Statistics

| **Statistic** | **Value** |
| --- | --- |
| Total images (raw, as downloaded) | 13,945 |
| Total images (final, after preprocessing) | 13,655 |
| Total annotated instances (raw, all 7 native classes) | 36,081 |
| Total annotated instances (retained, `lost_parts` excluded, pre-dedup) | 33,262 |
| Total annotated instances (final, after preprocessing & dedup) | 32,672 |
| Mean instances per image | 2.59 |
| Median instances per image | 2.0 |
| Max instances per image | 20 |
| Native class IDs (Vietnamese) | 7 |
| Project classes after remapping | 6 |

### 5.2 Class Distribution and Imbalance

**VehiDE native class distribution (raw, all 7 classes, before exclusion):**

| **VehiDE class (native)** | **Project class** | **Raw instances** | **% of raw total** |
| --- | --- | --- | --- |
| paint_scratches | scratch | 14,646 | 40.6% |
| dents | dent | 5,681 | 15.7% |
| torn_body | crack | 5,509 | 15.3% |
| broken_lamp | broken_lamp | 2,782 | 7.7% |
| lost_parts | EXCLUDED | 2,819 | 7.8% |
| puncture | flat_tyre | 2,423 | 6.7% |
| broken_glass | shattered_glass | 2,221 | 6.2% |
| **Total (raw)** | | **36,081** | **100%** |

Each native class maps 1:1 to a project class (no merging) except `lost_parts`, which has no visible-damage equivalent and is dropped entirely.

**After remapping (6-class project taxonomy, `lost_parts` excluded), pre-image-dedup:**

| **Project class** | **Instances** | **% of total** |
| --- | --- | --- |
| scratch | 14,646 | 44.0% |
| dent | 5,681 | 17.1% |
| crack | 5,509 | 16.6% |
| broken_lamp | 2,782 | 8.4% |
| flat_tyre | 2,423 | 7.3% |
| shattered_glass | 2,221 | 6.7% |
| **Retained total** | **33,262** | **100%** |

<br><br>

![Class distribution — instance count and proportion per project class](eda_outputs/plots/class_distribution.png)

**Imbalance ratio (most frequent vs least frequent project class):** scratch (14,646) vs shattered_glass (2,221) = **6.59:1**.

After the production preprocessing pipeline additionally removes 18 exact-duplicate and 272 near-duplicate images (Section 6.1), the retained instance count drops proportionally from 33,262 to 32,672, giving the following final training-set class distribution:

| **Project class** | **Final retained instances** | **% of total** |
| --- | --- | --- |
| scratch | 14,386 | 44.0% |
| dent | 5,580 | 17.1% |
| crack | 5,411 | 16.6% |
| broken_lamp | 2,733 | 8.4% |
| flat_tyre | 2,380 | 7.3% |
| shattered_glass | 2,182 | 6.7% |
| **Retained total** | **32,672** | **100%** |

The imbalance ratio is unchanged at the class level (deduplication removes images roughly uniformly across classes), so the final training set still carries a **6.59:1** imbalance between `scratch` and `shattered_glass`. This is significant and will cause the model to underperform on `shattered_glass` and `flat_tyre` detection without mitigation. Class-weighted loss and targeted augmentation of minority classes are applied as described in Section 8.

### 5.3 Bounding Box Area Distribution

Bounding box area is computed as `width x height` (both normalised to [0,1]). This metric directly drives the Severity Agent\'s area-ratio proxy.

<br><br>
![Bounding box area distribution and severity proxy per class](eda_outputs/plots/bbox_area_distribution.png)

| **Severity proxy bin** | **Instance count** | **% of instances** |
| --- | --- | --- |
| Minor | 13,720 | 41.3% |
| Moderate | 8,238 | 24.8% |
| Severe | 11,304 | 34.0% |

**Bounding-box area statistics (normalised to [0,1]):**

| **Statistic** | **Value** |
| --- | --- |
| Mean | 0.1201 |
| Std | 0.1977 |
| Min | 0.00002 |
| 25th percentile | 0.0070 |
| Median (50th percentile) | 0.0332 |
| 75th percentile | 0.1347 |
| Max | 1.0000 |

**Key observations:**
- Mean bbox area (0.120) is well above the median (0.033), confirming the distribution is strongly right-skewed, with a long tail of very large damage regions pulling the mean upward.
- Mean bbox area per class varies substantially: `shattered_glass` has by far the largest mean normalised area (windshields and windows span a large fraction of the frame), while `flat_tyre` has the smallest — a useful prior for the Severity Agent, since the area-ratio proxy will need per-class calibration rather than one fixed threshold set.

**Bounding box aspect ratio:**

![Bounding box aspect ratio distribution](eda_outputs/plots/bbox_aspect_ratio.png)

The vast majority of bounding boxes have an aspect ratio (width/height) below 5, clustered close to the square (AR=1) reference line, with a long tail of elongated boxes (AR up to ~60) corresponding to linear damage such as long scratches or cracks running along a panel edge.

### 5.4 Instances per Image

| **Instances per image** | **Image count** | **% of images** |
| --- | --- | --- |
| 1 | 5,559 | 43.1% |
| 2 | 2,481 | 19.2% |
| 3 | 1,689 | 13.1% |
| 4+ | 3,180 | 24.6% |

![Distribution of instance counts per image, and class co-occurrence matrix](eda_outputs/plots/instances_per_image.png)

Mean instances per image: 2.58; median: 2.0; max: 20 in a single image. A substantial share of images (24.6%) contain four or more co-occurring damage instances, alongside a large single-instance group (43.1%). This confirms that the system must handle multi-label detection outputs per image rather than assuming single-instance classification, while also ensuring the model performs well on the common single-damage case. Note: these per-image counts total 12,909 images rather than the full 13,945, since images whose only annotated instance was the excluded `lost_parts` class are not counted here (they have zero remaining instances after exclusion, discussed further in Section 5.6).

#### 5.4.1 Class Co-occurrence Analysis

![Class co-occurrence heatmap — how often each pair of damage classes appears together in the same image](eda_outputs/plots/class_cooccurrence_heatmap.png)

The diagonal of the co-occurrence matrix shows each class's total instance count; the off-diagonal cells show how often two classes appear together in the same image. `scratch` co-occurs most frequently with every other class, consistent with it being the most common damage type overall this is expected rather than a data quality issue, since a single accident photo often shows both a dominant damage type (e.g. a dent) and incidental scratching around it. `shattered_glass` has the lowest co-occurrence with other classes (a broken windshield is often photographed on its own), which is useful context for the Damage Agent: this class is somewhat easier to isolate as the sole detection in a frame.

### 5.5 Image Resolution Analysis

Resolution was measured on a random sample of 1,000 images.

![Image width, height, and width-vs-height scatter](eda_outputs/plots/image_resolution.png)

| **Statistic** | **Width (px)** | **Height (px)** |
| --- | --- | --- |
| Mean | 1,383.9 | 1,033.7 |
| Median | 1,632.0 | 1,224.0 |
| Min | 204 | 153 |
| Max | 2,164 | 2,176 |

All images will be resized to 640 x 640 using letterboxing before YOLO training (Section 6.1). The wide range of native resolutions (from 204px to over 2,100px wide) confirms that naive cropping would be inappropriate.

### 5.6 Missing Value and Orphan Analysis

Object detection datasets do not have tabular "missing values" in the traditional sense. The relevant missing-data concepts for this dataset are orphan files and incomplete annotations.

| **Issue** | **Count** | **Resolution** |
| --- | --- | --- |
| Images with no annotation file | 0 | No action required |
| Annotation files with no image | 0 | No action required |
| Malformed annotation regions (invalid polygon coordinates) | 1 | Corrected during preprocessing |
| Images with zero instances after remapping (`lost_parts` only images) | Included in the 2,703 excluded instances | Retained as background images where the image has no other instance (improves specificity) |

### 5.7 Duplicate Analysis

| **Duplicate type** | **Method** | **Images found** | **Action** |
| --- | --- | --- | --- |
| Exact duplicates | MD5 hash of raw bytes | 18 images | Duplicate copy removed; primary retained |
| Near-duplicates | Perceptual hash (pHash) | 272 images | Duplicate copy removed; primary retained |
| Cross-dataset near-duplicates (VehiDE vs CarDD) | pHash cross-matching | Pending CarDD data access (Section 11) | To be re-run once the CarDD archive is available |

After deduplication and class-based exclusion, 13,655 unique images and 32,672 retained instances remain.

### 5.8 Spatial Distribution of Damage Centres

![Spatial distribution of normalised damage bounding-box centres, per class](eda_outputs/plots/spatial_distribution.png)

Normalised `(x_center, y_center)` positions for each class cluster around the centre of the frame (0.5, 0.5) for all six classes, which is expected: claimants typically photograph damage by centring the camera on the affected area rather than capturing the whole vehicle. `scratch` and `crack` show the widest spatial spread, consistent with damage that can run across large or varied parts of a panel, while `shattered_glass` is the most tightly clustered, reflecting the fixed position of windshields and windows relative to the frame. No systematic edge-of-frame bias was found that would suggest cropping artefacts in the source photography.


---

## 6. Data Preprocessing

All preprocessing steps are implemented in version-controlled scripts. No manual edits were made to annotation files. Every dropped file is logged with its reason.

### 6.1 Vision Data Preprocessing

**Step 1: Corrupt file removal**

```python
from PIL import Image
import os

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

corrupt = [p for p in image_paths if not check_image(p)]
# Result: 0 corrupt files removed
```

**Step 2: Class remapping (7-to-6)**

A versioned lookup table (`configs/class_remap.json`) maps VehiDE\'s 7 native Vietnamese class IDs to the project\'s 6 target classes. `torn_body` is mapped to `crack` (kept as its own class, Section 3.1), and the single excluded category (`lost_parts`) is relabelled as background and excluded from the annotation files used for training. The original annotation files are preserved unmodified so the mapping is fully reversible.

```json
{
  "tray_son": "scratch",
  "mop_lom": "dent",
  "rach": "crack",
  "mat_bo_phan": "exclude",
  "be_den": "broken_lamp",
  "thung": "flat_tyre",
  "vo_kinh": "shattered_glass"
}
```

**Step 3: YOLO format verification**

VehiDE annotations are already in YOLO normalised format. A verification pass confirmed all 5-field rows and value ranges. The `damage.yaml` configuration file maps class indices to human-readable labels:

```yaml
path: ./data/vehide
train: images/train
val:   images/val
test:  images/test

nc: 6
names: ['scratch', 'dent', 'crack', 'broken_lamp', 'flat_tyre', 'shattered_glass']
```


**Step 4: Image resizing with letterboxing**

All images are resized to 640 x 640 using letterboxing (padding with grey fill value 114) to preserve aspect ratio. This is applied as a preprocessing step before training rather than at runtime, to reduce I/O overhead during training.

```python
from PIL import Image, ImageOps

def letterbox_resize(img_path, out_path, size=640):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    delta_w = size - img.size[0]
    delta_h = size - img.size[1]
    padding = (delta_w//2, delta_h//2,
               delta_w - delta_w//2, delta_h - delta_h//2)
    padded = ImageOps.expand(img, padding, fill=(114, 114, 114))
    padded.save(out_path, quality=95)
```


**Step 5: Annotation spot-check**

A random sample of converted annotations was visually verified by rendering bounding boxes over images. No systematic coordinate conversion bugs were found. The single malformed annotation region identified in Section 4.3 was corrected.

**Step 6: PII blurring**

A Haar-cascade-based face detector and a license plate pattern detector (aspect ratio and character density heuristics) were run across all 13,655 retained images. This pipeline step is applied automatically regardless of dataset content, since car damage photos can incidentally capture bystanders or number plates.

| **PII type** | **Images flagged** | **Action** |
| --- | --- | --- |
| Visible license plates | 0 | No action required |
| Human faces | 0 | No action required |
| No PII detected | 13,655 | No action |

No PII was detected in the current VehiDE image set; the detector remains in the pipeline for any future data additions.


### 6.2 Policy Document Preprocessing

**Step 1: PDF text extraction**

Both reference PDFs and all 5 synthetic PDFs were parsed using `pdfplumber`, which preserves paragraph boundaries more reliably than raw PyPDF2 for multi-column policy documents.

```python
import pdfplumber

def extract_policy_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)
```

**Step 2: Section-aware chunking**

A heading-aware splitter was implemented to ensure chunk boundaries respect clause boundaries. Section headings matching patterns such as `SECTION I`, numbered items (`1.`, `2.`), and lettered sub-items (`a)`, `b)`) are treated as natural chunk delimiters before applying the token-length splitter.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=40,
    separators=["\\n\\n", "\\n", ". ", " "]
)
chunks = splitter.split_text(policy_text)
```

**Chunking results (aggregate, all 5 synthetic PDFs):**

| **Metric** | **Value** |
| --- | --- |
| PDFs processed | 5 |
| Total chunks indexed | 179 |
| Chunk size | 300 tokens, 40-token overlap |

See Section 3.3 for the per-damage-class and per-clause-type breakdown of the 179 indexed chunks.

**Step 3: Embedding and indexing**

```python
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection("policy_clauses")

for i, chunk in enumerate(all_chunks):
    embedding = model.encode(chunk).tolist()
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i:04d}"],
        metadatas=[{"doc_id": chunk_metadata[i]["doc_id"],
                    "damage_classes": chunk_metadata[i]["damage_classes"]}]
    )
print(f"Indexed {collection.count()} chunks")
# Output: Indexed 179 chunks
```

**Step 4: Ground-truth clause mapping**

Every chunk was manually tagged with the damage classes it addresses, producing a JSON ground-truth file used for Retrieval Precision@3 and MRR evaluation (Milestone 1, Section 4.2):

```json
{
  "chunk_0042": {
    "text_preview": "The Company will indemnify against loss or damage by accidental external means...",
    "damage_classes": ["dent", "scratch", "crack", "broken_lamp", "shattered_glass", "flat_tyre"],
    "clause_type": "coverage",
    "doc_id": "synthetic_policy_2"
  },
  "chunk_0091": {
    "text_preview": "Damage to tyres and tubes is limited to 50% of replacement cost...",
    "damage_classes": ["flat_tyre"],
    "clause_type": "sub_limit",
    "doc_id": "synthetic_policy_1"
  }
}
```
---


## 7. Dataset Integration

For this milestone, only **VehiDE** is preprocessed, it is sufficient on its own (13,655 images, 32,672 instances across all 6 project classes) to produce a training-ready dataset. CarDD, the Car Damage Severity dataset, and COCO Car Damage (Section 2.1) are **not** merged into the training corpus at this stage. They are held in reserve as a contingency plan and will only be integrated if the VehiDE-only model's validation performance (mAP@50, per-class F1 — particularly for the minority classes `shattered_glass`, `flat_tyre`, and `crack` given the 6.59:1 imbalance, Section 5.2) falls short of the Milestone 1 targets once baseline training runs in Milestone 3. Because no merging occurs at this stage, no schema alignment, cross-dataset deduplication, or format conversion has been performed.

### 7.1 Training Corpus - VehiDE

| **Source** | **Images** | **Instances** | **Status** |
| --- | --- | --- | --- |
| VehiDE | 13,655 | 32,672 | Sole training corpus for this milestone |

### 7.2 Contingency Datasets

| **Dataset** | **Would supplement** | **Trigger condition** | **Access status** |
| --- | --- | --- | --- |
| CarDD | Pixel-level segmentation masks for irregular damage (scratches, cracks) | Segmentation-head mAP or minority-class F1 below target after baseline training | Requires manual licensing form (Section 3.2); not yet obtained |
| Car Damage Severity | Human-labelled Minor/Moderate/Severe ground truth for calibrating the Severity Agent's bounding-box area-ratio proxy | Severity proxy shows poor agreement with human judgment during calibration (Milestone 5) | EDA notebook not yet run to completion (Section 3.2) |
| COCO Car Damage | Architecture sanity-check against a differently-annotated source | Used only for comparison, not planned for training integration | Already downloaded and profiled (70 images, 379 instances) |

### 7.3 Planned Integration Approach

If underperformance triggers CarDD integration, the following alignment work scoped but not yet executed would be required before merging it with VehiDE:

| **Attribute** | **VehiDE** | **CarDD** | **Alignment action** |
| --- | --- | --- | --- |
| Annotation format | YOLO .txt (bbox) | COCO JSON (segmentation polygon) | Convert CarDD to YOLO-seg format using `scripts/coco_to_yolo_seg.py` |
| Class taxonomy | 7 native classes, 6 after remapping | 6 classes (CarDD's own taxonomy) | Remap both to the project's 6-class taxonomy |
| Image resolution | 204-2,164px (sampled) | Not yet measured | Letterbox both to 640 x 640 |
| Image naming | `<original_filename>.jpg` | `cardd_XXXXX.jpg` (planned) | Prefix to avoid filename collisions |

A cross-dataset perceptual-hash deduplication pass (same methodology as Section 5.7) would also be run against VehiDE before any CarDD images are added to the training set, to prevent leakage. Until this trigger condition is met, VehiDE alone remains the complete and sufficient training corpus for this milestone's deliverable.

---

## 8. Data Augmentation

Augmentation is applied at training time only using the Ultralytics YOLO augmentation engine. No augmented images are baked into the stored dataset files.

### 8.1 Augmentation Configuration

The following parameters are set in `damage.yaml` and `configs/augmentation.yaml`:

| **Augmentation** | **Parameter** | **Value** | **Rationale** |
| --- | --- | --- | --- |
| Horizontal flip | `fliplr` | 0.5 | Vehicle damage is equally likely on either side |
| Mosaic | `mosaic` | 1.0 | Combines 4 images; effective for multi-instance detection |
| Brightness/contrast (HSV-V) | `hsv_v` | 0.4 | Simulates different lighting conditions (day/night/overcast) |
| Saturation (HSV-S) | `hsv_s` | 0.7 | Simulates camera variability and weather effects |
| Rotation | `degrees` | 5.0 | Handles slightly tilted mobile phone camera angles |
| Translation | `translate` | 0.1 | Simulates off-centre framing by non-expert claimants |
| Scale | `scale` | 0.5 | Handles varying distances from vehicle |
| Motion blur | `blur` | 0.3 | Simulates handheld mobile camera shake; directly addresses domain shift risk (Milestone 1, Section 10.3) |
| JPEG compression | `jpeg_quality` | 75 | Simulates WhatsApp/email compression of claim photos |

### 8.2 Class-Targeted Oversampling

To address the 6.59:1 imbalance between `scratch` and `shattered_glass` (Section 5.2), minority classes (`shattered_glass` and `flat_tyre`) are oversampled during training by a factor of 2x using YOLO\'s `cls_pw` (class positive weight) parameter. The class weights (inverse-frequency, normalised to the largest class, computed on the final 32,672-instance training set) are:

| **Class** | **Instance count** | **Class weight** |
| --- | --- | --- |
| Scratch | 14,386 | 1.0 |
| Dent | 5,580 | 2.6 |
| Crack | 5,411 | 2.7 |
| Broken lamp | 2,733 | 5.3 |
| Flat tyre | 2,380 | 6.0 |
| Shattered glass | 2,182 | 6.6 |

---

## 9. Dataset Splitting

### 9.1 Splitting Approach

A stratified 70/15/15 train/validation/test split was applied to the integrated VehiDE dataset. Stratification was performed on the dominant damage class per image (the most frequent class label among all instances in that image) to ensure each split reflects the full class distribution.

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# dominant class per image
dominant = df.groupby("source_file")["class_id"].agg(
    lambda x: x.mode()[0]
)

train_imgs, temp = train_test_split(
    dominant.index, test_size=0.30, random_state=42,
    stratify=dominant.values
)
val_imgs, test_imgs = train_test_split(
    temp, test_size=0.50, random_state=42,
    stratify=dominant[temp].values
)
```

### 9.2 Split Sizes

| **Split** | **Images** | **Instances** | **% of total (images)** |
| --- | --- | --- | --- |
| Train | 9,558 | ~22,876 | 70.0% |
| Validation | 2,048 | ~4,901 | 15.0% |
| Test | 2,049 | ~4,895 | 15.0% |
| **Total** | **13,655** | **32,672** | |

### 9.3 Class Distribution per Split

Instance counts per class per split are estimated proportionally from the overall class distribution (Section 5.2) and the 70/15/15 image split ratio.

| **Class** | **Train** | **Validation** | **Test** |
| --- | --- | --- | --- |
| Scratch | 10,074 | 2,158 | 2,154 |
| Dent | 3,906 | 837 | 837 |
| Crack | 3,789 | 812 | 810 |
| Broken lamp | 1,914 | 410 | 409 |
| Flat tyre | 1,666 | 357 | 357 |
| Shattered glass | 1,528 | 327 | 327 |

The proportional class distributions are consistent across splits, confirming that stratification is expected to be effective.

### 9.4 Leakage Prevention

The following leakage checks were run after splitting:

| **Check** | **Method** | **Result** |
| --- | --- | --- |
| Exact image duplicates across splits | MD5 hash intersection | 0 cross-split duplicates found |
| Near-duplicate images across splits | pHash Hamming distance < 8 | 0 cross-split near-duplicates found |
| Vehicle-level leakage (same vehicle in multiple splits) | EXIF metadata clustering (where available) + visual similarity grouping | No systematic vehicle-level leakage detected |
| Policy document leakage | Policy chunks are not split; the full 5-document corpus is used exclusively for retrieval at inference time, not for training | Not applicable |

The pipeline halts and logs a warning if any cross-split hash match is detected, so future dataset updates cannot silently introduce leakage.

### 9.5 Escalation-Path Subset

A held-out subset of ~100 images will be deliberately selected from the test split to contain ambiguous damage (low-contrast scratches, partially occluded damage regions, damage near image boundaries). This subset will be used exclusively to test the orchestrator\'s escalation logic (routing low-confidence detections to the human review queue rather than auto-generating a report). These images will not used in any metric computation for the main evaluation.

---

## 10. Final Prepared Dataset

### 10.1 Vision Dataset Summary

| **Artefact** | **Size** | **Format** | **Location** | **Status** |
| --- | --- | --- | --- | --- |
| Training images | 9,558 images | JPEG 640x640, letterboxed | `data/vehide/images/train/` | Ready |
| Training annotations | 9,558 .txt files | YOLO normalised bbox | `data/vehide/labels/train/` | Ready |
| Validation images | 2,048 images | JPEG 640x640, letterboxed | `data/vehide/images/val/` | Ready |
| Validation annotations | 2,048 .txt files | YOLO normalised bbox | `data/vehide/labels/val/` | Ready |
| Test images | 2,049 images | JPEG 640x640, letterboxed | `data/vehide/images/test/` | Ready |
| Test annotations | 2,049 .txt files | YOLO normalised bbox | `data/vehide/labels/test/` | Ready |
| Escalation test subset | ~100 images + annotations | JPEG 640x640 | `data/vehide/escalation_test/` | To be selected once the model is trained and low confidence images are selected |
| YOLO config | 1 file | YAML | `data/damage.yaml` | Ready |
| Augmentation config | 1 file | YAML | `configs/augmentation.yaml` | Ready |
| Class remap lookup | 1 file | JSON | `configs/class_remap.json` | Ready |
| Split file lists | 3 .txt files | Plain text (one path per line) | `data/splits/` | Ready |

>Note: Due to repository size constraints, the full image dataset is not included. Only a small subset of sample images is available at the locations specified above. These samples are >provided for reference and may not be representative of the overall dataset distribution.

### 10.2 Policy Corpus Summary

| **Artefact** | **Size** | **Format** | **Location** | **Status** |
| --- | --- | --- | --- | --- |
| Synthetic policy PDFs | 5 PDFs | PDF | `data/policy_pdfs/synthetic/` | Ready |
| Reference policy PDFs | 2 PDFs | PDF | `data/policy_pdfs/reference/` | Ready (reference only) |
| ChromaDB vector index | 179 chunks | ChromaDB persistent collection | `data/chroma_db/` | Ready |
| Ground-truth clause mapping | 179 entries | JSON | `data/clause_groundtruth.json` | Ready |
| Synthetic incident descriptions | 50 records | JSON | `data/eval/incident_descriptions.json` | Ready |

### 10.3 Readiness Confirmation

**Any team member can begin YOLO fine-tuning immediately by running:**

```bash
yolo train \
  data=data/damage.yaml \
  model=yolo11m-seg.pt \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  optimizer=AdamW \
  project=runs/damage_detection \
  name=baseline
```

**The Policy Agent can begin retrieval testing immediately by running:**

```python
import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection("policy_clauses")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = "Is dent damage covered under accidental external means?"
results = collection.query(
    query_embeddings=[model.encode(query).tolist()],
    n_results=3
)
```

---

## 11. Challenges Encountered

| **Challenge** | **Details** | **Resolution / Remaining limitation** |
| --- | --- | --- |
| Class imbalance (6.59:1 ratio) | `scratch` accounts for 44.0% of retained instances; `shattered_glass` represents only 6.7% | Class-weighted loss and 2x oversampling applied; per-class F1 will be monitored separately |
| Mapping choice for `torn_body` | We initially merged `torn_body` into `scratch`, which would have inflated the imbalance ratio to over 9:1 and lost `crack` as a separately detectable class | Resolved by mapping `torn_body` to its own `crack` class instead (Section 3.1), which also keeps the vision taxonomy aligned with the policy corpus's crack-related clauses (Section 3.3) |
| CarDD dataset access | CarDD is distributed via a manual licensing form rather than a direct download, so the CarDD EDA notebook and dataset integration (Section 7) could not be completed with real data during this milestone | Licensing form to be submitted based on baseline model performance; Integration to be completed if it is required |
| Geographic bias in VehiDE | Dataset constructed primarily from Southeast Asian vehicle images; Indian vehicle types and claim conditions may be underrepresented | Addressed through augmentation (brightness, blur, compression); acknowledged as a domain shift risk in Section 4.4 |
| Annotation format mismatch (VehiDE vs CarDD) | VehiDE uses bounding-box YOLO format derived from VIA polygon annotations; CarDD uses COCO polygon segmentation format | Conversion script drafted (`scripts/coco_to_yolo_seg.py`); to be run and spot-checked if CarDD data is used |


---

## 12. Deliverables Produced

The following artefacts are committed to the project GitHub repository at [github.com/HiveCase/Group-1-DS-and-AI-Lab-Project](https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project):

| **Deliverable** | **File / Location** | **Description** |
| --- | --- | --- |
| EDA notebook - VehiDE | `notebooks/EDA/VehiDE_Dataset_EDA.ipynb` | Plots, statistics, and quality-check outputs for the primary dataset |
| EDA notebook - CarDD | `notebooks/EDA/CarDD_EDA.ipynb` | Structure in place; pending re-run once the licensed archive is obtained |
| EDA notebook - COCO Car Damage | `notebooks/EDA/COCO_Car_Damage_Detection_EDA.ipynb` | Complete; used for architecture sanity-check comparison |
| EDA notebook — Car Damage Severity | `notebooks/EDA/Car_Damage_Severity_EDA.ipynb` | Structure in place; pending re-run |
| Preprocessing script | `scripts/preprocess_vehide.py` | Corrupt/duplicate removal, class remapping, letterboxing, PII detection |
| Class remap lookup | `configs/class_remap.json` | Vietnamese-to-project class mapping |
| YOLO config | `data/vehide_processed/damage.yaml` | 6-class detection config |
| Cleaned and split datasets | `data/vehide/images/{train,val,test}/`, `data/vehide/labels/{train,val,test}/` | Training-ready image and label sets |
| Synthetic policy corpus + ChromaDB index | `data/policy_pdfs/synthetic/`, `data/chroma_db/` | 5 PDFs, 179 indexed chunks |
| This report | `Milestone2_Report.md` | Documentation of dataset identification, EDA, preprocessing, and readiness |

---

## 13. Summary and Next Steps

### 13.1 Summary of Work Completed

This milestone identified, verified, downloaded, and prepared the datasets required for the four agents of the multi-agent claim assessment system. VehiDE was confirmed as the primary training dataset (13,655 images, 32,672 retained instances after quality checks and class exclusion), with CarDD (segmentation masks) and the Car Damage Severity dataset (severity calibration) identified but not yet integrated pending data access (Section 11); COCO Car Damage was fully profiled for architecture comparison. A comprehensive EDA revealed a 6.59:1 class imbalance between `scratch` and `shattered_glass`, a right-skewed bounding box area distribution, and a mean of 2.58 instances per image. Preprocessing steps for VehiDE (corrupt-file check, PII detection, class remapping, deduplication, stratified splitting, leakage verification) have been executed and scripted. A synthetic policy corpus of 5 documents (179 chunks, embedded into ChromaDB) was authored, varied in phrasing, and indexed.

### 13.2 Key Observations from the Data

- The 6.59:1 scratch-to-shattered_glass imbalance is the most significant data quality concern. Without class weighting, the model will likely meet the overall mAP@50 target but fail the per-class F1 target of >= 0.65 for shattered_glass and flat_tyre.
- At 2.58 instances per image, multi-label detection will be the norm.
- `torn_body` is mapped to `crack` class (Section 3.1), which keeps the vision taxonomy aligned with the policy corpus's crack-related clauses (Section 3.3) and avoids further inflating the already-dominant `scratch` class.
- Phrasing variation in the synthetic policy corpus produced noticeably different retrieval difficulty across documents, which is the intended outcome.

### 13.3 Confirmation of Training Readiness

The VehiDE-based vision dataset is ready for model training: the train/validation/test split is finalised, leakage-checked, and the YOLO configuration file (`damage.yaml`, `nc: 6`) is verified. The ChromaDB policy index (179 chunks) is built and queryable. The YOLO fine-tuning can be done on VehiDE without performing any additional data preparation; CarDD-based segmentation augmentation and severity-proxy calibration remain as backup if the performance of the model falls below the original target values.

### 13.4 Planned Activities for Milestone 3

- Select final model architecture: YOLO11m-seg vs YOLOv8m-seg baseline comparison.
- Define the full multi-agent pipeline code structure: LangGraph orchestrator state schema, MCP tool I/O contracts for each agent.
- Run a YOLO baseline training run (50 epochs) to establish initial mAP@50 and per-class F1 benchmarks on the 6-class taxonomy.
- Wire up the Policy Agent\'s FastMCP retrieval tool against the ChromaDB index and run a first-pass retrieval precision check on the ground-truth test set.
- Validate prompt template for the Report Agent against 5 sample incident/image pairs.
- And if the model performs poorly, then integrate CarDD data into the training set.

  
---

## 14. References

[1] H. Scullen, "VehiDE: Vehicle Damage Detection Dataset," Kaggle, 2023. Available: [https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection)

[2] Van-Dung Hoang, Nhan T. Huynh et al., "VehiDE Dataset: New dataset for Automatic vehicle damage detection in Car insurance," IEEE Conference Publication / J. Inf. Telecommun., 2023.

[3] "Powering AI-driven car damage identification based on VeHIDE dataset," Journal article, Taylor & Francis Online, 2024.

[4] S. Wang et al., "CarDD: A New Dataset for Vision-Based Car Damage Detection," University of Science and Technology of China (USTC), 2023. Available: [https://cardd-ustc.github.io](https://cardd-ustc.github.io)

[5] "Coco Car Damage Detection Dataset," Kaggle. Available: [https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset)

[6] J. Johnson, M. Douze, and H. Jegou, "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535-547, 2021.

[7] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019.

[8] Universal Sompo General Insurance Co. Ltd, "Motor Private Car 3 Years Policy Wordings," UIN: IRDAN134RP0003V01201819. Available: publicly filed with IRDAI.

[9] United India Insurance Company Limited, "Private Car Standalone Own Damage Policy," UIN: IRDAN545RP0001V01201920. Available: publicly filed with IRDAI.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | 09-07-2026 | S.K. |
| Pranab Kumar Manna | 09-07-2026  | Pk Manna |
| Venkata Siva Kamal Guddanti | 09-07-2026 | Kamal G |
| Anuj Gautam |  |  |

---
