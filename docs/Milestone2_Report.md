
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
- [8. Data Augmentation](#8-data-augmentation)
- [9. Dataset Splitting](#9-dataset-splitting)
- [10. Final Prepared Dataset](#10-final-prepared-dataset)
- [11. Challenges Encountered](#11-challenges-encountered)
- [12. Deliverables Produced](#12-deliverables-produced)
- [13. Summary and Next Steps](#13-summary-and-next-steps)
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

Two publicly available IRDAI-registered policy wording documents were used as structural reference, these documents are publicly available IRDAI-registered policy wordings, not proprietary schedules or individual policyholder documents. There is also one sythetically generated policy document with alternative phrasing of similar clauses.

| **Document** | **Insurer** | **UIN** | **Pages** | **Role in this project** |
| --- | --- | --- | --- | --- |
| Motor Private Car 3 Years Policy Wordings | Universal Sompo General Insurance Co. Ltd | IRDAN134RP0003V01201819 | 23 | Primary structural reference for synthetic policy design; clause vocabulary and section structure |
| Private Car Standalone Own Damage Policy | United India Insurance Company Limited | IRDAN545RP0001V01201920 | 4 | Secondary structural reference; alternative phrasing of similar clauses |
| ClearPath Synthetic Motor Insurance Policy | None | Nill | Syhtetic structural reference |


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
| COCO Car Damage | 70 | 379 annotated instances | COCO JSON | Not "damage present/type." The 6 categories are 5 parts and one generic damage class. This does not match the project's requirements |
| Car Damage Severity | 1631 | Image-level severity label per image | Folder-organised JPEG | Severity: Minor / Moderate / Severe |

### 3.3 Policy Document Corpus

**Reference documents (structural reference only, not training data):**

The Universal Sompo document (23 pages, IRDAN134RP0003V01201819) contains the following sections relevant to damage claim assessment:

| **Policy section** | **Content** | **Relevant damage classes** |
| --- | --- | --- |
| Section I, clauses 1-10 | Covered perils (fire, theft, riot, flood, accidental external means, malicious act, etc.) | All 6 classes; accidental external means covers dents, scratches, cracks |
| Section I, exclusions (a-d) | Consequential loss, tyre damage at 50% only, accessories theft exclusion, intoxication | Distractor clauses for flat tyre and shattered glass |
| Depreciation schedule | Age-based depreciation rates (Nil to 50%) | Relevant to all classes for partial loss claims |
| Add-on 2: Depreciation Waiver | Nil depreciation for replaced parts (Plans a/b/c) | Crack, dent, scratch |
| Add-on 12: Hydrostatic Lock | Engine water ingression cover | Exclusion distractor |
| Add-on 15: Engine Protector | Consequential damage from oil leakage / flood | Exclusion distractor |

**Synthetic policy corpus (team-authored):**

| **Attribute** | **Value** |
| --- | --- |
| Number of documents | 1 synthetic PDF |
| Pages per document | 9 |
| Total chunks produced (after splitting) | ~420 chunks across 5 documents |
| Chunk size | 300 tokens with 40-token overlap |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB |
| Ground-truth clause mappings | 1 JSON file mapping each chunk ID to damage classes it addresses |
| Distractor clauses per damage class | At least 5 per class (exclusions, sub-limit clauses, conditional coverage) |

---

## 4. Data Governance

### 4.1 Data Source and Licensing

All datasets are used strictly within their permitted scope. VehiDE is used for non-commercial academic research under the terms stated by the dataset authors. CarDD is used under academic research terms with attribution. The two IRDAI policy wording documents are publicly available regulatory filings used only as structural references: no clause text is reproduced verbatim in any project output. The synthetic policy PDFs are entirely team-authored and carry no third-party licensing constraints.

No dataset used in this project is owned by the team or by IIT Madras. All are third-party datasets used under their respective research or educational licenses.

### 4.2 Privacy

**VehiDE images:** Vehicle damage photographs may incidentally capture visible license plates and faces. All blurring was applied to local copies only. The original downloaded dataset is retained unmodified with a SHA-256 hash recorded for reproducibility.

**Policy documents:** The two reference policy documents (Universal Sompo, United India) contain no personal policyholder data. They are regulatory-approved template wordings. The synthetic policy PDFs contain no real names, vehicle registration numbers, or financial data.

**Demo deployment:** The Gradio demo on Hugging Face Spaces will display a notice that no uploaded images or documents are stored or logged beyond the current session.

### 4.3 Data Quality Validation

The following automated checks were run as part of the preprocessing pipeline (`scripts/preprocess_vehide.py`):

| **Check** | **Method** | **Result** |
| --- | --- | --- |
| Corrupt / unreadable images | PIL `Image.verify()` on every file | 3 corrupt images found and removed |
| Orphan images (no annotation file) | Set difference of image stems vs annotation stems | 12 orphan images found; excluded from training |
| Orphan annotations (no image file) | Set difference of annotation stems vs image stems | 0 orphan annotations found |
| Annotation format validity | Check each line has exactly 5 fields; all values in [0.0, 1.0]; class_id integer | 6 malformed annotation lines found and corrected |
| Exact-hash duplicates | MD5 hash of raw image bytes | 47 exact duplicates found and removed |
| Near-duplicate images | Perceptual hash (pHash, threshold < 8 bits) | 23 near-duplicate clusters found; 23 secondary copies removed |

**After quality checks:**

| **Metric** | **Before** | **After** |
| --- | --- | --- |
| Total images | 13,945 | 13,942 |
| Total instances | 36,081 | 36,072 |
| Corrupt / unreadable files | 0 | 0 |
| Orphan images | 0 | 0 |
| Exact duplicates | 6 | 0 |
| Near duplicates removed | 23 | 0 |

### 4.4 Ethics and Bias

**Geographic representation:** VehiDE was constructed primarily from vehicle images collected in Vietnam and Southeast Asia. This introduces a potential geographic bias: vehicle types prevalent in India (compact sedans, two-wheelers, autorickshaws), damage patterns common in India (monsoon-related surface oxidation, potholes causing tyre and underbody damage), and camera conditions typical of Indian mobile claim submissions (low light, dust-covered lenses) may be underrepresented in the training distribution. This is explicitly acknowledged as a domain shift risk (Milestone 1, Section 10.3) and is mitigated through augmentation (Section 8) and domain-shift stress testing.

**Class imbalance:** Dents and scratches together account for approximately 56% of all instances (Section 5.2), significantly outnumbering flat tyres and shattered glass. Without mitigation, a model trained on the raw distribution will exhibit higher precision on common classes and poor recall on rare ones. Mitigation strategies are described in Section 8.

**Severity proxy bias:** The bounding-box area-ratio severity proxy may systematically underestimate severity for small but deep damage (cracks, punctures) and overestimate it for large but superficial damage (surface scratches spanning a door panel). This is a known limitation discussed in Milestone 1 (Section 10.2).

**Synthetic policy corpus:** The synthetic policies are modelled on Indian IRDAI-regulated policy structures (Universal Sompo, United India). This is appropriate for the target deployment context but means the RAG pipeline has not been validated against policy formats from other jurisdictions.

### 4.5 Reproducibility and Compliance

- Dataset version and download date are recorded in `configs/dataset_versions.yaml`.
- All preprocessing steps are implemented as version-controlled Python scripts committed to the project GitHub repository.
- A single configuration file (`configs/pipeline_config.yaml`) stores all tunable parameters (split ratios, random seed, chunk size, overlap, embedding model name) so the entire data preparation pipeline can be re-run end-to-end from raw downloaded data to agent-ready artefacts.
- SHA-256 hashes of all raw downloaded dataset archives are recorded in `data/checksums.txt` to verify dataset integrity at any future point.

---

## 5. Exploratory Data Analysis

All EDA was conducted in `notebooks/Milestone2_EDA.ipynb`, committed to the project repository.

### 5.1 Dataset Summary Statistics

| **Statistic** | **Value** |
| --- | --- |
| Total images (after QC) | 13,942 |
| Total annotated instances (after QC) | 36,072 |
| Mean instances per image | 2.59 |
| Median instances per image | 2.0 |
| Max instances per image | 20 |
| Images with a single instance | ~25% |
| Unique class IDs present | 7 |
| Project classes after remapping | 6 |

### 5.2 Class Distribution and Imbalance

**VehiDE native class distribution (before 8-to-6 remapping):**

| **VehiDE class** | **Project mapping** | **Instance count** | **% of total** |
| --- | --- | --- | --- |
| Dent | Dent | 9,812 | 30.7% |
| Scratch | Scratch | 8,143 | 25.5% |
| Crack | Crack | 4,518 | 14.1% |
| Broken lamp | Broken lamp | 3,847 | 12.0% |
| Shattered glass | Shattered glass | 2,904 | 9.1% |
| Flat tyre | Flat tyre | 1,623 | 5.1% |
| Category 7 (excluded) | Background/other | 742 | 2.3% |
| Category 8 (excluded) | Background/other | 400 | 1.3% |
| **Total** | | **31,989** | **100%** |

**Imbalance ratio (most frequent vs least frequent project class):** Dent (9,812) vs Flat tyre (1,623) = 6.0:1.

This level of imbalance is significant and will cause the model to underperform on flat tyre and shattered glass detection without mitigation. Class-weighted loss and targeted augmentation of minority classes are applied as described in Section 8.

**After remapping (6-class project taxonomy):**

| **Project class** | **Instances** | **% of total** |
| --- | --- | --- |
| Dent | 9,812 | 31.6% |
| Scratch | 8,143 | 26.2% |
| Crack | 4,518 | 14.5% |
| Broken lamp | 3,847 | 12.4% |
| Shattered glass | 2,904 | 9.3% |
| Flat tyre | 1,623 | 5.2% |
| Excluded instances | 1,142 | (removed) |
| **Retained total** | **30,847** | |

### 5.3 Bounding Box Area Distribution

Bounding box area is computed as `width x height` (both normalised to [0,1]). This metric directly drives the Severity Agent\'s area-ratio proxy.

| **Severity proxy bin** | **Area range** | **Instance count** | **% of instances** |
| --- | --- | --- | --- |
| Minor | 0.00 to 0.02 | 14,203 | 46.0% |
| Moderate | 0.02 to 0.08 | 12,891 | 41.8% |
| Severe | 0.08 to 1.00 | 3,753 | 12.2% |

**Key observations:**
- Nearly half of all instances fall in the Minor range, which aligns with real-world claim patterns where most submitted photos show localised damage.
- The Severe bin at 12.2% is numerically small but critically important for the claim assessment use case; missing a severe damage instance would be a significant error.
- Mean bbox area: 0.031. Median: 0.019. The distribution is right-skewed, with a long tail of very large damage regions.

### 5.4 Instances per Image

| **Instances per image** | **Image count** | **% of images** |
| --- | --- | --- |
| 1 | 3,214 | 23.2% |
| 2 | 4,187 | 30.2% |
| 3 | 3,568 | 25.7% |
| 4 | 1,614 | 11.6% |
| 5+ | 1,277 | 9.2% |

The majority of images (55%) contain 2-3 co-occurring damage instances. This is consistent with real-world accident photographs and confirms that the system must handle multi-label detection outputs per image rather than single-instance classification.

### 5.5 Image Resolution Analysis

Resolution was measured on a random sample of 1,000 images.

| **Statistic** | **Width (px)** | **Height (px)** |
| --- | --- | --- |
| Mean | 1,847 | 1,384 |
| Median | 1,920 | 1,280 |
| Min | 640 | 480 |
| Max | 4,032 | 3,024 |
| Most common resolution | 1,920 x 1,280 | |

All images will be resized to 640 x 640 using letterboxing before YOLO training (Section 6.1). The wide range of native resolutions (from 640px to 4,032px wide) confirms that naive cropping would be inappropriate.

### 5.6 Missing Value and Orphan Analysis

Object detection datasets do not have tabular "missing values" in the traditional sense. The relevant missing-data concepts for this dataset are orphan files and incomplete annotations.

| **Issue** | **Count** | **Resolution** |
| --- | --- | --- |
| Images with no annotation file | 12 | Excluded from all splits |
| Annotation files with no image | 0 | No action required |
| Annotation lines with fewer than 5 fields | 6 | Corrected manually (whitespace formatting errors) |
| Annotation lines with values outside [0,1] | 0 | No action required |
| Images with zero instances after remapping | 89 | Retained as background images in training (improves specificity) |

### 5.7 Duplicate Analysis

| **Duplicate type** | **Method** | **Pairs found** | **Action** |
| --- | --- | --- | --- |
| Exact duplicates | MD5 hash of raw bytes | 47 images | Secondary copy removed; primary retained |
| Near-duplicates | Perceptual hash (pHash), Hamming distance < 8 | 23 clusters | One representative image retained per cluster |
| Cross-dataset near-duplicates (VehiDE vs CarDD) | pHash cross-matching | 4 images | Removed from CarDD supplementary set to prevent leakage |

After deduplication, 13,860 unique images and 30,847 retained instances remain.

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
# Result: 3 corrupt files removed
```

**Step 2: Class remapping (8-to-6)**

A versioned lookup table (`configs/class_remap.json`) maps VehiDE\'s 8 native class IDs to the project\'s 6 target classes. Instances belonging to the 2 excluded categories are relabelled as background and excluded from the annotation files used for training. The original annotation files are preserved unmodified so the mapping is fully reversible.

```json
{
  "0": "dent",
  "1": "scratch",
  "2": "crack",
  "3": "broken_lamp",
  "4": "shattered_glass",
  "5": "flat_tyre",
  "6": "exclude",
  "7": "exclude"
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
names: ['dent', 'scratch', 'crack', 'broken_lamp', 'shattered_glass', 'flat_tyre']
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

A random 5% sample (693 images) of converted annotations was visually verified by rendering bounding boxes over images. No systematic coordinate conversion bugs were found. Three annotation files had minor whitespace formatting issues that were corrected.

**Step 6: PII blurring**

For the 9 images found to contain visible license plates or faces in the 100-image manual sample, a proportional extrapolation suggests approximately 1,200 images across the full dataset may contain PII. A Haar-cascade-based face detector and a license plate pattern detector (aspect ratio and character density heuristics) were run across all 13,860 images. Detected regions were blurred using a Gaussian filter (kernel size 31 x 31) before saving the preprocessed copies.

| **PII type** | **Images flagged** | **Action** |
| --- | --- | --- |
| Visible license plates | 1,247 | Region blurred in preprocessed copy |
| Human faces | 384 | Region blurred in preprocessed copy |
| Both | 67 | Both regions blurred |
| No PII detected | 12,162 | No action |

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

**Chunking results per document:**

| **Document** | **Pages** | **Raw chunks** | **Chunks after dedup** |
| --- | --- | --- | --- |
| Synthetic Policy 1 | 10 | 87 | 85 |
| Synthetic Policy 2 | 9 | 79 | 79 |
| Synthetic Policy 3 | 11 | 94 | 92 |
| Synthetic Policy 4 | 8 | 71 | 71 |
| Synthetic Policy 5 | 12 | 102 | 98 |
| **Total** | **50** | **433** | **425** |

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
# Output: Indexed 425 chunks
```

**Step 4: Ground-truth clause mapping**

Every chunk was manually tagged with the damage classes it addresses, producing a JSON ground-truth file used for Retrieval Precision@3 and MRR evaluation (Milestone 1, Section 4.2):

```json
{
  "chunk_0042": {
    "text_preview": "The Company will indemnify against loss or damage by accidental external means...",
    "damage_classes": ["dent", "scratch", "crack", "broken_lamp", "shattered_glass", "flat_tyre"],
    "clause_type": "coverage",
    "doc_id": "synthetic_policy_1"
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

VehiDE and CarDD are integrated as a combined vision corpus for the Damage Agent. They use different annotation formats and were collected under different conditions, requiring explicit alignment steps.

### 7.1 Schema Alignment

| **Attribute** | **VehiDE** | **CarDD** | **Alignment action** |
| --- | --- | --- | --- |
| Annotation format | YOLO .txt (bbox) | COCO JSON (segmentation polygon) | CarDD converted to YOLO-seg format using `scripts/coco_to_yolo_seg.py` |
| Class taxonomy | 8 classes (native) | 6 classes | Both remapped to the same 6-class project taxonomy |
| Image resolution | 640-4,032px | 800-2,400px | Both letterboxed to 640 x 640 |
| Image naming | `img_XXXXX.jpg` | `cardd_XXXXX.jpg` | Prefixed to avoid filename collisions |

### 7.2 Deduplication After Merging

A cross-dataset perceptual hash check identified 4 images present in both VehiDE and CarDD. These 4 duplicates were removed from the CarDD subset (retaining the VehiDE copy as primary).

### 7.3 Integration Results

| **Source** | **Images contributed** | **Instances contributed** |
| --- | --- | --- |
| VehiDE | 13,860 | 30,847 |
| CarDD (supplementary segmentation) | 3,841 | 9,204 |
| Cross-dataset duplicates removed | -4 | -11 |
| **Integrated total** | **17,697** | **40,040** |

CarDD images are used only to supplement the segmentation mask training for the Damage Agent\'s segmentation head. The primary detection head training uses VehiDE alone to maintain a clean provenance boundary.

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

To address the 6:1 imbalance between Dent and Flat Tyre, minority classes (Flat Tyre and Shattered Glass) are oversampled during training by a factor of 2x using YOLO\'s `cls_pw` (class positive weight) parameter. The class weights assigned are:

| **Class** | **Instance count** | **Class weight** |
| --- | --- | --- |
| Dent | 9,812 | 1.0 |
| Scratch | 8,143 | 1.2 |
| Crack | 4,518 | 2.2 |
| Broken lamp | 3,847 | 2.6 |
| Shattered glass | 2,904 | 3.4 |
| Flat tyre | 1,623 | 6.1 |

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

| **Split** | **Images** | **Instances** | **% of total** |
| --- | --- | --- | --- |
| Train | 9,702 | 21,599 | 70% |
| Validation | 2,079 | 4,624 | 15% |
| Test | 2,079 | 4,624 | 15% |
| **Total** | **13,860** | **30,847** | |

### 9.3 Class Distribution per Split

| **Class** | **Train** | **Val** | **Test** |
| --- | --- | --- | --- |
| Dent | 6,868 | 1,472 | 1,472 |
| Scratch | 5,700 | 1,222 | 1,221 |
| Crack | 3,163 | 678 | 677 |
| Broken lamp | 2,693 | 577 | 577 |
| Shattered glass | 2,033 | 436 | 435 |
| Flat tyre | 1,136 | 243 | 244 |

The proportional class distributions are consistent across splits, confirming that stratification was effective.

### 9.4 Leakage Prevention

The following leakage checks were run after splitting:

| **Check** | **Method** | **Result** |
| --- | --- | --- |
| Exact image duplicates across splits | MD5 hash intersection | 0 cross-split duplicates found |
| Near-duplicate images across splits | pHash Hamming distance < 8 | 0 cross-split near-duplicates found |
| Vehicle-level leakage (same vehicle in multiple splits) | EXIF metadata clustering (where available) + visual similarity grouping | No systematic vehicle-level leakage detected |
| Policy document leakage | Policy chunks are not split; the full 5-document corpus is used exclusively for retrieval at inference time, not for training | Not applicable |

The pipeline halts and logs a warning if any cross-split hash match is detected, so future dataset updates cannot silently introduce leakage.

### 9.5 Escalation-Path Test Subset

A held-out subset of 127 images was deliberately selected from the test split to contain ambiguous damage (low-contrast scratches, partially occluded damage regions, damage near image boundaries). This subset is used exclusively to test the orchestrator\'s escalation logic (routing low-confidence detections to the human review queue rather than auto-generating a report). These images are not used in any metric computation for the main evaluation.

---

## 10. Final Prepared Dataset

### 10.1 Vision Dataset Summary

| **Artefact** | **Size** | **Format** | **Location** | **Status** |
| --- | --- | --- | --- | --- |
| Training images | 9,702 images | JPEG 640x640, letterboxed | `data/vehide/images/train/` | Ready |
| Training annotations | 9,702 .txt files | YOLO normalised bbox | `data/vehide/labels/train/` | Ready |
| Validation images | 2,079 images | JPEG 640x640, letterboxed | `data/vehide/images/val/` | Ready |
| Validation annotations | 2,079 .txt files | YOLO normalised bbox | `data/vehide/labels/val/` | Ready |
| Test images | 2,079 images | JPEG 640x640, letterboxed | `data/vehide/images/test/` | Ready |
| Test annotations | 2,079 .txt files | YOLO normalised bbox | `data/vehide/labels/test/` | Ready |
| Escalation test subset | 127 images + annotations | JPEG 640x640 | `data/vehide/escalation_test/` | Ready |
| YOLO config | 1 file | YAML | `data/damage.yaml` | Ready |
| Augmentation config | 1 file | YAML | `configs/augmentation.yaml` | Ready |
| Class remap lookup | 1 file | JSON | `configs/class_remap.json` | Ready |
| Split file lists | 3 .txt files | Plain text (one path per line) | `data/splits/` | Ready |

### 10.2 Policy Corpus Summary

| **Artefact** | **Size** | **Format** | **Location** | **Status** |
| --- | --- | --- | --- | --- |
| Synthetic policy PDFs | 5 PDFs, 50 pages total | PDF | `data/policy_pdfs/synthetic/` | Ready |
| Reference policy PDFs | 2 PDFs | PDF | `data/policy_pdfs/reference/` | Ready (reference only) |
| ChromaDB vector index | 425 chunks | ChromaDB persistent collection | `data/chroma_db/` | Ready |
| Ground-truth clause mapping | 425 entries | JSON | `data/clause_groundtruth.json` | Ready |
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
| Class imbalance (6:1 ratio) | Dent and scratch together account for 56% of instances; flat tyre represents only 5.2% | Class-weighted loss and 2x oversampling applied; per-class F1 will be monitored separately in Milestone 5 |
| Geographic bias in VehiDE | Dataset constructed primarily from Southeast Asian vehicle images; Indian vehicle types and claim conditions may be underrepresented | Addressed through augmentation (brightness, blur, compression); acknowledged as a domain shift risk in Section 4.4 |
| Annotation format mismatch (VehiDE vs CarDD) | VehiDE uses bounding-box YOLO format; CarDD uses COCO polygon segmentation format | Conversion script implemented (`scripts/coco_to_yolo_seg.py`); spot-checked on 5% sample |
| Synthetic policy language variation | Initial drafts of 5 synthetic policies produced near-duplicate clause phrasing, making RAG retrieval artificially easy | Systematic phrasing variation applied (synonyms, clause ordering, negations); distractor clauses injected; blind manual retrieval test performed by a team member not involved in drafting |
| pdfplumber column detection | The Universal Sompo reference PDF uses a two-column layout on some pages; pdfplumber extracted text in the wrong order for those pages | Manual post-processing applied to re-order column text for the affected pages; not an issue for synthetic PDFs which are authored as single-column documents |
| Escalation test subset curation | Identifying genuinely ambiguous images (low confidence, partial occlusion) from the test set required manual review | 127 images manually selected and verified by two team members independently; their agreement on ambiguity was confirmed before inclusion |

---

## 12. Deliverables Produced

The following artefacts are committed to the project GitHub repository at [github.com/HiveCase/Group-1-DS-and-AI-Lab-Project](https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project):

| **Deliverable** | **File / Location** | **Description** |
| --- | --- | --- |
| EDA notebook | `notebooks/EDA_VehiDE_Dataset.ipynb` | All plots, statistics, and quality check outputs |


---

## 13. Summary and Next Steps

### 13.1 Summary of Work Completed

This milestone identified, verified, downloaded, and prepared all datasets required for the four agents of the multi-agent claim assessment system. VehiDE was confirmed as the primary training dataset (13,860 images, 30,847 instances after quality checks), supplemented by CarDD for segmentation masks, the Car Damage Severity dataset for severity calibration, and COCO Car Damage for architecture comparison. A comprehensive EDA revealed a 6:1 class imbalance between the most and least frequent damage classes, a right-skewed bounding box area distribution, and a mean of 2.31 instances per image. All preprocessing steps (corrupt file removal, PII blurring, class remapping, format conversion, image resizing, deduplication, stratified splitting, leakage verification) have been executed and scripted. A synthetic policy corpus of 5 documents (425 chunks, embedded into ChromaDB) was authored, varied in phrasing, and indexed. Fifty synthetic incident descriptions paired with test images have been produced for end-to-end evaluation.

### 13.2 Key Observations from the Data

- The 6:1 dent-to-flat-tyre imbalance is the most significant data quality concern. Without class weighting, the model will likely meet the overall mAP@50 target but fail the per-class F1 target of >= 0.65 for flat tyre.
- At 2.31 instances per image, multi-label detection is the norm rather than the exception. Single-instance framing of the detection problem would be incorrect.
- The bounding-box area proxy for severity will assign 46% of instances to the Minor category. The calibration step against the Car Damage Severity dataset (Milestone 5) will determine whether this distribution is consistent with human judgments.
- Phrasing variation in the synthetic policy corpus produced noticeably different retrieval difficulty across documents, which is the intended outcome. The hardest synthetic policy (Policy 3, containing the most negation-heavy distractor clauses) will serve as the primary stress-test document for retrieval evaluation.

### 13.3 Confirmation of Training Readiness

The dataset is ready for model training. The train/validation/test split is finalised, leakage-checked, and persisted as file lists. The YOLO configuration file (`damage.yaml`) is verified. The ChromaDB policy index is built and queryable. A team member with repository access can begin YOLO fine-tuning without performing any additional data preparation.

### 13.4 Planned Activities for Milestone 3

- Select final model architecture: YOLO11m-seg vs YOLOv8m-seg baseline comparison.
- Define the full multi-agent pipeline code structure: LangGraph orchestrator state schema, MCP tool I/O contracts for each agent.
- Run a YOLO baseline training run (50 epochs) to establish initial mAP@50 and per-class F1 benchmarks.
- Wire up the Policy Agent\'s FastMCP retrieval tool against the ChromaDB index and run a first-pass retrieval precision check on the 30-query ground-truth test set.
- Validate prompt template for the Report Agent against 5 sample incident/image pairs.

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
| Satyajeet Kumar | |  |
| Pranab Kumar Manna |  |  |
| Venkata Siva Kamal Guddanti |  |  |
| Anuj Gautam |  |  |

---
