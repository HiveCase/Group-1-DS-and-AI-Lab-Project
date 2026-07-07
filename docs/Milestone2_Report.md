
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


---

## 1. Introduction

### 1.1 Project Recap

Milestone 1 defined the problem, scope, and evaluation plan for a multimodal vehicle damage assessment system that accepts vehicle damage photographs and insurance policy documents as input, detects and classifies visible damage using a fine-tuned YOLO-based model, retrieves relevant policy clauses using a Retrieval-Augmented Generation (RAG) pipeline, and generates a structured preliminary claim assessment report using an LLM. Since Milestone 1, the system design has been refined into a multi-agent RAG architecture in which a LangGraph orchestrator routes each claim to four specialist agents: a Damage Agent (YOLOv8 detection), a Severity Agent (bounding-box area-ratio scoring), a Policy Agent (RAG over the synthetic policy corpus, exposed as an MCP tool), and a Report Agent (LLM-based report writing), with low-confidence outputs escalated to a human review queue. This refinement extends the modular separation-of-concerns argument already established in Milestone 1 (Section 3.4) by adding a routing layer that can act conditionally per claim, e.g. skipping the Policy Agent when no PDF is supplied, or escalating low-confidence detections to human review rather than always executing a fixed three-stage sequence.

### 1.2 Objectives of Milestone 2

The three primary objectives of this milestone are:

1. **Dataset verification and download**: Identify, verify provenance, confirm licensing, and download all datasets required by each agent.
2. **Vision data preparation**: Assess quality, preprocess, augment, and split the VehiDE dataset and supplementary vision datasets into training-ready artefacts for the Damage and Severity Agents.
3. **Policy corpus construction**: Author, review, chunk, embed, and index the synthetic insurance policy document corpus for the Policy Agent, using publicly available IRDAI-registered policy wordings as structural references.

By the end of this milestone, any team member with repository access should be able to begin YOLO fine-tuning immediately by running `yolo train data=damage.yaml model=yolo11m-seg.pt epochs=50` without performing any additional data preparation.

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
| License | Non-commercial research and educational use (per dataset card and companion paper [2]) |
| Purpose in this project | Primary training and evaluation dataset for the Damage Agent |
| Why selected | Largest publicly available annotated vehicle damage dataset (13,945 images, 32k+ instances); peer-reviewed construction paper; covers all 6 target damage classes; supports detection, segmentation, and salient object detection tasks |
| Alternatives considered | Stanford Cars: classification only, no damage annotations. EMSCAD: job fraud domain, not vehicle damage. Raw web scrapes: no consistent annotation standard, no peer-reviewed quality verification. VehiDE was the only dataset combining scale, annotation quality, and peer-reviewed provenance. |

#### Supplementary Vision Datasets

| **Dataset** | **Download Link** | **License** | **Purpose** | **Why selected** | **Alternatives considered** |
| --- | --- | --- | --- | --- | --- |
| CarDD | [cardd-ustc.github.io](https://cardd-ustc.github.io/) | Academic research use | Pixel-level segmentation masks for irregularly shaped damage (scratches, cracks) not well-represented by bounding boxes alone | Only public dataset with pixel-level damage segmentation across 6 damage categories with a peer-reviewed benchmark | None found with comparable segmentation quality |
| COCO Car Damage | [kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset) | Community (Kaggle) | Architecture comparison and pipeline sanity-checking | COCO format allows direct benchmark comparison against published COCO-trained baselines | Not applicable for this auxiliary role |
| Car Damage Severity | [kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset) | Community (Kaggle) | Calibrating the Severity Agent\'s bounding-box area-ratio proxy against human-labelled severity | Only public dataset with human-assigned Minor/Moderate/Severe labels matching our three-category scheme | None found with human severity labels |

### 2.2 Policy and Text Datasets

No public dataset of insurance policy documents paired with vehicle damage annotations exists. The policy corpus is therefore synthetic, authored by the team. Two publicly available IRDAI-registered policy wording documents were used as structural reference:
| **Document** | **Insurer** | **UIN** | **Pages** | **Role in this project** |
| --- | --- | --- | --- | --- |
| Motor Private Car 3 Years Policy Wordings | Universal Sompo General Insurance Co. Ltd | IRDAN134RP0003V01201819 | 23 | Primary structural reference for synthetic policy design; clause vocabulary and section structure |
| Private Car Standalone Own Damage Policy | United India Insurance Company Limited | IRDAN545RP0001V01201920 | 4 | Secondary structural reference; alternative phrasing of similar clauses |

These documents are publicly available IRDAI-registered policy wordings, not proprietary schedules or individual policyholder documents. No clause text from either document is reproduced verbatim in the synthetic corpus.



### 2.3 Ownership, Licensing, and Usage Constraints

| **Dataset** | **Ownership** | **Permitted use** | **Restrictions** |
| --- | --- | --- | --- |
| VehiDE | Dataset authors (Scullen et al.) | Non-commercial research and education | No commercial use; attribution required |
| CarDD | USTC research group | Academic research | Attribution to original paper required |
| COCO Car Damage | Kaggle community uploader | Community use | Cite dataset page |
| Car Damage Severity | Kaggle community uploader | Community use | Cite dataset page |
| Universal Sompo policy | Universal Sompo General Insurance Co. Ltd | Publicly available IRDAI filing | Used as structural reference only; no verbatim reproduction |
| United India Insurance policy | United India Insurance Co. Ltd | Publicly available IRDAI filing | Used as structural reference only; no verbatim reproduction |
| Synthetic policy PDFs | This project team | Fully team-owned | No restrictions |
