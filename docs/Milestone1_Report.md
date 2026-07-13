
---

<div align="center">


<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">


<h1 style="font-size:26em;">Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 1: Problem Definition & Literature Review</h2>

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

- [1. Problem Statement](#1-problem-statement)
  - [1.1 What problem are we solving?](#11-what-problem-are-we-solving)
  - [1.2 Who are the stakeholders?](#12-who-are-the-stakeholders)
  - [1.3 Scope Definition](#13-scope-definition)
- [2. Problem Motivation](#2-problem-motivation)
- [3. Existing Solutions and Prior Research](#3-existing-solutions-and-prior-research)
  - [3.1 Computer Vision Approaches to Vehicle Damage Detection](#31-computer-vision-approaches-to-vehicle-damage-detection)
  - [3.2 Retrieval-Augmented Generation (RAG) for Document Understanding](#32-retrieval-augmented-generation-rag-for-document-understanding)
  - [3.3 Multimodal Insurance AI - Industry and Academic Work](#33-multimodal-insurance-ai---industry-and-academic-work)
  - [3.4 Comparison of Modern VLMs and a Modular YOLO + RAG + LLM Architecture](#34-comparison-of-modern-vlms-and-a-modular-yolo--rag--llm-architecture)
  - [3.5 Literature Comparison Summary](#35-literature-comparison-summary)
- [4. Metrics and Success Definitions](#4-metrics-and-success-definitions)
  - [4.1 Vision Model Metrics](#41-vision-model-metrics)
  - [4.2 RAG Pipeline Metrics](#42-rag-pipeline-metrics)
  - [4.3 End-to-End and Usability Metrics](#43-end-to-end-and-usability-metrics)
- [5. Gaps in Existing Solutions](#5-gaps-in-existing-solutions)
- [6. Nature of Our Contribution](#6-nature-of-our-contribution)
- [7. System Architecture Overview](#7-system-architecture-overview)
  - [7.1 Orchestrator (LangGraph)](#71-orchestrator-langgraph)
  - [7.2 Damage Agent](#72-damage-agent)
  - [7.3 Severity Agent](#73-severity-agent)
  - [7.4 Policy Agent (MCP Tool)](#74-policy-agent-mcp-tool)
  - [7.5 Report Agent](#75-report-agent)
  - [7.6 Escalation Path](#76-escalation-path)
  - [7.7 Deployment](#77-deployment)
- [8. Evaluation Plan](#8-evaluation-plan)
  - [8.1 Vehicle Damage Detection](#81-vehicle-damage-detection)
  - [8.2 Policy Retrieval](#82-policy-retrieval)
  - [8.3 Generated Claim Reports](#83-generated-claim-reports)
- [9. Dataset Plan](#9-dataset-plan)
  - [9.1 Vision Datasets](#91-vision-datasets)
  - [9.2 Synthetic Data](#92-synthetic-data)
- [10. Expected Challenges and Project Risks](#10-expected-challenges-and-project-risks)
  - [10.1 Dataset Imbalance](#101-dataset-imbalance)
  - [10.2 Severity Estimation Reliability](#102-severity-estimation-reliability)
  - [10.3 Domain Shift Between Training Data and Real Claims](#103-domain-shift-between-training-data-and-real-claims)
  - [10.4 RAG Faithfulness with Synthetic Policies](#104-rag-faithfulness-with-synthetic-policies)
  - [10.5 LLM Hallucination on Edge Cases](#105-llm-hallucination-on-edge-cases)
  - [10.6 Compute Resources and Infrastructure](#106-compute-resources-and-infrastructure)
  - [10.7 API and Deployment Risks](#107-api-and-deployment-risks)
  - [10.8 Expected Failure Scenarios](#108-expected-failure-scenarios)
- [11. Ethical Considerations](#11-ethical-considerations)
  - [11.1 Fairness and Bias in Insurance Decision Support](#111-fairness-and-bias-in-insurance-decision-support)
  - [11.2 Privacy of Uploaded Images and Documents](#112-privacy-of-uploaded-images-and-documents)
  - [11.3 Transparency of AI-Generated Reports](#113-transparency-of-ai-generated-reports)
- [12. References](#12-references)

---

## 1. Problem Statement

### 1.1 What problem are we solving?

Insurance claim processing for vehicle damage is a slow, labour-intensive, and inconsistent process. When a vehicle is damaged, a claim assessor must manually examine submitted photographs, cross-reference the relevant sections of the policyholder's insurance document, and produce a written preliminary assessment report, a workflow that is both time-consuming and susceptible to inter-assessor variability.

This project builds an AI-powered decision-support system that automates the initial stage of this assessment pipeline.

### 1.2 Who are the stakeholders?

| **Stakeholder** | **Type** | **Interest in the system** |
| --- | --- | --- |
| Insurance claim assessors | Primary | Faster, consistent first-pass reports; reduced repetitive manual work |
| Insurance companies | Primary | Reduced processing time per claim; standardised initial assessments |
| Policyholders (vehicle owners) | Secondary | Faster claim decisions; transparent, traceable damage documentation |

### 1.3 Scope Definition

**In Scope**

- Detection and localisation of visible vehicle damage from uploaded photographs using a fine-tuned YOLO object detection model.

- Classification of damage into types: dent, scratch, crack, broken lamp, flat tyre, shattered glass.

- Severity estimation per detected damage region, categorised as Minor, Moderate, or Severe, based on the proportion of the damaged area relative to the vehicle surface visible in the image. This is a practical approximation with known limitations (discussed in Section 10.2), which also evaluates alternative severity estimation approaches.

- Retrieval of relevant insurance policy clauses from a user-provided policy PDF using a RAG pipeline.

- LLM-generated preliminary claim assessment report containing: detected damage summary table, estimated severity per damage, applicable policy coverage, and recommended next steps for the assessor.

- A Gradio-based web interface accessible via Hugging Face Spaces for live demonstration.

**Out of Scope**

- Final claim approval or rejection: The system produces a preliminary report only, all final decisions remain with a qualified human assessor.

- Repair costs depend on numerous variables such as vehicle make, model, manufacturing year, spare-part prices, labour rates, geographic location of service centre, and other policies none of which are determinable from photographs alone. Hence the final repair cost estimation is out of scope of this project.

- Detection of damage not visible in photographs such as internal mechanical damage, frame damage or anything that requires expertise is out of scope for this project.

- Synthetic policy will be used throughout the project due to the proprietary nature of real insurer documents.

- Multi-vehicle accident scenarios, fraud detection, or third-party liability assessment are out of scope for this project.

---

## 2. Problem Motivation

Vehicle insurance is one of the largest lines of general insurance globally. In India alone, the motor insurance market was valued at over $10 billion in 2026 and continues to grow rapidly and is expected to cross $15 billion by 2031 [13]. Despite this scale, the claims assessment process remains heavily manual at its initial stage.

After a vehicle is damaged, the policyholder submits photographs and a written incident description through an insurer's app or portal. A claims assessor then reviews this submission, identifies the relevant policy sections, and writes a preliminary report. This process typically takes one to several business days [14]. Several pain points are well-documented in the insurance technology literature:

- **Throughput bottleneck:** A single assessor may handle dozens of claims per day. Manual photo review for each claim is the largest time cost in the pipeline.

- **Assessor variability:** Two assessors examining identical photographs may classify damage severity differently, leading to inconsistent outcomes for policyholders.

- **Policy cross-referencing:** Identifying which policy clauses apply to a given damage type requires reading through multi-page policy documents repeatedly, creating additional latency.

- **Scalability:** Surge events such as hailstorms or floods produce claim volumes that cannot be absorbed at the same pace as normal operations.

Automating the initial assessment stage addresses all four pain points simultaneously, and does so in a setting where the consequences of an error are bounded. The system outputs a preliminary report reviewed by a human, not a final binding decision. This makes it an appropriate and high-impact application for AI-assisted decision support.

---

## 3. Existing Solutions and Prior Research

### 3.1 Computer Vision Approaches to Vehicle Damage Detection

Vehicle damage detection has been an active research topic since at least 2017. The foundational work by [Kalpesh Patil (2017)](https://ieeexplore.ieee.org/document/8260613) demonstrated that convolutional neural networks could distinguish damaged from undamaged vehicles with reasonable accuracy on small datasets. Subsequent work shifted from binary classification toward damage localisation and type classification.

YOLO-series models (You Only Look Once) have become the dominant architecture for this task due to their speed and accuracy trade-off. Multiple published studies have fine-tuned YOLOv5, YOLOv8, and YOLOv11 on vehicle damage datasets:

- **YOLOv8 for damage segmentation:** A 2024 IEEE study trained YOLOv8 on a dataset of over 4,000 high-resolution vehicle images annotated with 21 car part classes and 8 damage type classes, achieving strong [mAP scores](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173) for both part and damage segmentation simultaneously.

- **HL-YOLO:** A 2025 MDPI Vehicles paper proposed HL-YOLO, a heterogeneous convolution variant of YOLO11, reporting gains of 2.5% precision, 5.8% recall, and approximately 3-4% mAP over the YOLO11 baseline on vehicle damage detection.

- **Mask R-CNN:** He et al.'s Mask R-CNN (ICCV 2017) has been applied in a two-stage pipeline: first segmenting the vehicle body, then classifying damage within detected regions. This achieves higher segmentation fidelity but at significantly greater computational cost.

- **CarDD dataset paper:** The CarDD dataset (USTC, 2023) introduced pixel-level damage annotations across six damage categories and served as a benchmark for segmentation-based damage models.

>Among single-stage and two-stage object detectors, Faster R-CNN (Ren et al. [19]), DETR (Carion et al. [20]), and SSD (Liu et al., [21]) were also considered as alternatives to YOLO. Faster R-CNN achieves higher precision on small objects through its region proposal network but is significantly slower at inference, making it unsuitable for a Gradio demo that must run on a CPU-basic Hugging Face Spaces instance within a reasonable response time. DETR offers a cleaner transformer-based detection formulation that eliminates hand-crafted anchors, but it is known to require substantially more training data and epochs to converge compared to convolutional detectors, a constraint given the eight-week project timeline and limited GPU hours. SSD achieves faster inference than Faster R-CNN but consistently reports lower mAP than YOLO on comparable object detection benchmarks, particularly on small and occluded objects. YOLO11 and YOLOv8 were therefore selected as the primary detector family, offering the best combination of detection accuracy, native segmentation head support, training speed on the available hardware, and a well-maintained deployment ecosystem via Ultralytics.

### 3.2 Retrieval-Augmented Generation (RAG) for Document Understanding

Lewis et al. (2020, NeurIPS) introduced RAG as a framework for grounding LLM outputs in retrieved document context, reducing hallucination in knowledge-intensive tasks. Since then, RAG has been applied extensively to legal, medical, and financial document understanding domains closely analogous to insurance policy retrieval.

Key findings from RAG literature relevant to this project:

- **Chunk size matters:** Smaller chunks (200-400 tokens with overlap) consistently outperform large-chunk retrieval for precise clause-level recall in legal documents.

- **Embedding model choice:** Bi-encoder models (e.g., MiniLM-L6-v2, MPNet) outperform BM25 sparse retrieval for semantic matching of insurance-style queries.

- **Faithfulness is critical:** Without explicit grounding, LLMs hallucinate coverage entitlements. RAG with source attribution substantially reduces this problem (ES-RAG, 2024).

### 3.3 Multimodal insurance AI - industry and academic work

Several insurtech companies (Tractable, CCC Intelligent Solutions, Mitchell) have deployed computer vision systems for vehicle damage assessment in production. Published technical details are limited due to proprietary constraints, but disclosed capabilities include:

- Automated identification of damaged parts (hood, door, bumper) from photographs.

- Integration with repair cost databases using vehicle VIN and local labour rates which is outside the scope of this project.

- Human-in-the-loop review for all final decisions.

Academic work combining vision and language for insurance is sparse. The closest published analogues are medical report generation systems (e.g., MIMIC-CXR radiology report generation using vision-language models), which share the similar structure as this project: a vision model produces detections, and an LLM generates a structured natural-language report grounded in those detections. This project adapts that paradigm to the vehicle damage domain.

### 3.4 Comparison of Modern VLMs and a Modular YOLO + RAG + LLM Architecture

Recent multimodal Vision Language Models (VLMs) such as Florence-2 [15], Qwen2.5-VL [16], LLaVA [17], and GPT-4V [18] are capable of jointly reasoning over images and text in a single model, making them a natural baseline architecture for this task. Furthermore, given an image of a damaged vehicle and a policy document, a sufficiently capable VLM could in principle produce an assessment report in a single end-to-end inference process. However, despite their strong multimodal reasoning capabilities, a modular architecture combining YOLO, Retrieval-Augmented Generation (RAG), and an LLM is better aligned with this project's objectives and implementation constraints for the following reasons:

- **Separation of concerns and independent debuggability.** In a modular design, each component can be tested and diagnosed in isolation. If the final report is incorrect, it is possible to determine whether the fault lies in the detection stage, the retrieval stage, or the generation stage, and fix it independently. A monolithic VLM is a black box in this regard: a poor output provides little signal as to what went wrong, slowing down iteration significantly within a time-boxed academic project.

- **Measurable, ground-truth-comparable detection.** YOLO produces precise, structured outputs (bounding boxes, class labels, and confidence scores) that can be directly evaluated against the annotated ground truth in datasets such as VehiDE using standard metrics (mAP@50, F1 per class). VLMs describe damage in natural language, which is not directly comparable to bounding-box annotations. Calibrated localisation and severity scoring from VLM outputs would require an additional post-processing step and would still be difficult to score reliably, undermining the evaluation framework defined in Section 4.1.

- **Cost, latency, and deployment.** A fine-tuned YOLO small or nano variant runs on CPU or a small GPU and can be hosted on Hugging Face Spaces within the available memory and compute budget. Large VLMs require either paid API access or dedicated GPU memory that Spaces cannot reliably provide. The modular pipeline keeps each component lightweight and independently replaceable.

Therefore, although modern VLMs offer an attractive end-to-end paradigm, the proposed modular pipeline better satisfies the project's requirements for interpretability, quantitative evaluation, computational efficiency, and iterative development.

### 3.5 Literature Comparison Summary

The table below provides a structured critical comparison of the key prior works reviewed, covering datasets, models, reported metrics, and limitations.

| **Paper / System** | **Dataset** | **Model / Method** | **Metrics Reported (Scores)** | **Key Limitations** |
| --- | --- | --- | --- | --- |
| Patil et al. (2017) | Custom small dataset | CNN (classification only) | Binary accuracy: ~84% on held-out test set | No localisation; very small dataset; no severity estimation; no report generation |
| He et al. Mask R-CNN (ICCV 2017) | COCO | Mask R-CNN | Box AP: 37.1, Mask AP: 35.7 on COCO test-dev | High compute cost; requires pre-segmented vehicle; no insurance domain adaptation |
| CarDD benchmark (USTC, 2023) | CarDD (pixel-level) | Various (benchmark evaluation) | Best mAP@50: ~0.71 across evaluated models | Detection only; no downstream report generation; limited damage classes |
| YOLOv8 damage segmentation (IEEE, 2024) | Custom 4k images, 21 part + 8 damage classes | YOLOv8-seg | mAP@50: ~0.73 for damage classes; part segmentation mAP@50: ~0.81 | No policy integration; no structured report; severity not defined; proprietary dataset |
| HL-YOLO (MDPI, 2025) | Custom vehicle dataset | YOLO11 + heterogeneous convolutions | Precision: +2.5%, Recall: +5.8%, mAP: +3-4% over YOLO11 baseline; absolute mAP@50: ~0.79 | Detection only; no NLP or policy component; severity not addressed |
| Lewis et al. RAG (NeurIPS, 2020) | NaturalQuestions, TriviaQA | DPR + BART | NQ Exact Match: 44.5; TriviaQA Exact Match: 56.8 | Text-only; not applied to visual or insurance contexts; hallucination risk without strict grounding |


---

## 4. Metrics and Success Definitions

### 4.1 Vision Model Metrics

Object detection and segmentation research uses the following standard metrics:

| **Metric** | **Definition** | **Target** |
| --- | --- | --- |
| mAP@50 | Mean average precision at IoU threshold 0.50. Primary detection metric. | ≥ 0.70 overall |
| mAP@50-95 | mAP averaged over IoU thresholds 0.50-0.95. Stricter localisation metric. | ≥ 0.50 overall |
| Per-class F1 | Harmonic mean of precision and recall per damage class. | ≥ 0.65 all classes |
| Inference speed | Frames per second on a CPU/GPU at 640px input. | Report only |

### 4.2 RAG Pipeline Metrics

| **Metric** | **Definition** | **Target** |
| --- | --- | --- |
| Retrieval Precision@3 | Proportion of test queries for which the relevant policy clause is retrieved within the top three retrieved chunks. | ≥ 0.80 |
| Recall@3 | Proportion of all ground-truth relevant clauses that are retrieved within the top three results, averaged across queries. | ≥ 0.75 |
| MRR (Mean Reciprocal Rank) | Average of the reciprocal rank of the first correct clause retrieved. Captures how highly the correct clause is ranked, not just whether it appears. | ≥ 0.70 |
| Faithfulness score | Proportion of generated reports whose conclusions are fully supported by the retrieved policy clauses, assessed through manual evaluation of a 20-sample test set. | ≥ 0.85 |


### 4.3 End-to-End and Usability Metrics

| **Metric** | **Definition** | **Target** |
| --- | --- | --- |
| Human evaluation - accuracy | 3 raters score each generated report for factual accuracy on a 1-5 scale. | Mean ≥ 4.0 |
| Human evaluation - clarity | 3 raters score report clarity and usefulness to a claim assessor. | Mean ≥ 4.0 |
| Ablation delta | mAP and report quality improvement of full system vs. baseline (ResNet50 classifier, no RAG, no LLM). | Positive across all metrics |
| Severity accuracy | Agreement rate between model-assigned Minor/Moderate/Severe and human-assigned severity on a 30-image test set. | ≥ 0.75 |
| BERTScore (F1) | Token-level semantic similarity between generated report text and a human-authored reference summary, measured using contextual BERT embeddings. Captures semantic fidelity beyond exact word match. | ≥ 0.80 F1 |

> **Note on target values:** The mAP@50 >= 0.70 target is consistent with the threshold commonly accepted for production-grade automotive object detection and is achievable on VehiDE given the scale of the dataset and YOLO11's pretrained backbone. The Retrieval Precision@3 >= 0.80 target reflects performance reported by Lewis et al. on comparable document-retrieval tasks. The human evaluation Mean >= 4.0 threshold is set conservatively relative to the 5-point scale, ensuring it reflects genuine assessor utility rather than marginal acceptability. All targets will be revisited if dataset or compute constraints make them infeasible within the project timeline.


---

## 5. Gaps in Existing Solutions

Despite meaningful progress in both vehicle damage detection and document-grounded generation, no publicly available system integrates all three components: vision-based damage detection, policy RAG retrieval, and LLM report generation into a single end-to-end pipeline. The following specific gaps motivate this project:

**Gap 1: Detection without structured reporting**

Existing vision models for vehicle damage (YOLOv8 fine-tunes, CarDD benchmark models) produce bounding boxes and class labels, but do not generate human-readable structured outputs. A claim assessor receiving a list of bounding-box coordinates and class indices still needs to manually interpret and write the assessment. No published open-source system bridges the detection output and the final report.

**Gap 2: LLM reports without grounding in policy documents**

General-purpose LLMs (GPT-4, Gemini) can produce plausible insurance-related text, but without access to the specific policy document, they hallucinate coverage entitlements, cite incorrect exclusions, or fabricate deductible values. RAG over the actual policy document is necessary for any claim report to be trustworthy. This grounding step is absent in all existing publicly demonstrated systems.

**Gap 3: No accessible decision-support demo for this domain**

Industry systems (Tractable, CCC) are closed, proprietary, and inaccessible to researchers and small insurers. There is no open, deployable demonstration of a vision-plus-language claim assessment tool that a claim assessor could realistically interact with. Deployment on Hugging Face Spaces addresses this accessibility gap directly.

---

## 6. Nature of Our Contribution

This project's primary contribution is not a new model architecture. It lies in three other dimensions:

- **Deployment context and usability:** We will develop publicly accessible, open-source end-to-end pipeline combining vision-based damage detection with policy-aware LLM report generation. The system is specifically designed for practical use by claim assessors significantly reducing the time required to produce the first draft of insurance claim assessment report.

- **Pipeline integration:** The YOLO detection results will be converted into a structured format and provided to an LLM together with relevant policy information retrieved using RAG. This enables the LLM to generate responses that are accurate, grounded in policy, and easy for non-technical assessors to understand. The integration of YOLO-based object detection with a RAG-supported LLM forms the core of this project.

- **Defined scope and reproducible benchmark:** This project draws a deliberate boundary around what a vision-based pipeline operating on photographs alone can reliably deliver at an academic scale. Capabilities such as repair cost estimation require vehicle-specific data (make, model, manufacturing year, local labour rates, spare-part prices) that lie outside the image, and integrating them is a separate engineering problem beyond the scope of this work. By restricting severity to three human-interpretable categories (Minor, Moderate, Severe), focusing on visible damage only, and grounding all report conclusions in retrieved policy clauses with a documented evaluation protocol, this project establishes a clearly scoped, reproducible benchmark for the preliminary claim assessment task. This scope definition is itself a contribution: it gives future work a well-defined baseline to build on and compare against. 


---

## 7. System Architecture Overview

The system is implemented as a **multi-agent RAG architecture** in which a LangGraph orchestrator routes each incoming claim to four specialist agents. The agents run in a defined sequence, but the orchestrator can short-circuit or branch the path depending on what the claim contains and how confident the outputs are.

![High-Level Architecture Diagram](multiagent_architecture_staged.svg)

### 7.1 Orchestrator (LangGraph)

A LangGraph state machine holds the shared claim context - uploaded image, uploaded policy PDF (if any), detection results, retrieved clauses, severity scores, and a confidence flag - and routes it through the agents below. At each node, the orchestrator reads the current state and decides the next action. If no policy PDF is supplied, the Policy Agent step is skipped. If the Damage Agent's detection confidence falls below a configurable threshold, the claim is routed to a **human review queue** rather than auto-generating a report, implementing an explicit escalation path.

### 7.2 Damage Agent

The uploaded vehicle image is passed through a fine-tuned vision model. The model outputs a list of detected damage regions, each with:

- a class label (dent, scratch, crack, broken lamp, flat tyre, shattered glass),
- a bounding box and segmentation mask,
- a confidence score, and
- a severity category (Minor / Moderate / Severe) derived from the normalised bounding-box area relative to the visible vehicle surface.

The detection output is written back into the LangGraph state, and the orchestrator checks whether the minimum per-detection confidence meets the escalation threshold before proceeding.

### 7.3 Severity Agent

The Severity Agent reads the bounding-box area statistics from the Damage Agent's output and applies a per-class calibrated severity proxy to assign a final severity rating to each detected damage instance. Class-specific calibration is necessary because the raw bounding-box area has different distributional properties per class - `shattered_glass` instances, for example, naturally span a much larger area than `flat_tyre` instances, so a single global threshold would systematically mislabel them.

### 7.4 Policy Agent (MCP Tool)

The Policy Agent is exposed as a **FastMCP tool** callable by the orchestrator. When invoked, it:

1. Constructs a query string from the detected damage classes and severity ratings written to the LangGraph state.
2. Encodes the query using `sentence-transformers/all-MiniLM-L6-v2` (a lightweight bi-encoder producing 384-dimensional dense vectors).
3. Retrieves the top-k most relevant 300-token chunks from a ChromaDB persistent vector index built over 5 synthetic insurance policy PDFs.
4. Returns the retrieved chunks and their metadata (clause type, damage classes tagged, source document ID) to the orchestrator state.

Exposing retrieval as an MCP tool rather than embedding it inline in the orchestrator allows the Policy Agent to be called conditionally (skipped if no policy PDF context is needed), replaced independently (e.g. swapping ChromaDB for a different vector store) without changing the orchestrator logic, and tested in isolation against the ground-truth clause mapping (Section 8.2).

### 7.5 Report Agent

The Report Agent receives the full LangGraph state (detection results, severity ratings, and retrieved policy clauses) and issues a structured prompt to an LLM. GPT-4o (primary, via the OpenAI API) generates the preliminary claim assessment report, constrained to the retrieved policy context to minimise hallucination. Gemini 1.5 Flash is maintained as a cost-efficient fallback. The generated report is written back to the state as the final output.

### 7.6 Escalation Path

If the Damage Agent's output contains any detection with confidence below the escalation threshold, or if no damage is detected at all, the orchestrator does not invoke the Severity, Policy, or Report Agents. Instead, it writes the claim to a **human review queue** with a structured flag identifying the low-confidence regions, so that a human assessor can review the raw detections before any policy lookup or report is generated. This prevents the downstream agents from producing authoritative-sounding outputs on uncertain inputs.

### 7.7 Deployment

The full pipeline is served as a **Gradio application deployed on Hugging Face Spaces**. A user uploads a vehicle damage photograph and optionally a policy PDF; the orchestrator runs the agent sequence; and the interface displays the annotated detection image, the severity breakdown, the retrieved policy clauses, and the generated report in a structured tabbed view. All processing is stateless per submission (no image, document, or report is retained beyond the current session).


---


## 8. Evaluation Plan 

The proposed system will be evaluated at both the component and system levels. Evidence will be gathered to assess the accuracy of vehicle damage detection, the relevance of retrieved policy information, and the quality of the generated claim reports, providing a comprehensive evaluation of the end-to-end pipeline.

### 8.1 Vehicle Damage Detection

The computer vision component will be evaluated using standard object detection metrics, including mAP@50, mAP@50–95, Precision, Recall, and F1-score on unseen test images. These metrics will indicate how accurately the YOLO model detects and classifies different types of vehicle damage.

### 8.2 Policy Retrieval

The RAG component will be evaluated by measuring whether it retrieves the correct policy clauses for detected damage. Retrieval Precision@3 will be used to determine how often the relevant policy information appears within the top three retrieved results. This demonstrates that the language model is provided with appropriate supporting evidence before generating a report.

### 8.3 Generated Claim Reports

The final reports will be assessed for three key qualities:

- **Accuracy:** whether the report correctly reflects the detected damage.

- **Faithfulness:** whether policy recommendations are supported by the retrieved policy clauses rather than generated from the LLM's internal knowledge.

- **Clarity:** whether the report is understandable and useful for a claims assessor.

A sample of 20 generated reports will be independently scored by three evaluators: two team members who were not involved in generating the specific reports being evaluated, and one external reviewer with familiarity with insurance documents. Each evaluator will score each report on Accuracy (1-5), Faithfulness (1-5), and Clarity (1-5) using a fixed rubric provided in advance. The rubric defines each score point explicitly: for example, a Faithfulness score of 5 requires all coverage recommendations to be directly traceable to a retrieved clause, while a score of 1 indicates at least one fabricated entitlement. Evaluators will score reports independently and without seeing each other's ratings. Disagreements larger than one point on any dimension will be resolved through discussion and a majority vote. Inter-rater agreement will be reported using Cohen's kappa. BERTScore F1 will be computed automatically against a human-authored reference summary as an additional objective signal.

Together, these evaluations will demonstrate the effectiveness of each individual component as well as the overall end-to-end claim assessment pipeline.

---

## 9. Dataset Plan

### 9.1 Vision Datasets

| **Dataset** | **Size** | **Annotation type** | **Role** | **Justification for selection** |
| --- | --- | --- | --- | --- |
| VehiDE | 13,945 images | Bounding boxes, 32k+ instances | Primary training and evaluation dataset | Largest publicly available annotated vehicle damage dataset; diverse damage types across real vehicle images; sufficient scale for YOLO fine-tuning without data augmentation alone. |
| CarDD | Varies by split | Pixel-level segmentation masks | Supplementary segmentation fine-tuning | Provides pixel-level masks unavailable in VehiDE; enables training and evaluating the segmentation head of YOLO11-seg, which underpins the bounding-box severity proxy. |
| COCO Car Damage | ~500 images | COCO-format bounding boxes | Supplementary for architecture comparison | COCO-format annotations allow direct integration with standard detection training pipelines and enable comparison against published COCO-trained baselines. |
| Car Damage Severity | ~2,300 images | Minor / Moderate / Severe labels | Severity classifier calibration | The only publicly available dataset with human-assigned severity labels matching our three-category scheme; used to calibrate and validate the bounding-box-area severity proxy against human judgment. |


### 9.2 Synthetic Data

No public dataset of insurance policy documents paired with vehicle damage annotations exists. The team will produce:

- Five synthetic insurance policy PDFs (approximately 8-12 pages each) covering collision coverage, comprehensive coverage, deductibles, exclusions, claim limits, and third-party liability.

- Each policy will have coverage clauses mapped to all six damage classes so that the RAG pipeline can be evaluated against known ground-truth clause retrievals.

- Fifty synthetic incident descriptions paired with test images, for end-to-end report quality evaluation.

To ensure the synthetic policies are a reasonable approximation of real-world documents and do not make retrieval artificially easy, the following quality measures will be applied. First, the policies will be modelled on the structure and clause vocabulary of publicly available sample insurance policy documents from Indian general insurers, adapting clause phrasing without reproducing proprietary text. Second, each policy will include at least five distractor clauses per damage class: clauses that are semantically related but do not grant coverage, such as exclusion clauses, sub-limit clauses, and clauses conditioned on circumstances not present in the test scenarios. Third, clause phrasing will be deliberately varied across the five policies (synonyms, different sentence structures, negations) to stress-test the embedding model's ability to retrieve semantically equivalent clauses under surface variation. Finally, a team member not involved in writing the policies will verify that the retrieval task is non-trivial by attempting to match test queries to ground-truth clauses manually before running the automated RAG evaluation.

---


## 10. Expected Challenges and Project Risks

### 10.1 Dataset Imbalance

Real-world vehicle damage distributions are highly skewed. Dents and scratches are far more frequent than flat tyres or shattered glass, and this imbalance is reflected in publicly available datasets including VehiDE. A model trained on an imbalanced dataset will likely exhibit high precision on common classes but poor recall on rare ones, directly threatening the per-class F1 target of >= 0.65 set in Section 4.1. Mitigation strategies include class-weighted loss functions, oversampling minority classes with augmentation, and monitoring per-class metrics separately rather than relying solely on overall mAP.

### 10.2 Severity Estimation Reliability

The severity estimation approach in this project relies on the ratio of bounding box area to visible vehicle surface area as a proxy for damage extent. This is a practical approximation but has known failure modes: a large but shallow scratch may be classified as Severe, while a small but deep crack may be classified as Minor. Additionally, image angle, zoom level, and occlusion all affect the apparent size of a damage region. Alternative approaches considered include training a dedicated severity classification head on the Car Damage Severity dataset, or querying a VLM with a structured prompt to assign severity from the image. The dedicated classifier approach was not selected as the primary method because the Car Damage Severity dataset (~2,300 images) is insufficient to train a reliable standalone classifier without severe overfitting; however, it will be used for calibration. The VLM approach introduces API cost and latency that are incompatible with the deployment target. The bounding-box proxy therefore remains the primary method, with the Car Damage Severity dataset used to validate its accuracy against human labels, as described in Section 4.3. The severity accuracy target of >= 0.75 reflects this constraint.

### 10.3 Domain Shift Between Training Data and Real Claims

The YOLO model will be trained on datasets collected under controlled or near-controlled conditions (studio photography, consistent lighting, unoccluded vehicles). Real insurance claim photographs are submitted by policyholders using mobile phones under variable lighting, angles, and occlusion conditions. This domain shift is a standard challenge in applied computer vision and may cause a significant drop in mAP when moving from the test set to realistic inputs. Mitigation includes augmenting training data with brightness, contrast, and perspective transforms, and stress-testing the model on a manually collected set of realistic claim-style photographs.

### 10.4 RAG Faithfulness with Synthetic Policies

Because real insurer policy documents cannot be used due to proprietary constraints, the RAG pipeline will be developed and evaluated against synthetic policies authored by the project team. This introduces a risk that the synthetic policies are structurally simpler or more consistently formatted than real documents, making retrieval artificially easy. The team will deliberately vary clause phrasing, introduce negations and exceptions, and include distractor clauses to stress-test the retrieval component and produce a more robust faithfulness evaluation.

### 10.5 LLM Hallucination on Edge Cases

Even with RAG grounding, LLMs can introduce inaccuracies when the retrieved clauses are ambiguous or when no relevant clause is found for a detected damage type. In such cases, the model may fall back on parametric knowledge and fabricate plausible-sounding but incorrect coverage details. This risk is mitigated by the faithfulness evaluation (Section 4.2) and by including explicit instructions in the prompt to state "not covered under retrieved policy" rather than infer coverage from general knowledge.

### 10.6 Compute Resources and Infrastructure

YOLO fine-tuning will be conducted on a single NVIDIA T4 GPU available via Google Colab Pro or Kaggle Notebooks (up to 30 free GPU hours per week each). A T4 provides 16 GB VRAM, sufficient for YOLO11m-seg training at 640px input with a batch size of 16. Estimated training time for 50 epochs on VehiDE is approximately 2-4 hours per run. RAG index construction will be performed on CPU using FAISS. LLM inference will use the OpenAI API (GPT-4o) or Gemini API as a fallback. The Gradio demo will be deployed on Hugging Face Spaces using a CPU-basic instance (2 vCPU, 16 GB RAM), sufficient for inference-only use. If GPU availability on free tiers is interrupted, the team will use Kaggle's GPU quota as a secondary environment.

### 10.7 API and Deployment Risks

The report generation component depends on access to the OpenAI GPT-4o API or an equivalent paid service. Risks include API rate limiting during evaluation, unexpected cost overruns if the number of evaluation samples grows beyond budget, and potential changes to API response format. Mitigations include caching all API responses during development, using Gemini 1.5 Flash as a cost-efficient fallback, and testing prompt templates with smaller models before large-scale evaluation. Deployment on Hugging Face Spaces introduces a further risk of resource contention during the live demonstration. A local fallback environment will be prepared as a contingency for the Milestone 6 presentation.

### 10.8 Expected Failure Scenarios

Beyond the technical risks described above, the system has identifiable expected failure modes that evaluators should be aware of:

- **Poor lighting and low image quality:** The YOLO model will likely underperform on photographs taken at night, in heavy rain, or with significant motion blur. Scratches and small cracks are particularly sensitive to lighting conditions, as their visual texture is disrupted under low light.

- **Heavy occlusion:** If a damaged area is obscured by another vehicle, debris, or an unfavourable camera angle, the model cannot detect it. A dented wheel arch hidden behind an open door, for example, will not appear in the detection output.

- **Internal and hidden damage:** Damage to internal components (engine, gearbox, frame, wiring) is entirely outside the scope of a vision-based system and will not be flagged in any report. All generated reports will include a disclaimer reminding assessors to conduct a physical inspection for non-visible damage.

- **Novel damage types:** Damage classes not seen during training (fire damage, water submersion marks, vandalism beyond standard scratching) may be missed or misclassified.

- **Multi-vehicle images:** If multiple damaged vehicles appear in a single image, the model may confuse detections across vehicles. The current scope restricts use to single-vehicle photographs.

---

## 11. Ethical Considerations

This project is positioned as a decision-support tool and not a replacement for qualified human assessors. Nevertheless, several ethical dimensions warrant explicit discussion.

### 11.1 Fairness and Bias in Insurance Decision Support

If the training datasets (VehiDE, CarDD) are not representative of the full diversity of vehicle types, damage patterns, or photographic conditions encountered by real policyholders, the model may systematically underperform for certain groups. For example, if certain vehicle colours, body types, or damage patterns are underrepresented in training data, detection rates may differ across policyholders, introducing a form of indirect algorithmic bias into the claim handling process. As a mitigation, we will include a stratified error analysis across damage classes and, where metadata is available, across vehicle types in the later stages of project. All generated reports will explicitly state that they are AI-generated preliminary assessments subject to human review.

### 11.2 Privacy of Uploaded Images and Documents

Vehicle damage photographs and insurance policy documents submitted by users are sensitive personal data that may contain Personally Identifiable Information (PII) such as names, addresses, vehicle registration numbers, and contact details. The project team will not collect, store, share, or disclose any original user-submitted data to third parties at any stage of the project.
For the Gradio demo deployed on Hugging Face Spaces, a clear on-screen notice will inform users that no submitted images or documents are retained beyond the current session. In any internal testing that requires handling of sample data containing PII, the team will apply one of the following two anonymisation strategies before use:
- **PII detection and masking:** Sensitive fields in text inputs and policy documents will be identified and redacted using the Microsoft Presidio SDK (a Python-based open-source PII detection and anonymisation framework), which supports entity types such as names, phone numbers, addresses, and national identifiers. Detected entities will be replaced with type-level placeholders (for example, <PERSON>, <ADDRESS>) before the data enters any pipeline component.
- **Attribute shuffling:** Where structured sample records are used for testing (for example, rows of claimant profiles), personal attributes will be shuffled across records so that no combination of values in a single row corresponds to a real individual. For instance, a dataset containing records `[Ram, 23, IT, Kolkata]` and `[Shyam, 34, CSE, HP]` would be shuffled to `[Shyam, 23, CSE, Kolkata]` and `[Ram, 34, IT, HP]`, preserving statistical distributions without retaining any real person's complete profile.
  
In a production deployment, data handling would need to comply with applicable data protection regulations, including India's Digital Personal Data Protection (DPDP) Act and, where relevant, the GDPR.

### 11.3 Transparency of AI-Generated Reports

All reports generated by the system will include a prominent disclaimer stating that the report is a preliminary AI-assisted assessment, has not been verified by a licensed insurance assessor, and must not be used as the sole basis for a final claim decision. This is consistent with the system's intended role as a first-pass tool that supports, rather than replaces, the assessor's judgment.


---

# 12. References

[1] K. Patil, S. Kulkarni, S. M. P. B., and V. K. Bairagi, "Car Damage Detection Using Convolutional Neural Networks," International Journal of Engineering Research & Technology (IJERT), vol. 6, no. 2, 2017.

[2] K. He, G. Gkioxari, P. Dollár, and R. Girshick, "Mask R-CNN," in Proceedings of the IEEE International Conference on Computer Vision (ICCV), Venice, Italy, 2017, pp. 2961-2969.

[3] J. Redmon and A. Farhadi, "YOLOv3: An Incremental Improvement," arXiv preprint, arXiv:1804.02767, 2018.

[4] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459-9474, 2020.

[5] S. Wang et al., "CarDD: A New Dataset for Vision-Based Car Damage Detection," University of Science and Technology of China (USTC), 2023.

[6] H. Scullen, "VehiDE: Vehicle Damage Detection Dataset," Kaggle, 2023. Available: [https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection](https://www.kaggle.com/datasets/hendrichscullen/vehide-dataset-automatic-vehicle-damage-detection)

[7] G. Jocher et al., "YOLO by Ultralytics," Zenodo, 2023. doi:10.5281/zenodo.7347926.

[8] "Advanced Car Damage Assessment Using YOLOv8: A Hybrid Approach to Detection and Masking," IEEE, 2024. doi:10.1109/ICCV.2025.10983960.

[9] A. E. W. Johnson et al., "MIMIC-CXR: A Large Publicly Available Database of Labelled Chest Radiographs," arXiv preprint, arXiv:1901.07042, 2019.

[10] "HL-YOLO: Improving Vehicle Damage Detection with Heterogeneous Convolutions," Vehicles (MDPI), 2025.

[11] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019.

[12] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535-547, 2021.

[13] Mordor Intelligence, "India Motor Insurance Market Size & Share Analysis - Growth Trends & Forecasts (2026-2031)," Mordor Intelligence Industry Reports, 2026. Available: https://www.mordorintelligence.com/industry-reports/india-motor-insurance-market

[14] Bajaj Allianz General Insurance, "Motor Insurance Claim Process," Bajaj General Insurance, 2024. Available: https://www.bajajgeneralinsurance.com/motor-insurance/motor-insurance-claim-process.html

[15] B. Xiao et al., "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

[16] Qwen Team, "Qwen2.5-VL Technical Report," Alibaba Group, arXiv preprint, arXiv:2502.13923, 2025.

[17] H. Liu, C. Li, Q. Wu, and Y. J. Lee, "Visual Instruction Tuning," in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023.

[18] OpenAI, "GPT-4 Technical Report," arXiv preprint, arXiv:2303.08774, 2023.

[19] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 28, 2015.

[20] N. Carion, F. Massa, G. Synnaert, N. Usunier, A. Kirillov, and S. Zagoruyko, "End-to-End Object Detection with Transformers," in Proceedings of the European Conference on Computer Vision (ECCV), 2020, pp. 213-229.

[21] W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, "SSD: Single Shot MultiBox Detector," in Proceedings of the European Conference on Computer Vision (ECCV), 2016, pp. 21-37.


---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | 02 July 2026 | S.K. |
|Pranab Kumar Manna | 02 July 2026| P.K.Manna|
| Venkata Siva Kamal Guddanti | 02 July 2026 | Kamal G |
| Anuj Gautam | 02 July 2026 | Anuj Gautam |

---
