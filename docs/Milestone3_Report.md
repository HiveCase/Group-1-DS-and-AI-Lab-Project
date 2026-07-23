
---

<div align="center">

<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">

<h1>Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 3: Model Architecture</h2>

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

1. [Introduction](#1-introduction)
2. [Overall System Architecture](#2-overall-system-architecture)
3. [End-to-End Workflow](#3-end-to-end-workflow)
4. [Model Architecture Selection](#4-model-architecture-selection)
5. [Justification of Model Choices](#5-justification-of-model-choices)
6. [Model Inputs and Outputs](#6-model-inputs-and-outputs)
7. [Training Strategy](#7-training-strategy)
8. [Model Pipeline](#8-model-pipeline)
9. [Retrieval and Knowledge Components](#9-retrieval-and-knowledge-components)
10. [Prompt Engineering](#10-prompt-engineering-for-llm-based-systems)
11. [System Integration](#11-system-integration)
12. [Computational Requirements](#12-computational-requirements)
13. [Design Decisions and Trade-offs](#13-design-decisions-and-trade-offs)
14. [Risks and Limitations](#14-risks-and-limitations)
15. [Deliverables Produced](#15-deliverables-produced)
16. [Summary and Next Steps](#16-summary-and-next-steps)
17. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Project recap

The project builds a multimodal pipeline that assesses vehicle damage from
a claim photo and cross-references the finding against a policy document to
produce a grounded, human-readable claim report. Milestone 1 scoped the
problem and data sources; Milestone 2 delivered a training-ready image
dataset (13,655 deduplicated VehiDE images, 32,672 instances across 6 damage
classes, letterboxed to 1,280×1,280, stratified 70/15/15 split, leakage
checked) and a validated 185-chunk insurance-policy corpus indexed in
ChromaDB behind a hybrid dense+sparse retriever (Precision@3 = 0.913).

### 1.2 Objectives of Milestone 3

- Select and justify the model architecture for the damage-detection module.
- Justify the associated training configuration ahead of Milestone 4 runs.
- Design and implement the complete end-to-end multi-agent pipeline that
  takes a claim image (and optional policy PDF) as input and produces a
  claim report as output, including routing, fallback, and escalation logic.
- Demonstrate that the pipeline runs end-to-end (with stub/mock weights
  where trained weights are not yet available) and that all module
  contracts (inputs/outputs, schemas, tool interfaces) are defined.

### 1.3 Relationship between the model architecture and project goals

The project's core deliverable is a claim report that is fast enough for
interactive use, accurate enough to be trustworthy for triage, and
explainable enough that a human adjuster can verify it against policy text.
This drives every architecture choice in this milestone: a single-stage
instance-segmentation detector for interactive-latency damage localisation,
a deterministic severity-scoring layer for auditability, a hybrid
retrieval-augmented-generation (RAG) component so report language is
grounded in the actual policy rather than model memory, and an
orchestration layer (LangGraph) that makes the routing logic (escalation,
fallback) explicit and testable rather than implicit in prompt text.

---

## 2. Overall System Architecture

### 2.1 High-level architecture

![Architecture](multiagent_architecture_staged.svg)

The system is a four-agent, graph-orchestrated pipeline with two
conditional branch points:

```
damage_agent ─► severity_agent ─┬─(escalate)──────► human_review ─► END
                                ├─(policy PDF)────► policy_agent ─► report_agent ─► END
                                └─(no PDF)────────► report_agent ─► END
```

### 2.2 Major modules and their interactions

| Module | Role | Talks to |
| --- | --- | --- |
| Damage Agent | Localises and classifies damage in the claim image | Severity Agent (via shared state) |
| Severity Agent | Converts detections into a per-detection and overall severity score | Policy Agent / Report Agent (branch dependent) |
| Policy Agent | Retrieves the policy clauses relevant to the detected damage and severity | ChromaDB vector store; also exposed as a FastMCP tool | 
| Report Agent | Synthesises the final claim report, grounded in retrieved clauses | LLM providers (GPT-4o, Gemini 1.5 Flash) with offline fallback |
| Human Review (sink node) | Terminal node for low-confidence detections that need manual adjudication | — |

### 2.3 Data flow between modules

All inter-agent communication happens through a single shared `ClaimState`
object (see §4.2) rather than direct function calls — each agent reads the
keys it needs and writes the keys it owns, and the LangGraph orchestrator
decides the next node based on the state's routing fields
(`escalate`, presence of `policy_pdf_path`).

### 2.4 External services / APIs

| Service | Purpose | Fallback if unavailable |
| --- | --- | --- |
| OpenAI GPT-4o | Primary report-generation LLM | Gemini 1.5 Flash |
| Google Gemini 1.5 Flash | Secondary report-generation LLM | Deterministic offline template |
| ChromaDB (local/embedded) | Vector store for policy clauses | N/A — required for policy branch only |
| Hugging Face Spaces | Deployment/demo target | N/A |

### 2.5 Technology stack

- **Detection/segmentation:** Ultralytics YOLO11m-seg (PyTorch)
- **Orchestration:** LangGraph
- **Retrieval:** ChromaDB + hybrid dense (embedding) / sparse (BM25-style) retriever, reciprocal-rank fusion
- **LLMs:** GPT-4o (primary), Gemini 1.5 Flash (fallback)
- **Tool exposure:** FastMCP (Policy Agent exposed as an MCP tool)
- **Language/runtime:** Python 3, `src/pipeline/` package
- **Testing:** `pytest` (`tests/test_pipeline.py`)
- **Target deployment:** Hugging Face Spaces (CPU tier for inference demo)

---

## 3. End-to-End Workflow

### 3.1 Complete workflow: input to output

1. User submits a claim image and, optionally, a policy PDF and free-text
   incident description.
2. **Damage Agent** runs YOLO11m-seg inference on the image, keeps
   detections with confidence ≥ 0.45, and flags escalation if any
   detection confidence is < 0.30.
3. **Severity Agent** converts kept detections into per-class,
   area-ratio-calibrated severity bins and an overall severity label.
4. **Routing decision:**
   - If escalation was flagged → route to **Human Review** (terminal).
   - Else if a policy PDF was supplied → route to **Policy Agent**.
   - Else → route directly to **Report Agent**.
5. **Policy Agent** (if reached) retrieves the top-k relevant clauses from
   ChromaDB using hybrid dense+sparse retrieval (75:25 reciprocal-rank
   fusion), conditioned on damage classes, severity, and incident
   description.
6. **Report Agent** synthesises the final claim report from the full
   state, grounding language in retrieved clauses when available; falls
   back from GPT-4o → Gemini 1.5 Flash → offline deterministic template if
   LLM calls are unavailable.
7. Output: a structured claim report (plus annotated image and, where
   applicable, cited policy clauses) or a human-review flag with reason.

### 3.2 Inputs and outputs per module

| Module | Inputs | Outputs |
| --- | --- | --- |
| Damage Agent | `image_path` | `detections`, `annotated_image_path`, `escalate` |
| Severity Agent | `detections` | per-detection severity, `overall_severity` |
| Policy Agent | `detections`, `overall_severity`, `incident_description` | `retrieved_clauses` |
| Report Agent | full state | `report` |
| Human Review | `escalation_reason` | (terminal — routed to adjuster queue) |

### 3.3 Error handling and fallback mechanisms

- **Low-confidence detections** never silently proceed to reporting — they
  are routed to human review with the triggering reason recorded in
  `escalation_reason`.
- **Missing policy PDF** is not an error — the graph has an explicit
  no-PDF branch straight to the Report Agent.
- **LLM unavailability** (no API key, rate limit, network failure) is
  handled by a two-level provider fallback (GPT-4o → Gemini 1.5 Flash)
  followed by a deterministic offline template, so the pipeline is
  guaranteed to terminate with a report even with zero LLM keys configured.
- **Memory-constrained training** has a documented contingency preset
  (960px / batch 8) if 1280px / batch 4 does not fit on the available GPU
  (see §14).

### 3.4 Storage and retrieval components

- Damage detection weights (`models/best.pt`) loaded once at pipeline
  start.
- Policy corpus persisted in ChromaDB (embedded/local mode), built during
  Milestone 2 and reused unchanged in Milestone 3.
- Annotated images and generated reports are written to disk per run for
  later audit/inspection (paths carried in `ClaimState`).

### 3.5 User interaction flow

A user (claims handler or, in the demo, an end user) uploads an image and
optionally a policy PDF and description through the Hugging Face Spaces
interface. The pipeline runs synchronously and returns either a completed
report or a "flagged for human review" result with the reason surfaced in
the UI.

---

## 4. Model Architecture Selection

### 4.1 Models selected, by module

| Module | Model | Pre-trained vs custom |
| --- | --- | --- |
| Damage detection/segmentation | YOLO11m-seg (Ultralytics) | Pre-trained (COCO) backbone, fine-tuned end-to-end on the M2 dataset |
| Severity scoring | Rule-based calibrated area-ratio bins (no learned model) | Custom, configuration-driven (`configs/pipeline_config.yaml`) |
| Policy retrieval | Hybrid dense (embedding) + sparse retriever, RRF fusion | Pre-trained embedding model, no fine-tuning in scope |
| Report generation | GPT-4o (primary) / Gemini 1.5 Flash (fallback) | Pre-trained, closed-weight, used via API with retrieval-grounded prompting — no fine-tuning |

### 4.2 Candidate comparison for the detection module

Two same-size-class instance-segmentation architectures were compared with
identical 10-epoch probe runs (imgsz=1280, batch=4, AdamW, lr0=0.001,
seed=42):

| | YOLO11m-seg | YOLOv8m-seg |
| --- | --- | --- |
| Parameters | 22.4M (measured, see supplementary note below) | ~26M |
| Release | 2024 (current Ultralytics flagship) | 2023 (established baseline) |
| mAP@50 @ epoch 10 | *`<fill from runs/probe/yolo11m_probe/results.csv>`* | *`<fill from runs/probe/yolov8m_probe/results.csv>`* |
| Training time / epoch | *`<fill>`* | *`<fill>`* |
| Inference ms / image (1280px) | *`<fill>`* | *`<fill>`* |
| Peak GPU memory | *`<fill>`* | *`<fill>`* |

> **Note:** the probe-run numbers above are pending the actual
> `results.csv` outputs from `runs/probe/`. Populate this table before
> submission — do not estimate these figures.

#### Supplementary: measured architecture footprint (`notebooks/YOLO11_Capability_Analysis.ipynb`, 2026-07-23)

A separate capability-exploration notebook loaded real YOLO11 checkpoints and read their
architecture directly via `model.info()`, rather than estimating. This is a different
comparison from the table above (YOLO11n-seg vs YOLO11m-seg, within the YOLO11 family) and does
**not** replace the pending YOLO11m-seg vs YOLOv8m-seg probe numbers:

| | YOLO11n-seg | YOLO11m-seg |
| --- | --- | --- |
| Layers | 203 | 253 |
| Parameters | 2,876,848 | 22,420,896 |
| GFLOPs | 9.9 | 113.9 |

This corrects the "~27M parameters" estimate used elsewhere in this report for YOLO11m-seg to
the measured **22.4M** — the mAP/training-time/GPU-memory columns in the table above are
unaffected and still require the real 10-epoch GPU probe run. The same notebook also verified
that a COCO-pretrained YOLO11n-seg forward pass runs correctly on this project's own sample
claim photos (`data/vehide/images/test/`), returning boxes and instance masks in the expected
shape (`[N, 640, 640]`).

### 4.3 Decision

*`<State the winner and cite the probe numbers once available. Expected
outcome: YOLO11m-seg, on the strength of equal-or-better probe mAP with the
newer C3k2/C2PSA backbone blocks that improve small-object performance —
relevant here because the M2 EDA found a minimum normalised bbox area of
0.00002 (M2 §5.3).>`*

### 4.4 Model size and complexity

- YOLO11m-seg: 22.4M parameters (measured, §4.2), single-stage detector
  with an instance segmentation head; runs on a single T4/P100 for
  training and on the free Hugging Face Spaces CPU tier for inference.
- Report-generation LLMs are used as hosted APIs (no local weights,
  no fine-tuning); size/complexity is out of scope for this pipeline's
  compute budget.
- Retrieval embedding model: lightweight, CPU-inferable (carried over
  unchanged from Milestone 2; see M2 report for exact model and dimension).

### 4.5 Input and output formats

| Model | Input format | Output format |
| --- | --- | --- |
| YOLO11m-seg | RGB image, letterboxed to 1280×1280 | Bounding boxes + instance masks + class + confidence per detection |
| Severity scorer | List of detections (class, mask/box area) | Per-detection severity label + overall severity label |
| Retriever | Query string (damage classes + severity + incident description) | Ranked list of `{chunk_id, text, score, heading, clause_type, doc_id, damage_classes}` |
| Report LLM | Structured state (detections, severity, retrieved clauses, incident description) | Free-text claim report, grounded in cited clauses |

### 4.6 Integration between multiple models

Models are not chained directly; they are integrated through the shared
`ClaimState` object and the LangGraph orchestrator, which decouples each
model's I/O contract from the others (see §11).

---

## 5. Justification of Model Choices

### 5.1 Why YOLO-family single-stage detection

- **Real-time constraint:** the claim demo targets interactive latency on
  Hugging Face Spaces; two-stage detectors (Mask R-CNN family) are 3–5×
  slower at inference for marginal accuracy gains at this dataset scale.
- **Instance segmentation head:** damage regions (cracks, shattered glass)
  have irregular boundaries; the -seg variant keeps mask-level supervision
  available from CarDD if its contingency trigger is met (M2 §7.2), without
  a change of architecture.
- **Transfer-learning fit:** a COCO-pretrained backbone transfers well to
  vehicle imagery; the 32,672-instance corpus is large enough to fine-tune
  the full network (no layer freezing needed) but far too small to train
  from scratch.
- **Deployment weight:** 22.4M parameters (measured, §4.2) runs on the
  free HF Spaces CPU tier at acceptable latency and trains on a single
  T4/P100.

### 5.2 Why YOLO11m-seg over YOLOv8m-seg specifically

Both are same-size-class instance-segmentation models from the same
family, so the comparison is apples-to-apples (§4.2). YOLO11's newer
C3k2/C2PSA backbone blocks are reported by Ultralytics to improve
small-object performance, which matters here given the M2 EDA finding of a
minimum normalised bbox area of 0.00002 (M2 §5.3). Final selection is
confirmed by the probe-run numbers once populated (§4.2–4.3).

### 5.3 Why rule-based severity scoring instead of a learned model

No labelled severity ground truth exists in the M2 dataset (only damage
class and location). A calibrated, configuration-driven area-ratio
approach is transparent, requires no additional labelled data, and is
directly auditable by a human adjuster — an important property for an
insurance use case where a black-box severity score would be hard to
justify to a policyholder.

### 5.4 Why hybrid dense+sparse retrieval

Pure dense retrieval can miss exact policy terminology (clause numbers,
named exclusions); pure sparse (keyword) retrieval misses paraphrase and
semantic matches. The hybrid RRF-fused approach, already validated in
Milestone 2 at Precision@3 = 0.913, covers both failure modes and is
carried forward unchanged.

### 5.5 Why GPT-4o / Gemini 1.5 Flash with offline fallback for reporting

Report generation needs fluent, well-structured natural language, which
favours a capable general-purpose LLM over a smaller fine-tuned model, and
fine-tuning is out of scope given the absence of labelled report examples.
A two-provider fallback plus a deterministic offline template guards
against API/key unavailability, which is a realistic constraint for a
student project demo (see §3.3).

### 5.6 Advantages and disadvantages summary

| Model | Advantages | Disadvantages |
| --- | --- | --- |
| YOLO11m-seg | Fast, single-stage, good small-object handling, deployable on CPU | Slightly newer/less battle-tested than YOLOv8; needs probe confirmation |
| Rule-based severity | Fully transparent, no extra training data needed | Less adaptive than a learned model; calibration is manual |
| Hybrid retriever | Robust to both lexical and semantic query types | Slightly more complex than a single-method retriever |
| GPT-4o / Gemini fallback | High-quality fluent output, resilient to outages | External dependency, cost per call, latency variance |

### 5.7 Computational considerations and expected performance

Detailed in §12; expected detection performance is pending probe-run
completion (§4.2). Retrieval performance is inherited from the Milestone 2
validated figure (Precision@3 = 0.913).

### 5.8 Suitability for the dataset and problem

The 6-class, 32,672-instance, imbalance-corrected (6.68:1, addressed via
`cls=2.0`) dataset is well matched to a mid-sized single-stage segmentation
model fine-tuned end-to-end rather than a much larger model that would risk
overfitting at this data scale, or a much smaller model that would
underfit the small-object cases identified in the M2 EDA.

---

## 6. Model Inputs and Outputs

### 6.1 Damage detection module

- **Input:** RGB claim image, letterboxed to 1,280×1,280 (matches the
  training-time preprocessing established in Milestone 2).
- **Output:** per-instance bounding box, segmentation mask, class label
  (one of 6 damage classes), and confidence score.
- **Preprocessing:** letterbox resize/pad to 1280×1280, normalisation
  consistent with COCO-pretrained YOLO weights.

### 6.2 Severity module

- **Input features:** per-detection class and mask/box area, normalised
  against image area.
- **Output:** per-detection severity bin and an aggregated overall
  severity label, produced via calibrated bins defined in
  `configs/pipeline_config.yaml`.

### 6.3 Retrieval module

- **Input:** structured query composed of damage classes, severity label,
  and (optional) free-text incident description.
- **Embedding/chunking strategy:** inherited unchanged from Milestone 2 —
  185-chunk policy corpus, hybrid dense+sparse index in ChromaDB.
- **Output:** ranked list of clause objects
  (`chunk_id, text, score, heading, clause_type, doc_id, damage_classes`).

### 6.4 Report generation module

- **Input:** full `ClaimState` (detections, severity, retrieved clauses,
  incident description).
- **Output:** free-text claim report, grounded in and citing the retrieved
  policy clauses.
- **Feature representation:** no additional embedding step — the LLM
  consumes structured state fields directly via prompt templating (§10).

---

## 7. Training Strategy

*(Applies to the Damage detection module — the only trainable model in the
pipeline; severity scoring is rule-based and the LLMs/retriever are used
without fine-tuning.)*

| Aspect | Choice | Justification |
| --- | --- | --- |
| Approach | Fine-tuning (full network) | Corpus is large enough (32,672 instances) to fine-tune the full COCO-pretrained network; too small to train from scratch |
| Frozen vs trainable layers | None frozen | Domain gap (COCO → vehicle damage) is large enough to warrant full-network fine-tuning |
| imgsz | 1280 | Data-driven: instance-count-weighted mean resolution is 1,395×1,038 px (M2 §6.1 Step 4); 1280 is the nearest YOLO-compatible size — 640 would discard roughly half the available resolution |
| batch | 4 | 1280px + 22.4M-param seg model (measured, §4.2) saturates a 16 GB P100 at batch 4 |
| Optimizer | AdamW | Standard for fine-tuning; less LR-sensitive than SGD on small domain datasets |
| Learning rate (lr0) | 0.001 | 10× below the from-scratch default — appropriate fine-tuning regime on COCO-pretrained weights |
| Loss function | YOLO composite (box + segmentation + objectness + classification), `cls=2.0` | Class-loss gain partially corrects the 6.68:1 class imbalance (M2 §8.2) |
| Epochs | 50 | Probe curves show *`<flattening/still improving>`* at epoch 10; 50 balances compute budget vs convergence |
| Early stopping | Patience 15 | Prevents wasted compute once validation metrics plateau |
| Checkpointing | Best-weights checkpoint saved to `models/best.pt` | Standard Ultralytics checkpointing; best-epoch weights carried into the pipeline |

---

## 8. Model Pipeline

### 8.1 Data flow into the model

Claim image → letterbox to 1280×1280 → normalisation → YOLO11m-seg
forward pass.

### 8.2 Preprocessing before inference

Identical to training-time preprocessing (letterboxing, normalisation) to
avoid train/inference skew.

### 8.3 Intermediate outputs

Raw per-instance boxes/masks/scores prior to confidence filtering.

### 8.4 Post-processing

- Non-max suppression (standard YOLO post-processing).
- Confidence filtering: detections ≥ 0.45 kept; any detection < 0.30
  triggers escalation to human review.
- Severity binning applied to kept detections (§6.2).
- Clause retrieval conditioned on the post-processed detection/severity
  output (§6.3).

### 8.5 Final prediction generation

The Report Agent combines post-processed detections, severity, and (if
applicable) retrieved clauses into the final structured claim report
(§6.4), or the pipeline terminates at Human Review with an
`escalation_reason`.

---

## 9. Retrieval and Knowledge Components

### 9.1 Retrieval pipeline

Query (damage classes + severity + incident description) → hybrid
dense+sparse retrieval → reciprocal-rank fusion (75:25 dense:sparse) →
top-k clause list returned to the Report Agent.

### 9.2 Embedding model / vector database

Vector database: ChromaDB, populated in Milestone 2 with the 185-chunk
policy corpus. Embedding model choice and chunking parameters are carried
over unchanged from Milestone 2 (see the M2 report for exact model name
and chunking configuration).

### 9.3 Similarity search algorithm

Hybrid retrieval: dense (embedding cosine/L2 similarity) combined with a
sparse keyword-based method, fused via reciprocal rank fusion at a 75:25
dense:sparse weighting — validated in Milestone 2 at Precision@3 = 0.913.

### 9.4 RAG workflow

Retrieved clauses are passed into the Report Agent's prompt as grounding
context; the report is required to cite retrieved clauses rather than
generate policy language from the LLM's own knowledge (see §10.5,
hallucination mitigation).

### 9.5 Re-ranking strategy

The reciprocal-rank fusion step itself serves as the re-ranking mechanism
across the two retrieval methods; no separate cross-encoder re-ranker is
used in the current scope.

### 9.6 MCP tool exposure

The Policy Agent's retrieval function is additionally exposed as a FastMCP
tool:

```
tool: retrieve_clauses
args: damage_classes: list[str], severity: str, incident_description: str|None, top_k: int
returns: list of {chunk_id, text, score, heading, clause_type, doc_id, damage_classes}
```

Run standalone: `python -m src.pipeline.policy_agent`.

---

## 10. Prompt Engineering (for LLM-based systems)

*(Applies to the Report Agent.)*

### 10.1 Prompt template

The report prompt is constructed from a fixed template that injects: the
detected damage classes and confidences, the overall severity label, the
incident description (if provided), and the retrieved policy clauses (if
the policy branch was taken).

### 10.2 System prompt

Instructs the model to act as a claims-report writer that must (a) only
state damage findings present in the structured input, (b) only cite
policy language present in the retrieved clauses, and (c) explicitly note
when no policy document was supplied.

### 10.3 Few-shot / zero-shot strategy

Zero-shot, structured-input prompting — the state schema itself provides
the necessary grounding, so no in-context examples are currently required.

### 10.4 Structured output format

The report follows a fixed section structure (findings summary, severity,
relevant policy clauses with citations, recommended next step), enforced
via the prompt template rather than a strict JSON schema, to keep the
output human-readable for adjusters.

### 10.5 Hallucination mitigation

- Report generation is grounded strictly in retrieved clauses (§9.4); the
  prompt instructs the model not to state policy terms that are not present
  in the retrieved context.
- Detection facts come directly from `ClaimState`, not from the LLM.
- The offline template fallback (§10.6) uses no LLM at all and therefore
  carries zero hallucination risk when triggered.

### 10.6 Guardrails / function calling

- Provider fallback chain (GPT-4o → Gemini 1.5 Flash → deterministic
  offline template) guarantees a valid output even without any LLM
  available.
- The Policy Agent's retrieval function is exposed as a callable MCP tool
  (§9.6), keeping retrieval a separate, auditable step from generation
  rather than delegating retrieval to the LLM itself.

---

## 11. System Integration

### 11.1 How different models communicate

All agents communicate exclusively through the shared `ClaimState`
(TypedDict), read/written by each LangGraph node — no direct
agent-to-agent calls.

### 11.2 State schema

`src/pipeline/state.py` defines `ClaimState`: inputs (`image_path`,
`policy_pdf_path`, `incident_description`); per-agent outputs
(`detections`, `overall_severity`, `retrieved_clauses`, `report`); and
routing fields (`escalate`, `escalation_reason`).

### 11.3 Agent contracts

| Agent | Input (from state) | Output (to state) | Implementation |
| --- | --- | --- | --- |
| Damage | `image_path` | `detections`, `annotated_image_path`, `escalate` | YOLO11m-seg inference, conf ≥ 0.45 kept, any < 0.30 ⇒ escalate |
| Severity | `detections` | per-detection severity, `overall_severity` | Per-class calibrated area-ratio bins (`configs/pipeline_config.yaml`) |
| Policy | `detections`, `overall_severity`, `incident_description` | `retrieved_clauses` | Hybrid dense+sparse RRF (75:25) over ChromaDB; also exposed as a FastMCP tool |
| Report | full state | `report` | GPT-4o primary, Gemini 1.5 Flash fallback, offline template fallback; grounded to retrieved clauses |

### 11.4 APIs between modules / database interactions

- Internal: shared-state read/write via the LangGraph orchestrator (no
  network calls between internal agents).
- External: Report Agent → OpenAI/Gemini APIs; Policy Agent → ChromaDB
  (local/embedded, no network call required).

### 11.5 Orchestration framework

LangGraph. The orchestrator encodes the two conditional branches
(escalate vs. continue; policy-PDF vs. no-PDF) as explicit graph edges
rather than in-agent conditional logic, which keeps routing testable in
isolation (§11.6).

### 11.6 End-to-end verification

Routing was verified for all three paths (image+PDF → policy → report;
image-only → report; low-confidence → human review) via
`tests/test_pipeline.py` (6 tests) and a stub-graph invocation test. A full
run with real trained weights: `python -m src.pipeline.orchestrator <image> --policy <pdf>`.

---

## 12. Computational Requirements

| Resource | Requirement |
| --- | --- |
| Training hardware | Single T4/P100 GPU (16 GB), batch=4 at 1280px |
| Training contingency | 960px / batch 8 preset if 16 GB is insufficient (`--experiment small_imgsz`) |
| Inference hardware | Free Hugging Face Spaces CPU tier (target: acceptable interactive latency) |
| Memory | 22.4M-parameter model (measured, §4.2); peak GPU memory *`<fill from probe runs>`* at batch 4/1280px |
| Expected inference latency | *`<fill from probe runs — ms/image at 1280px>`* |
| Storage | Policy vector index (ChromaDB, from M2), model checkpoints (`models/best.pt`), per-run annotated images and reports |
| External API cost | Pay-per-call for GPT-4o / Gemini 1.5 Flash (Report Agent only); zero cost when offline template fallback is used |

### 12.1 Supplementary: measured CPU inference latency (`notebooks/YOLO11_Capability_Analysis.ipynb`, 2026-07-23)

Measured on the CPU (no GPU in this dev environment) — a supplementary baseline, not a
substitute for the GPU probe-run numbers still marked `<fill from probe runs>` above:

| Variant | imgsz | ms/image |
| --- | --- | --- |
| YOLO11n-seg | 640 | 704.3 |
| YOLO11n-seg | 1280 | 2,309.6 |
| YOLO11m-seg | 640 | 2,930.1 |
| YOLO11m-seg | 1280 | 9,955.1 |

YOLO11m-seg at the project's chosen 1280px training resolution takes **~10 seconds/image on
CPU** — not interactive latency. This is a real data point to weigh against the §2.6 CPU-tier
Hugging Face Spaces deployment assumption, alongside the eventual GPU/CPU inference numbers
from the real probe run.

---

## 13. Design Decisions and Trade-offs

| Decision | Alternative rejected | Trade-off reasoning |
| --- | --- | --- |
| Single-stage YOLO11m-seg | Mask R-CNN family (two-stage) | 3–5× slower inference for marginal accuracy gain at this dataset scale; interactive latency is a hard constraint |
| Full fine-tuning | Frozen-backbone feature extraction | Domain gap (COCO → vehicle damage) is large enough that freezing would likely underfit; dataset is large enough to support full fine-tuning |
| 1280px training resolution | 640px default | 640px would discard roughly half the available image resolution given the data-driven mean resolution (M2 §6.1); accepts higher compute cost for better small-object recall |
| Rule-based severity | Learned severity classifier | No severity ground truth available; transparency/auditability valued over adaptivity for an insurance use case |
| Hybrid dense+sparse retrieval | Dense-only or sparse-only | Covers both semantic and exact-terminology query types; already validated in M2 at Precision@3 = 0.913 |
| Cloud LLM APIs (GPT-4o/Gemini) with offline fallback | Local/self-hosted LLM | No labelled report data for fine-tuning; cloud APIs give higher output quality at the cost of external dependency and per-call cost, mitigated by the offline fallback |
| CPU-tier inference deployment | GPU-hosted inference | 22.4M-parameter model (measured, §4.2) was chosen specifically to fit the free HF Spaces CPU tier, trading some inference speed for zero hosting cost |

### 13.1 Scalability considerations

The rule-based severity module and hybrid retriever both scale
independently of the detection model's throughput; the main scalability
constraint is CPU inference latency for the detection model under
concurrent load, which is out of scope for the current milestone but noted
as a Milestone-4-and-beyond consideration.

---

## 14. Risks and Limitations

| Risk / limitation | Description | Mitigation |
| --- | --- | --- |
| Probe metrics not yet finalised | §4.2 architecture comparison table is pending real `results.csv` outputs | Populate before submission; do not proceed to M4 baseline training on estimated numbers |
| Small-object detection difficulty | M2 EDA found instances with normalised bbox area as low as 0.00002 | Addressed via 1280px training resolution and YOLO11's C3k2/C2PSA blocks; will be monitored via per-class mAP in M4 |
| Class imbalance (6.68:1) | Risk of the model under-predicting minority damage classes | `cls=2.0` loss weighting; to be re-evaluated with per-class metrics in M4 |
| GPU memory ceiling | 1280px + batch 4 saturates a 16 GB P100 | Documented 960px/batch 8 contingency preset |
| LLM key unavailability | Demo may run without API keys configured | Two-level provider fallback plus deterministic offline template — pipeline always terminates with a report |
| Hallucination risk in report generation | LLM could state policy terms not actually present in the policy document | Prompting constrains the model to cite only retrieved clauses (§10.5); still requires human spot-checking, especially for edge-case clauses |
| No severity ground truth | Severity bins are calibrated manually, not learned/validated against labelled data | Documented as a known limitation; flagged for future validation against adjuster-labelled examples if available |
| CPU inference latency at scale | Free HF Spaces CPU tier may not sustain concurrent users at low latency | Out of scope for this milestone; noted for future scalability work |
| Bias | Dataset is sourced from VehiDE and may not represent all vehicle types, lighting conditions, or damage severities equally | Carried over from M2 EDA discussion; no additional bias analysis performed in M3 |
| Segmentation labels not yet available | `data/vehide/labels/` holds plain detection boxes, not polygons; confirmed by a reproducible `ValueError` when fine-tuning YOLO11n-seg (`notebooks/YOLO11_Capability_Analysis.ipynb`): "Segment dataset requires equal numbers of boxes and segments, but got len(segments) = 0, len(boxes) = 37" | Open decision for Milestone 4: extend `scripts/preprocess_images.py` to emit polygon labels from the VIA JSON already at `data/raw/vehide_raw/`, or fall back to the plain detection variant (would require revising §2.5/§4.1/§4.4/§5.1-5.2/§6.1/§7/§11.3/§13 accordingly). The plain detection variant (YOLO11n, no `-seg`) trained successfully on the identical labels/taxonomy in the same notebook. |
| Stale dataset config path | `data/damage.yaml`'s `path: ./vehide_processed` does not exist anywhere in the repo; the actual committed sample data is at `data/vehide/` | Needs a one-line fix to `data/damage.yaml` before any real training run (probe or Milestone 4 baseline) is attempted |

---

## 15. Deliverables Produced

- Model architecture diagram: `multiagent_architecture_staged.svg`
- Architecture comparison table and decision record (§4.2–4.3)
- Training configuration table with justification (§7)
- Pipeline implementation: `src/pipeline/` (`state.py`, `damage_agent.py`,
  `severity_agent.py`, `policy_agent.py`, `report_agent.py`,
  `orchestrator.py` — module names as implemented in the repository)
- Pipeline configuration: `configs/pipeline_config.yaml`
- MCP tool contract for the Policy Agent (§9.6)
- Test suite: `tests/test_pipeline.py` (6 tests) plus a stub-graph
  invocation test
- YOLO11 capability-exploration notebook: `notebooks/YOLO11_Capability_Analysis.ipynb` —
  architecture/CPU-latency measurements and a small-scale end-to-end fine-tuning smoke test
  on the sample dataset (see §4.2, §12, §14)
- This report: `Milestone3_Report.md`

### 15.1 Repository structure (relevant subset)

```
src/pipeline/
├── state.py
├── damage_agent.py
├── severity_agent.py
├── policy_agent.py
├── report_agent.py
└── orchestrator.py
configs/
└── pipeline_config.yaml
scripts/
├── train_yolo.py
└── compare_experiments.py
tests/
└── test_pipeline.py
runs/probe/
├── yolo11m_probe/results.csv
└── yolov8m_probe/results.csv
models/
└── best.pt   # populated in Milestone 4
```

---

## 16. Summary and Next Steps

### 16.1 Summary of architecture decisions

Milestone 3 selects a fine-tuned YOLO11m-seg (pending final confirmation
against YOLOv8m-seg probe numbers) for damage detection/segmentation,
paired with a transparent rule-based severity scorer, a hybrid
dense+sparse retriever carried over from Milestone 2, and an LLM-based
report generator with a two-provider fallback plus a deterministic offline
template. All four components are wired into a LangGraph-orchestrated,
four-agent pipeline with explicit escalation and branching logic, and the
routing has been verified end-to-end via automated tests.

### 16.2 Readiness for model training (Milestone 4)

The dataset, architecture, and training configuration are all finalised
and justified; the pipeline can already run end-to-end with stub/mock
weights. The only remaining prerequisite for Milestone 4 is populating the
probe-run results table (§4.2) and confirming the final model choice
(§4.3).

### 16.3 Planned implementation activities (Milestone 4)

- Baseline 50-epoch training run of the selected architecture
  (`scripts/train_yolo.py`).
- Three single-variable ablation experiments: `cls=3.0`, cosine LR
  schedule, label smoothing.
- Comparison of experiment results via `scripts/compare_experiments.py`.
- Best weights wired into the pipeline at `models/best.pt`, replacing the
  current stub weights used for end-to-end testing.

---

## Appendices

### A. Architecture diagrams

- `multiagent_architecture_staged.svg` — multi-agent pipeline architecture

### B. Sequence / flow diagram

```
damage_agent ─► severity_agent ─┬─(escalate)──────► human_review ─► END
                                ├─(policy PDF)────► policy_agent ─► report_agent ─► END
                                └─(no PDF)────────► report_agent ─► END
```

### C. Model configuration table

| Parameter | Value |
| --- | --- |
| Model | YOLO11m-seg (pending final confirmation, §4.3) |
| Parameters | 22.4M (measured, §4.2) |
| imgsz | 1280 |
| batch | 4 |
| optimizer | AdamW |
| lr0 | 0.001 |
| cls | 2.0 |
| epochs | 50 (patience 15) |

### D. Hyperparameter table

See §7 (Training Strategy) for the full justified table.

### E. Prompt templates

See §10.1–10.4 for the Report Agent prompt structure (full literal prompt
text to be attached as `prompts/report_agent_prompt.md` in the repository).

### F. API specifications

```
tool: retrieve_clauses
args: damage_classes: list[str], severity: str, incident_description: str|None, top_k: int
returns: list of {chunk_id, text, score, heading, clause_type, doc_id, damage_classes}
```

### G. References to selected models

- Ultralytics YOLO11 / YOLOv8 (Ultralytics, 2023–2024)
- OpenAI GPT-4o
- Google Gemini 1.5 Flash
- ChromaDB (vector store)

### H. Change log

| Date | Change |
| --- | --- |
| May 2026 | Milestone 3 report drafted from Milestone 2 deliverables and pipeline implementation |
| 2026-07-23 | Added YOLO11 capability-analysis notebook findings (measured param/GFLOPs/CPU-latency numbers, segmentation-label-format blocker, stale `damage.yaml` path) — §4.2, §12, §14, §15. Seg-vs-detection architecture choice remains open pending resolution before Milestone 4. |
| *`<TBD>`* | Probe-run results populated (§4.2); final model decision confirmed (§4.3) |

---

## 5 (original). Challenges

| Challenge | Resolution |
| --- | --- |
| 1280px training memory | batch=4 on P100; contingency preset at 960px/batch 8 (`--experiment small_imgsz`) |
| LLM key availability for the demo | Report Agent has a deterministic offline template fallback, so the pipeline never fails without keys |
| *`<add anything encountered>`* | |
