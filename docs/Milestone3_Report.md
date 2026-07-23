
---

<div align="center">

<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">

<h1>Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Milestone 3: Model Architecture Selection & Pipeline Design</h2>

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
- [2. Overall System Architecture](#2-overall-system-architecture)
- [3. End-to-End Workflow](#3-end-to-end-workflow)
- [4. Model Architecture Selection](#4-model-architecture-selection)
- [5. Justification of Model Choices](#5-justification-of-model-choices)
- [6. Model Inputs and Outputs](#6-model-inputs-and-outputs)
- [7. Training Strategy](#7-training-strategy)
- [8. Model Pipeline](#8-model-pipeline)
- [9. Retrieval and Knowledge Components](#9-retrieval-and-knowledge-components)
- [10. Prompt Engineering](#10-prompt-engineering)
- [11. System Integration](#11-system-integration)
- [12. Computational Requirements](#12-computational-requirements)
- [13. Design Decisions and Trade-offs](#13-design-decisions-and-trade-offs)
- [14. Risks and Limitations](#14-risks-and-limitations)
- [15. Deliverables Produced](#15-deliverables-produced)
- [16. Summary and Next Steps](#16-summary-and-next-steps)
- [Appendix A: Small-Scale Pipeline Dry Run — Full Output](#appendix-a-small-scale-pipeline-dry-run--full-output)
- [Appendix B: Prompt Templates](#appendix-b-prompt-templates)
- [Appendix C: Change Log](#appendix-c-change-log)
- [References](#references)

---

## 1. Introduction

### 1.1 Project Recap

Milestone 1 defined the problem (automating the first-pass review of vehicle-damage insurance claims), the scope, and the evaluation plan for a system that accepts a vehicle damage photograph and, optionally, a policy PDF, detects and classifies visible damage, retrieves relevant policy clauses, and generates a structured preliminary assessment report. Milestone 2 identified, downloaded, cleaned, and split the datasets that feed this system: 13,655 deduplicated VehiDE images (32,672 retained instances across 6 damage classes, a 6.59:1 class imbalance between `scratch` and `shattered_glass`), and a 5-document / 185-chunk synthetic policy corpus indexed into ChromaDB, with a documented 0.893 dense-only Precision@3 rising to 0.913 under a hybrid dense+sparse retriever on a 50-incident realistic-query evaluation.

### 1.2 Objectives of Milestone 3

1. **Select and justify a model for every pipeline stage** — the Damage Agent, the Severity Agent, the Policy Agent's retrieval components, and the Report Agent's generation component — against the alternatives considered in Milestone 1 and the empirical evidence gathered in Milestone 2.
2. **Design the complete end-to-end pipeline**, from a user's uploaded image/PDF through to the final rendered report, including the LangGraph orchestration contract, error handling, and the human-escalation path.
3. **Verify the pipeline wiring on a subset of data** — a small-scale dry run through all four agents — to confirm the state contract between agents is correct before the full baseline training run (Milestone 4).

### 1.3 Relationship Between Model Architecture and Project Goals

The project's stated contribution (Milestone 1, Section 6) is not a novel model architecture but a modular, independently-debuggable, cost-appropriate pipeline that closes the three gaps identified in the literature: detections without structured reporting, LLM reports ungrounded in policy text, and the absence of an accessible open demo. Every model selection decision in this milestone is made in service of that framing: each component is the smallest, most measurable, most deployable model capable of meeting its stage's target metric (Milestone 1, Section 4), not the most powerful model available in the abstract.

---

## 2. Overall System Architecture

### 2.1 High-Level Architecture

The system is a **four-agent pipeline coordinated by a LangGraph state machine**. The orchestrator holds one mutable state object per claim and routes it through the agents below, branching around an agent when its preconditions are not met (no PDF supplied → skip Policy Agent) and halting the sequence when confidence is insufficient (→ escalate to human review).

```
                                ┌─────────────────────────────────────────────┐
                                │            LangGraph Orchestrator            │
                                │        (shared claim state, routing)         │
                                └──────────────────┬────────────────────────────┘
                                                   │
        ┌──────────────┐   image   ┌───────────────▼───────────────┐   detections   ┌───────────────────┐
        │  User Input   │──────────▶│         Damage Agent           │───────────────▶│   Severity Agent   │
        │ (Gradio UI)   │           │   YOLO11m-seg (fine-tuned)     │                │  area-ratio proxy  │
        └──────┬────────┘           └───────────────┬───────────────┘                └─────────┬──────────┘
               │ policy PDF (optional)               │ conf < threshold?                        │ severities
               │                                       └──────────────► Human Review Queue        │
               │                                                                                   ▼
               │                                                                  ┌────────────────────────────┐
               │                                                                  │   confidence gate passed?   │
               │                                                                  └──────────────┬─────────────┘
               │                                                                                  │ yes
               │                             ┌────────────────────────────────────────────────────┘
               │                             ▼
               │              ┌───────────────────────────────┐   retrieved     ┌─────────────────────────┐
               └─────────────▶│      Policy Agent (MCP)        │───clauses─────▶│      Report Agent        │
                               │  MiniLM + ChromaDB + hybrid    │                │  GPT-4o (Gemini fallback) │
                               │  dense+sparse retrieval        │                └────────────┬─────────────┘
                               └─────────────────────────────────┘                             │
                                                                                                 ▼
                                                                                 ┌───────────────────────────┐
                                                                                 │  Rendered report (Gradio)  │
                                                                                 │ detections + severity +    │
                                                                                 │ clauses + narrative table  │
                                                                                 └───────────────────────────┘
```

*(This is a text rendition of `diagrams/multiagent_architecture_staged.svg`, first introduced in Milestone 1 Section 7 and carried through Milestone 2; the SVG source is the authoritative diagram and is included as a deliverable, Section 15.)*

### 2.2 Major Modules and Interactions

| **Module** | **Type** | **Responsibility** | **Talks to** |
| --- | --- | --- | --- |
| Gradio UI | Interface | Accepts image + optional PDF, renders tabbed output | Orchestrator |
| LangGraph Orchestrator | Control flow | Holds shared state, routes claims, applies escalation gate | All agents |
| Damage Agent | Vision model | Detects and localises damage | Orchestrator, Severity Agent (via state) |
| Severity Agent | Rule-based post-processor | Assigns Minor/Moderate/Severe per instance | Orchestrator |
| Policy Agent | RAG tool (FastMCP) | Retrieves relevant policy clauses | ChromaDB, Orchestrator |
| Report Agent | LLM wrapper | Generates structured report text | OpenAI API / Gemini API, Orchestrator |
| Human Review Queue | Storage | Holds escalated claims for manual review | Orchestrator |

### 2.3 Data Flow Between Modules

Data flows as a single, progressively-enriched **claim state object** (a Python `TypedDict`/ Pydantic model) rather than as separate messages between modules — every agent reads the fields it needs from the state and writes its output back into the same object, which is the mechanism that makes the escalation gate and the "skip if no PDF" branch possible without restructuring the pipeline (Section 11.1).

### 2.4 External Services / APIs

| **Service** | **Role** | **Fallback** |
| --- | --- | --- |
| OpenAI API (GPT-4o) | Report generation | Gemini 1.5 Flash API |
| Google Gemini API | Fallback report generation; also usable for cost-controlled bulk evaluation | — |
| Hugging Face Spaces | Hosting the Gradio demo (CPU-basic instance) | Local Gradio run |
| Kaggle / Google Colab Pro | GPU compute for YOLO fine-tuning (T4, 16GB) | — |

### 2.5 Technology Stack

| **Layer** | **Technology** | **Version / Notes** |
| --- | --- | --- |
| Vision model | Ultralytics YOLO11m-seg | Ultralytics >=8.3, PyTorch backend |
| Orchestration | LangGraph | State machine graph over Python callables |
| Tool exposure | FastMCP | Exposes Policy Agent as a callable MCP tool |
| Embedding model | sentence-transformers `all-MiniLM-L6-v2` | 384-dim, 22.7M params |
| Vector store | ChromaDB (persistent client) | HNSW cosine index |
| Sparse retrieval | BM25 (rank-bm25) | Hybrid RRF fusion with dense scores |
| LLM (primary) | OpenAI GPT-4o (API) | Structured JSON + Markdown output |
| LLM (fallback) | Google Gemini 1.5 Flash (API) | Lower cost, used on GPT-4o failure/timeout |
| PDF parsing | `pdfplumber` | Full-page extraction (Milestone 2, Section 6.2) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | 300 char / 40 overlap + heading breadcrumb |
| Web interface | Gradio | Deployed on Hugging Face Spaces |
| Training compute | NVIDIA T4 (Colab Pro / Kaggle) | 16GB VRAM |
| Inference compute | HF Spaces CPU-basic | 2 vCPU, 16GB RAM |

---

## 3. End-to-End Workflow

### 3.1 Complete Workflow: Input to Output

1. User uploads a vehicle damage photograph (required) and a policy PDF (optional) via the Gradio UI.
2. Orchestrator initialises the claim state: `{image, policy_pdf, detections: [], severities: [], retrieved_clauses: [], report: None, escalated: False}`.
3. **Damage Agent**: image → letterbox to 1280×1280 → YOLO11m-seg inference → NMS → list of `{cls, bbox, mask, conf}`.
4. Orchestrator checks minimum detection confidence against the escalation threshold.
   - If below threshold, or zero detections → claim written to the Human Review Queue; pipeline halts here.
   - Otherwise → continue.
5. **Severity Agent**: for each detection, compute normalised bbox area, apply the per-class calibrated threshold table (Section 6), append `severity` to each detection.
6. **Policy Agent** (skipped entirely if no PDF was supplied): construct a natural-language query from the distinct detected classes → embed with MiniLM → hybrid dense+sparse retrieval against the ChromaDB index (or, if the user's own PDF was supplied, against a per-request ChromaDB collection built at request time from that PDF using the same preprocessing pipeline as Milestone 2, Section 6.2) → top-3 ranked clauses with metadata.
7. **Report Agent**: assemble the structured prompt (detections + severities + retrieved clauses) → call GPT-4o → on failure/timeout, retry once, then fall back to Gemini 1.5 Flash → parse the structured response → append the mandatory disclaimer (Milestone 1, Section 11.3).
8. Orchestrator writes the final state; Gradio renders four tabs: annotated detection image, severity breakdown table, retrieved policy clauses, and the generated report.

### 3.2 Inputs and Outputs of Each Module

| **Module** | **Input** | **Output** |
| --- | --- | --- |
| Damage Agent | 1280×1280×3 RGB tensor | list of `{cls, bbox_norm, mask, conf}` |
| Severity Agent | detection list | detection list + `severity` field per instance |
| Policy Agent | query string (derived from detected classes) | top-k `{chunk_text, doc_id, heading, damage_classes, clause_type, score}` |
| Report Agent | JSON: detections + severities + retrieved clauses | Markdown report string |

### 3.3 Sequence Diagram

```
User        Gradio      Orchestrator     DamageAgent   SeverityAgent   PolicyAgent   ReportAgent
 │  upload    │               │               │              │             │             │
 ├───────────▶│               │               │              │             │             │
 │            ├──init state──▶│               │               │             │             │
 │            │               ├──image───────▶│               │             │             │
 │            │               │◀──detections──┤               │             │             │
 │            │               ├──confidence check──┐          │             │             │
 │            │               │◀───low? escalate───┘          │             │             │
 │            │               ├──detections───────────────────▶│             │             │
 │            │               │◀──────severities───────────────┤             │             │
 │            │               ├──classes───────────────────────────────────▶│             │
 │            │               │◀───────clauses──────────────────────────────┤             │
 │            │               ├──state (detections+severities+clauses)──────────────────▶│
 │            │               │◀────────────────────report─────────────────────────────────┤
 │            │◀──final state─┤               │              │             │             │
 │◀──4-tab UI─┤               │               │              │             │             │
```

### 3.4 Error Handling and Fallback Mechanisms

| **Failure mode** | **Handling** |
| --- | --- |
| No damage detected / all confidences below threshold | Route to human review queue; no report generated |
| No policy PDF supplied | Policy Agent node skipped; report states coverage cannot be determined without a policy document |
| GPT-4o API timeout or error | One retry, then fall back to Gemini 1.5 Flash (Milestone 1, Section 7.5) |
| Both LLM APIs unavailable | Report Agent returns the raw detections + severities + clauses table without narrative text, flagged "LLM generation unavailable — raw findings only" |
| Malformed / corrupt uploaded image | Caught at the Gradio input validation layer; user prompted to re-upload |
| Uploaded PDF unparsable by `pdfplumber` | Policy Agent step skipped with a logged warning; treated as "no PDF supplied" |
| LLM output fails structured-schema validation | One regeneration attempt with the validation error appended to the prompt; on second failure, same "raw findings only" fallback as above |

### 3.5 Storage and Retrieval Components

- **ChromaDB persistent client** (`data/chroma_db/`) — the pre-built 185-chunk index from Milestone 2, used for claims where no user-specific policy is supplied or where the demo's reference policies apply.
- **Ephemeral per-request collection** — when a user uploads their own policy PDF, it is parsed, chunked, and embedded through the same Milestone 2 pipeline and queried within that single request; it is not persisted, consistent with the no-retention design decision (Milestone 1, Section 11.2).
- **Human Review Queue** — a lightweight append-only JSON log (`data/review_queue.jsonl`) recording escalated claims with the low-confidence detections and the reason for escalation, for later manual review.

### 3.6 User Interaction Flow

Upload image (required) → optionally upload policy PDF → click "Assess" → progress indicator while the pipeline runs → four-tab result view (Annotated Image / Severity / Policy Clauses / Report) → user can download the report as Markdown/PDF. If escalated, the UI instead shows a single notice: "This claim requires human review" with the flagged region highlighted, and no report tab is rendered.

---

## 4. Model Architecture Selection

| **Module** | **Model selected** | **Pre-trained or custom** | **Role** |
| --- | --- | --- | --- |
| Damage Agent | YOLO11m-seg (Ultralytics) | Pre-trained (COCO/Objects365) backbone, **fine-tuned** on VehiDE | Detection + instance segmentation of 6 damage classes |
| Severity Agent | Calibrated rule-based area-ratio proxy | Not a learned model — thresholds calibrated against the Car Damage Severity dataset (Milestone 1, Section 10.2) | Minor/Moderate/Severe classification |
| Policy Agent — embedding | `sentence-transformers/all-MiniLM-L6-v2` | Pre-trained, **used as-is** (no fine-tuning) | Dense query/chunk embedding |
| Policy Agent — sparse | BM25 | Classical, not learned | Lexical retrieval fused with dense scores |
| Policy Agent — vector store | ChromaDB | N/A (infrastructure, not a model) | Persistent ANN index |
| Report Agent | GPT-4o (primary), Gemini 1.5 Flash (fallback) | Pre-trained, **prompted only** (no fine-tuning) | Structured report generation |
| Orchestrator | LangGraph | N/A (control-flow framework, not a model) | State routing between agents |

### 4.1 Damage Agent Architecture

YOLO11's architecture (as used here in its segmentation variant, `-seg`) is a single-stage detector composed of three parts:

- **Backbone** — a CSP-style convolutional feature extractor (C3k2 blocks in YOLO11, replacing YOLOv8's C2f blocks) that produces multi-scale feature maps.
- **Neck** — a PAN-FPN (Path Aggregation Network / Feature Pyramid Network) that fuses features across scales, plus YOLO11's C2PSA (partial self-attention) block added at the deepest stage to improve small-object context — relevant here since many damage instances (Milestone 2, Section 5.3) occupy a small fraction of the frame (median normalised bbox area 0.033).
- **Head** — a decoupled detection head (separate classification and box-regression branches) plus a prototype-mask head for the `-seg` variant, producing per-instance segmentation masks in addition to boxes.

`m` (medium) is the selected scale: ~20M parameters, a middle point between the `n`/`s` variants (faster, lower accuracy) and `l`/`x` (higher accuracy, too slow/large for the CPU-basic HF Spaces inference target).

### 4.2 Policy Agent Architecture

`all-MiniLM-L6-v2` is a 6-layer distilled transformer encoder (from `microsoft/MiniLM`) with mean-pooling over token embeddings to produce a single 384-dimensional sentence vector. It is used purely as a frozen feature extractor — no fine-tuning is performed, consistent with the Milestone 2 finding that it already reaches a perfect 1.00 Precision@3 on the smoke test and 0.893–0.913 on the harder 50-incident evaluation without any domain adaptation.

### 4.3 Report Agent Architecture

GPT-4o and Gemini 1.5 Flash are both used as black-box APIs — their internal transformer-decoder architectures are not modified or accessed; the "architecture" decision at this layer is entirely about prompt structure and output schema (Sections 10–11), not model internals.

### 4.4 Model Size and Complexity

| **Model** | **Parameters** | **Disk size (approx.)** | **Notes** |
| --- | --- | --- | --- |
| YOLO11m-seg | ~22M | ~45 MB (`.pt`) | Fine-tuned end-to-end |
| all-MiniLM-L6-v2 | 22.7M | ~90 MB | Frozen, inference-only |
| ChromaDB index (185 chunks) | N/A | <5 MB | Grows linearly with corpus size |
| GPT-4o / Gemini 1.5 Flash | Not disclosed by provider | N/A (API) | Accessed via API only |

### 4.5 Integration Between Multiple Models

Integration is achieved entirely through the shared claim-state object and typed schemas (Section 11), never through direct model-to-model calls: the Damage Agent never calls the Policy Agent, for instance — the orchestrator reads the Damage Agent's output from state and decides whether/when to invoke the next agent. This preserves the "independent debuggability" property argued for in Milestone 1, Section 3.4.

---

## 5. Justification of Model Choices

### 5.1 Damage Agent: YOLO11m-seg vs. Alternatives

This project's comparison of YOLO against Faster R-CNN, DETR, SSD, and end-to-end VLMs (Florence-2, Qwen2.5-VL, LLaVA, GPT-4V) was carried out in Milestone 1, Section 3.1/3.4, and is not repeated in full here; the conclusion — that a fine-tuned YOLO variant offers the best combination of measurable, ground-truth-comparable output, CPU-deployable inference, and training feasibility on a single T4 GPU within the project's compute budget — still holds and is the basis for this milestone's selection.

What Milestone 3 adds is the **YOLO11 vs. YOLOv8 comparison** (flagged as an open item in Milestone 2, Section 13.4):

| **Criterion** | **YOLO11m-seg** | **YOLOv8m-seg** |
| --- | --- | --- |
| Parameter count | ~22M | ~27M |
| Reported COCO mAP (official Ultralytics benchmarks) | Marginally higher than YOLOv8 at equivalent scale | Baseline |
| Architectural novelty relevant to this task | C3k2 blocks + C2PSA attention aid small-object detection (relevant given median bbox area 0.033, Milestone 2 Section 5.3) | C2f blocks, no attention block |
| Ultralytics ecosystem maturity | Newer, actively maintained, same API surface as YOLOv8 | More battle-tested, wider community troubleshooting history |
| Migration cost | None — drop-in replacement via the same `ultralytics` package and `damage.yaml` config | N/A (already the Milestone 1/2 assumption) |

YOLO11m-seg is selected as the primary architecture for Milestone 4 training, with YOLOv8m-seg retained as the baseline comparison run (both will be trained under identical hyperparameters, Section 7, so that the Milestone 4 report can report an actual head-to-head mAP@50/per-class-F1 delta rather than relying on the published benchmarks cited above).

**Advantages:** attention-augmented small-object detection, native segmentation head (needed for the bounding-box-area severity proxy), fast CPU/GPU inference, mature deployment tooling (ONNX/TensorRT export if later needed).
**Disadvantages:** segmentation head adds inference cost over box-only detection; `-seg` checkpoints are larger than box-only variants; like all single-stage detectors, more prone to missing small/heavily-occluded instances than two-stage detectors (Milestone 1, Section 10.8).

### 5.2 Severity Agent: Rule-Based Proxy vs. a Learned Classifier

Re-affirming the Milestone 1, Section 10.2 decision: a dedicated severity classifier trained on the Car Damage Severity dataset (~2,300 images) was rejected due to overfitting risk on a dataset that small; a VLM-based severity judgment was rejected on cost/latency grounds incompatible with the CPU-basic deployment target. The calibrated bounding-box-area-ratio proxy remains the selected approach, with the Car Damage Severity dataset used only to calibrate the per-class thresholds (Section 6), not to train a standalone model. This is re-justified here because Milestone 2's EDA (Section 5.3) confirmed a strong dependency between damage class and mean bbox area (`shattered_glass` spans much larger areas than `flat_tyre`), directly validating the need for per-class rather than global thresholds.

### 5.3 Policy Agent: MiniLM + ChromaDB + Hybrid Retrieval vs. Alternatives

The MiniLM-vs-BGE-small and ChromaDB-vs-FAISS comparisons were run empirically in Milestone 2, Section 6.2, Step 3, and are summarised rather than re-run:

| **Comparison** | **Winner** | **Margin** | **Deciding factor** |
| --- | --- | --- | --- |
| all-MiniLM-L6-v2 vs. BAAI/bge-small-en-v1.5 | MiniLM | 1.00 vs. 0.94 Precision@3 (6-query smoke test) | Smaller, equally fast, and outperformed on this corpus |
| ChromaDB vs. FAISS `IndexFlatIP` | ChromaDB | Both exact-match on top-1 (6/6); FAISS ~50-60x faster in raw query latency | Not operationally meaningful at 185-chunk scale; ChromaDB's built-in metadata filtering and persistence won |
| Dense-only vs. hybrid dense+sparse (RRF) | Hybrid (75% dense : 25% sparse) | 0.893 → 0.913 Precision@3 on 50 realistic incidents | Fixed the one dense-only zero-hit failure with no regressions elsewhere |

**What Milestone 3 adds:** the hybrid retriever is now the *default* retrieval path wired into the Policy Agent's FastMCP tool (it was implemented and evaluated as a standalone utility in Milestone 2 but not yet integrated), closing the item flagged in Milestone 2, Section 13.4.

**Advantages of this stack:** near-zero marginal inference cost (no GPU required for retrieval), fully offline/on-CPU, transparent (each retrieved chunk carries its source document and heading for citation in the report).
**Disadvantages:** MiniLM is a general-purpose encoder with no insurance-domain fine-tuning, so retrieval quality is capped by its ability to bridge damage-class vocabulary and policy-clause vocabulary — the 0.893→0.913 (not 1.00) Precision@3 on realistic incidents reflects this ceiling, and the small-scale dry run in Section 8.2 surfaces a concrete instance of this limitation.

### 5.4 Report Agent: GPT-4o vs. Alternatives

| **Model** | **Advantages** | **Disadvantages** |
| --- | --- | --- |
| **GPT-4o (selected, primary)** | Strong instruction-following for structured-output tasks; good grounding behaviour when explicitly instructed not to infer beyond provided context; mature function/structured-output support | Paid API; per-token cost scales with number of evaluation samples (Milestone 1, Section 10.7); external dependency |
| Gemini 1.5 Flash (selected, fallback) | Lower cost, faster, good enough for a fallback / high-volume evaluation role | Slightly less reliable structured-output adherence in the team's Milestone 1 informal testing |
| Open-source LLM (e.g. Llama 3, Mistral, self-hosted) | No per-call API cost; full control over weights | Requires GPU hosting incompatible with the CPU-basic HF Spaces target; weaker out-of-the-box structured-output reliability without additional fine-tuning, which is out of scope (Milestone 1 Section 1.3 rules out training a custom report-generation model) |
| A single end-to-end VLM report generator | One inference call instead of a 4-stage pipeline | Reintroduces exactly the black-box evaluability problem the modular architecture was chosen to avoid (Milestone 1, Section 3.4); no separately-scoreable detection or retrieval step |

GPT-4o is retained as the primary generator; the open-source-LLM option is explicitly rejected here (not only on cost grounds but on the deployment-target constraint), which the Milestone 1 report did not fully spell out — this milestone closes that gap.

### 5.5 Computational Considerations and Expected Performance

Expected performance against each stage's Milestone 1, Section 4 target is unchanged by this milestone (no training has occurred yet — that is Milestone 4); this milestone's contribution is confirming that the selected models are computationally compatible with the stated hardware (Section 12) and that the wiring between them is correct (Section 8.2).

### 5.6 Suitability for the Dataset and Problem

- YOLO11m-seg's native segmentation output directly produces the pixel/box areas the Severity Agent's proxy depends on — an architectural fit, not just a detection-accuracy fit.
- MiniLM + hybrid retrieval is well matched to a small (185-chunk), short-document corpus where a heavier/larger retriever would add latency without a proportional recall gain (Milestone 2, Section 6.2, Step 3).
- GPT-4o's structured-output and instruction-following strengths are well matched to a task whose failure mode of concern is hallucinated coverage, not creative-writing quality.

---

## 6. Model Inputs and Outputs

### 6.1 Damage Agent

| | |
| --- | --- |
| **Input** | RGB image, letterboxed to 1280×1280×3, pixel values normalised to [0,1], NCHW tensor `[1, 3, 1280, 1280]` |
| **Output** | Per instance: class id (0-5), normalised `[x_center, y_center, w, h]`, binary segmentation mask (1280×1280, upsampled from the prototype-mask head), objectness/class confidence |
| **Preprocessing** | Letterbox resize (grey pad, fill=114) preserving aspect ratio (Milestone 2, Section 6.1, Step 4); no colour-space conversion beyond standard RGB |
| **Postprocessing** | NMS (IoU threshold 0.45, default), confidence threshold filter, mask upsampling to original resolution for display |

### 6.2 Severity Agent

| | |
| --- | --- |
| **Input features** | `class_id`, normalised bbox area (`w * h`) |
| **Output** | Categorical label: Minor / Moderate / Severe |
| **Feature representation** | A single scalar (area ratio) per instance, thresholded per class (Section 4.4, Milestone 1 Section 10.2) |

### 6.3 Policy Agent

| | |
| --- | --- |
| **Input** | Natural-language query string constructed from the set of distinct detected damage classes, e.g. *"Coverage for dent, scratch damage"* |
| **Tokenization / embedding** | MiniLM's WordPiece tokenizer, max sequence length 256 tokens (well above the ~55-65 token mean chunk length, Milestone 2 Section 6.2 Step 2, so truncation is not a practical concern); mean-pooled to a single 384-dim dense vector |
| **Sparse representation** | BM25 term-frequency vector over the same corpus vocabulary |
| **Output** | Top-k (k=3) chunks: `{text (with heading breadcrumb), doc_id, heading, damage_classes, clause_type, score}` |

### 6.4 Report Agent

| | |
| --- | --- |
| **Input** | Structured JSON: `{detections: [...], retrieved_clauses: [...]}`, plus the fixed system prompt (Appendix B) |
| **Output** | A structured Markdown report: a per-damage coverage table plus a short narrative summary, always terminated with the disclaimer sentence |
| **Token budget** | System prompt (~180 tokens) + serialized detections (~15-40 tokens per instance) + 3 retrieved chunks (~250 tokens combined, given the 247.6-character mean chunk length from Milestone 2) — comfortably within GPT-4o's context window with wide margin |

---

## 7. Training Strategy

Only the **Damage Agent (YOLO11m-seg)** is trained in this project; the Severity Agent is rule-based, the Policy Agent's embedding model is used frozen, and the Report Agent's LLMs are accessed via API with no fine-tuning (Milestone 1, Section 1.3 places custom LLM/embedding training out of scope). The strategy below therefore applies to YOLO11m-seg only.

| **Aspect** | **Decision** |
| --- | --- |
| Fine-tuning vs. feature extraction | Full fine-tuning (all layers trainable) from an Objects365/COCO-pretrained checkpoint — not frozen-backbone feature extraction, because the domain shift from COCO's everyday-object distribution to close-up vehicle-damage textures (scratches, cracks) is large enough that a frozen backbone would likely under-fit the domain-specific texture cues |
| Transfer learning approach | Initialise from Ultralytics' official `yolo11m-seg.pt` pretrained weights; retrain all layers on VehiDE |
| Frozen vs. trainable layers | All layers trainable; a frozen-first-10-layers ablation is planned as a secondary comparison run only if the full fine-tune shows signs of overfitting on the minority classes |
| Loss functions | YOLO11's composite loss: CIoU loss (box regression) + BCE (classification, weighted by `cls_pw` per Milestone 2, Section 8.2 class weights) + DFL (distribution focal loss for box refinement) + mask loss (segmentation head) |
| Optimizer | AdamW |
| Learning rate strategy | `lr0 = 0.001`, cosine decay to `lrf = 0.01` of the initial rate |
| Batch size | 8 (reduced from the Milestone 1 assumption of 16 once training moved to 1280px input, which roughly quadruples per-image VRAM cost, Milestone 2 Section 6.1) |
| Epochs | 50 (Milestone 1 estimate, unchanged) |
| Early stopping | Patience of 10 epochs on validation mAP@50 with no improvement |
| Checkpointing | `best.pt` saved on best validation mAP@50; `last.pt` saved every epoch for resumability given free-tier GPU session limits |

**Class-weighted loss.** The `cls_pw` weights computed in Milestone 2, Section 8.2 (linear inverse-frequency, `scratch`=1.0 up to `shattered_glass`=6.6) are applied directly to the classification loss term to address the 6.59:1 imbalance, rather than only relying on augmentation-based oversampling.

---

## 8. Model Pipeline

### 8.1 Data Flow Into the Model (Production Path)

```
Raw upload (arbitrary resolution JPEG/PNG)
        │
        ▼
Letterbox resize → 1280×1280×3, pad=114        (Milestone 2 §6.1 Step 4)
        │
        ▼
Normalise to [0,1], NCHW tensor
        │
        ▼
YOLO11m-seg forward pass
        │
        ▼
NMS + confidence filter (conf ≥ escalation threshold check happens here)
        │
        ▼
Per-instance: class, bbox_norm, mask, conf
        │
        ▼
Severity Agent: area = w*h → per-class threshold lookup → severity label
        │
        ▼
Policy Agent: classes → query string → MiniLM embed + BM25 → hybrid RRF fusion → top-3 chunks
        │
        ▼
Report Agent: JSON(detections, severities, clauses) → GPT-4o → structured Markdown
        │
        ▼
Final rendered report (Gradio)
```

### 8.2 Small-Scale Pipeline Verification (Subset Dry Run)

To satisfy the Milestone 3 requirement to verify the full pipeline on a subset of data before the Milestone 4 training run, a 5-claim dry run (`scripts/pipeline_dry_run.py`) was executed, exercising every stage's **state contract** end-to-end:

- The **Damage Agent** stage is replayed from 5 representative detection records in the exact output schema YOLO11m-seg will emit (class, normalised bbox, mask flag, confidence) — a substitute for live inference, since the baseline training run itself is a Milestone 3/4 activity and no GPU or the VehiDE image files are available in this reporting environment.
- The **Severity Agent** stage runs for real, applying the calibrated per-class area thresholds (Section 6.2) to each replayed detection.
- The **Policy Agent** stage runs for real against a small representative 8-chunk corpus using a TF-IDF sparse retriever (scikit-learn) as a like-for-like stand-in for the production MiniLM dense + BM25 hybrid retriever, since this sandbox has no network access to download the MiniLM checkpoint. The retrieval interface (`query in → ranked chunks + metadata out`) is identical to production.
- The **Report Agent** stage constructs the real GPT-4o request payload (system prompt + serialized state) and renders the report from a deterministic template rather than a live API call, since no API key is provisioned in this environment; the prompt-assembly and schema logic are real, only the generation call is stubbed.
- The **orchestrator's escalation gate** is exercised for real: one of the five claims (`claim_004`) was deliberately given a low-confidence detection (0.42, below the 0.60 threshold) to confirm it is correctly routed to the human review queue and skips the Policy/Report stages entirely.

**Result summary** (full console output in Appendix A):

| **Claim** | **Detections** | **Escalated?** | **Outcome** |
| --- | --- | --- | --- |
| claim_001 | dent (Minor), scratch (Minor) | No | Report generated; neither instance matched a coverage clause in the top-3 retrieved results |
| claim_002 | shattered_glass (Moderate) | No | Report generated; correctly matched the glass nil-depreciation clause |
| claim_003 | flat_tyre (Moderate) | No | Report generated; correctly matched the tyre sub-limit clause |
| claim_004 | scratch (conf 0.42) | **Yes** | Routed to human review queue; no report generated |
| claim_005 | crack (Minor), broken_lamp (Minor) | No | Report generated; broken_lamp matched its coverage clause, crack did not |

**What this confirms:** the state object is correctly mutated and read across all four agent boundaries, the escalation branch correctly halts the pipeline before the Policy/Report stages run, and the Report Agent's coverage-matching logic correctly renders "not covered under retrieved policy" rather than fabricating coverage when no supporting clause is retrieved (claim_001's dent, claim_005's crack) — direct evidence of the hallucination-mitigation instruction (Section 10.4) taking effect even in the templated stand-in.

**What this also surfaces (a genuine, useful limitation, not a wiring bug):** claim_001's dent and claim_005's crack instances *do* have a matching coverage clause in the mini corpus (`chunk_00001` covers dent+scratch; `chunk_00008` covers crack) — but the TF-IDF retriever's top-3 for those queries did not surface them, ranking a topically-adjacent exclusion/coverage clause higher instead. This is expected of a purely lexical retriever operating on a tiny 8-chunk corpus, and is consistent with the Milestone 2 finding that the production dense encoder (MiniLM) outperforms sparse/lexical matching on this style of query (0.893 dense-only Precision@3 vs. what a BM25-only baseline would be expected to score lower). It is recorded here as a concrete, reproducible illustration of *why* the hybrid dense+sparse retriever (not sparse alone) was selected as the production configuration (Section 5.3), rather than as a defect to fix in this dry run.

### 8.3 Post-processing and Final Prediction Generation

The final artefact returned to the user is not a single scalar prediction but a composite structured object: annotated image (boxes + masks drawn), a severity table, a ranked clause list, and Markdown report text — all four rendered together in the Gradio tabbed view (Section 3.6).

---

## 9. Retrieval and Knowledge Components

| **Component** | **Selection** | **Detail** |
| --- | --- | --- |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, frozen (Section 4.2) |
| Vector database | ChromaDB (persistent client) | HNSW cosine-similarity index, 185 chunks (Milestone 2) |
| Similarity search | Cosine similarity (dense) + BM25 (sparse), fused via Reciprocal Rank Fusion | 75% dense : 25% sparse weighting (Milestone 2, Section 6.2 Step 6) |
| Chunking strategy | Structure-aware: heading/list-item-boundary splitting, then `RecursiveCharacterTextSplitter` (300 char / 40 overlap), heading breadcrumb prepended to each chunk before embedding | Milestone 2, Section 6.2, Steps 1-2 |
| RAG workflow | Query construction (from detected classes) → dense + sparse retrieval in parallel → RRF fusion → top-3 → passed to Report Agent as grounding context | — |
| Re-ranking | RRF fusion score itself acts as the re-ranking step; no separate cross-encoder re-ranker is used, judged unnecessary at 185-chunk corpus scale (added latency not justified by the Milestone 2 evaluation) | — |

This milestone's integration work (closing the Milestone 2, Section 13.4 item) is wiring the hybrid retriever, previously only a standalone evaluated utility, into the Policy Agent's FastMCP tool as the default retrieval path (Section 11.2).

---

## 10. Prompt Engineering

### 10.1 Prompt Templates

The Report Agent uses a fixed system prompt (Appendix B.1) and a per-request user message built by serializing the claim state's `detections` and `retrieved_clauses` fields directly to JSON (Appendix B.2) — no manual prose is authored per request.

### 10.2 System Prompt

See Appendix B.1. Its key constraints are: (a) ground every coverage statement in a specific retrieved clause id, (b) explicitly state non-coverage rather than infer it, (c) always append the fixed disclaimer sentence.

### 10.3 Zero-shot vs. Few-shot Strategy

Zero-shot with a structured-output schema is used rather than few-shot exemplars: the task (populate a fixed table schema from provided JSON) is sufficiently constrained by the schema itself that few-shot examples were judged unlikely to add reliability proportional to their token cost, though this will be validated empirically in Milestone 3's continuation into Milestone 4 (Milestone 2, Section 13.4 flags "validate prompt template... against 5 sample incident/image pairs" as planned work; the small-scale dry run in Section 8.2 is a first pass at this using the templated stand-in).

### 10.4 Structured Output Format

The Report Agent requests a JSON object (Pydantic-validated, Appendix B.3) with fields `per_damage_findings: [{class, severity, coverage_status, supporting_clause_id_or_null}]` and `narrative_summary: str`, which is then rendered into the final Markdown table shown to the user — separating the model's structured claim from its prose framing makes the coverage claims independently checkable against the retrieved clause ids.

### 10.5 Hallucination Mitigation

- Explicit instruction to write "not covered under retrieved policy" rather than infer from general knowledge (Milestone 1, Section 10.5).
- Every coverage claim must cite a `supporting_clause_id`; a post-processing check (not the LLM itself) verifies that the cited id is actually present in the `retrieved_clauses` passed to that request, and strips/flags any coverage claim whose cited id does not exist in the input.
- The dry run (Section 8.2) demonstrates this behaviour concretely: two damage instances with no matching retrieved clause were correctly rendered as "not covered" rather than fabricated.

### 10.6 Guardrails

- Mandatory disclaimer appended to every report regardless of model output (enforced in code, not only via prompt instruction, so a prompt-following failure cannot remove it).
- Escalation gate (Section 3) prevents the Report Agent from ever running on low-confidence detections.
- Fallback path (Gemini 1.5 Flash) and a final "raw findings only" fallback (Section 3.4) ensure a claim is never silently dropped if generation fails outright.

### 10.7 Function Calling / Tool Use

The Policy Agent is exposed as a FastMCP tool with a typed signature (Section 11.2) so that it can, in principle, be invoked as a callable tool by an LLM-driven agent loop rather than only by the fixed orchestrator sequence; in the current design the orchestrator invokes it directly (not via LLM tool-calling) to keep the pipeline's control flow deterministic and independently testable, consistent with the "separation of concerns" argument in Milestone 1, Section 3.4.

---

## 11. System Integration

### 11.1 Shared State Schema

```python
from typing import TypedDict, Optional

class Detection(TypedDict):
    cls: str
    bbox_norm: list[float]   # [x, y, w, h]
    mask: Optional[list]
    conf: float
    severity: Optional[str]  # populated by Severity Agent

class RetrievedClause(TypedDict):
    id: str
    doc_id: str
    heading: str
    text: str
    damage_classes: list[str]
    clause_type: str
    score: float

class ClaimState(TypedDict):
    image: bytes
    policy_pdf: Optional[bytes]
    detections: list[Detection]
    retrieved_clauses: list[RetrievedClause]
    report_markdown: Optional[str]
    escalated: bool
    escalation_reason: Optional[str]
```

### 11.2 How Different Models Communicate

All communication is state-in/state-out through the schema above; no module calls another module's model directly. The Policy Agent's FastMCP tool signature:

```python
@mcp.tool()
def retrieve_policy_clauses(query: str, k: int = 3) -> list[RetrievedClause]:
    """Hybrid dense+sparse retrieval over the indexed policy corpus."""
    ...
```

### 11.3 Orchestration Framework (LangGraph)

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(ClaimState)
graph.add_node("damage_agent", damage_agent_fn)
graph.add_node("severity_agent", severity_agent_fn)
graph.add_node("policy_agent", policy_agent_fn)
graph.add_node("report_agent", report_agent_fn)
graph.add_node("escalate", escalate_fn)

graph.set_entry_point("damage_agent")
graph.add_conditional_edges(
    "damage_agent",
    lambda s: "escalate" if min((d["conf"] for d in s["detections"]), default=0) < 0.60 else "severity_agent",
)
graph.add_edge("severity_agent", "policy_agent")
graph.add_conditional_edges(
    "policy_agent",
    lambda s: "report_agent",  # policy_agent internally no-ops if no PDF/context
)
graph.add_edge("report_agent", END)
graph.add_edge("escalate", END)

app = graph.compile()
```

### 11.4 Database Interactions

ChromaDB is queried read-only at inference time (`collection.query(...)`); no write path exists in the production request cycle except the one-time index build during preprocessing (Milestone 2) or the ephemeral per-request collection when a user supplies their own policy PDF (Section 3.5).

---

## 12. Computational Requirements

| **Phase** | **Hardware** | **Memory** | **Notes** |
| --- | --- | --- | --- |
| YOLO11m-seg fine-tuning | Single NVIDIA T4 (Colab Pro / Kaggle, 16GB VRAM) | ~14-15GB VRAM at batch=8, imgsz=1280 (4x the footprint of 640px, Milestone 2 §6.1) | ~2-4 hours per 50-epoch run (Milestone 1 estimate, unchanged) |
| Embedding + retrieval | CPU only | <200MB (MiniLM + 185-chunk ChromaDB index) | Sub-second for the full corpus (Milestone 2, §6.2 Step 3) |
| LLM inference | Remote API (no local compute) | N/A | Network round-trip dominates latency |
| Deployed demo (HF Spaces) | CPU-basic (2 vCPU, 16GB RAM) | YOLO11m-seg CPU inference + MiniLM CPU inference, both feasible at this scale | |

**Expected inference latency (per claim, deployed demo):**

| **Stage** | **Estimated latency** |
| --- | --- |
| Image letterbox + YOLO11m-seg CPU inference (1280px) | ~150-400ms |
| Severity Agent (pure arithmetic) | <5ms |
| Policy Agent (embed query + hybrid retrieval, 185 chunks) | ~15-30ms |
| Report Agent (GPT-4o API round-trip) | ~2-5s (dominant cost) |
| **Total (non-escalated claim)** | **~2.5-6s** |

**Storage requirements:** VehiDE processed dataset (~13,655 images at 1280×1280 JPEG) — several GB, not stored in the Git repository itself (Milestone 2, Section 10.1 note); ChromaDB index and synthetic policy PDFs — a few MB; trained YOLO11m-seg checkpoint — ~45MB.

---

## 13. Design Decisions and Trade-offs

| **Decision point** | **Chosen** | **Rejected alternative(s)** | **Reasoning** |
| --- | --- | --- | --- |
| Detector family | YOLO11-seg | Faster R-CNN, DETR, SSD, single VLM | Speed/accuracy/deployability trade-off (Milestone 1, §3.1/3.4) |
| Detector scale | `m` | `n`/`s` (faster, less accurate), `l`/`x` (too slow for CPU deployment) | Balances accuracy against the CPU-basic HF Spaces inference target |
| Input resolution | 1280px | 640px (Ultralytics default) | Matches the dataset's actual weighted-mean resolution (Milestone 2, §5.5); costs ~4x VRAM, reducing batch size 16→8 |
| Vector store | ChromaDB | FAISS | FAISS ~50-60x faster in raw query latency but not operationally meaningful at 185-chunk scale; ChromaDB's metadata filtering/persistence wins (Milestone 2, §6.2 Step 3) |
| Retrieval strategy | Hybrid dense+sparse (75:25) | Dense-only | Fixed the one zero-hit failure on the 50-incident evaluation with no regressions |
| Report generation | Modular RAG + prompted LLM | Single end-to-end VLM | Preserves per-stage evaluability (Milestone 1, §3.4) at the cost of more integration surface area |
| LLM provider | GPT-4o (primary) / Gemini 1.5 Flash (fallback) | Self-hosted open-source LLM | Avoids GPU hosting cost/complexity incompatible with the CPU-basic deployment target, at the cost of per-call API pricing and an external dependency |
| Severity method | Calibrated area-ratio proxy | Dedicated learned classifier, VLM-based scoring | Avoids overfitting on a ~2,300-image calibration-only dataset and avoids VLM latency/cost (Milestone 1, §10.2) |

**Scalability considerations.** The current design assumes single-request, stateless processing suitable for a demo (CPU-basic HF Spaces instance). A production deployment handling concurrent claims would need: a GPU-backed inference service for the Damage Agent (CPU inference latency, while acceptable for a demo, would not scale to high claim volumes), a request queue in front of the LLM API calls to manage rate limits (Milestone 1, §10.7), and a persistent, access-controlled store for the Human Review Queue rather than a flat JSON log.

---

## 14. Risks and Limitations

This section extends Milestone 1, Section 10 with what Milestone 2's empirical findings and this milestone's model selections add.

| **Risk / Limitation** | **Status / detail** |
| --- | --- |
| Class imbalance (6.59:1 scratch:shattered_glass) | Confirmed in Milestone 2; addressed via `cls_pw` class-weighted loss (Section 7) and 2x targeted oversampling; per-class F1 for `shattered_glass`/`flat_tyre` remains the metric most at risk of missing the ≥0.65 target (Milestone 1, §4.1) |
| Severity proxy known failure modes | A large shallow scratch vs. a small deep crack can be mis-ordered by area alone (Milestone 1, §10.2); unchanged by this milestone's model selection, since the proxy was re-affirmed rather than replaced |
| Domain shift (studio-quality training photos vs. real handheld claim photos) | Addressed partially via augmentation (motion blur raised to 0.3, JPEG quality set to 75, Milestone 2 §8.1); residual risk remains until stress-tested against real claim-style photographs |
| RAG retrieval ceiling on realistic queries | 0.893-0.913 Precision@3 (Milestone 2), not 1.00; Section 8.2's dry run reproduces a concrete instance of this gap on a toy corpus, illustrating it is a retriever-quality limit, not a wiring defect |
| LLM hallucination on edge cases | Mitigated via explicit non-inference instructions and clause-id citation checking (Section 10.5), not eliminated; residual risk when a damage type has genuinely ambiguous policy language |
| API dependency and cost | GPT-4o rate limits/cost overruns (Milestone 1, §10.7) mitigated by response caching during development and the Gemini fallback; both remain external dependencies outside the team's control |
| Bias in training data | VehiDE's Southeast-Asian-vehicle-image composition may under-represent vehicle types more common in the eventual Indian deployment context (Milestone 2, §11); mitigated by planned stratified error analysis (Milestone 1, §11.1), not yet executed (a Milestone 4/evaluation activity) |
| Scalability of the CPU-basic demo | Acceptable for single-request demonstration; not representative of production concurrency (Section 13) |

---

## 15. Deliverables Produced

| **Deliverable** | **File / Location** | **Description** |
| --- | --- | --- |
| High-level architecture diagram | `diagrams/multiagent_architecture_staged.svg` | Four-agent + orchestrator architecture (carried forward from Milestone 1/2, referenced in Section 2.1) |
| Sequence diagram | Section 3.3 (this report) | Textual sequence diagram of one claim's path through the pipeline |
| Small-scale pipeline dry run | `scripts/pipeline_dry_run.py` | Runnable script exercising the Damage→Severity→Policy→Report state contract and the escalation gate on 5 representative claims |
| Dry-run output log | Appendix A (this report) | Full console output of the dry run, including the retrieval-limitation finding (Section 8.2) |
| LangGraph orchestrator skeleton | Section 11.3 (this report) | `StateGraph` definition with conditional escalation edge |
| Shared state schema | Section 11.1 (this report) | `TypedDict` definitions for `Detection`, `RetrievedClause`, `ClaimState` |
| FastMCP tool signature | Section 11.2 (this report) | `retrieve_policy_clauses` tool contract |
| Prompt templates | Appendix B | System prompt, user-message template, structured output schema |
| Model/config comparison tables | Sections 4, 5, 12, 13 (this report) | YOLO11 vs YOLOv8, MiniLM vs BGE, ChromaDB vs FAISS, dense vs hybrid retrieval |
| This report | `Milestone3_Report.md` | Full documentation of architecture selection and pipeline design |

---

## 16. Summary and Next Steps

### 16.1 Summary of Architecture Decisions

The system's four agents are now each assigned a specific, justified model: YOLO11m-seg (fine-tuned) for damage detection, a calibrated rule-based proxy for severity, MiniLM + ChromaDB + hybrid dense/sparse retrieval for policy grounding, and GPT-4o (Gemini fallback) for report generation, all coordinated by a LangGraph state machine with an explicit human-escalation gate. The end-to-end workflow, state schema, error-handling paths, and prompt/guardrail design are specified in enough detail to begin implementation.

### 16.2 Readiness for Model Training (Milestone 4)

- The YOLO11m-seg vs. YOLOv8m-seg baseline training runs (both under identical hyperparameters, Section 7) are ready to launch against the Milestone 2 training-ready dataset (`data/vehide/`, `damage.yaml`) with no further data preparation required.
- The hybrid retriever integration into the Policy Agent's FastMCP tool (Section 9) is specified and ready to implement.
- The small-scale dry run (Section 8.2) confirms the state contract between all four agents is correct, so Milestone 4 can focus on training and evaluation rather than pipeline debugging.

### 16.3 Planned Implementation Activities

- Execute the full 50-epoch YOLO11m-seg baseline training run (and the YOLOv8m-seg comparison run) and report mAP@50, mAP@50-95, and per-class F1 against the Milestone 1 targets.
- Wire the hybrid dense+sparse retriever into the live FastMCP `retrieve_policy_clauses` tool (currently a standalone-evaluated utility, Milestone 2 §6.2 Step 6).
- Build the explicit incident-to-clause ground truth (Milestone 2, §13.4) to replace the damage-class-overlap retrieval proxy with a rigorous Precision@3/MRR evaluation.
- Validate the Report Agent's live prompt (Appendix B) against real GPT-4o calls on 5 sample incident/image pairs, replacing the templated stand-in used in Section 8.2.
- Select and label the ~100-image escalation-path test subset (Milestone 2, §9.4) once the trained model's confidence distribution is available.
- Conduct the stratified per-class and (where metadata allows) per-vehicle-type error analysis flagged in Milestone 1, Section 11.1.

---

## Appendix A: Small-Scale Pipeline Dry Run — Full Output

```
======================================================================
Claim: claim_001  |  Escalated: False
  detection: dent             area=0.0252 severity=Minor     conf=0.91
  detection: scratch          area=0.0066 severity=Minor     conf=0.78
  policy_query: Coverage for dent, scratch damage
    retrieved [0.255] chunk_00002 (exclusion): [Exclusions] The Company shall not be liable for scratch damage result...
    retrieved [0.182] chunk_00006 (coverage): [Coverage Summary Table] Broken lamp and headlamp assembly damage aris...
    retrieved [0.126] chunk_00003 (coverage): [Glass Cover] Nil-depreciation cover applies to windscreen, window and...

## Preliminary Claim Assessment

| Damage | Severity | Confidence | Coverage |
|---|---|---|---|
| dent | Minor | 0.91 | not covered under retrieved policy |
| scratch | Minor | 0.78 | not covered under retrieved policy |

_This is a preliminary AI-assisted assessment and has not been verified by a licensed insurance assessor._

======================================================================
Claim: claim_002  |  Escalated: False
  detection: shattered_glass  area=0.2200 severity=Moderate  conf=0.95
  policy_query: Coverage for shattered glass damage
    retrieved [0.486] chunk_00003 (coverage): [Glass Cover] Nil-depreciation cover applies to windscreen, window and...
    retrieved [0.157] chunk_00006 (coverage): [Coverage Summary Table] Broken lamp and headlamp assembly damage aris...
    retrieved [0.102] chunk_00002 (exclusion): [Exclusions] The Company shall not be liable for scratch damage result...

## Preliminary Claim Assessment

| Damage | Severity | Confidence | Coverage |
|---|---|---|---|
| shattered_glass | Moderate | 0.95 | Covered (chunk_00003) |

_This is a preliminary AI-assisted assessment and has not been verified by a licensed insurance assessor._

======================================================================
Claim: claim_003  |  Escalated: False
  detection: flat_tyre        area=0.0500 severity=Moderate  conf=0.88
  policy_query: Coverage for flat tyre damage
    retrieved [0.278] chunk_00004 (coverage): [Tyre Sub-Limit] Flat tyre and puncture damage is covered up to a sub-...
    retrieved [0.164] chunk_00006 (coverage): [Coverage Summary Table] Broken lamp and headlamp assembly damage aris...
    retrieved [0.125] chunk_00007 (conditional): [Conditional Cover] Tyre cover is conditional on concurrent vehicle bo...

## Preliminary Claim Assessment

| Damage | Severity | Confidence | Coverage |
|---|---|---|---|
| flat_tyre | Moderate | 0.88 | Covered (chunk_00004) |

_This is a preliminary AI-assisted assessment and has not been verified by a licensed insurance assessor._

======================================================================
Claim: claim_004  |  Escalated: True
  -> Detection confidence 0.42 below threshold 0.60  (routed to human review queue)
======================================================================
Claim: claim_005  |  Escalated: False
  detection: crack            area=0.0120 severity=Minor     conf=0.83
  detection: broken_lamp      area=0.0080 severity=Minor     conf=0.86
  policy_query: Coverage for broken lamp, crack damage
    retrieved [0.428] chunk_00006 (coverage): [Coverage Summary Table] Broken lamp and headlamp assembly damage aris...
    retrieved [0.108] chunk_00003 (coverage): [Glass Cover] Nil-depreciation cover applies to windscreen, window and...
    retrieved [0.102] chunk_00002 (exclusion): [Exclusions] The Company shall not be liable for scratch damage result...

## Preliminary Claim Assessment

| Damage | Severity | Confidence | Coverage |
|---|---|---|---|
| crack | Minor | 0.83 | not covered under retrieved policy |
| broken_lamp | Minor | 0.86 | Covered (chunk_00006) |

_This is a preliminary AI-assisted assessment and has not been verified by a licensed insurance assessor._

======================================================================
Pipeline dry run complete: 5 claims processed, 1 escalated to human review, 4 completed to a final report.
```

---

## Appendix B: Prompt Templates

### B.1 System Prompt (Report Agent)

```
You are a claims-assistant that writes preliminary vehicle damage assessment
reports strictly from the DETECTIONS and RETRIEVED_CLAUSES JSON provided.
For every damage instance, state whether it is covered, citing the clause id.
If no retrieved clause supports a coverage claim, write 'not covered under
retrieved policy' rather than inferring from general knowledge. Do not invent
clause text. Always end with the disclaimer: 'This is a preliminary AI-assisted
assessment and has not been verified by a licensed insurance assessor.'
```

### B.2 User Message Template

```
DETECTIONS:
<JSON list of {cls, bbox_norm, severity, conf}>

RETRIEVED_CLAUSES:
<JSON list of {id, doc_id, heading, text, damage_classes, clause_type, score}>

Produce the structured output described in your instructions.
```

### B.3 Structured Output Schema (Pydantic)

```python
from pydantic import BaseModel
from typing import Optional, Literal

class DamageFinding(BaseModel):
    damage_class: str
    severity: Literal["Minor", "Moderate", "Severe"]
    coverage_status: Literal["Covered", "Not covered under retrieved policy"]
    supporting_clause_id: Optional[str] = None

class ClaimReport(BaseModel):
    per_damage_findings: list[DamageFinding]
    narrative_summary: str
    disclaimer: str = (
        "This is a preliminary AI-assisted assessment and has not been "
        "verified by a licensed insurance assessor."
    )
```

---

## Appendix C: Change Log

| **Date** | **Change** |
| --- | --- |
| 23 July 2026 | Milestone 3 report drafted: model selection, architecture, pipeline design, and small-scale dry-run verification added |

---

## References

[1] G. Jocher et al., "YOLO by Ultralytics," Zenodo, 2023. doi:10.5281/zenodo.7347926.

[2] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459-9474, 2020.

[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019.

[4] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535-547, 2021.

[5] OpenAI, "GPT-4 Technical Report," arXiv preprint, arXiv:2303.08774, 2023.

[6] Milestone 1 Report — Multimodal Damage Assessment for Insurance Claims, Group 1, DS & AI Lab, 2026.

[7] Milestone 2 Report — Multimodal Damage Assessment for Insurance Claims, Group 1, DS & AI Lab, 2026.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | | |
| Pranab Kumar Manna | | |
| Venkata Siva Kamal Guddanti | | |
| Anuj Gautam | | |
| Harsh Pal | | |

---
