
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

1. **Select and justify a model for every pipeline stage:** the Damage Agent, the Severity Agent, the Policy Agent's retrieval components, and the Report Agent's generation component against the alternatives considered in Milestone 1 and the empirical evidence gathered in Milestone 2.
2. **Design the complete end-to-end pipeline:** from a user's uploaded image/PDF through to the final rendered report, including the LangGraph orchestration contract, error handling, and the human-escalation path.

### 1.3 Relationship Between Model Architecture and Project Goals

The project's stated contribution is not a novel model architecture but a modular, independently-debuggable, cost-appropriate pipeline that closes the three gaps identified in the literature: detections without structured reporting, LLM reports ungrounded in policy text, and the absence of an accessible open demo. Every model selection decision in this milestone is made in service of that framing: each component is the smallest, most measurable, most deployable model capable of meeting its stage's target metric (Milestone 1, Section 4), not the most powerful model available in the abstract.

---

## 2. Overall System Architecture

### 2.1 High-Level Architecture

The system is a **four-agent pipeline coordinated by a LangGraph state machine**. The orchestrator holds one mutable state object per claim and routes it through the agents below, branching around an agent when its preconditions are not met (no PDF supplied → skip Policy Agent) and halting the sequence when confidence is insufficient (→ escalate to human review).

```
                                ┌───────────────────────────────────────────────┐
                                │            LangGraph Orchestrator             │
                                │        (shared claim state, routing)          │
                                └──────────────────┬────────────────────────────┘
                                                   │
        ┌───────────────┐   image   ┌───────────────▼───────────────┐   detections      ┌────────────────────┐
        │  User Input   │─────────▶│         Damage Agent           │─────────────────▶│   Severity Agent   │
        │ (Gradio UI)   │           │   YOLO11m (fine-tuned)        │                   │  area-ratio proxy  │
        └──────┬────────┘           └───────────────┬───────────────┘                   └─────────┬──────────┘
               │ policy PDF (optional)              │ conf < threshold?                          │ severities
               │                                    └──────────────► Human Review Queue          │
               │                                                                                 ▼
               │                                                                  ┌────────────────────────────┐
               │                                                                  │   confidence gate passed?  │
               │                                                                  └──────────────┬─────────────┘
               │                                                                                 │ yes
               │                             ┌───────────────────────────────────────────────────┘
               │                             ▼
               │               ┌───────────────────────────────┐   retrieved    ┌─────────────────────────┐
               └─────────────▶│      Policy Agent (MCP)        │ ──clauses────▶│      Report Agent       │
                               │  MiniLM + ChromaDB + hybrid   │                │ GPT-4o (Gemini fallback) │
                               │  dense+sparse retrieval       │                └────────────┬─────────────┘
                               └───────────────────────────────┘                             │
                                                                                             ▼
                                                                                 ┌───────────────────────────┐
                                                                                 │  Rendered report (Gradio) │
                                                                                 │ detections + severity +   │
                                                                                 │ clauses + narrative table │
                                                                                 └───────────────────────────┘
```

*(This is a text rendition of `multiagent_architecture_staged.svg`, first introduced in Milestone 1 Section 7 and carried through Milestone 2; the SVG source is the authoritative diagram and is included as a deliverable, Section 15.)*

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

Data flows as a single, progressively-enriched **claim state object** (a Python `TypedDict`/Pydantic model) rather than as separate messages between modules — every agent reads the fields it needs from the state and writes its output back into the same object, which is the mechanism that makes the escalation gate and the "skip if no PDF" branch possible without restructuring the pipeline (Section 11.1).

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
| Vision model | Ultralytics YOLO11m | Ultralytics >=8.3, PyTorch backend |
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

1. User uploads a vehicle damage photograph (required) and selects a policy from the catalog (required) and/or supplies an incident narrative (optional).
2. The claim's shared state is initialised: `{image, detections: [], overall_severity: None, policy: {}, retrieved_clauses: {}, report: None, escalated: False}` (Section 6).
3. **Damage Agent**: image → letterbox to 1280×1280 → YOLO11m inference → NMS → list of `{class_id, class_name, confidence, bbox_normalized, area_ratio}`.
4. Minimum detection confidence is checked against the escalation threshold (0.50, Section 8.2) by the calling code (Section 4.5).
   - If below threshold, or zero detections → claim written to the human review queue; pipeline halts here.
   - Otherwise → continue.
5. **Severity Agent**: for each detection, apply the per-class calibrated threshold table (Section 6.2) to the normalised bbox area, append `severity` to each detection.
6. **Stage 1 — Policy selection**: the claimant's selected policy (Section 5.3/6.3) resolves to a validated `doc_id`, scoping all retrieval that follows.
7. **Stage 2 — Policy Agent**: for each distinct detected class, run a coverage query and a separate exclusion query, both scoped to `doc_id` → MiniLM embed + hybrid dense/sparse retrieval against the ChromaDB index → up to 5 ranked clauses per query, filtered by the minimum score floor (Section 6.3).
8. **Report Agent**: assemble the context bundle (detections, severities, policy selection, retrieved clauses, incident narrative — Section 6.4) → call `llama-3.3-70b-versatile` and `openai/gpt-oss-20b` (Section 5.4) → parse the structured JSON response (verdict per damage class with citations, Section 6.4).
9. The final state is rendered to the user. A tabbed interface (annotated detection image, severity breakdown, retrieved policy clauses, generated report) is planned via Gradio; this UI is not yet built (Section 15).

### 3.2 Inputs and Outputs of Each Module

| **Module** | **Input** | **Output** |
| --- | --- | --- |
| Damage Agent | 1280×1280×3 RGB tensor | list of `{class_id, class_name, confidence, bbox_normalized, area_ratio}` |
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
| GPT-4o API timeout or error | One retry, then fall back to Gemini 1.5 Flash |
| Both LLM APIs unavailable | Report Agent returns the raw detections + severities + clauses table without narrative text, flagged "LLM generation unavailable — raw findings only" |
| Malformed / corrupt uploaded image | Caught at the Gradio input validation layer; user prompted to re-upload |
| Uploaded PDF unparsable by `pdfplumber` | Policy Agent step skipped with a logged warning; treated as "no PDF supplied" |
| LLM output fails structured-schema validation | One regeneration attempt with the validation error appended to the prompt; on second failure, same "raw findings only" fallback as above |

### 3.5 Storage and Retrieval Components

- **ChromaDB persistent client** (`data/chroma_db/`) the pre-built 185-chunk index from Milestone 2, used for claims where no user-specific policy is supplied or where the demo's reference policies apply.
- **Ephemeral per-request collection** when a user uploads their own policy PDF, it is parsed, chunked, and embedded through the pipeline and queried within that single request; it is not persisted, consistent with the no-retention design decision.
- **Human Review Queue** a lightweight append-only JSON log (`data/review_queue.jsonl`) recording escalated claims with the low-confidence detections and the reason for escalation, for later manual review.

### 3.6 User Interaction Flow

Upload image (required) → optionally upload policy PDF → click "Assess" → progress indicator while the pipeline runs → four-tab result view (Annotated Image / Severity / Policy Clauses / Report) → User can download the report as Markdown/PDF. If escalated, the UI instead shows a single notice: "This claim requires human review" with the flagged region highlighted, and no report tab is rendered.

---

## 4. Model Architecture Selection

| **Module** | **Model selected** | **Pre-trained or custom** | **Role** |
| --- | --- | --- | --- |
| Damage Agent | YOLO11m (Ultralytics) | Pre-trained backbone, **fine-tuned** on VehiDE | Bounding-box detection of 6 damage classes |
| Severity Agent | Calibrated rule-based area-ratio proxy | Not a learned model, thresholds calibrated against the Car Damage Severity dataset | Minor/Moderate/Severe classification |
| Policy Agent - embedding | `sentence-transformers/all-MiniLM-L6-v2` | Pre-trained, **used as-is** (no fine-tuning) | Dense query/chunk embedding |
| Policy Agent - sparse | TF-IDF (`sklearn.TfidfVectorizer`) | Fit once on the 185-chunk corpus (not learned in the ML sense - vocabulary/IDF weights only) | Lexical retrieval fused with dense scores via weighted RRF |
| Policy Agent - vector store | ChromaDB | N/A (infrastructure, not a model) | Persistent ANN index |
| Report Agent | `llama-3.3-70b-versatile` and `openai/gpt-oss-20b` (both run and compared), via Groq API | Pre-trained, **prompted only** (no fine-tuning) | Structured report generation |
| Orchestrator | None implemented yet — plain sequential Python function calls. LangGraph is the target framework for wrapping the existing stage functions, not yet built | N/A | Currently: fixed call sequence, not state-graph routing |

### 4.1 Damage Agent Architecture

YOLO11's architecture is a single-stage detector composed of three parts:

- **Backbone:** A CSP-style convolutional feature extractor (C3k2 blocks in YOLO11, replacing YOLOv8's C2f blocks) that produces multi-scale feature maps.
- **Neck:** A PAN-FPN (Path Aggregation Network / Feature Pyramid Network) that fuses features across scales, plus YOLO11's C2PSA (partial self-attention) block added at the deepest stage to improve small-object context relevant here since many damage instances occupy a small fraction of the frame (median normalised bbox area 0.033).
- **Head:** A decoupled detection head (separate classification and box-regression branches) producing per-instance bounding boxes and class confidence.

`m` (medium) is the selected scale: ~20M parameters, a middle point between the `n`/`s` variants (faster, lower accuracy) and `l`/`x` (higher accuracy, too slow/large for the CPU-basic HF Spaces inference target).

### 4.2 Policy Agent Architecture

`all-MiniLM-L6-v2` is a 6-layer distilled transformer encoder (from `microsoft/MiniLM`) with mean-pooling over token embeddings to produce a single 384-dimensional sentence vector. It is used purely as a frozen feature extractor no fine-tuning is performed, consistent with the Milestone 2 finding that it already reaches a perfect 1.00 Precision@3 on the smoke test and 0.893–0.913 on the harder 50-incident evaluation without any domain adaptation.

### 4.3 Report Agent Architecture

`llama-3.3-70b-versatile` and `openai/gpt-oss-20b`, both accessed via the Groq API, are used as black-box models and their internal transformer-decoder architectures are not modified or accessed. The "architecture" decision at this layer is entirely about prompt structure and output schema (Sections 10–11), not model internals. Both models are run and compared head-to-head (Section 5.4) rather than one being designated primary with the other as fallback.

### 4.4 Model Size and Complexity

| **Model** | **Parameters** | **Disk size (approx.)** | **Notes** |
| --- | --- | --- | --- |
| YOLO11m | 20.1M | ~40.7 MB (`.pt`) | Fine-tuned end-to-end |
| all-MiniLM-L6-v2 | 22.7M | ~90 MB | Frozen, inference-only |
| ChromaDB index (185 chunks) | N/A | <5 MB | Grows linearly with corpus size |
| `llama-3.3-70b-versatile` | 70B (remote) | N/A (API) | Accessed via Groq API only |
| `openai/gpt-oss-20b` | 20B (remote, MoE) | N/A (API) | Accessed via Groq API only |


### 4.5 Integration Between Multiple Models

Integration is achieved entirely through a shared state object and typed schemas (Section 11), never through direct model-to-model calls: the Damage Agent never calls the Policy Agent. For instance, each stage is currently invoked as a **fixed sequence of plain Python function calls** (Section 2.5) rather than through an orchestrator making dynamic routing decisions a stage function's output (e.g. minimum detection confidence) is checked by simple conditional logic in the calling code to decide whether the next stage runs at all (e.g. the escalation gate skips the Policy/Report Agent calls entirely). This still preserves the "independent debuggability" property argued for in Milestone 1, Section 3.4: each stage function is testable in isolation against the shared schema, whether or not a routing framework sits in front of it. Wrapping these stage functions as LangGraph nodes (Section 11.3) is planned but not yet implemented.

---

## 5. Justification of Model Choices


### 5.1 Damage Agent: YOLO11m vs. Alternatives

This project's comparison of YOLO against Faster R-CNN, DETR, SSD, and end-to-end VLMs (Florence-2, Qwen2.5-VL, LLaVA, GPT-4V) was carried out in Milestone 1, Section 3.1/3.4, and is not repeated in full here. The conclusion is that a fine-tuned YOLO variant offers the best combination of measurable, ground-truth-comparable output, CPU-deployable inference, and training feasibility on a GPU within the project's compute budget and still holds and is the basis for this milestone's selection.

**YOLO11 vs. YOLOv8 comparison:**

Rather than relying on published COCO benchmarks (which reflect a different dataset and task), both architectures were actually trained and measured on this project's data. A fixed 3,000-image training subsample (with a 500-image validation subsample), identical across both architectures, was trained progressively at **3, 5, and then 15 epochs** and each stage checked whether a clear winner had emerged before committing more GPU time to the next. At 15 epochs (the most informative run), both models remained far from converged (mAP@50 ≈ 0.03-0.04 on a 6-class task starting from a reinitialised detection head), but that is expected at this data/epoch scale and was not the point of the probe the point was the **relative** comparison between the two architectures under identical conditions.

| **Criterion** | **YOLO11m** | **YOLOv8m** |
| --- | --- | --- |
| Parameters (measured) | 20.1M | 25.9M |
| mAP@50 @ 15-epoch probe (3,000-image subsample) | 0.0305 | 0.0371 |
| mAP@50-95 @ 15-epoch probe | 0.0046 | 0.0081 |
| Training time / epoch (measured) | 1,370.8 s | 1,265.8 s |
| Inference speed (measured, 1280px) | 99.9 ms/image | 92.3 ms/image |
| Peak inference memory (measured) | 1.45 GB | 1.44 GB |
| Architectural novelty relevant to this task | C3k2 blocks + C2PSA attention aid small-object detection | C2f blocks, no attention block |
| Migration cost | None — drop-in replacement via the same `ultralytics` package and `damage.yaml` config | N/A (already the Milestone 1/2 assumption) |

**No clear winner emerged from the probe.** The mAP@50 delta (0.0066) and mAP@50-95 delta (0.0035) between the two architectures are very much comparable and at this data scale (3,000 images, 15 epochs), the two architectures are statistically indistinguishable on accuracy; YOLOv8m's edge is noise-level, not a demonstrated advantage. YOLOv8m is marginally faster per epoch and at inference, and has more parameters; YOLO11m is marginally slower on both counts but smaller.

**Decision: YOLO11m is selected**, on the tie-break criterion established before the probe was run: since accuracy was inconclusive, the decision falls to architectural reasoning - YOLO11's C3k2/C2PSA blocks are reported by Ultralytics to improve small-object detection, which is directly relevant given the Milestone 2 EDA's minimum normalised bbox area of 0.00002. YOLOv8m is retained as the Milestone 4 baseline comparison run (both trained under identical hyperparameters, Section 7, on the **full** training set) so that the Milestone 4 report can report an actual head-to-head result at full scale, rather than relying on this inconclusive subsample probe as the final word.

**Advantages:** attention-augmented small-object detection, lightweight box-only inference (no segmentation head to carry), fast CPU/GPU inference, mature deployment tooling (ONNX/TensorRT export if later needed).
**Disadvantages:** bounding-box area over-estimates true damaged area for irregular or elongated damage (e.g. a diagonal crack or scratch) relative to a pixel-precise mask, since the box necessarily includes undamaged background — a known bias in the Severity Agent's area-ratio proxy (Section 6.2), most pronounced for `crack`/`scratch` and smallest for roughly-rectangular classes like `shattered_glass`; like all single-stage detectors, more prone to missing small/heavily-occluded instances than two-stage detectors (Milestone 1, Section 10.8); the probe above found no measured accuracy advantage over YOLOv8m at this data scale, so the architectural rationale is a prior, not (yet) an empirical result.


### 5.2 Severity Agent: Rule-Based Proxy vs. a Learned Classifier

The calibrated bounding-box-area-ratio proxy is therefore the viable approach, not merely the preferred one. Severity is derived, not labelled: each detection's normalised bbox area is binned against fixed per-class thresholds into Minor/Moderate/Severe, following the same binning logic used in the Milestone 2 EDA (`scripts/eda_vehide.py`, `SEVERITY_BINS = [0.0, 0.02, 0.08, 1.0]`). The per-class threshold structure (rather than one global threshold) is justified by Milestone 2's EDA (Section 5.3), which found mean bbox area varies substantially by class — `shattered_glass` spans much larger areas than `flat_tyre` for comparable real-world severity, so a single global cutoff would systematically over-rate large-footprint classes and under-rate small-footprint ones. The specific threshold values are analyst-set from this bbox-area distribution, not empirically fitted against human severity judgments, since no such judgments exist in scope — this is recorded as a standing limitation in Section 14, not a temporary gap.


### 5.3 Policy Agent: MiniLM + ChromaDB + Hybrid Retrieval vs. Alternatives

The MiniLM-vs-BGE-small and ChromaDB-vs-FAISS comparisons were run empirically in Milestone 2, Section 6.2, Step 3, and are summarised rather than re-run:

| **Comparison** | **Winner** | **Margin** | **Deciding factor** |
| --- | --- | --- | --- |
| all-MiniLM-L6-v2 vs. BAAI/bge-small-en-v1.5 | MiniLM | 1.00 vs. 0.94 Precision@3 (6-query smoke test) | Smaller, equally fast, and outperformed on this corpus |
| ChromaDB vs. FAISS `IndexFlatIP` | ChromaDB | Both exact-match on top-1 (6/6); FAISS ~50-60x faster in raw query latency | Not operationally meaningful at 185-chunk scale; ChromaDB's built-in metadata filtering and persistence won |
| Dense-only vs. hybrid dense+sparse (RRF) | Hybrid (75% dense : 25% sparse) | 0.893 → 0.913 Precision@3 on 50 realistic incidents | Fixed the one dense-only zero-hit failure with no regressions elsewhere |

**What Milestone 3 adds:** `HybridRetriever` was extended with a `doc_filter` parameter, enabling doc-scoped retrieval — dense-side via a ChromaDB `where` clause, sparse-side by masking non-matching rows before ranking. Retrieval now runs as **two scoped passes per damage class** (a coverage query and a separate exclusion query, each restricted to the claimant's selected policy) rather than a single mixed top-k call, so that an exclusion clause capping or voiding coverage is not buried beneath the coverage clause it qualifies. Re-running the original 50-incident evaluation after this change reproduced the same 0.913 Precision@3 / 0.977 MRR exactly, confirming the unscoped retrieval path is unaffected — this closes the integration item flagged in Milestone 2, Section 13.4.

**Advantages of this stack:** near-zero marginal inference cost (no GPU required for retrieval), fully offline/on-CPU, transparent (each retrieved chunk carries its source document and heading for citation in the report), and doc-scoping prevents cross-policy clause leakage into a single claim's context.
**Disadvantages:** MiniLM is a general-purpose encoder with no insurance-domain fine-tuning, so retrieval quality is capped by its ability to bridge damage-class vocabulary and policy-clause vocabulary — the 0.893→0.913 (not 1.00) Precision@3 on realistic incidents reflects this ceiling; two scoped passes per damage class also roughly doubles retrieval calls per claim relative to a single mixed query, though this remains sub-second in aggregate (Section 12) given the corpus size.

### 5.4 Report Agent: Open-Weight Models via Groq vs. Alternatives

| **Model** | **Advantages** | **Disadvantages** |
| --- | --- | --- |
| **`llama-3.3-70b-versatile` (selected, compared)** | Free-tier via Groq (no per-token cost); fast (~0.7-0.9s per report); OpenAI-compatible API (low migration cost to another provider later); dense 70B general reasoning; reached a full 1.0 composite score on the mechanical faithfulness eval (Section 10) | Remote dependency (network, rate limits) — the same class of risk as any hosted API |
| **`openai/gpt-oss-20b` (selected, compared)** | Free-tier via Groq; open-weight MoE, ~3.5x smaller than the model above yet matched it exactly (1.0 composite) on the same faithfulness eval; running two differently-sized models head-to-head is what surfaced that report correctness is gated by retrieval/context quality rather than model choice (Section 10.4) | Same remote-dependency risk as above; smaller model, so headroom on harder reasoning cases is untested at this milestone |
| Paid frontier API (e.g. GPT-4o, named in Milestone 1) | Possibly stronger instruction-following/grounding on harder cases (not tested against the two selected models here) | Per-token cost scales with evaluation volume (Milestone 1, Section 10.7); external dependency no more reliable than Groq's; the empirical finding that context quality — not model choice — gates correctness (Section 10.4) weakens the case for a cost premium at this corpus/task scale |
| Self-hosted open-source LLM (own GPU) | No per-call API cost; full control over weights | Requires GPU hosting incompatible with the CPU-basic HF Spaces deployment target; Groq's hosted API already provides open-weight models at no per-call cost, so self-hosting adds infrastructure burden without an offsetting benefit |
| A single end-to-end VLM report generator | One inference call instead of a 4-stage pipeline | Reintroduces exactly the black-box evaluability problem the modular architecture was chosen to avoid (Milestone 1, Section 3.4); no separately-scoreable detection or retrieval step |

Two open-weight models — `llama-3.3-70b-versatile` and `openai/gpt-oss-20b` — are run and compared head-to-head via the Groq API, rather than one being designated primary with the other as fallback. Paid frontier APIs (GPT-4o, named in Milestone 1) are explicitly rejected for this milestone, both on cost grounds and on the empirical finding (Section 10.4) that model choice is a second-order factor here: a stronger, more expensive model cannot recover a policy clause that retrieval never surfaced, and the two Groq models converged on identical, correct verdicts once a context-quality bug was fixed (Section 10.3) regardless of which model was used. Self-hosting is separately rejected on the deployment-target constraint (GPU hosting incompatible with the CPU-basic HF Spaces target) — this milestone closes a gap the Milestone 1 report did not fully spell out, by choosing a hosted-API path that gets open-weight models' cost and transparency benefits without the self-hosting burden.


### 5.5 Computational Considerations and Expected Performance

Expected performance against each stage's Milestone 1, Section 4 target is not yet demonstrated at full scale — the **full baseline training run** (50 epochs, full training set) is Milestone 4. This milestone does include real training: the architecture-comparison probe (Section 5.1) trained both YOLO11m and YOLOv8m on a 3,000-image subsample up to 15 epochs, but that probe was explicitly sized to compare architectures relatively, not to reach production-level accuracy, and its measured mAP values (≈0.03-0.04) are far below the Milestone 1 targets by design, not by concern. This milestone's contribution toward the full targets is: confirming the selected models are computationally compatible with the stated hardware (Section 12), confirming the RAG side already exceeds its retrieval target empirically (Precision@3, Section 5.3), and confirming the report-generation side already passes its faithfulness checks on synthetic claims (Section 10) — leaving the vision side's full-scale accuracy as the one target still awaiting Milestone 4.

### 5.6 Suitability for the Dataset and Problem

- YOLO11m's bounding-box output directly produces the `area_ratio` (normalised box area) the Severity Agent's proxy depends on (Section 6.2) — the Severity Agent consumes box area, not a segmentation mask, so plain detection is a direct architectural fit, not a reduced substitute.
- MiniLM + hybrid retrieval is well matched to a small (185-chunk), short-document corpus where a heavier/larger retriever would add latency without a proportional recall gain (Milestone 2, Section 6.2, Step 3).
- `llama-3.3-70b-versatile` and `openai/gpt-oss-20b`'s structured-output adherence and grounding behaviour — both scored a full 1.0 composite on the mechanical faithfulness eval (Section 10.2), including zero citation-validity and zero currency-violation failures — are well matched to a task whose failure mode of concern is hallucinated coverage, not creative-writing quality; the eval also showed model choice is second-order to context quality (Section 10.4), so neither model's specific capability ceiling is the binding constraint on report correctness.
---

## 6. Model Inputs and Outputs

### 6.1 Damage Agent

| | |
| --- | --- |
| **Input** | RGB image, letterboxed to 1280×1280×3, pixel values normalised to [0,1], NCHW tensor `[1, 3, 1280, 1280]` |
| **Output** | Per instance: class id (0-5), normalised `[x_center, y_center, w, h]` bounding box, objectness/class confidence |
| **Preprocessing** | Letterbox resize (grey pad, fill=114) preserving aspect ratio (Milestone 2, Section 6.1, Step 4); no colour-space conversion beyond standard RGB |
| **Postprocessing** | NMS (IoU threshold 0.45, default), confidence threshold filter, box coordinates rescaled to original resolution for display |

### 6.2 Severity Agent

| | |
| --- | --- |
| **Input features** | `class_id`, normalised bbox area (`w * h`) |
| **Output** | Categorical label: Minor / Moderate / Severe |
| **Feature representation** | A single scalar (area ratio) per instance, thresholded per class (Section 5.2) |

### 6.3 Policy Agent

| | |
| --- | --- |
| **Input** | Per detected damage class: a **coverage query and a separate exclusion query**, both scoped to the claimant-selected policy's `doc_id` (Section 5.3, Section 8.1/8.3) — e.g. coverage query *"Coverage for dent damage"*, exclusion query *"Exclusions or conditions for dent damage"* — not a single combined query across all detected classes |
| **Tokenization / embedding** | MiniLM's WordPiece tokenizer, max sequence length 256 tokens (well above the ~55-65 token mean chunk length, Milestone 2 Section 6.2 Step 2, so truncation is not a practical concern); mean-pooled to a single 384-dim dense vector |
| **Sparse representation** | TF-IDF (`sklearn.TfidfVectorizer`) term vector over the same corpus vocabulary, fit once at load time |
| **Output** | Per damage class: up to 5 `coverage`/`definition`-tagged chunks and up to 5 `exclusion`/`sub_limit`/`condition`-tagged chunks (Section 5.3), each `{chunk_id, text (with heading breadcrumb), doc_id, heading, clause_type, score}`, plus a `coverage_clause_found` boolean; chunks scoring below a `MIN_CLAUSE_SCORE` floor (0.01) are dropped |

### 6.4 Report Agent

| | |
| --- | --- |
| **Input** | Full context bundle JSON: `claim_id`, `incident_narrative`, `detections[]`, policy selection (`doc_id`, `selection_method`, insurer/product metadata), per-class `clauses` (coverage/exclusion arrays from Section 6.3), and `escalation` flags — plus the fixed system prompt |
| **Output** | A structured **JSON** object (`response_format=json_object`): `{claim_id, policy_doc_id, items: [{damage_class, verdict, rationale, cited_chunk_ids}], overall_recommendation, escalate_to_human, escalation_reason}`, with `verdict` drawn from a controlled vocabulary (`covered`/`excluded`/`conditional`/`needs_review`) — rendered into Markdown/UI display downstream (Section 3.6), not generated as Markdown directly |
| **Token budget** | System prompt + serialized detections + up to 10 retrieved chunks per damage class (coverage + exclusion, Section 6.3), scaling with the number of distinct damage classes in a claim — comfortably within `llama-3.3-70b-versatile` / `openai/gpt-oss-20b`'s context windows via the Groq API, with wide margin even for a multi-class claim |

### 6.5 Dataset Organization and Directory Structure

This section closes the Milestone 3 additional-rubric requirement to show the directory layout the models above actually read from — the splits and paths below were established in Milestone 2 and are reproduced here so this report is self-contained for implementation.

```
data/
├── vehide_raw/                        # Original VehiDE download, pre-deduplication (Milestone 2 §4)
│   ├── images/                        # 13,655 deduplicated source images
│   └── annotations/                   # Original per-image annotation files
│
├── vehide_processed/
│   └── damage.yaml                    # YOLO 6-class detection config (nc: 6)
│
├── vehide/                            # Training-ready, letterboxed 1280×1280 JPEGs + YOLO-format labels
│   ├── images/
│   │   ├── train/                     # 9,558 images  (70%, stratified on dominant damage class)
│   │   ├── val/                       # 2,048 images  (15%)
│   │   └── test/                      # 2,049 images  (15%)
│   ├── labels/
│   │   ├── train/                     # 9,558 .txt   — normalised [cls, x, y, w, h] per instance
│   │   ├── val/                       # 2,048 .txt
│   │   └── test/                      # 2,049 .txt
│   └── escalation_test/               # ~100 images set aside for the low-confidence escalation
│                                       #   test subset (selected post-training, Section 16.3)
│
├── splits/                            # train.txt / val.txt / test.txt — plain-text image path lists
│                                       #   used to reproduce the split deterministically
│
├── policies/                          # 5 synthetic policy source PDFs (Milestone 2 §6.1)
├── chroma_db/                         # Persistent ChromaDB collection — 185 embedded chunks
│
├── eval/
│   ├── incident_descriptions.json     # 50 synthetic realistic incident narratives
│   └── retrieval_smoke_test.json      # 6-query, one-per-class smoke-test results
├── rag_outputs/eval/
│   └── incident_retrieval_eval.json   # 50-incident Precision@3 / MRR evaluation output
│
└── review_queue.jsonl                 # Append-only human-review escalation log (production, §3.5)
```

**Raw vs. processed separation.** `vehide_raw/` is never mutated in place; `scripts/preprocess_images.py` (Milestone 2) reads from it and writes the deduplicated, letterboxed, split, and re-annotated result into `vehide_processed/`, so the raw source can always be reproduced from or re-run against without destructive edits.

**Split leakage guarantee.** The 70/15/15 stratified split was verified to have zero cross-split filename-stem or MD5-hash duplicates (Milestone 2, Section 9.4), so the directory boundaries above are also the leakage boundary, not just a filing convention.

**Alignment with model input format.** `vehide_processed/images/{train,val,test}/` already stores images at the exact 1280×1280 resolution the Damage Agent consumes (Section 6.1) — the only transform left at inference time is the pixel normalisation to `[0,1]` and NCHW tensor packing, since letterboxing was performed once during preprocessing rather than repeated per training epoch.

---

## 7. Training Strategy

Only the **Damage Agent (YOLO11m)** is trained in this project; the Severity Agent is rule-based, the Policy Agent's embedding model is used frozen, and the Report Agent's LLMs are accessed via API with no fine-tuning (Milestone 1, Section 1.3 places custom LLM/embedding training out of scope). The strategy below therefore applies to YOLO11m only, and reflects what was **actually run** in the Milestone 3 architecture-probe training (Section 5.1), not an unvalidated plan.

| **Aspect** | **Decision** |
| --- | --- |
| Fine-tuning vs. feature extraction | Full fine-tuning (all layers trainable) from an Objects365/COCO-pretrained checkpoint — not frozen-backbone feature extraction, because the domain shift from COCO's everyday-object distribution to close-up vehicle-damage textures (scratches, cracks) is large enough that a frozen backbone would likely under-fit the domain-specific texture cues |
| Transfer learning approach | Initialise from Ultralytics' official `yolo11m.pt` pretrained weights |
| Frozen vs. trainable layers | All layers trainable; a frozen-first-10-layers ablation is planned as a secondary comparison run only if the full fine-tune shows signs of overfitting on the minority classes |
| Loss functions | YOLO11's composite detection loss: CIoU loss (box regression) + BCE (classification, gain `cls`) + DFL (distribution focal loss for box refinement) — no mask loss, since the plain detection head is used (Section 4.1) |
| Optimizer | AdamW |
| Learning rate strategy | `lr0 = 0.001`, **linear** decay to `lrf = 0.01` of the initial rate (`cos_lr=False`, the Ultralytics default) — cosine decay is a separate, not-yet-run experiment variant planned for Milestone 4 (`scripts/train_yolo.py`, `cosine_lr` preset) |
| Batch size | 4, as actually run in the Milestone 3 probe (Section 5.1) on a 14,912 MiB T4 at 1280px, `batch=8` has not yet been tested and may be attempted at full-scale Milestone 4 training if a larger-VRAM GPU is available |
| Epochs | 50 for the full Milestone 4 baseline run (Milestone 1 estimate, unchanged), distinct from the 3/5/15-epoch architecture-comparison probes already run in Milestone 3 (Section 5.1), which were deliberately short and are not the baseline |
| Early stopping | Patience of 15 epochs on validation fitness with no improvement (matches `scripts/train_yolo.py`'s baseline preset) |
| Checkpointing | `best.pt` saved on best validation fitness (Ultralytics' default weighted combination of mAP@50 and mAP@50-95, not pure mAP@50); `last.pt` saved every epoch for resumability given free-tier GPU session limits |

**Class-weighted loss — current status.** Milestone 2, Section 8.2 designed per-class inverse-frequency weights (`scratch`=1.0 up to `shattered_glass`=6.6) to address the 6.68:1 imbalance. In the Milestone 3 probe runs actually executed, only a **uniform** class-loss gain (`cls=2.0`, applied equally across all 6 classes) was used; Ultralytics' per-class weighting mechanism (`cls_pw`) was left at its default (0.0, unused) in every run. The per-class weight vector is therefore not yet wired into training as originally planned doing so would require either a custom loss modification or a class-weighted sampler, neither implemented yet. This is tracked as an open item for Milestone 4, alongside the `cls_weight` experiment preset (`cls=3.0`, still a uniform gain, not per-class) already defined in `scripts/train_yolo.py` as an interim step.

---

## 8. Model Pipeline

### 8.1 Data Flow Into the Model (Production Path)

```
Raw upload (arbitrary resolution JPEG/PNG)
        │
        ▼
Letterbox resize → 1280×1280×3, pad=114     
        │
        ▼
Normalise to [0,1], NCHW tensor
        │
        ▼
YOLO11m forward pass
        │
        ▼
NMS + confidence filter (confidence ≥ escalation threshold check happens here)
        │
        ▼
Per-instance: class_id, class_name, confidence, bbox_normalized
        │
        ▼
Severity Agent: area_ratio = w*h → per-class threshold lookup → severity label
        │
        ▼
Stage 1 — Policy selection: claimant selects policy from catalog (explicit
input, not inferred from damage) → validated doc_id
        │
        ▼
Stage 2 — Policy Agent: per detected class, two doc-scoped queries
(coverage + exclusion) → MiniLM embed + hybrid retrieval → up to 5 chunks
each, filtered by MIN_CLAUSE_SCORE 
        │
        ▼
Report Agent: context bundle JSON (detections, severities, policy, clauses,
incident narrative) → llama-3.3-70b-versatile / openai/gpt-oss-20b (Groq)
→ structured JSON output 
        │
        ▼
Final rendered report
```

### 8.2 Small-Scale Pipeline Verification (10-Payload Faithfulness Evaluation)

The Policy and Report Agent stages were verified against 10 deliberately contrastive claim scenarios, each run through the actual retrieval and generation stack.

| **Claim** | **Damage** | **Selected policy** | **Stress-tests** |
| --- | --- | --- | --- |
| 01 | dent (minor) | policy_1 | Clean baseline |
| 02 | glass (severe) | policy_4 | PDF table-row garbling (see below) |
| 03 | flat_tyre alone | policy_5 | Tyre coverage conditional on concurrent damage |
| 04 | dent + crack + lamp | policy_3 | Multi-class, dense-exclusion policy |
| 05 | scratch (vandalism) | policy_2 | Vandalism wording vs. malicious-act clause |
| 06 | dent (confidence 0.35) | policy_1 | Escalation path |
| 07 | glass (window) | policy_2 | Hybrid-retrieval fix case, end-to-end |
| 08 | crack + lamp | policy_4 | Scope-based exclusion reasoning |
| 09 | dent ×3 (hail) | policy_1 | Multi-instance; the `chunk_00004` fix below |
| 10 | flat_tyre + scratch | policy_5 | Mixed severity, multi-class |

Each payload was run through both selected Report Agent models (`llama-3.3-70b-versatile` and `openai/gpt-oss-20b`, Section 5.4), producing 20 reports total, scored by a mechanical (not LLM-judged) faithfulness evaluator (`scripts/eval_report_agent.py`) so every result is reproducible from the JSON output alone. The evaluator checks: `schema_valid`, `class_coverage_complete` (every detected class received a verdict), `citation_validity` (every cited chunk was actually offered to the model), `verdict_evidence_consistent` — a hard check that a `covered` verdict must cite at least one coverage-type chunk — `escalation_consistent`, and `currency_violation` (a ₹/rupee figure appearing in the output not present in any offered clause). Two additional soft flags are surfaced for manual review but not scored.

Both models scored a full **1.0 composite** across all 10 payloads on every hard check — fully grounded, zero fabricated citations, zero currency violations.

**Two data-quality issues were found through this evaluation:**

- **Found and fixed** — `chunk_00004` (policy_1's umbrella coverage grant) was mistagged by the Milestone 2 auto-tagger's bare `\bmeans\b` keyword firing on "external **means**," causing a coverage-only clause filter to drop it. Before the fix, the two models disagreed on claim 09 (hail dents): one verdict was `covered`, the other `excluded`, both citing a substitute chunk because the real coverage clause had been filtered out. After correcting the filter to also accept `definition`-tagged chunks, both models converged on the same correct verdict, citing `chunk_00004`.
- **Found, not yet fixed** — in policy_4, `pdfplumber` linearised a coverage table such that a glass row's value and a tyre row's condition merged into one chunk (`chunk_00122`); both models then read the tyre condition as if it applied to glass (claim 02). This is citation-valid but semantically wrong, and is only partially detectable via a soft flag (36 of 185 corpus chunks carry more than one damage-class tag). The recommended fix is to re-extract the affected table pages with `pdfplumber.extract_tables()` rather than plain text extraction.

**Headline finding:** the claim-09 episode is the clearest evidence that context quality, not model choice, gates report correctness — given the same flawed context, two different models produced opposite confident verdicts; given the same corrected context, both converged on the same correct answer, with no change to prompt, temperature, or model. This supports the Report Agent model selection in Section 5.4: retrieval and chunking quality matter more than model size or cost, since a stronger model cannot recover a clause that was never retrieved.

**Scope of this verification:** the detections feeding these 10 payloads are hand-constructed per scenario (matching the confirmed detection schema, Section 6.1), not live Damage Agent inference output — no trained YOLO checkpoint exists yet to produce them (Milestone 4). The Severity Agent's per-class thresholds are applied within each payload's construction, not exercised as a standalone live stage. A full end-to-end run — real image in, real YOLO inference, through to a rendered report — has not yet been performed; this is planned work (Section 16.3), gated on the Milestone 4 baseline training run producing real detection output.

**Escalation threshold used in this evaluation:** claim 06 tests the escalation path at a detection confidence of 0.35, below the escalation threshold of 0.50 used throughout this pipeline (Section 3.1), pending calibration against real detection-confidence data from the Milestone 4 training run.

### 8.3 Post-processing and Final Prediction Generation

The final artefact returned to the user is a composite structured object: annotated image (bounding boxes drawn), a severity table, a ranked clause list, and a structured report (Section 6.4). A tabbed rendering of this output via Gradio is planned (Section 3.6) but not yet built (Section 15).

### 8.4 Evaluation Metrics

No model has been trained yet — that is Milestone 4 — so the table below consolidates the metrics and targets each component will be scored against (carried forward from Milestone 1, Section 4.1) with what is already empirically known from Milestone 2 and this milestone's evaluation, rather than reporting new results.

| **Component** | **Metric** | **Target** | **Status at end of Milestone 3** |
| --- | --- | --- | --- |
| Damage Agent | mAP@50 | ≥ 0.70 | Not yet measured at full scale — requires the Milestone 4 baseline training run; the Milestone 3 architecture probe (Section 5.1) measured 0.0305-0.0371 on a 15-epoch, 3,000-image subsample, well below target by design |
| Damage Agent | mAP@50-95 | ≥ 0.50 | Not yet measured at full scale |
| Damage Agent | Per-class F1 (all 6 classes) | ≥ 0.65 | Not yet measured; `shattered_glass`/`flat_tyre` flagged as most at risk given the 6.68:1 class imbalance (Section 14) |
| Policy Agent (retrieval) | Precision@3 | ≥ 0.80 | **Already exceeded**: 0.893 dense-only, 0.913 hybrid, measured empirically in Milestone 2, Section 6.2 Step 6; reproduced exactly after the Milestone 3 doc-scoping extension (Section 5.3) |
| Policy Agent (retrieval) | Mean Reciprocal Rank | Not separately targeted in Milestone 1 | 0.977 hybrid (Milestone 2/3) |
| Report Agent | Faithfulness (mechanical evaluation, Section 8.2) | Full grounding, zero fabricated citations | **Already achieved**: 1.0 composite score, both models, all 10 payloads (Section 8.2) |
| Report Agent | Human evaluation (Accuracy / Faithfulness / Clarity, 1-5 scale) | Mean ≥ 4.0 | Not yet measured; rubric and inter-rater protocol already defined (Milestone 1, Section 4.3) |
| Full pipeline | Qualitative wiring correctness, all four agent boundaries, with live detection input | N/A — pass/fail | Not yet performed; the Policy/Report Agent side is verified on synthetic detections (Section 8.2), but a run using live Damage Agent output has not yet occurred (Milestone 4) |

**Loss functions used to measure training performance** are specified in full in Section 7 (YOLO11's composite CIoU + BCE + DFL); the retrieval and report-generation components are not trained in this project (Section 7 preamble) and so have no associated loss function — they are scored purely against the evaluation metrics above.

### 8.5 Example Model Outputs

No trained Damage Agent checkpoint exists yet (Milestone 4), so the Damage Agent and Severity Agent examples below show the confirmed output schema (Section 6.1/6.2) with representative values rather than a live inference result. The Policy and Report Agent examples are drawn from the real evaluation in Section 8.2.

**Damage Agent — raw output for one image (illustrative):**

```json
{
  "image_id": "claim_0001.jpg",
  "detections": [
    {"class_id": 0, "class_name": "dent",    "confidence": 0.91, "bbox_normalized": [0.42, 0.55, 0.11, 0.08], "area_ratio": 0.0088},
    {"class_id": 1, "class_name": "scratch", "confidence": 0.78, "bbox_normalized": [0.61, 0.30, 0.06, 0.03], "area_ratio": 0.0018}
  ]
}
```

**Severity Agent — same instances after severity assignment:**

```json
[
  {"class_id": 0, "class_name": "dent",    "confidence": 0.91, "bbox_normalized": [0.42, 0.55, 0.11, 0.08], "area_ratio": 0.0088, "severity": "minor"},
  {"class_id": 1, "class_name": "scratch", "confidence": 0.78, "bbox_normalized": [0.61, 0.30, 0.06, 0.03], "area_ratio": 0.0018, "severity": "minor"}
]
```

**Policy Agent — real retrieval output, claim 09 (Section 8.2), before the `chunk_00004` fix:** the coverage-pass query for `dent` surfaced a substitute chunk rather than the correct umbrella coverage clause, causing the two models to disagree on the claim's verdict. After the fix, `chunk_00004` ranks first for the same query.

**Report Agent — final rendered output:** the two Report Agent models' outputs on claim 09, before and after the `chunk_00004` fix, are the concrete illustration of both the target output format and the headline finding of Section 8.2; they are not reproduced in full here.

---

## 9. Retrieval and Knowledge Components

| **Component** | **Selection** | **Detail** |
| --- | --- | --- |
| Embedding model | `all-MiniLM-L6-v2` | 384-dim, frozen (Section 4.2) |
| Vector database | ChromaDB (persistent client) | HNSW cosine-similarity index, 185 chunks, metadata-filterable by `doc_id` (Milestone 2; Section 5.3) |
| Similarity search | Cosine similarity (dense) + TF-IDF (sparse), fused via weighted Reciprocal Rank Fusion | 75% dense : 25% sparse weighting (Milestone 2, Section 6.2 Step 6) |
| Chunking strategy | Structure-aware: heading/list-item-boundary splitting, then `RecursiveCharacterTextSplitter` (300 char / 40 overlap), heading breadcrumb prepended to each chunk before embedding | Milestone 2, Section 6.2, Steps 1-2 |
| RAG workflow | Claimant selects a policy (Section 5.3/8.1), resolving a `doc_id` → for each detected damage class, a coverage query and a separate exclusion query, both scoped to `doc_id` → dense + sparse retrieval in parallel → RRF fusion → up to 5 chunks per query, filtered by a minimum score floor → passed to the Report Agent as grounding context | Section 6.3 |
| Re-ranking | RRF fusion score itself acts as the re-ranking step; no separate cross-encoder re-ranker is used, judged unnecessary at 185-chunk corpus scale (added latency not justified by the Milestone 2 evaluation) | — |

`HybridRetriever` was extended with a `doc_filter` parameter in Milestone 3 to support the doc-scoped, two-pass retrieval described above (Section 5.3); this closes the integration item flagged in Milestone 2, Section 13.4.

---

## 10. Prompt Engineering

### 10.1 Prompt Templates

The Report Agent uses a fixed system prompt and a per-request user message built by serializing the context bundle (Section 6.4) to JSON — no manual prose is authored per request. The template shown in Appendix B is an earlier draft; see the note there for its confirmation status.

### 10.2 System Prompt

See Appendix B.1. Its intended constraints are: (a) ground every coverage statement in a specific retrieved clause id, (b) explicitly state non-coverage rather than infer it, (c) always append the fixed disclaimer sentence.

### 10.3 Zero-shot vs. Few-shot Strategy

Zero-shot with a structured-output schema is used rather than few-shot exemplars: the task (populate a fixed schema from provided JSON) is sufficiently constrained by the schema itself that few-shot examples were judged unlikely to add reliability proportional to their token cost. This is supported by the Section 8.2 evaluation: both models reached a full 1.0 faithfulness composite zero-shot on all 10 payloads.

### 10.4 Structured Output Format

The Report Agent requests a JSON object with a controlled verdict vocabulary (`covered`/`excluded`/`conditional`/`needs_review`) and mandatory clause-id citations per damage class, which is then rendered into a report view for the user — separating the model's structured claim from its prose framing makes coverage claims independently checkable against the retrieved clause ids, and is what the Section 8.2 mechanical evaluator checks directly. The exact schema in Appendix B.3 is an earlier draft; see the note there for its confirmation status.

### 10.5 Hallucination Mitigation

- Explicit instruction to write a `needs_review` or non-coverage verdict rather than infer coverage from general knowledge.
- Every verdict must cite at least one retrieved clause id; the Section 8.2 evaluator's `citation_validity` and `verdict_evidence_consistent` checks verify this mechanically, not just via prompt instruction.
- A currency-violation check (Section 8.2) independently catches any monetary figure not present in the retrieved clauses, since no policy-schedule or repair-cost data exists to ground such a figure.
- The Section 8.2 evaluation demonstrates this behaviour concretely: both models scored zero currency violations and zero invalid citations across all 10 payloads and 20 reports.

### 10.6 Guardrails

- Mandatory disclaimer appended to every report (enforced in code, not only via prompt instruction, so a prompt-following failure cannot remove it) — **[TO CONFIRM WITH RAG OWNER]** exact enforcement point.
- Escalation gate (Section 3.1) prevents the Report Agent from running on low-confidence detections.
- Two models are run and compared (Section 5.4) rather than one primary model with a fallback; this is a comparison design for evaluation purposes, not a runtime fallback mechanism, so a single-model production configuration is still to be decided.

### 10.7 Function Calling / Tool Use

The current implementation does not use LLM-driven function calling: the context bundle is fully assembled before the Report Agent is called, so each Report Agent invocation is a single constrained generation call, not an agent loop. This keeps the guardrails simple to verify (Section 10.5) and keeps the pipeline's control flow deterministic, consistent with the "separation of concerns" argument in Milestone 1, Section 3.4. Exposing the Policy Agent as a callable tool for an LLM-driven agent loop is a possible future direction, not part of the current design.

---

## 11. System Integration

### 11.1 Shared State Schema

```python
from typing import TypedDict, Optional, List, Dict, Any

class Detection(TypedDict):
    class_id: int
    class_name: str
    confidence: float
    bbox_normalized: list[float]   # [x, y, w, h]
    area_ratio: float
    severity: Optional[str]  # populated by Severity Agent

class RetrievedClause(TypedDict):
    chunk_id: str
    doc_id: str
    heading: str
    text: str
    clause_type: str
    score: float

class PolicySelection(TypedDict):
    doc_id: str
    selection_method: str  # "claimant_selected" — see Section 8.1
    insurer: str
    product: str
    description: str

class ClaimState(TypedDict):
    claim_id: str
    image: bytes
    incident_narrative: Optional[str]
    detections: List[Detection]
    overall_severity: Optional[str]
    policy: Optional[PolicySelection]
    clauses: Dict[str, Dict[str, Any]]  # per damage class: coverage[] / exclusion_or_condition[]
    report: Optional[dict]
    escalated: bool
    escalation_reason: Optional[str]
```

**[TO CONFIRM WITH RAG OWNER]** — this schema is aligned to the confirmed detection format (Section 6.1) and the confirmed context bundle structure (Section 6.4), but has not been checked field-by-field against the actual `report_context.py` implementation.

### 11.2 How Different Models Communicate

All communication is state-in/state-out through the schema above; no module calls another module's model directly. Each stage is currently invoked as a plain Python function call (Section 4.5, Section 2.5) rather than through a tool-calling interface — there is no MCP or similar tool wrapper around the Policy Agent in the current implementation.

### 11.3 Orchestration Framework (Planned)

No orchestration framework is implemented yet. The current pipeline is a fixed sequence of plain Python function calls (Section 2.5, Section 4.5); a stage's output is checked by simple conditional logic in the calling code to decide whether the next stage runs (e.g. the escalation gate skips the Policy/Report Agent calls entirely). Wrapping the existing stage functions as LangGraph nodes is planned, since each stage already has a clean, schema-defined input/output boundary (Section 11.1):

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
    lambda s: "escalate" if min((d["confidence"] for d in s["detections"]), default=0) < 0.50 else "severity_agent",
)
graph.add_edge("severity_agent", "policy_agent")
graph.add_edge("policy_agent", "report_agent")
graph.add_edge("report_agent", END)
graph.add_edge("escalate", END)

app = graph.compile()
```

### 11.4 Database Interactions

ChromaDB is queried read-only at inference time (`collection.query(...)`); no write path exists in the production request cycle except the one-time index build during preprocessing (Milestone 2).

---

## 12. Computational Requirements

| **Phase** | **Hardware** | **Memory** | **Notes** |
| --- | --- | --- | --- |
| YOLO11m fine-tuning | Single NVIDIA T4 (Colab Pro / Kaggle, 16GB VRAM) | ~8.5-9.4GB VRAM at batch=4, imgsz=1280, as measured in the Milestone 3 probe (Section 5.1) | Milestone 3 probe: ~1,371s/epoch at 15 epochs on a 3,000-image subsample; the full 50-epoch Milestone 4 baseline run on the full training set has not yet been timed |
| Embedding + retrieval | CPU only | <200MB (MiniLM + 185-chunk ChromaDB index + TF-IDF matrix) | Cold init (MiniLM load + TF-IDF fit) ~16.5s one-time per process; warm scoped query ~10ms median |
| LLM inference | Remote API (Groq, no local compute) | N/A | ~0.7-0.9s per report round-trip (Section 8.2 evaluation) |
| Deployed demo (HF Spaces) | CPU-basic (2 vCPU, 16GB RAM) | YOLO11m CPU inference + MiniLM CPU inference, both feasible at this scale | UI not yet built (Section 8.3, Section 15) |

**Expected inference latency (per claim, once deployed):**

| **Stage** | **Latency** |
| --- | --- |
| Image letterbox + YOLO11m CPU inference (1280px) | ~150-400ms (estimated; not yet measured on CPU) |
| Severity Agent (pure arithmetic) | <5ms |
| Policy Agent (per damage class, coverage + exclusion queries) | ~10-20ms per class (measured, Section 8.2 context) |
| Report Agent (Groq API round-trip) | ~0.7-0.9s (measured, Section 8.2) |
| **Total (non-escalated, single-class claim)** | **~1-1.5s** |

**Storage requirements:** VehiDE processed dataset (13,655 images at 1280×1280 JPEG) — several GB, not stored in the Git repository itself; ChromaDB index and synthetic policy PDFs — a few MB; trained YOLO11m checkpoint — ~40.7MB (measured, Section 5.1 probe; the full-training-run checkpoint size is expected to be comparable).

---

## 13. Design Decisions and Trade-offs

| **Decision point** | **Chosen** | **Rejected alternative(s)** | **Reasoning** |
| --- | --- | --- | --- |
| Detector family | YOLO11 (plain detection) | Faster R-CNN, DETR, SSD, single VLM | Speed/accuracy/deployability trade-off (Milestone 1, §3.1/3.4) |
| Detector scale | `m` | `n`/`s` (faster, less accurate), `l`/`x` (too slow for CPU deployment) | Balances accuracy against the CPU-basic HF Spaces inference target |
| Input resolution | 1280px | 640px (Ultralytics default) | Matches the dataset's actual weighted-mean resolution (Milestone 2, §5.5); costs ~4x VRAM, reducing batch size from an assumed 16 to a measured 4 (Section 7) |
| Vector store | ChromaDB | FAISS | FAISS ~50-60x faster in raw query latency but not operationally meaningful at 185-chunk scale; ChromaDB's metadata filtering (needed for doc-scoping, Section 5.3) and persistence wins |
| Retrieval strategy | Hybrid dense+sparse (75:25), doc-scoped two-pass | Dense-only; single mixed top-k query | Hybrid fixed the one zero-hit failure on the 50-incident evaluation with no regressions; two-pass avoids an exclusion clause being buried beneath the coverage clause it qualifies (Section 5.3) |
| Report generation | Modular RAG + prompted LLM | Single end-to-end VLM | Preserves per-stage evaluability (Milestone 1, §3.4) at the cost of more integration surface area |
| LLM provider | Open-weight models (`llama-3.3-70b-versatile`, `openai/gpt-oss-20b`) via Groq API, run and compared | Paid frontier API (GPT-4o); self-hosted open-source LLM | Free-tier, fast, OpenAI-compatible; avoids GPU hosting incompatible with the CPU-basic deployment target; empirical evidence (Section 8.2) that context quality gates correctness more than model choice weakens the case for a paid alternative |
| Severity method | Calibrated area-ratio proxy, permanently rule-based | Dedicated learned classifier, VLM-based scoring | No severity-labelled ground truth exists in the VehiDE-only dataset scope (Section 5.2), so a learned classifier has nothing to train against; VLM scoring rejected on latency/cost grounds (Milestone 1, §10.2) |

**Scalability considerations.** The current design assumes single-request, stateless processing suitable for a demo (CPU-basic HF Spaces instance). A production deployment handling concurrent claims would need: a GPU-backed inference service for the Damage Agent (CPU inference latency, while acceptable for a demo, would not scale to high claim volumes), a request queue in front of the LLM API calls to manage rate limits, and a persistent, access-controlled store for the human review queue rather than a flat JSON log.

---

## 14. Risks and Limitations

This section extends Milestone 1, Section 10 with what Milestone 2's empirical findings and this milestone's model selections add.

| **Risk / Limitation** | **Status / detail** |
| --- | --- |
| Class imbalance (6.68:1 scratch:shattered_glass) | Confirmed in Milestone 2; a uniform class-loss gain (`cls=2.0`) was applied in the Milestone 3 probe runs, but Milestone 2's per-class inverse-frequency weight vector is not yet wired in (Section 7); per-class F1 for `shattered_glass`/`flat_tyre` remains the metric most at risk of missing the ≥0.65 target |
| Severity is derived, not labelled, and permanently so | No severity ground truth exists anywhere in the VehiDE-only dataset scope (Section 5.2); the area-ratio proxy's known failure mode — a large shallow scratch vs. a small deep crack can be mis-ordered by area alone — cannot be corrected by acquiring more labelled data within this project's scope |
| Domain shift (studio-quality training photos vs. real handheld claim photos) | Addressed partially via augmentation (motion blur raised to 0.3, JPEG quality set to 75, Milestone 2 §8.1); residual risk remains until stress-tested against real claim-style photographs |
| RAG retrieval ceiling on realistic queries | 0.893-0.913 Precision@3 (Milestone 2/3), not 1.00 — a retriever-quality limit, not a wiring defect; the PDF table-garbling finding (Section 8.2) is a separate, corpus-level source of the same class of error |
| LLM hallucination on edge cases | Mitigated via explicit non-inference instructions and clause-id citation checking, verified mechanically to score zero failures on all 10 evaluated payloads (Section 8.2), but not exhaustively — residual risk remains on claim types not covered by those 10 scenarios |
| API dependency | Groq API rate limits or outages remain an external dependency outside the team's control; no caching or fallback provider is currently implemented |
| Policy selection cannot be automated from damage | Proven exhaustively across all 315 possible damage-class/policy combinations (Section 5.3/8.1) — the applicable policy is a contract fact, not a damage fact, so this is treated as a permanent input requirement, not a gap to close |
| Bias in training data | VehiDE's Southeast-Asian-vehicle-image composition may under-represent vehicle types more common in the eventual Indian deployment context (Milestone 2, §11); mitigated by planned stratified error analysis (Milestone 1, §11.1), not yet executed (a Milestone 4/evaluation activity) |
| Scalability of the CPU-basic demo | Acceptable for single-request demonstration; not representative of production concurrency (Section 13) |
| No orchestration framework yet | The pipeline runs as a fixed sequence of function calls (Section 11.3); this limits the escalation/routing logic to what plain conditional code can express, and complicates adding new branching logic later without a state-graph framework |

---

## 15. Deliverables Produced

| **Deliverable** | **File / Location** | **Description** |
| --- | --- | --- |
| High-level architecture diagram | `diagrams/multiagent_architecture_staged.svg` | Four-agent architecture (carried forward from Milestone 1/2, referenced in Section 2.1) |
| Sequence diagram | Section 3.3 (this report) | Textual sequence diagram of one claim's path through the pipeline |
| Policy catalog (Stage 1) | `scripts/policy_catalog.py` | Claimant-facing policy selection menu; each option described (Section 8.1) |
| Rejected auto-selection census | `scripts/policy_selector.py`, `data/rag_outputs/mile3/policy_selection_eval.json` | Exhaustive 315-case evaluation proving damage-based policy auto-selection is indistinguishable from chance (Section 5.3/8.1) |
| Doc-scoped clause retriever | `scripts/hybrid_retrieval.py` (`doc_filter` extension), `scripts/report_context.py` | Two-pass coverage/exclusion retrieval per damage class, scoped to the selected policy (Section 9) |
| Contrastive claim payloads | `scripts/generate_claim_payloads.py`, `data/rag_outputs/mile3/payloads_all.json` | 10 contrastive claim scenarios used for verification (Section 8.2) |
| Report Agent | `scripts/report_agent.py` | Report generation via Groq, both models |
| Faithfulness evaluator | `scripts/eval_report_agent.py`, `data/rag_outputs/mile3/faithfulness_eval.json` | Mechanical grounding/faithfulness checks and results (Section 8.2) |
| Shared state schema | Section 11.1 (this report) | `TypedDict` definitions for `Detection`, `RetrievedClause`, `PolicySelection`, `ClaimState` |
| Prompt templates | Appendix B | System prompt, user-message template, structured output schema (draft status noted) |
| Model/config comparison tables | Sections 4, 5, 12, 13 (this report) | YOLO11 vs YOLOv8, MiniLM vs BGE, ChromaDB vs FAISS, dense vs hybrid retrieval, Groq models vs alternatives |
| Dataset directory structure | Section 6.5 (this report) | Full `data/` tree — raw vs. processed separation, train/val/test split paths, leakage guarantee |
| Consolidated evaluation metrics table | Section 8.4 (this report) | Per-component metrics and Milestone 1 targets, with results already achieved flagged |
| This report | `Milestone3_Report.md` | Full documentation of architecture selection and pipeline design |

---

## 16. Summary and Next Steps

### 16.1 Summary of Architecture Decisions

The system's four agents are each assigned a specific, justified model: YOLO11m (fine-tuned, plain detection) for damage detection, a permanently rule-based proxy for severity, MiniLM + ChromaDB + doc-scoped hybrid dense/sparse retrieval for policy grounding, and two open-weight models via Groq (run and compared) for report generation. The pipeline currently runs as a fixed sequence of function calls, with an explicit human-escalation gate implemented in the calling code; wrapping this sequence as a LangGraph state machine is planned but not yet built (Section 11.3). The end-to-end workflow, state schema, error-handling paths, and prompt/guardrail design are specified in enough detail to begin implementation.

### 16.2 Readiness for Model Training (Milestone 4)

- The YOLO11m vs. YOLOv8m baseline training runs (both under identical hyperparameters, Section 7) are ready to launch against the Milestone 2 training-ready dataset (`data/vehide_processed/`, `damage.yaml`) with no further data preparation required.
- The doc-scoped hybrid retriever (Section 9) is implemented and already reproduces its Milestone 2 evaluation numbers exactly after the Milestone 3 extension.
- The Report Agent's faithfulness has already been verified on synthetic claim scenarios (Section 8.2) with a full composite score, so Milestone 4 can focus on training the vision model and on the true end-to-end run with live detections, rather than on RAG-side pipeline debugging.

### 16.3 Planned Implementation Activities

- Execute the full 50-epoch YOLO11m baseline training run (and the YOLOv8m comparison run) and report mAP@50, mAP@50-95, and per-class F1 against the Milestone 1 targets.
- Wire the Milestone 2 per-class inverse-frequency class weights into training (Section 7), replacing the uniform `cls` gain used in the Milestone 3 probe.
- Perform a true end-to-end run — real image in, real YOLO inference, through Severity, Policy, and Report Agents, to a rendered output — once a trained checkpoint exists.
- Reconcile the escalation confidence threshold (currently 0.50, a placeholder, Section 8.2) against real detection-confidence calibration data from the trained model.
- Apply the `chunk_00004` root-cause tag fix in the policy corpus, and re-extract the garbled table pages identified in policy_4 (Section 8.2).
- Build the explicit incident-to-clause ground truth (Milestone 2, §13.4) to replace the damage-class-overlap retrieval proxy with a rigorous Precision@3/MRR evaluation.
- Select and label the ~100-image escalation-path test subset (Milestone 2, §9.4) once the trained model's confidence distribution is available.
- Conduct the stratified per-class and (where metadata allows) per-vehicle-type error analysis flagged in Milestone 1, Section 11.1.
- If an orchestrator is wanted for Milestone 4, wrap the existing stage functions as LangGraph nodes (Section 11.3) — they already have clean, schema-defined input/output boundaries.

---

## Appendix A: Prompt Templates

### A.1 System Prompt (Report Agent)

```
You are a claims-assistant that writes preliminary vehicle damage assessment
reports strictly from the DETECTIONS and RETRIEVED_CLAUSES JSON provided.
For every damage instance, state whether it is covered, citing the clause id.
If no retrieved clause supports a coverage claim, write 'not covered under
retrieved policy' rather than inferring from general knowledge. Do not invent
clause text. Always end with the disclaimer: 'This is a preliminary AI-assisted
assessment and has not been verified by a licensed insurance assessor'.
```

### A.2 User Message Template

The user message is the full context bundle, serialized directly as JSON there is no separate `DETECTIONS:`/`RETRIEVED_CLAUSES:` sectioning; the entire payload is passed as-is:

```json
{
  "claim_id": "CLAIM_09_multi_instance_hail_dents",
  "incident_narrative": "Hailstorm left multiple small dents ...",
  "detections": [
    {"class_id": 0, "class_name": "dent", "confidence": 0.80,
     "bbox_normalized": [0.3, 0.4, 0.11, 0.11], "area_ratio": 0.012, "severity": "minor"}
  ],
  "policy": {
    "doc_id": "policy_1_bharat_suraksha",
    "selection_method": "claimant_selected",
    "insurer": "Bharat Suraksha Motor Insurance Co. Ltd",
    "product": "Motor Private Car Comprehensive (Own Damage + Third Party Liability)",
    "description": "Comprehensive cover ... glass at nil depreciation ... no add-ons.",
    "clauses": {
      "dent": {
        "coverage": [
          {"chunk_id": "chunk_00004", "text": "...", "heading": "...",
           "clause_type": "definition", "doc_id": "policy_1_bharat_suraksha", "score": 0.0656}
        ],
        "exclusion_or_condition": [ "..." ],
        "coverage_clause_found": true
      }
    }
  },
  "escalation": {
    "low_confidence_detections": [],
    "missing_coverage_clause_for": [],
    "needs_human_review": false
  }
}
```

No user-facing instruction text is added beyond the JSON itself — the system prompt (A.1) alone specifies the task and output schema.
> Field names in A.2 are confirmed directly from `scripts/report_agent.py`.
> The system prompt (A.1) and output schema (A.3) below are drafts and still
> need the same confirmation pass against that script.

### A.3 Structured Output Schema (Draft)

```python
from pydantic import BaseModel
from typing import Optional, Literal

class DamageFinding(BaseModel):
    damage_class: str
    severity: Literal["minor", "moderate", "severe"]
    coverage_status: Literal["covered", "excluded", "conditional", "needs_review"]
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

## Appendix B: Change Log

| **Date** | **Change** | 
| --- | --- |
| | |

---

## Appendix C: References

[1] G. Jocher et al., "YOLO by Ultralytics," Zenodo, 2023. doi:10.5281/zenodo.7347926.

[2] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459-9474, 2020.

[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019.

[4] J. Johnson, M. Douze, and H. Jégou, "Billion-Scale Similarity Search with GPUs," IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535-547, 2021.

[5] Milestone 1 Report — Multimodal Damage Assessment for Insurance Claims, Group 1, DS & AI Lab, 2026.

[6] Milestone 2 Report — Multimodal Damage Assessment for Insurance Claims, Group 1, DS & AI Lab, 2026.

---

***Declaration:***

I have read and reviewed this submission in its entirety and confirm that it accurately represents the work of our group. By entering my initials and the date below, I acknowledge my approval of this submission.

| Name | Date of Review | Sign |
|---|---|---|
| Satyajeet Kumar | 23-07-2026 | S.K. |
| Pranab Kumar Manna | 23-07-2026|Pk Manna |
| Venkata Siva Kamal Guddanti | | |
| Anuj Gautam | | |
| Harsh Pal | | |

---
