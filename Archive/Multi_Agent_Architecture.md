
---

<div align="center">

<b>***Data Science & AI Lab May 2026***</b>
<br>

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/IITM_logo.png" width="520">

<h1 style="font-size:26em;">Multimodal Damage Assessment for Insurance Claims</h1>

<h2>Open-Source Multi-Agent RAG Architecture</h2>

<h3>Group 1</h3>

<br>

  ***Prepared by:***

  
| **Name** | **Email ID** | **GitHub Profile** |
| --- | --- | --- |
| SATYAJEET KUMAR | 23f1003132@ds.study.iitm.ac.in | [HiveCase](https://github.com/HiveCase) |
| AGNESH KUNDU | 22f1000768@ds.study.iitm.ac.in | [AgneshK](https://github.com/AgneshK) |
| ANUJ GAUTAM | 21f1002407@ds.study.iitm.ac.in | [anujgautam1](https://github.com/anujgautam1) |
| PRANAB KUMAR MANNA | 22f1000887@ds.study.iitm.ac.in | [pranab92](https://github.com/pranab92) |
| VENKATA SIVA KAMAL GUDDANTI | 22f2000094@ds.study.iitm.ac.in | [22f2000094](https://github.com/22f2000094) |

</div>

---

# Table of Contents

- [1. Introduction](#1-introduction)
- [2. End-to-End Request Flow](#2-end-to-end-request-flow)
- [3. Multi-Agent Orchestration Detail](#3-multi-agent-orchestration-detail)
- [4. Open-Source Component Mapping](#4-open-source-component-mapping)
- [5. Design Notes](#5-design-notes)

---

## 1. Introduction

This document adapts a multi-agent, MCP-tool-based reference architecture (originally designed for a travel concierge system) to our vehicle damage insurance claim assessment project. Every managed/proprietary service in the reference design has been substituted with a self-hostable, open-source equivalent, since this is an academic project with no cloud budget. The architecture is presented in two parts: the end-to-end request flow (Section 2), and the internal detail of the multi-agent orchestrator (Section 3), followed by a full component mapping table (Section 4).

---

## 2. End-to-End Request Flow

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/data/images/multi_agent_architecture_overview.svg" width="700">

A user-submitted claim (photo + optional policy PDF) enters through an open-source reverse proxy, is handled by a FastAPI backend backed by PostgreSQL for session/state storage, and is then handed to the agent orchestrator. The orchestrator calls out to MCP-exposed tools (damage detection, PDF parsing, VIN lookup) on one side and a memory store (Redis for short-term session state, Chroma for long-term/vector memory) on the other. All agent activity is traced through an AgentOps layer before any LLM call is routed through a local LLM gateway to self-hosted open-source models.

---

## 3. Multi-Agent Orchestration Detail

<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/Archive/insurance_claim_agent_architecture_overview.png" width="700">

The orchestrator (built on LangGraph) routes each claim to four specialist agents, each wrapping one stage of the pipeline defined in the project's system architecture: a damage detection agent (YOLOv8), a severity agent (bounding-box area-ratio scoring), a policy retrieval agent (RAG over the synthetic policy corpus), and a report generation agent (LLM-based report writing). Low-confidence outputs from any agent can be escalated by the orchestrator to a human review queue rather than being auto-finalized, consistent with the project's scope decision that the system produces a preliminary report only.
<img src="https://github.com/HiveCase/Group-1-DS-and-AI-Lab-Project/blob/main/Archive/insurance_claim_multiagent_detail.png" width="700">

---

## 4. Open-Source Component Mapping

| **Layer** | **Proprietary reference** | **Open-source substitute** |
| --- | --- | --- |
| Load balancer | AWS Load Balancer | Nginx or Traefik |
| Application runtime | AWS ECS + DynamoDB | FastAPI (or Gradio for the demo UI) + PostgreSQL |
| Agent framework | Bedrock AgentCore, Strands Agents | LangGraph or CrewAI |
| External platform APIs | Amadeus, OpenStreetMap, OpenMeteo | Fine-tuned YOLOv8 service, PDF parser/chunker, open VIN-decoder API |
| Tool exposure | MCP Server | FastMCP |
| Memory | Bedrock AgentCore Memory, S3 | Redis (short-term) + ChromaDB or FAISS (long-term/vector) |
| AgentOps | Langfuse + OpenTelemetry | Unchanged - already open source |
| LLM load balancing/caching | LiteLLM | Unchanged - already open source |
| LLMs | Azure OpenAI, Bedrock, Vertex | Self-hosted via Ollama or vLLM - Llama 3, Mistral, or Qwen2.5 |
| Governance | AWS IAM, KMS, SIEM | Keycloak (identity), HashiCorp Vault (secrets), Wazuh (SIEM), Prometheus + Grafana (monitoring) |
| Deployment | AWS ECS, Kubernetes | Docker Compose (dev) / k3s (single-node Kubernetes) |

---

## 5. Design Notes

- **Why split detection and severity into separate agents**: this keeps each agent independently testable against its own metric (mAP for detection, severity-agreement rate for scoring), consistent with the modular-over-monolithic rationale already established for this project.
- **Why human review sits beside, not inside, the agent loop**: it mirrors the "user in the loop" pattern from the reference architecture and enforces the project's stated scope boundary that the system outputs a preliminary report only, with final decisions remaining with a human assessor.
- **Why a small, local vector store (Chroma/FAISS) is sufficient**: the RAG corpus is a handful of synthetic policy documents, so a managed vector database is unnecessary overhead for this project's scale.
- **Deployment target**: the full stack (Postgres, Redis, Chroma, Ollama/vLLM, LiteLLM, Langfuse) is designed to run via Docker Compose on a single machine or a modest GPU box for development and demo purposes, with k3s available if a multi-node demo is needed.

---
