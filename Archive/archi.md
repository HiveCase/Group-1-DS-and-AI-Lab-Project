# Open-Source Architecture for Car Damage Claim System

## Overview

This document outlines a fully open-source system architecture for a **Car Damage Claim Processing System**, inspired by the travel concierge multi-agent pattern. All proprietary and managed cloud services have been replaced with self-hostable, open-source alternatives.

The system is designed to run entirely on infrastructure you control, making it ideal for academic projects, local deployments, or cost-efficient production setups.

---

## Request Flow Architecture

The system follows a structured request-response pipeline:

1. User uploads car damage images and claim details
2. Request passes through a load balancer
3. Application backend processes input
4. Multi-agent orchestrator coordinates tasks
5. Agents invoke tools (ML models, APIs, parsers)
6. Results are aggregated into a structured report
7. Optional human review step before final output

---

## Agent Orchestrator Design

Inside the **Agent Orchestrator**, the system is divided into multiple specialized agents:

### 1. Damage Detection Agent

* Uses YOLOv8 or similar model
* Identifies damaged regions in vehicle images

### 2. Damage Classification Agent

* Categorizes type of damage (scratch, dent, crack, etc.)

### 3. Severity Assessment Agent

* Assigns severity score based on damage extent
* Independent from detection for better validation and retries

### 4. Claim Report Generation Agent

* Aggregates outputs
* Generates structured insurance report (JSON/PDF)

---

## Tooling Used by Agents

Each agent interacts with specific tools:

* **YOLOv8 Service** → Damage detection
* **PDF Parser / Chunker** → Policy document analysis
* **VIN Decoder API (Open Source)** → Vehicle metadata extraction
* **Vector DB (Chroma / FAISS)** → Policy matching and semantic search

---

## Open-Source Technology Mapping

| Layer               | AWS / Proprietary         | Open-Source Alternative                   |
| ------------------- | ------------------------- | ----------------------------------------- |
| Load Balancer       | AWS Load Balancer         | Nginx / Traefik                           |
| Application Runtime | AWS ECS + DynamoDB        | FastAPI + PostgreSQL                      |
| Agent Framework     | Bedrock Agents / Strands  | LangGraph / CrewAI                        |
| External APIs       | Amadeus, OpenStreetMap    | YOLOv8 service, VIN API, PDF tools        |
| Tool Exposure       | MCP Server                | FastMCP                                   |
| Memory (Short-term) | Bedrock Memory            | Redis                                     |
| Memory (Long-term)  | S3                        | ChromaDB / FAISS                          |
| Observability       | AgentOps                  | Langfuse + OpenTelemetry                  |
| LLM Routing         | LiteLLM                   | LiteLLM (unchanged)                       |
| LLMs                | OpenAI / Bedrock / Vertex | Ollama / vLLM (Llama 3, Mistral, Qwen2.5) |
| Identity & Security | AWS IAM                   | Keycloak                                  |
| Secrets Management  | AWS KMS                   | HashiCorp Vault                           |
| SIEM                | AWS Security Hub          | Wazuh                                     |
| Monitoring          | CloudWatch                | Prometheus + Grafana                      |
| Deployment          | ECS / Kubernetes          | Docker Compose / k3s                      |

---

## Key Design Decisions

### 1. Multi-Agent Separation

The system splits processing into **four independent agents** instead of a single pipeline:

* Improves modularity
* Enables independent testing
* Allows retry logic per stage
* Supports disagreement resolution (e.g., detection vs severity)

This directly improves upon the earlier sequential pipeline design.

---

### 2. Human-in-the-Loop Validation

A **human review stage** is included for:

* Low-confidence detections
* Ambiguous insurance policy matches
* Edge cases

This ensures:

* Better reliability
* Alignment with real-world insurance workflows
* Compliance with project scope (preliminary report generation only)

---

### 3. Fully Self-Hosted Stack

All components run on your own infrastructure:

* No dependency on managed cloud services
* Works on:

  * Local machine
  * Single GPU workstation
  * Small server setup

Typical stack:

* FastAPI + PostgreSQL
* Redis
* FAISS / ChromaDB
* Ollama / vLLM
* Docker Compose

---

### 4. Scalable Deployment Path

The system supports gradual scaling:

* **Development** → Docker Compose (single machine)
* **Intermediate** → GPU-enabled node
* **Advanced** → k3s (lightweight Kubernetes cluster)

---

## Benefits of This Architecture

* Fully open-source and cost-efficient
* Modular and extensible design
* Easy to debug and test
* Suitable for academic and production use
* Supports offline or private deployments

---

## Conclusion

This architecture provides a robust, scalable, and entirely self-hostable alternative to cloud-native AI systems. By combining multi-agent orchestration with open-source infrastructure, it enables efficient and transparent processing of car damage insurance claims.

---

## Future Enhancements

* Add real-time video damage assessment
* Integrate repair cost estimation models
* Improve policy matching using RAG pipelines
* Add mobile app interface for claim submission
