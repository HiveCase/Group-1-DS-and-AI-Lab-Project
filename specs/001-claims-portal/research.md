# Research Notes

## Backend architecture

### Decision
Use FastAPI with SQLAlchemy ORM, Pydantic v2 schemas, and repository-style service modules for claims, policies, and AI pipeline orchestration.

### Rationale
This keeps the API layer thin, supports clear validation, and isolates business logic so the data-access layer can later move from SQLite to PostgreSQL without changing route logic.

### Alternatives considered
- Direct route-level database access: rejected because it would mix validation, persistence, and business rules in one place.
- Django REST Framework: rejected because the project is already centered on FastAPI and a lighter stack is preferable for the MVP.

## AI pipeline design

### Decision
Use a small orchestrated flow with three services: DamageAnalysisService, PolicyClauseService, and an orchestration layer powered by LangGraph.

### Rationale
This satisfies the requirement for an agentic workflow without over-engineering the MVP. The orchestration step can run locally with explainable rule-based fallbacks when a local LLM is not available.

### Alternatives considered
- Hand-written glue code: rejected because it would be harder to explain and maintain than a structured orchestration layer.
- Full LLM-only reasoning: rejected because it adds unnecessary runtime and dependency risk for a local capstone build.

## Computer vision and retrieval choices

### Decision
Load the existing YOLO weights with Ultralytics, normalize detections into a single severity label, and store annotated images on disk under the uploads directory.

### Rationale
This directly reuses the existing model asset and keeps the inference path explicit and explainable for the defense presentation.

### Alternatives considered
- Using a different detector: rejected because the project explicitly requires the existing YOLO model.
- Embedding the entire inference logic in the route layer: rejected because service isolation is required by the constitution.

## Policy retrieval and vector storage

### Decision
Use sentence-transformers to embed the policy clause dataset once at startup and persist the index in Chroma on disk.

### Rationale
This provides semantic retrieval without requiring a remote service or a heavier database setup.

### Alternatives considered
- Simple keyword search only: rejected because the requirement calls for semantic retrieval.
- External hosted vector search: rejected because the project must stay local and self-hosted.

## Frontend approach

### Decision
Use Vue 3 with the Composition API, Vue Router, and plain CSS variables for a hand-rolled design system. Pinia is intentionally avoided unless shared client state becomes necessary.

### Rationale
The project is intentionally lightweight and should stay close to plain HTML and CSS to simplify implementation and defense.

### Alternatives considered
- Full component libraries such as Vuetify or Quasar: rejected because the requirement calls for a minimal custom design system.
- Pinia from the start: rejected because the initial workflows can be managed with local component state and composables.

## Deployment strategy

### Decision
Use Docker Compose with backend and frontend services, and make Ollama optional because the explanation text can fall back to a rule-based template.

### Rationale
This keeps local deployment straightforward while preserving a path to optional richer explanation generation.

### Alternatives considered
- Cloud deployment: rejected because the project explicitly targets local Docker Compose.
- Requiring Ollama in the MVP: rejected because it adds setup complexity that is not necessary for the core workflow.
