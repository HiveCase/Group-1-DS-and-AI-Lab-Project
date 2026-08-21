# Contributing

This is the Car Damage Insurance Claim Portal — a FastAPI backend (`backend/`) and Vue 3
frontend (`frontend/`) implementing a 5-agent AI claims pipeline (YOLO damage detection, severity
scoring, RAG policy-clause retrieval, Groq report synthesis, deterministic fraud scoring)
coordinated by LangGraph. Full architecture, setup, and API reference: [`README.md`](README.md).
This file covers *how to work on it*, not what it is.

## Before you start

- Follow [`README.md`](README.md) §2–§3 to get a working local environment (Python 3.12, Node 20,
  `.env` from `.env.example`). Don't duplicate that setup here — if it drifts out of date, fix it
  in the README, not in a second copy.
- Read [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) if your change touches anything the README
  claims is reproducible (the app itself, the RAG evaluation, the YOLO model).

## Project layout (where things actually live)

```
backend/app/
  routes/       FastAPI endpoints -- thin: parse request, call a service, return
  services/     the actual logic (claim_service, damage_detection_service,
                policy_clause_service, report_synthesis_service, fraud_agent_service,
                langgraph_orchestrator, ...)
  db/           SQLAlchemy models (models.py) and engine/session (database.py)
  schemas/      Pydantic request/response models
  core/         config.py (Settings), security.py (get_current_user / require_admin)
  rag_scripts/  the RAG retrieval library (src/) + standalone eval CLI tools (scripts/)
backend/tests/  pytest suite -- conftest.py isolates the test DB
frontend/src/
  views/        one file per portal (ClaimantView, AdjusterView, SIUView, SupervisorView,
                LoginView, SignupView) + LandingView
  components/   shared UI (e.g. AiAnalysisPanel.vue)
  services/     api.ts (axios client), auth.js (session state)
  router.js     route table + the auth/role guard
frontend/tests/ Vitest + @vue/test-utils
scripts/        reproduce.py -- the single entry point for reproducing key results
.github/workflows/  build-ghcr.yml (test+build+push), model-evaluation.yml (on-demand eval)
```

A new backend feature almost always means: a service in `services/`, a thin route in `routes/`
that calls it, a schema in `schemas/` for the request/response shape, and a test in `tests/`. A
new frontend feature almost always means: a view or component, a function in `services/api.ts`,
and a test in `frontend/tests/`.

## Coding conventions

There's no linter or formatter configured in this repo (no ruff/black/flake8, no eslint/prettier)
— match the style of the file you're editing rather than reformatting it wholesale.

- **Services are plain classes**, usually constructed with `db: Session` (backend) or built from
  injected dependencies with sensible defaults (e.g. `ReportSynthesisService(groq_model=None)`),
  never global mutable state. `ClaimAnalysisOrchestrator` (`backend/app/services/
  claim_analysis_graph.py`) is built once as a module-level singleton in `routes/claims.py`
  specifically so the YOLO model and embedding model load once, not per request — follow that
  pattern for anything else with real model-loading cost.
- **Routes stay thin.** Validation and business logic belong in the service; a route function
  should read like "get a service, call one method, shape the response."
- **New routes need an explicit auth decision.** Every route except `/auth/*`, `/health`, and the
  documented `annotated-photo` exception must depend on `get_current_user` or `require_admin`
  (`backend/app/core/security.py`) — decide which tier a new route belongs to (any logged-in
  user, or admin-only) rather than leaving it open by omission. See the `users` table note in
  `README.md` §1 for the current admin/user split and why it's split that way.
- **Comments explain *why*, not *what*.** This codebase leans heavily on comments that record a
  non-obvious constraint, a bug that was found and fixed, or the reasoning behind a design
  choice (e.g. `policy_clauses`' uniqueness constraint in `backend/app/db/models.py`, or the
  occlusion-sensitivity docstring in `damage_detection_service.py`) — not comments restating what
  the next line does. Follow that pattern: if you make a non-obvious choice, write down why, not
  just what.
- **No secrets in git, ever.** `.env` is gitignored; `.env.example` must only ever contain
  placeholder values. If you add a new required environment variable, add it to `.env.example`
  with a placeholder and document it in `README.md` §3.

## Testing

Both suites must pass before opening a PR:

```bash
cd backend && python -m pytest -q
cd frontend && npm test
```

- New backend routes that require auth need a test proving both the happy path and the
  rejection — see `backend/tests/test_authorization.py` for the pattern (401 with no token, 403
  for the wrong role, 200 for the right one), and `backend/tests/_auth_helpers.py` for the shared
  `admin_auth_headers(client)` helper used to authenticate a module-level `TestClient`.
- Prefer exercising real components over mocks where the cost is reasonable — several tests in
  `backend/tests/test_ai_pipeline.py` call the real YOLO model, the real RAG retriever, or the
  real Groq API and `pytest.skip()` if the dependency (model file, embedding cache, API key)
  isn't available in the current environment, rather than always mocking. Follow that pattern for
  new tests where it's not prohibitively slow.
- If your change affects a number cited in `README.md` (test counts, evaluation metrics, the
  reproduce script's expected output), update the doc in the same PR — see
  `scripts/reproduce.py` and `REPRODUCIBILITY.md` for what's expected to stay in sync.

## Branch naming

- `feature/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `experiment/<short-description>`

## Pull Request Process

- Describe the purpose of the change and link related issues/tasks.
- Include test output (or screenshots for UI changes).
- If the change touches auth, the RAG pipeline, or the YOLO model, say so explicitly in the
  description — these are the areas with the most cross-cutting effects (see `README.md`'s
  `users` table note, `backend/app/rag_scripts/README.md`, and §6.4 respectively).
- Request review from the responsible team member (see the contribution table below).

## Team Responsibilities

TODO: contribution percentages not yet assigned — keep this table current as ownership shifts.

### Contribution Table

| Group Member | Contribution Percentage | Ownership Area | Remarks |
|--------------|--------------|---------|---------|
| **Satyajeet Kumar** | 100%| Problem-statement definition and requirement gathering; VehiDE dataset preprocessing (dedup, PII scan, class remapping, letterboxing); YOLO11m-seg model training (Optuna hyperparameter search across baseline/extended/tuned runs); milestone presentations and technical report authoring | Primary owner of the data/vision track, dataset curation through model training |
| **Pranab Kumar Manna** | 100%| Problem-statement definition and requirement gathering; UI/UX wireframing; relational schema design; system architecture; Project Plan; Multi-agent orchestration ; Agent Observability & Monitoring; containerized/Kubernetes deployment; Vue 3 SPA implementation ; Pytest cases; API designed; and other backend/frontend code as committed | Primary architect and full-stack owner across backend, orchestration, and deployment |
| **Venkata Siva Kamal Guddanti** |100% | RAG retrieval pipeline implementation and evaluation (hybrid dense+sparse retrieval) | Scope limited to RAG |
| **Anuj Gautam** | 100%| YOLO damage-detection model fine-tuning | Scope limited to YOLO fine-tuning |
| **Harsh Pal** | 100% | Frontend framework exploration (Vue 3 prototyping);technical report authoring; documentaion; testing | Scope limited to frontend exploration |
