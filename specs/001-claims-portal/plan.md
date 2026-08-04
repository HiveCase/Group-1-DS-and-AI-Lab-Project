# Implementation Plan: AI-Assisted Claim Portal

**Branch**: `001-claims-portal` | **Date**: 2026-08-04 | **Spec**: [specs/001-claims-portal/spec.md](specs/001-claims-portal/spec.md)

**Input**: Feature specification from /specs/001-claims-portal/spec.md

## Summary
Build a local-first claims portal with four core user experiences: claimant, adjuster, SIU, and supervisor. The implementation uses FastAPI and SQLAlchemy for the backend, Vue 3 and Vite for the frontend, and a background AI workflow that reuses the existing YOLO model and policy clause data through service-based components and an orchestrated pipeline. The SIU and Supervisor experiences reuse the existing DamageAnalysisService and PolicyClauseService results rather than duplicating AI logic, and the frontend is planned around a shared AppShell and a shared design system.

## Technical Context

**Language/Version**: Python 3.11+, Node.js 20+, Vue 3, Vite

**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic v2, Ultralytics, sentence-transformers, Chroma, LangGraph, Vue Router, Axios, Vitest, pytest, httpx

**Storage**: SQLite for MVP data, local disk storage under uploads for claim photos, Chroma index persisted on disk

**Testing**: pytest and httpx for backend endpoint tests; Vitest and Vue Test Utils for frontend unit tests

**Target Platform**: Local development and Docker Compose

**Project Type**: Web application

**Performance Goals**: Claim CRUD operations under 300 ms locally, excluding AI analysis; AI analysis ready within roughly one minute for typical local test claims

**Constraints**: Must stay self-hosted and open-source; no paid services; existing YOLO model and policy dataset must be reused as-is; background processing must not block claim submission

**Scale/Scope**: MVP for a small local capstone deployment with seeded policies, a modest claim volume, and no production-grade fraud or payment workflows

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Pass: Backend uses FastAPI, Pydantic v2, service-layer separation, environment-based configuration, and background tasks for AI processing.
- Pass: Frontend uses Vue 3 Composition API and a shared design system that will be implemented through an open-source UI library such as PrimeVue or Naive UI rather than hand-rolled elements.
- Pass: Testing strategy covers backend endpoints, frontend component behavior, and AI pipeline tests with mocked external dependencies.
- Pass: Performance and storage expectations align with background processing, photo preprocessing, and indexed SQLite lookups.
- No exemptions required.

## Project Structure

### Documentation (this feature)

```text
specs/001-claims-portal/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (not created by this step)
```

### Source Code (repository root)

```text
backend/
├── app/
│   ├── agents/
│   ├── core/
│   ├── db/
│   ├── routes/
│   │   ├── claims.py
│   │   ├── policies.py
│   │   ├── investigations.py
│   │   └── analytics.py
│   ├── schemas/
│   │   ├── claim_schema.py
│   │   ├── policy_schema.py
│   │   ├── investigation_schema.py
│   │   └── analytics_schema.py
│   └── services/
│       ├── claim_service.py
│       ├── damage_analysis_service.py
│       ├── policy_clause_service.py
│       ├── investigation_service.py
│       └── analytics_service.py
├── models/
├── data/
├── tests/
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   ├── services/
│   ├── stores/
│   ├── styles/
│   │   ├── tokens.css
│   │   └── main.css
│   └── views/
│       ├── ClaimantView.vue
│       ├── AdjusterView.vue
│       ├── SIUView.vue
│       └── SupervisorView.vue
├── tests/
└── package.json

uploads/
docker-compose.yml
```

**Structure Decision**: Use a split backend/frontend layout with a clearly separated service layer in the backend and lightweight Vue views and components in the frontend. The AI orchestration and storage concerns live under backend/app so business logic remains independent of the route layer. The SIU and Supervisor experiences each get their own route and service layer, but both reuse the existing DamageAnalysisService and PolicyClauseService results already associated with a claim rather than invoking AI workflows again. The frontend will introduce a shared AppShell, design tokens, and an open-source component library for buttons, tables, forms, cards, and layout primitives. Existing Claimant and Adjuster components will be retrofitted onto this shared layout where possible rather than rebuilt from scratch.

## Phase 0 Research Notes

- The SIU workflow will use a lightweight investigations table with claim_id, investigator_id, status, notes, and timestamp fields.
- The Supervisor workflow will use aggregate SQL queries over existing claims, damage detections, and policy-clause-check tables, with precomputed or cached summaries to satisfy the performance requirement.
- The frontend will use a shared design token layer for colors, typography, and spacing, then layer portal-specific accents on top of that foundation.
- Existing DamageAnalysisService and PolicyClauseService remain the single source of truth for AI output; the SIU and Supervisor views only read and summarize those already-stored results.

## Phase 1 Design Notes

- Backend contracts will expose SIU investigation actions and Supervisor analytics payloads through dedicated routes and schemas.
- Frontend route planning will include a portal-selection landing page and a shared AppShell that wraps the Claimant, Adjuster, SIU, and Supervisor portals.
- Claimant and Adjuster components will be reviewed for reuse and retrofitting onto the new shared layout; only the parts that conflict with the new layout will be reworked.

## Complexity Tracking

No constitution violations were identified, so no special complexity exceptions are required.
