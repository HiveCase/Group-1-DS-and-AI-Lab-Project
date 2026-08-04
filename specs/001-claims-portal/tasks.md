# Tasks: AI-Assisted Claim Portal

**Input**: Design documents from `/specs/001-claims-portal/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by milestone and user-story scope to support incremental implementation and testing.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the project skeleton, local services, and shared configuration.

- [X] T001 Create backend and frontend project structure per implementation plan in backend/ and frontend/
- [X] T002 Initialize FastAPI backend with SQLAlchemy, Pydantic v2, and dependency configuration in backend/app/
- [X] T003 Initialize Vue 3 + Vite frontend with Vue Router and Axios in frontend/
- [X] T004 [P] Configure Docker Compose for backend, frontend, and optional Ollama service in docker-compose.yml
- [X] T005 [P] Configure environment files and shared settings for uploads, database path, and model/data locations
- [X] T006 Create SQLite schema and ORM models for policies, claims, claim photos, analysis results, decision records, and clause metadata in backend/app/db/

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared application foundation before feature work begins.

- [X] T007 Implement repository/service abstractions for claims, policies, and photo storage in backend/app/services/
- [X] T008 Create FastAPI application wiring, startup hooks, and health endpoints in backend/app/
- [X] T009 Implement upload handling and local file persistence under uploads/ with metadata stored in SQLite
- [X] T010 Create seeded policy data and initial clause metadata fixtures for local development in backend/data/
- [X] T011 Add database indexes for claim_id, policy_number, and status to support claim and policy lookups
- [X] T012 Add basic error handling, validation, and structured logging across backend routes and services

---

## Phase 3: Milestone M2 - Core Claim CRUD (Priority: P1)

**Goal**: Support the core claim lifecycle for claimant and adjuster workflows.

**Independent Test**: A claimant can submit a claim and an adjuster can fetch or list it without needing AI processing.

### Implementation for M2

- [X] T013 [P] [US1] Implement POST /claims endpoint for claim creation in backend/app/routes/claims.py
- [X] T014 [P] [US1] Implement GET /claims/{claim_id} endpoint for claim lookup in backend/app/routes/claims.py
- [X] T015 [P] [US1] Implement GET /claims filtering by status for dashboard usage in backend/app/routes/claims.py
- [X] T016 [US1] Implement claim service methods for create, fetch, and list operations in backend/app/services/claim_service.py
- [X] T017 [US1] Add request/response schemas for claim CRUD in backend/app/schemas/claim_schema.py
- [X] T018 [US1] Add claim submission validation for required fields and photo count (1-5) in backend/app/schemas/claim_schema.py

---

## Phase 4: Milestone M3 - Claimant Flow End-to-End (Priority: P1)

**Goal**: Deliver the claimant experience from policy lookup through confirmation and status lookup.

**Independent Test**: A claimant can look up a policy, submit a claim with photos, see a confirmation screen, and look up the claim by ID.

### Implementation for M3

- [X] T019 [P] [US1] Implement policy lookup endpoint POST /policies/lookup in backend/app/routes/policies.py
- [X] T020 [US1] Implement claimant policy lookup UI and policy details view in frontend/src/views/ClaimantView.vue
- [X] T021 [US1] Implement claimant claim submission form with incident fields and photo upload in frontend/src/views/ClaimantView.vue
- [X] T022 [US1] Implement claimant confirmation and claim status lookup screens in frontend/src/views/ClaimantView.vue
- [X] T023 [US1] Implement frontend API client methods for policy lookup, claim submission, and claim lookup in frontend/src/services/api.ts
- [X] T024 [US1] Add shared claimant-facing styles and loading/error/success states in frontend/src/styles/

---

## Phase 5: Milestone M4 - Damage Severity Agent (Priority: P1)

**Goal**: Wrap the existing YOLO model and generate structured severity output for each uploaded photo.

**Independent Test**: A claim with uploaded photos produces an analysis record with a normalized severity label and annotated image output.

### Implementation for M4

- [ ] T025 [US2] Implement DamageAnalysisService in backend/app/services/damage_analysis_service.py
- [ ] T026 [US2] Load the existing YOLO weights from backend/models/ and run inference for each claim photo; mark as blocker until the YOLO model file is available locally
- [ ] T027 [US2] Save annotated images alongside originals under uploads/ and expose their paths via the backend
- [ ] T028 [US2] Normalize YOLO detections into Minor/Moderate/Severe using an explicit rule and store the structured result in SQLite
- [ ] T029 [US2] Add a background-safe analysis result persistence flow so the analysis status can be recorded without blocking claim submission

---

## Phase 6: Milestone M5 - Policy Clause Agent (Priority: P1)

**Goal**: Semantically retrieve relevant policy clauses and evaluate the claimed amount against clause limits.

**Independent Test**: For a claim description and amount, the system returns relevant clauses and a simple coverage check.

### Implementation for M5

- [ ] T030 [US2] Implement PolicyClauseService in backend/app/services/policy_clause_service.py
- [ ] T031 [US2] Embed the policy clause dataset with sentence-transformers and store the index using Chroma on disk; mark as blocker until the clause dataset is available locally
- [ ] T032 [US2] Add retrieval logic for top-k relevant clauses and a simple numeric-limit check based on retrieved clause text
- [ ] T033 [US2] Persist policy findings alongside the claim analysis result in SQLite

---

## Phase 7: Milestone M6 - Agentic Orchestration (Priority: P1)

**Goal**: Combine severity and policy findings into a single orchestrated decision flow.

**Independent Test**: A submitted claim triggers background orchestration that produces recommendation, confidence, and explanation data.

### Implementation for M6

- [ ] T034 [US2] Implement an orchestrator layer in backend/app/agents/ or backend/app/services/ that wires damage analysis and policy retrieval into one flow
- [ ] T035 [US2] Use a local orchestration framework such as LangGraph or CrewAI to coordinate the three-step pipeline
- [ ] T036 [US2] Implement a Decision Agent that produces recommendation, confidence score, and explanation text using rule-based fallback when an LLM is not configured
- [ ] T037 [US2] Hook the orchestration flow into FastAPI BackgroundTasks on claim submission so the claimant request completes immediately
- [ ] T038 [US2] Expose analysis readiness and results through the claim detail endpoints for the adjuster view

---

## Phase 8: Milestone M7 - Adjuster Flow End-to-End (Priority: P1)

**Goal**: Deliver the adjuster workflow with dashboard, detail review, AI analysis panel, and decision submission.

**Independent Test**: An adjuster can view pending claims, inspect AI analysis, and submit an approve/deny/request-more-info decision.

### Implementation for M7

- [ ] T039 [US3] Implement adjuster dashboard list endpoint and summary counts in backend/app/routes/claims.py
- [ ] T040 [US3] Implement adjuster claim detail and analysis retrieval endpoints in backend/app/routes/claims.py
- [ ] T041 [US3] Implement decision submission endpoint and persistence logic in backend/app/routes/claims.py
- [ ] T042 [US3] Build the adjuster dashboard view in frontend/src/views/AdjusterView.vue
- [ ] T043 [US3] Build the claim detail view with annotated photo, severity summary, policy findings, and recommendation panel in frontend/src/views/AdjusterView.vue
- [ ] T044 [US3] Build the decision form for approve/deny/request more info with settlement amount and reasoning note in frontend/src/views/AdjusterView.vue
- [ ] T045 [US3] Add frontend API methods for dashboard data, claim detail data, and decision submission in frontend/src/services/api.ts
- [ ] T046 [US3] Add shared adjuster-facing styles and consistent status/color indicators in frontend/src/styles/

---

## Phase 9: Milestone M8 - Testing Pass (Priority: P2)

**Goal**: Validate functionality and prevent regressions across backend, frontend, and pipeline behavior.

**Independent Test**: The project has automated backend/frontend tests and a manual validation script for the core flows.

### Implementation for M8

- [ ] T047 [P] Add backend endpoint tests for policy lookup, claim submission, claim retrieval, and decision submission in backend/tests/
- [ ] T048 [P] Add frontend component tests for claimant and adjuster views in frontend/tests/
- [ ] T049 [P] Add pipeline tests that mock YOLO inference and Chroma retrieval so CI does not depend on GPU or the full dataset
- [ ] T050 Create a manual end-to-end validation script documenting the claimant and adjuster workflow in specs/001-claims-portal/quickstart.md

---

## Phase 10: Milestone M10 - SIU Portal (Priority: P2)

**Goal**: Add the SIU investigation workflow with a fraud-focused dashboard and investigation actions that reuse the existing AI analysis results.

**Independent Test**: An SIU analyst can view high-risk claims, open a claim detail screen, and record an investigation or clear action without creating duplicate AI processing.

### Implementation for M10

- [ ] T051 [US4] Create the investigations table and ORM model for claim_id, investigator_id, status, notes, and timestamp in backend/app/db/
- [ ] T052 [US4] Implement InvestigationService in backend/app/services/investigation_service.py for listing, fetching, and updating investigations
- [ ] T053 [US4] Add SIU routes and schemas in backend/app/routes/investigations.py and backend/app/schemas/investigation_schema.py
- [ ] T054 [US4] Add the SIU dashboard endpoint that auto-filters claims above the fraud-score threshold and returns summary stats for high-risk, under-investigation, and confirmed-fraud counts
- [ ] T055 [US4] Add the SIU claim-detail read path that reuses the existing DamageAnalysisService and PolicyClauseService results already stored for the claim
- [ ] T056 [US4] Implement the investigate/clear action flow so the SIU analyst can open a formal investigation or clear the flag and log the action with a timestamp and investigator ID
- [ ] T057 [US4] Build the SIU dashboard UI in frontend/src/views/SIUView.vue with claim cards showing claim ID, claimant, claim type, amount, and fraud score prominently
- [ ] T058 [US4] Build the SIU claim detail view in frontend/src/views/SIUView.vue reusing the existing AI analysis panel and showing fraud factors that drove the score
- [ ] T059 [US4] Add frontend API methods for SIU dashboard data, claim detail data, and investigation actions in frontend/src/services/api.ts

---

## Phase 11: Milestone M11 - Supervisor Dashboard (Priority: P2)

**Goal**: Add a read-only supervisor analytics view with aggregate KPIs and AI-specific metrics.

**Independent Test**: A supervisor can view portfolio metrics, AI metric summaries, and system-status information without editing claims.

### Implementation for M11

- [ ] T060 [US5] Implement AnalyticsService in backend/app/services/analytics_service.py with aggregate SQL queries over claims, damage detections, and policy-clause-check tables
- [ ] T061 [US5] Add the Supervisor analytics endpoint and schemas in backend/app/routes/analytics.py and backend/app/schemas/analytics_schema.py
- [ ] T062 [US5] Include the required KPI cards for total claims by status, average fraud score, claims by type, approved vs. denied ratio, and average submission-to-decision time
- [ ] T063 [US5] Include the two AI-specific metrics: severity distribution (Minor/Moderate/Severe) and coverage-flag rate (outside coverage / over limit)
- [ ] T064 [US5] Include a read-only system status section with AI pipeline operability and throughput metrics such as average analysis time and claims processed today
- [ ] T065 [US5] Cache or precompute analytics summaries to satisfy the performance requirement rather than recomputing on every page load
- [ ] T066 [US5] Build the Supervisor dashboard UI in frontend/src/views/SupervisorView.vue with KPI cards, one chart, and the system-status panel
- [ ] T067 [US5] Add frontend API methods for Supervisor analytics data in frontend/src/services/api.ts

---

## Phase 12: Milestone M12 - Design-System Migration (Priority: P2)

**Goal**: Introduce the shared AppShell, design tokens, and component-library-based UI system across all four portals.

**Independent Test**: The Claimant, Adjuster, SIU, and Supervisor views render through the same shared shell and design system with no major layout regressions.

### Implementation for M12

- [ ] T068 [P] Add shared design tokens for colors, typography, spacing, and portal accents in frontend/src/styles/tokens.css
- [ ] T069 [P] Add a shared AppShell layout and portal-selection landing page update in frontend/src/App.vue and frontend/src/router.js
- [ ] T070 [P] Integrate PrimeVue or Naive UI as the base UI library for buttons, tables, forms, cards, and layout primitives in frontend/
- [ ] T071 [US1] Retrofit the Claimant view onto the shared AppShell and design tokens without rebuilding it from scratch
- [ ] T072 [US2] Retrofit the Adjuster view onto the shared AppShell and design tokens without rebuilding it from scratch
- [ ] T073 [US4] Ensure the SIU view uses the shared shell and the same design tokens as the other portals
- [ ] T074 [US5] Ensure the Supervisor view uses the shared shell and the same design tokens as the other portals

---

## Phase 13: Milestone M13 - Polish Pass (Priority: P2)

**Goal**: Tighten consistency and responsiveness across all four portals before handoff.

**Independent Test**: The portal experience is visually consistent and usable at a standard laptop size and smaller widths.

### Implementation for M13

- [ ] T075 [P] Review and align spacing, typography, status colors, and component usage across Claimant, Adjuster, SIU, and Supervisor views in frontend/src/styles/
- [ ] T076 [P] Verify the shared AppShell and landing page create a cohesive portal-selection experience across all four portals
- [ ] T077 [P] Run a responsive check for the major flows and adjust layouts for standard laptop and narrower widths
- [ ] T078 [P] Add documentation for the severity normalization rule, policy-limit rule, and SIU/Supervisor analytics conventions in docs/ or backend/app/

---

## Phase 14: Milestone M14 - Testing Pass (Priority: P2)

**Goal**: Add regression coverage for the new SIU, Supervisor, and shared-layout work.

**Independent Test**: Backend and frontend tests cover the new service and view behaviors, and a mocked end-to-end flow covers all four portals.

### Implementation for M14

- [ ] T079 [P] Add backend pytest coverage for InvestigationService and AnalyticsService in backend/tests/
- [ ] T080 [P] Add frontend Vitest coverage for SIU and Supervisor components in frontend/tests/
- [ ] T081 [P] Add a mocked end-to-end test or manual validation script covering Claimant, Adjuster, SIU, and Supervisor portal flows in specs/001-claims-portal/quickstart.md
- [ ] T082 [P] Ensure the new tests validate the shared AppShell and portal-selection flow without depending on full AI execution

---

## Dependencies & Execution Order

### Milestone Dependencies

- **M1**: No dependencies; starts immediately
- **M2**: Depends on M1
- **M3**: Depends on M2
- **M4**: Depends on M2 and the local YOLO model file
- **M5**: Depends on M2 and the local clause dataset
- **M6**: Depends on M4 and M5
- **M7**: Depends on M2, M3, M4, M5, and M6
- **M8**: Depends on M3 and M7
- **M9**: Completed earlier as part of the initial implementation scope
- **M10**: Depends on M2, M3, M4, M5, M6, and M7 because the SIU detail view reuses the existing AI analysis data
- **M11**: Depends on M10 because the Supervisor coverage-flag metric depends on InvestigationService/PolicyClauseService data existing
- **M12**: Depends on M10 and M11 for the new portal views, and on M3/M7 for existing Claimant/Adjuster retrofit work
- **M13**: Depends on M12
- **M14**: Depends on M10, M11, and M12

### Parallel Opportunities

- T051-T059 can proceed in parallel once M7 is complete
- T060-T067 can proceed once M10 has introduced the investigation data shape and the existing AI detail data is available
- T068-T074 can proceed once M3 and M7 are stable and the shared shell is defined
- T075-T078 can proceed once M12 is in place
- T079-T082 can proceed once M10-M12 are implemented

### Suggested MVP scope

- Deliver the completed M1-M7 and M2-M3 frontend flow first for the claim lifecycle.
- Then implement M10 and M11 for the new portal analytics and investigation workflows.
- M12-M14 should be treated as required before project handoff because they improve the four-portal experience and regression coverage.

---

## Dependencies & Execution Order

### Milestone Dependencies

- **M1**: No dependencies; starts immediately
- **M2**: Depends on M1
- **M3**: Depends on M2
- **M4**: Depends on M2 and the local YOLO model file
- **M5**: Depends on M2 and the local clause dataset
- **M6**: Depends on M4 and M5
- **M7**: Depends on M2, M3, M4, M5, and M6
- **M8**: Depends on M3 and M7
- **M9**: Depends on M7 and remains optional

### Blockers to resolve first

- YOLO model file available locally under backend/models/ before T026 can be completed
- Clause dataset available locally under backend/data/ before T031 can be completed
- If either asset is missing, the relevant AI milestone should be treated as blocked until the asset is provided

### Parallel Opportunities

- T004 and T005 can run in parallel with T001-T003
- T013-T015 can run in parallel once T012 is complete
- T020-T024 can run in parallel once M2 is ready
- T025-T033 can be implemented in parallel once the AI assets are available
- T039-T046 can be implemented in parallel once M6 analysis is available

### Suggested MVP scope

- Deliver M1 through M3 first for a working claimant journey.
- Then complete M4 and M5 as blockers for M6 and M7.
- M8 should be treated as required before project handoff.
- M9 is optional and should be skipped unless the core milestones complete early.
