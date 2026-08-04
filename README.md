# Car Damage Insurance Claim Portal

A local-first MVP portal for car damage insurance claims with four role-based interfaces:
- **Claimant**: submit claims, upload photos, and check claim status
- **Adjuster**: review claims, inspect AI-derived damage analysis, and decide claims
- **SIU**: investigate suspicious or high-risk claims using fraud scoring and claim evidence
- **Supervisor**: monitor portfolio analytics, risk exposure, and AI performance
![Home Page](https://raw.githubusercontent.com/HiveCase/Group-1-DS-and-AI-Lab-Project/car-damage-claim-frontend/backend/static/images/HomePage.png)
---

## Architecture

This repository contains two main applications:

- `backend/`: FastAPI service with SQLite persistence and local file upload handling
- `frontend/`: Vue 3/Vite single-page app for the four portal views

### System architecture

```
[Browser] -> [Frontend SPA] -> [Backend FastAPI] -> [SQLite + uploads/]
                        |              |
                        v              v
                    [API adapter]  [AI rule engine]
```

- The frontend is a Vue 3 single-page application that routes users to four portals.
- The backend exposes REST endpoints for claim intake, policy lookup, adjuster actions, SIU investigations, and supervisor analytics.
- Uploaded claim photos are stored locally in `uploads/`.
- SQLite stores the claim, policy, analysis, decision, and investigation data.

### Backend

Key backend components:

- `backend/app/main.py`: FastAPI application entry point and route registration
- `backend/app/routes/claims.py`: claim intake, detail lookup, adjuster actions, SIU actions
- `backend/app/routes/policies.py`: policy lookup support for claimant intake
- `backend/app/routes/analytics.py`: supervisor analytics summary
- `backend/app/db/models.py`: SQLAlchemy ORM models for policies, claims, photos, analysis, decisions, and investigations
- `backend/app/services/`: domain services for claims, analysis, investigations, and analytics
- `backend/uploads/`: locally stored uploaded claim photos

### Frontend

Key frontend components:

- `frontend/src/App.vue`: shared app shell with portal navigation
- `frontend/src/router.js`: router configuration for landing page and portal routes
- `frontend/src/views/`: role-specific portal views and the landing page
- `frontend/src/services/api.ts`: API adapter methods for backend interactions
- `frontend/src/styles/main.css`: shared design tokens, portal cards, and responsive layout

---

## Data flow

### Claim submission and review flow

```
Claimant fills form -> frontend POST /claims -> backend creates Claim
                        |                              |
                        +- upload photos -> uploads/    v
                                                    analysis service
                                                        |
                                           saves ClaimAnalysis -> Claim
                                                        |
                                           response includes claim + analysis
                                                        v
                                                   Adjuster/SIU fetch claim
```

### Role-based portal flow

- Claimant: `/claimant` to submit claims and view status
- Adjuster: `/adjuster` to review claims, see analysis, and decide
- SIU: `/siu` to inspect flagged claims and open investigations
- Supervisor: `/supervisor` to view aggregated analytics

---

## Agentic AI flow

The current implementation is a lightweight, local rule-based analysis engine that simulates an agentic AI workflow for damage scoring and fraud detection.

```
[Incoming claim data]
       |
       v
[DamageAnalysisService]
       |
       +-- evaluate claim details (description, policy, price, location)
       +-- compute severity label
       +-- compute confidence and fraud score
       +-- summarize findings for adjuster review
       v
[persist ClaimAnalysis]
       |
       v
[Claim response includes analysis result]
```

- The service acts like a lightweight reasoning agent by evaluating structured claim fields and deriving explainable outputs.
- Fraud score and severity are returned as structured evidence for downstream UI workflows.

---

## Database schema

### Models and key fields

```
Policy
  id: int
  number: str
  holder_name: str
  coverage_type: str
  deductible: float
  active: bool

Claim
  id: int
  policy_id: int
  claimant_name: str
  vehicle_make: str
  vehicle_model: str
  vehicle_year: int
  date_of_loss: date
  damage_description: str
  estimated_repair_cost: float
  status: str
  created_at: datetime

ClaimPhoto
  id: int
  claim_id: int
  filename: str
  uploaded_at: datetime

ClaimAnalysis
  id: int
  claim_id: int
  severity: str
  confidence: float
  explanation: str
  fraud_score: float

ClaimDecision
  id: int
  claim_id: int
  adjuster_name: str
  decision: str
  notes: str
  decided_at: datetime

Investigation
  id: int
  claim_id: int
  investigator_name: str
  findings: str
  status: str
  opened_at: datetime
```

### Database schema diagram

```
Policy 1---* Claim 1---* ClaimPhoto
            |  \        \
            |   \        * ClaimAnalysis
            |    * ClaimDecision
            |
            * Investigation
```

- `Policy` is referenced by `Claim`
- `ClaimPhoto`, `ClaimAnalysis`, `ClaimDecision`, and `Investigation` are child records of `Claim`

---

## Features

- Role-based portal navigation for Claimant, Adjuster, SIU, and Supervisor
- Claim submission with photo upload and policy lookup
- Rule-based AI analysis for damage severity, claim confidence, and fraud risk
- Adjuster decision workflow and claim status updates
- SIU investigation dashboard for suspicious investigations
- Supervisor analytics summary for claims and model quality
- Local-first deployment with SQLite and static frontend build

---

## Setup

### Prerequisites

- Python 3.11+ (recommended)
- Node.js 20+ and npm
- Optional: Docker, if you want containerized deployment

### Backend install

```powershell
cd "c:\Users\Pranab\OneDrive\Desktop\IIT Assignment\DS AI LAB\Car-Damage-Insurance-Claim\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend install

```powershell
cd "c:\Users\Pranab\OneDrive\Desktop\IIT Assignment\DS AI LAB\Car-Damage-Insurance-Claim\frontend"
npm install
```

---

## Running the app

### Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```powershell
cd frontend
npm run dev
```

Then open the app in your browser at the URL shown by Vite (typically `http://localhost:5173`).

---

## Testing

### Backend tests

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
pytest
```

### Frontend tests

```powershell
cd frontend
npm test
```

---

## Deployment

### Build frontend for production

```powershell
cd frontend
npm run build
```

### Run backend with production settings

Use the current `backend` app and point the frontend to the deployed backend endpoint. The backend uses SQLite and local file storage by default.

---

## Notes

- The app is designed as a self-hosted prototype and does not require external AI services.
- File uploads are stored locally under `uploads/`.
- The current AI/analysis workflow is a lightweight rule-based prototype built for MVP speed.

---

## Project structure

```text
backend/
  app/
    main.py
    routes/
    services/
    db/
  uploads/
frontend/
  src/
    views/
    services/
    styles/
  tests/
```

---

## Future improvements

- Add real ML-based damage detection and policy-matching inference
- Expand claimant task history and adjuster claim detail pages
- Add authentication and role-based access control
- Persist investigation notes and supervisor audit logs
- Improve responsive mobile layout and accessibility
