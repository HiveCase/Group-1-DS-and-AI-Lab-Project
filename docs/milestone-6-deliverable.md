# Milestone 6 delivery status

Status checked on 2026-08-13 against the repository and its automated tests.

| Deliverable | Status | Evidence / current limitation |
| --- | --- | --- |
| Docker packaging | Complete in source | Root Dockerfile builds the frontend and FastAPI service; Docker Desktop was not running locally, so an image run remains to be verified. |
| GitHub Actions CI | Complete in source | Workflow installs backend dependencies, runs backend/frontend tests, and builds the frontend. It deploys to GKE only after required GitHub/GCP secrets are configured. |
| Render backend configuration | Ready | `render.yaml` deploys `backend/` with Uvicorn. SQLite and local uploads are ephemeral on the free tier. |
| Vercel frontend configuration | Ready | `frontend/vercel.json` includes the SPA rewrite; configure the Vercel project root directory as `frontend`. |
| PostgreSQL, Alembic, object storage | Not started | Deferred by scope. |
| Celery/Redis, retries, circuit breaker | Not started | Deferred by scope. |
| Health probes and graceful shutdown | Partially complete | `/health`, Kubernetes readiness/liveness probes, and FastAPI lifespan cleanup exist. A dedicated readiness endpoint and explicit process-draining policy are still needed for production scaling. |
| Autoscaling (HPA/KEDA) | Not started | Deferred until shared state and background jobs are moved out of the container. |
| Backups, alerting, IaC | Not started | Deferred by scope. |

## Validation

- Backend: `42 passed, 3 skipped` (`pytest -q`). The skipped tests are optional ML tests because the local virtual environment does not include the heavyweight ML packages.
- Frontend: run `npm test` and `npm run build` from `frontend/` before publishing.
- Docker runtime verification is blocked locally until Docker Desktop is started.

## Deployment now

Push the `car-damage-claim-frontend` branch to trigger the existing Render/Vercel Git integrations. Before the first Vercel deployment, set its project Root Directory to `frontend` and set `VITE_API_BASE_URL` to the public Render backend URL. For Render, configure a non-default `JWT_SECRET_KEY` environment variable.
