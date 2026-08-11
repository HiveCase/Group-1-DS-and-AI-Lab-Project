import logging
import sys
import time
from pathlib import Path

# Ensure the app package can be imported when main.py is executed from the repo root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.db.database import SessionLocal, init_db
from app.routes.analytics import router as analytics_router
from app.routes.claims import router as claims_router
from app.routes.policies import router as policies_router
from app.services.policy_clause_service import PolicyClauseService
from app.services.policy_service import PolicyService


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_data()
    yield


app = FastAPI(title='Claims Portal API', lifespan=lifespan)
logger = logging.getLogger("claims_portal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(claims_router)
app.include_router(policies_router)
app.include_router(analytics_router)


def frontend_dist_dir() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[1] / "frontend" / "dist",
        Path(__file__).resolve().parents[2] / "frontend" / "dist",
    ]
    return next((path for path in candidates if (path / "index.html").exists()), None)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "request method=%s path=%s status=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
    )
    return response


def initialize_app_data():
    settings = get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    db = SessionLocal()
    try:
        PolicyService(db).seed_defaults()
    finally:
        db.close()
    PolicyClauseService().ensure_all_seeded_policies_ingested()


initialize_app_data()


@app.get('/health')
def health():
    return {'status': 'ok'}


frontend_dist = frontend_dist_dir()
if frontend_dist:
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        requested_file = frontend_dist / full_path
        if full_path and requested_file.is_file():
            return FileResponse(requested_file)
        return FileResponse(frontend_dist / "index.html")
