import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.database import SessionLocal, init_db
from app.routes.analytics import router as analytics_router
from app.routes.claims import router as claims_router
from app.routes.policies import router as policies_router
from app.services.policy_service import PolicyService

app = FastAPI(title='Claims Portal API')
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


initialize_app_data()


@app.on_event("startup")
def startup():
    initialize_app_data()

@app.get('/health')
def health():
    return {'status': 'ok'}
