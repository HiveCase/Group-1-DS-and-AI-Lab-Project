from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix='/analytics', tags=['analytics'])


@router.get('/summary')
def analytics_summary(db: Session = Depends(get_db)):
    return {'summary': AnalyticsService(db).build_summary()}
