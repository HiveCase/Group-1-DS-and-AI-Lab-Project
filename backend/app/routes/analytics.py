from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import require_admin
from app.db.database import get_db
from app.db.models import User
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix='/analytics', tags=['analytics'])


@router.get('/summary')
def analytics_summary(db: Session = Depends(get_db), _admin: User = Depends(require_admin)):
    return {'summary': AnalyticsService(db).build_summary()}
