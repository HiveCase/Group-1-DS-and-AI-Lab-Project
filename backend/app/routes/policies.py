from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.policy_schema import PolicyLookupRequest, PolicyRead
from app.services.policy_service import PolicyService

router = APIRouter(prefix='/policies', tags=['policies'])

@router.post('/lookup', response_model=PolicyRead)
def lookup_policy(payload: PolicyLookupRequest, db: Session = Depends(get_db)):
    policy = PolicyService(db).get_by_number(payload.policy_number)
    if not policy:
        raise HTTPException(status_code=404, detail='policy not found')
    return policy
