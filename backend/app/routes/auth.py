from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.auth_schema import LoginRequest, SignupRequest, TokenResponse, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/signup', response_model=UserRead, status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    # Public self-signup can only ever create a "user" account -- role is
    # deliberately not client-supplied here (the request schema has no role
    # field). "admin" is the internal-staff role that gates the
    # Adjuster/SIU/Supervisor portals; the only admin account is the one
    # seeded at startup (AuthService.seed_default_admin), whose credentials
    # must be rotated before any real deployment.
    try:
        user = AuthService(db).signup(payload.email, payload.password, "user")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return user


@router.post('/login', response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    service = AuthService(db)
    user = service.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail='invalid email or password')
    token = service.create_access_token(user)
    return TokenResponse(access_token=token, user=user)
