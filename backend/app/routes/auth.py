from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.auth_schema import LoginRequest, SignupRequest, TokenResponse, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/signup', response_model=UserRead, status_code=201)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    try:
        user = AuthService(db).signup(payload.email, payload.password, payload.role)
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
