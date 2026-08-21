from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.services.auth_service import AuthService

# auto_error=False so a missing header raises our own 401 with a clear
# detail message, instead of FastAPI's generic 403 "Not authenticated".
_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Every route that isn't explicitly public depends on this. Validates
    the bearer token, loads the user it names, and rejects a deactivated
    account -- signup/login issuing a token has never been enough on its
    own; this is the piece that was missing before."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = AuthService(db).decode_access_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == int(user_id)).first() if user_id is not None else None
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Gates the internal-staff portals (Adjuster/SIU/Supervisor). Only two
    roles exist (`user`/`admin` -- see auth_schema.VALID_ROLES); `admin` is
    the internal-staff role, never self-assignable via public signup (see
    routes/auth.py)."""
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user
