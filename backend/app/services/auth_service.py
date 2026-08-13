from __future__ import annotations

from datetime import datetime, timedelta

import bcrypt
import jwt
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import User
from app.schemas.auth_schema import VALID_ROLES

DEFAULT_ADMIN_EMAIL = "admin@gmail.com"
DEFAULT_ADMIN_PASSWORD = "admin"


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    def signup(self, email: str, password: str, role: str) -> User:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {', '.join(VALID_ROLES)}")
        if self.get_by_email(email):
            raise ValueError("email already registered")
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user = User(email=email, hashed_password=hashed, role=role)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def authenticate(self, email: str, password: str) -> User | None:
        user = self.get_by_email(email)
        if not user or not user.is_active:
            return None
        if not bcrypt.checkpw(password.encode("utf-8"), user.hashed_password.encode("utf-8")):
            return None
        user.last_login_at = datetime.utcnow()
        self.db.commit()
        return user

    def get_by_email(self, email: str) -> User | None:
        return self.db.query(User).filter(User.email == email).first()

    def seed_default_admin(self) -> None:
        if self.get_by_email(DEFAULT_ADMIN_EMAIL):
            return
        self.signup(DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD, "admin")

    def create_access_token(self, user: User) -> str:
        expire = datetime.utcnow() + timedelta(minutes=self.settings.jwt_access_token_expire_minutes)
        payload = {"sub": str(user.id), "email": user.email, "role": user.role, "exp": expire}
        return jwt.encode(payload, self.settings.jwt_secret_key, algorithm="HS256")

    def decode_access_token(self, token: str) -> dict:
        return jwt.decode(token, self.settings.jwt_secret_key, algorithms=["HS256"])
