from pydantic import BaseModel, ConfigDict, Field

VALID_ROLES = ("user", "admin")


class SignupRequest(BaseModel):
    email: str = Field(min_length=3)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: str = Field(min_length=1)
    password: str = Field(min_length=1)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: str
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead
