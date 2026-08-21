import jwt
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)


def _signup(email="user1@example.com", password="supersecret1"):
    return client.post('/auth/signup', json={'email': email, 'password': password})


def test_signup_creates_user():
    response = _signup(email="signup1@example.com")
    assert response.status_code == 201
    body = response.json()
    assert body['email'] == "signup1@example.com"
    assert body['role'] == "user"
    assert body['is_active'] is True
    assert 'password' not in body
    assert 'hashed_password' not in body


def test_signup_ignores_client_supplied_role():
    """Public self-signup can only ever create a "user" account -- role is
    not part of SignupRequest, so a client trying to smuggle role="admin"
    into the request body must be silently ignored (extra/unknown fields
    are dropped by the schema), not honored and not rejected."""
    response = client.post('/auth/signup', json={
        'email': "wouldbeadmin@example.com",
        'password': "supersecret1",
        'role': "admin",
    })
    assert response.status_code == 201
    assert response.json()['role'] == "user"


def test_signup_rejects_duplicate_email():
    _signup(email="dup@example.com")
    response = _signup(email="dup@example.com")
    assert response.status_code == 422
    assert "already registered" in response.json()['detail']


def test_login_succeeds_with_correct_password():
    _signup(email="login1@example.com", password="correcthorse1")
    response = client.post('/auth/login', json={'email': "login1@example.com", 'password': "correcthorse1"})
    assert response.status_code == 200
    body = response.json()
    assert body['token_type'] == "bearer"
    assert body['user']['email'] == "login1@example.com"
    assert body['user']['role'] == "user"

    decoded = jwt.decode(body['access_token'], get_settings().jwt_secret_key, algorithms=["HS256"])
    assert decoded['email'] == "login1@example.com"
    assert decoded['role'] == "user"


def test_login_fails_with_wrong_password():
    _signup(email="login2@example.com", password="correcthorse1")
    response = client.post('/auth/login', json={'email': "login2@example.com", 'password': "wrongpassword"})
    assert response.status_code == 401


def test_login_fails_for_unknown_email():
    response = client.post('/auth/login', json={'email': "nobody@example.com", 'password': "whatever1"})
    assert response.status_code == 401


def test_password_is_hashed_not_stored_plaintext():
    from app.db.database import SessionLocal
    from app.db.models import User

    _signup(email="hashcheck@example.com", password="plaintextpassword1")
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == "hashcheck@example.com").first()
        assert user is not None
        assert user.hashed_password != "plaintextpassword1"
        assert user.hashed_password.startswith("$2b$")  # bcrypt hash prefix
    finally:
        db.close()
