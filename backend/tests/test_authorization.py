"""Proves the enforcement in app/core/security.py actually blocks what it
claims to, not just that existing tests were updated to route around it."""
from fastapi.testclient import TestClient

from app.main import app
from _auth_helpers import admin_auth_headers

client = TestClient(app)


def _signup_and_login(email: str, password: str = "supersecret1") -> dict:
    signup = client.post('/auth/signup', json={'email': email, 'password': password})
    assert signup.status_code == 201
    assert signup.json()['role'] == "user"
    login = client.post('/auth/login', json={'email': email, 'password': password})
    assert login.status_code == 200
    return {'Authorization': f"Bearer {login.json()['access_token']}"}


def test_public_routes_need_no_token():
    assert client.get('/health').status_code == 200
    assert client.post('/policies/lookup', json={'policy_number': 'POL-001'}).status_code == 401


def test_protected_route_without_token_is_401():
    response = client.get('/claims')
    assert response.status_code == 401
    assert response.headers.get('www-authenticate') == 'Bearer'


def test_protected_route_with_garbage_token_is_401():
    response = client.get('/claims', headers={'Authorization': 'Bearer not-a-real-token'})
    assert response.status_code == 401


def test_user_role_can_reach_user_level_routes():
    headers = _signup_and_login("plainuser@example.com")
    assert client.post('/policies/lookup', json={'policy_number': 'POL-001'}, headers=headers).status_code == 200
    assert client.get('/claims', headers=headers).status_code == 200


def test_user_role_cannot_reach_admin_only_routes():
    headers = _signup_and_login("nonadmin@example.com")
    assert client.get('/claims/adjuster-dashboard', headers=headers).status_code == 403
    assert client.get('/claims/siu-dashboard', headers=headers).status_code == 403
    assert client.get('/analytics/summary', headers=headers).status_code == 403


def test_admin_role_can_reach_admin_only_routes():
    headers = admin_auth_headers(client)
    assert client.get('/claims/adjuster-dashboard', headers=headers).status_code == 200
    assert client.get('/claims/siu-dashboard', headers=headers).status_code == 200
    assert client.get('/analytics/summary', headers=headers).status_code == 200


def test_signup_cannot_self_assign_admin_role():
    """The core of this fix: a client cannot grant itself the role that
    unlocks the admin-only routes above just by asking for it at signup."""
    response = client.post('/auth/signup', json={
        'email': "selfmadeadmin@example.com",
        'password': "supersecret1",
        'role': "admin",
    })
    assert response.status_code == 201
    assert response.json()['role'] == "user"
