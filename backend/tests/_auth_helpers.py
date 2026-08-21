"""Shared auth helper for tests that call routes through a TestClient.

Every route except /auth/*, /health, and the annotated-photo image now
requires a bearer token (see app/core/security.py). The seeded default
admin (app/services/auth_service.py::seed_default_admin, run at app
startup -- the same startup that seeds POL-001 and friends, which is why
existing tests could already assume those policies exist) can reach every
route: admin-only routes because it has the admin role, and user-level
routes because require_admin depends on get_current_user first. Logging
test clients in as that account is therefore the one login that covers
both tiers without needing per-test role bookkeeping.
"""
from app.services.auth_service import DEFAULT_ADMIN_EMAIL, DEFAULT_ADMIN_PASSWORD


def admin_auth_headers(client) -> dict:
    response = client.post('/auth/login', json={'email': DEFAULT_ADMIN_EMAIL, 'password': DEFAULT_ADMIN_PASSWORD})
    response.raise_for_status()
    return {'Authorization': f"Bearer {response.json()['access_token']}"}
