from fastapi.testclient import TestClient

from app.main import app


def test_signup_creates_account_with_the_signup_fields():
    with TestClient(app) as client:
        response = client.post(
            "/auth/signup",
            json={
                "username": "new_claimant",
                "email": "new_claimant@example.com",
                "password": "safe-password",
                "full_name": "New Claimant",
            },
        )

    assert response.status_code == 201
    assert response.json()["username"] == "new_claimant"
    assert "role" not in response.json()
