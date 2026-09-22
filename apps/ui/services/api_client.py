import requests

from apps.ui.config import API_BASE_URL


class SessionExpiredError(RuntimeError):
    """The API rejected the current bearer token."""


def _authorization_headers(
    access_token: str,
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
    }


def _raise_for_authenticated_request(
    response: requests.Response,
) -> None:
    if response.status_code == 401:
        raise SessionExpiredError(
            "Session expired"
        )

    response.raise_for_status()


def login(
    email: str,
    password: str,
) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/auth/login",
        json={
            "email": email,
            "password": password,
        },
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def get_api_health() -> dict:

    response = requests.get(
        f"{API_BASE_URL}/api/v1/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()

def get_model_health() -> dict:

    response = requests.get(
        f"{API_BASE_URL}/api/v1/health/model",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()

def send_chat(
    messages: list[dict],
    session_id: str,
    access_token: str,
) -> dict:

    response = requests.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={
            "messages": messages,
            "session_id": session_id,
        },
        headers=_authorization_headers(
            access_token
        ),
        timeout=120,
    )

    _raise_for_authenticated_request(
        response
    )

    return response.json()


def submit_approval(
    session_id: str,
    approved: bool,
    comment: str | None,
    access_token: str,
) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/chat/approval",
        json={
            "session_id": session_id,
            "approved": approved,
            "comment": comment,
        },
        headers=_authorization_headers(access_token),
        timeout=120,
    )
    _raise_for_authenticated_request(response)
    return response.json()


def create_failed_transaction_report(
    *,
    days: int,
    access_token: str,
) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/jobs/failed-transactions",
        json={"days": days},
        headers=_authorization_headers(access_token),
        timeout=30,
    )
    _raise_for_authenticated_request(response)
    return response.json()


def get_background_job(
    *,
    job_id: str,
    access_token: str,
) -> dict:
    response = requests.get(
        f"{API_BASE_URL}/api/v1/jobs/{job_id}",
        headers=_authorization_headers(access_token),
        timeout=10,
    )
    _raise_for_authenticated_request(response)
    return response.json()


def cancel_background_job(
    *,
    job_id: str,
    access_token: str,
) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/jobs/{job_id}/cancel",
        headers=_authorization_headers(access_token),
        timeout=10,
    )
    _raise_for_authenticated_request(response)
    return response.json()
