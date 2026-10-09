
import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError

from app import api


client = TestClient(api.app)


def test_openai_timeout_returns_504(monkeypatch):
    def mock_run_question(question):
        request = httpx.Request(
            "POST",
            "https://api.openai.com/v1/chat/completions",
        )
        raise APITimeoutError(request=request)

    monkeypatch.setattr(api, "run_question", mock_run_question)

    response = client.post(
        "/ask",
        json={"question": "What is Nexora's meal allowance?"},
    )

    assert response.status_code == 504
    assert response.json() == {
        "detail": "The AI service timed out. Please try again."
    }


def test_empty_question_returns_422():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422
