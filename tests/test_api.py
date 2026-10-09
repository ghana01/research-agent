
import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError
import pytest
from pydantic import ValidationError

from app.api import AskResponse


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

def test_response_contract_rejects_unknown_answer_status():
    with pytest.raises(ValidationError):
        AskResponse(
            answer="INR 2,400",
            answer_status="UNKNOWN",
            decision="ACCEPT",
        )
        

def test_ask_returns_expected_response_contract(monkeypatch):
    def mock_run_question(question):
        assert question == "What is Nexora's meal allowance?"
        return {
            "answer": "Employees receive INR 2,400.",
            "answer_status": "ANSWERED",
            "decision": "ACCEPT",
        }

    monkeypatch.setattr(api, "run_question", mock_run_question)

    response = client.post(
        "/ask",
        json={"question": "What is Nexora's meal allowance?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Employees receive INR 2,400.",
        "answer_status": "ANSWERED",
        "decision": "ACCEPT",
    }
    assert response.headers.get("X-Request-ID")

def test_ask_returns_abstention_with_200(monkeypatch):
    def mock_run_question(question):
        return {
            "answer": "The provided context does not contain enough information to answer this question.",
            "answer_status": "ABSTAINED",
            "decision": "ACCEPT",
        }

    monkeypatch.setattr(api, "run_question", mock_run_question)

    response = client.post(
        "/ask",
        json={"question": "What is Nexora's office address?"},
    )

    assert response.status_code == 200
    assert response.json()["answer_status"] == "ABSTAINED"
    assert response.json()["decision"] == "ACCEPT"
