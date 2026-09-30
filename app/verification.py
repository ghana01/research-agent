from typing import Literal

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from app.llm import get_llm
from langsmith import traceable

def _extract_usage_metadata(response):
    if response is None:
        return {}

    usage = getattr(response, "usage_metadata", None)
    if usage:
        return usage

    metadata = getattr(response, "response_metadata", None) or {}
    if isinstance(metadata, dict):
        for key in ("usage", "token_usage", "usage_metadata"):
            value = metadata.get(key)
            if isinstance(value, dict):
                return value

    return {}


class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED"
    ]
    reason: str
    evidence_chunk_id: str | None = None


class VerificationResult(BaseModel):
    claims: list[ClaimVerification]


def verify_answer(question, answer, context):
    result, _ = verify_answer_with_usage(question, answer, context)
    return result

@traceable(name="verification", run_type="chain")
def verify_answer_with_usage(question, answer, context):
    llm = get_llm().with_structured_output(
        VerificationResult,
        include_raw=True
    )

    prompt_template = PromptTemplate(
        input_variables=[
            "question",
            "context",
            "answer"
        ],
        template="""
You are a factual verification system.

Verify the answer using ONLY the provided context.

Do not use your own knowledge.

Question:
{question}

Context:
{context}

Answer:
{answer}

For every important factual claim in the answer,
determine whether it is:

SUPPORTED:
The context directly supports the complete claim.

PARTIALLY_SUPPORTED:
The context supports only part of the claim.

UNSUPPORTED:
The context does not support the claim.

Return each claim with:
- claim
- SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED
- reason
- evidence_chunk_id: the most relevant chunk ID from the context that supports or refutes the claim. If no chunk clearly supports it, use null.

Important: use only chunk IDs that appear in the context.
"""
    )

    prompt = prompt_template.format(
        question=question,
        context=context,
        answer=answer
    )

    response = llm.invoke(prompt)
    result = response["parsed"]
    raw_response = response["raw"]
    usage = _extract_usage_metadata(raw_response)
    return result, usage


def decide_result(verification_result):

    verdicts = [
        claim.verdict
        for claim in verification_result.claims
    ]

    if "UNSUPPORTED" in verdicts:
        return "REJECT"

    if "PARTIALLY_SUPPORTED" in verdicts:
        return "WARN"

    return "ACCEPT"