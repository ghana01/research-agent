from typing import Literal

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from app.llm import get_llm


class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal[
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED"
    ]
    reason: str


class VerificationResult(BaseModel):
    claims: list[ClaimVerification]


def verify_answer(question, answer, context):

    llm = get_llm().with_structured_output(
        VerificationResult
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

Provide a reason for every verdict.
"""
    )

    prompt = prompt_template.format(
        question=question,
        context=context,
        answer=answer
    )

    result = llm.invoke(prompt)

    return result


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