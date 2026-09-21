from typing import Literal

from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate

from app.llm import get_llm


class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal["SUPPORTED", "UNSUPPORTED"]
    reason: str


class VerificationResult(BaseModel):
    claims: list[ClaimVerification]


def verify_claims(question: str, answer: str, context: str):
    prompt_template = PromptTemplate(
        input_variables=["question", "answer", "context"],
        template="""You are a strict factual grounding verifier.

Your job is to examine the answer claim by claim.

Question:
{question}

Context:
{context}

Answer:
{answer}

Instructions:

1. Identify every factual claim made in the answer.
2. Check each claim ONLY against the provided context.
3. Mark a claim SUPPORTED only when the context explicitly supports it.
4. Mark a claim UNSUPPORTED when the context does not explicitly support it.
5. Do not use your own knowledge.
6. Do not treat assumptions, implications, or common sense as evidence.
7. Check every factual claim in the answer.

For each claim, provide:
- the claim
- SUPPORTED or UNSUPPORTED
- a short reason
"""
    )

    prompt = prompt_template.format(
        question=question,
        answer=answer,
        context=context,
    )

    llm = get_llm().with_structured_output(VerificationResult)

    result = llm.invoke(prompt)

    return result