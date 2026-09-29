from typing import Literal

from pydantic import BaseModel

from app.llm import get_llm


class ClaimVerification(BaseModel):
    claim: str
    verdict: Literal["SUPPORTED", "UNSUPPORTED", "PARTIALLY_SUPPORTED"]
    reason: str


class VerificationResult(BaseModel):
    claims: list[ClaimVerification]


def main():
    llm = get_llm().with_structured_output(VerificationResult)

    prompt = """
Context:
RAG combines retrieval with language generation.
A RAG system retrieves relevant information from a knowledge base
and provides that information to an LLM as context.

Answer:
RAG combines retrieval with language generation and uses a database
to train the LLM.

Check the answer against the context.
Identify each factual claim and determine whether it is supported.
"""

    result = llm.invoke(prompt)

    print("TYPE:")
    print(type(result))

    print("\nFULL RESULT:")
    print(result)

    print("\nCLAIMS:")
    for claim in result.claims:
        print("\nClaim:", claim.claim)
        print("Verdict:", claim.verdict)
        print("Reason:", claim.reason)


if __name__ == "__main__":
    main()