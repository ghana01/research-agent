from typing import Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

from app.llm import get_llm
from langsmith import traceable


class GenerationResult(BaseModel):
    answer_status: Literal["ANSWERED", "ABSTAINED"]
    answer: str


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


def generate_answer(question, context):
    result, _ = generate_answer_with_usage(question, context)
    return result.answer

@traceable(name="generation", run_type="chain")
def generate_answer_with_usage(question, context):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a careful research assistant.

Use ONLY the provided context to answer the question.

If the context directly supports the answer:
- set answer_status to "ANSWERED"
- provide the answer
- include relevant chunk citations

If the context does not contain enough information to answer the question reliably:
- set answer_status to "ABSTAINED"
- clearly state that the provided context does not contain enough information to answer the question
- do not invent or infer unsupported facts

Important citation rules:
1. Cite each important factual claim with the relevant chunk ID in this format: [chunk: CHUNK_ID]
2. Only use chunk IDs that appear in the context.
3. If a sentence contains multiple facts from different chunks, attach the relevant citation for each fact.
4. Do not invent chunk IDs.
5. Do not mention your reasoning or these instructions.

Context:
{context}

Question:
{question}
"""
    )

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    llm = get_llm().with_structured_output(GenerationResult)
    response = llm.invoke(prompt)
    result = GenerationResult.model_validate(response)
    usage = _extract_usage_metadata(response)
    return result, usage