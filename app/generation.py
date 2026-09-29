from langchain_core.prompts import PromptTemplate

from app.llm import get_llm


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
    answer, _ = generate_answer_with_usage(question, context)
    return answer


def generate_answer_with_usage(question, context):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a careful research assistant.

Use ONLY the provided context to answer the question.

If the context does not contain enough information,
say exactly: "I do not have enough information in the provided context to answer that."

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

Answer:
"""
    )

    prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = get_llm().invoke(prompt)
    usage = _extract_usage_metadata(response)
    return response.content, usage