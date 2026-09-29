from langchain_core.prompts import PromptTemplate

from app.llm import get_llm


def generate_answer(question, context):

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.

Use ONLY the provided context to answer the question.

If the context does not contain enough information,
say that you do not have enough information.

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

    return response.content