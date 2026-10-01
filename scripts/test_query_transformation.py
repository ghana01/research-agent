from app.llm import get_llm
from langchain_core.prompts import PromptTemplate


def transform_query(question: str) -> str:
    llm = get_llm()

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
You are a query transformation system for a RAG pipeline.

Rewrite the user's question into a concise retrieval-friendly search query.

Rules:
1. Preserve the original meaning.
2. Do not answer the question.
3. Do not add facts that are not present in the question.
4. Extract important entities, concepts, and constraints.
5. Use terminology that is likely to appear in a company policy/document.
6. Return ONLY the rewritten query.

Original question:
{question}

Retrieval query:
"""
    )

    prompt = prompt_template.format(question=question)

    response = llm.invoke(prompt)

    return response.content.strip()


if __name__ == "__main__":
    questions = [
        "What is Nexora's monthly meal allowance?",
        "What is Nexora's annual learning allowance?",
        "Can Nexora's learning allowance be used to purchase a smartphone?",
        "What is Nexora's remote-work limit?",
        "Does Nexora provide free lunch at its offices?",
    ]

    print("========== QUERY TRANSFORMATION ==========")

    for i, question in enumerate(questions, start=1):
        transformed = transform_query(question)

        print(f"\n===== QUESTION {i} =====")
        print(f"Original:    {question}")
        print(f"Transformed: {transformed}")