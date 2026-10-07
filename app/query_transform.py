from app.llm import get_llm


def transform_query(question: str) -> str:
	"""Rewrite a user question into a concise retrieval-friendly query."""
	llm = get_llm()
	prompt = f"""
Rewrite the following user question into a concise search query optimized
for retrieving relevant documents from a knowledge base.

Keep the important entities, keywords, and intent.
Do not answer the question.

User question:
{question}

Retrieval query:
"""
	response = llm.invoke(prompt)
	return response.content.strip()
