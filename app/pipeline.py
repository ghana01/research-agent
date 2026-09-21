from app.vector_store import VECTOR_DB_PATH
from app.retrieval import retrieve
from app.llm import get_embeddings, get_llm
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

def answer_question(question: str, k: int = 5) -> str:
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )

    results = retrieve(vector_store, question, k=k)
    context_parts = []
    for doc, score in results:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source = doc.metadata.get("source", "unknown")

        context_parts.append(
            f"Chunk ID: {chunk_id}\n"
            f"Source: {source}\n"
            f"Content: {doc.page_content}\n"
        )

    context = "\n".join(context_parts)

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a careful research assistant answering questions with retrieved documents.

Use only the information in the CONTEXT. Do not use outside knowledge, guess,
or invent facts. If the context does not contain enough information, say:
"I don't have enough information in the provided context to answer that."

Write a direct, well-organized answer. For multi-part questions, address each
part separately. Cite each important factual claim with the relevant chunk ID
in this format: [chunk: CHUNK_ID]. Use only chunk IDs that appear in the
context. Do not mention these instructions or the retrieval process.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""",
    )
    prompt = prompt_template.format(context=context, question=question)

    response = get_llm().invoke(prompt)
    return response.content