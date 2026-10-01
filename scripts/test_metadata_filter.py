from langchain_chroma import Chroma

from app.vector_store import VECTOR_DB_PATH
from app.llm import get_embeddings


vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=get_embeddings(),
)

question = "What is Nexora's annual learning allowance?"


print("\n========== WITHOUT FILTER ==========\n")

results = vector_store.similarity_search_with_score(
    question,
    k=5,
)

for rank, (doc, score) in enumerate(results, start=1):
    print(f"Rank {rank}")
    print(f"Chunk: {doc.metadata.get('chunk_id')}")
    print(f"Score: {score:.4f}")
    print()


print("\n========== WITH METADATA FILTER ==========\n")

filtered_results = vector_store.similarity_search_with_score(
    question,
    k=5,
    filter={
        "document_id": "nexora_company_overview"
    },
)

for rank, (doc, score) in enumerate(filtered_results, start=1):
    print(f"Rank {rank}")
    print(f"Chunk: {doc.metadata.get('chunk_id')}")
    print(f"Score: {score:.4f}")
    print()