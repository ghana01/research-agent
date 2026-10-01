from sentence_transformers import CrossEncoder
from app.vector_store import VECTOR_DB_PATH
from app.retrieval import retrieve
from app.llm import get_embeddings
from langchain_chroma import Chroma


QUESTION = "Does Nexora provide free lunch at its offices?"

# 1. Load our existing vector DB
vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=get_embeddings(),
)

# 2. Retrieve candidates using our existing retrieval
results = retrieve(
    vector_store,
    QUESTION,
    k=5,
)

print("\n========== VECTOR SEARCH ==========")

for rank, (doc, similarity_score) in enumerate(results, start=1):
    chunk_id = doc.metadata.get("chunk_id", "unknown")

    print(f"\nRank {rank}")
    print(f"Chunk: {chunk_id}")
    print(f"Similarity score: {similarity_score}")
    print(f"Content: {doc.page_content}")


# 3. Load reranker
reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# 4. Create query-document pairs
pairs = [
    (QUESTION, doc.page_content)
    for doc, _score in results
]

# 5. Calculate reranker scores
reranker_scores = reranker.predict(pairs)

# 6. Combine documents + scores
reranked = list(
    zip(
        [doc for doc, _score in results],
        reranker_scores,
    )
)

# 7. Sort highest relevance first
reranked.sort(
    key=lambda x: x[1],
    reverse=True,
)

print("\n========== RERANKED RESULTS ==========")

for rank, (doc, score) in enumerate(reranked, start=1):
    chunk_id = doc.metadata.get("chunk_id", "unknown")

    print(f"\nRank {rank}")
    print(f"Chunk: {chunk_id}")
    print(f"Reranker score: {score:.4f}")
    print(f"Content: {doc.page_content}")