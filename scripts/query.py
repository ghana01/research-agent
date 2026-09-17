from langchain_chroma import Chroma

from app.llm import get_embeddings
from app.retrieval import retrieve, retrieve_with_mmr


VECTOR_DB_PATH = "./data/chroma"


vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=get_embeddings(),
)

collection = vector_store._collection

print("Total stored chunks:", collection.count())
query = input("\nEnter your question: ")


print("\n========== SIMILARITY SEARCH ==========")

results = retrieve(
    vector_store,
    query,
    k=5,
)

for i, (doc, score) in enumerate(results):

    print(f"\n===== RESULT {i + 1} =====")
    print(f"Score: {score}")
    print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
    print(f"Source: {doc.metadata.get('source')}")

    print("\nCONTENT:")
    print(doc.page_content)

print("\n========== MMR SEARCH ==========")

results_mmr = retrieve_with_mmr(
    vector_store,
    query,
    k=5,
)

for i, doc in enumerate(results_mmr):
    print(f"\n===== RESULT {i + 1} =====")
    print(doc.page_content)
    print("\nMETADATA:")
    print(doc.metadata)