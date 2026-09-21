from langchain_chroma import Chroma

from app.llm import get_embeddings
from app.retrieval import retrieve, retrieve_with_mmr
from app.pipeline import answer_question
from app.verifier import  verify_claims
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
    
answer = answer_question(query)
print(f"\n===== ANSWER =====")
print(answer)

verification = verify_claims(query, answer, context)
print(f"\n===== VERIFICATION =====")
print(verification)