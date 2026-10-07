from langchain_community.retrievers import BM25Retriever
from app.ingestion import load_markdown, split_documents


SOURCE = "data/documents/nexora_company_overview.md"


# 1. Load documents
documents = load_markdown(SOURCE)

print(f"Documents loaded: {len(documents)}")


# 2. Split into the same chunks used by our RAG
chunks = split_documents(
    documents,
    source_path=SOURCE,
    file_type="markdown",
)

print(f"Chunks created: {len(chunks)}")


# 3. Create BM25 retriever
bm25 = BM25Retriever.from_documents(chunks)

bm25.k = 5


# 4. Test query
question = input("\nEnter your question: ")


# 5. Retrieve using BM25
results = bm25.invoke(question)


print("\n========== BM25 RESULTS ==========")

for rank, doc in enumerate(results, start=1):

    chunk_id = doc.metadata.get(
        "chunk_id",
        "unknown"
    )

    print(f"\nRank {rank}")
    print(f"Chunk: {chunk_id}")

    print("\nCONTENT:")
    print(doc.page_content)