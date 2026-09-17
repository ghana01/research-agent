from app.ingestion import load_markdown, split_documents
from app.vector_store import create_vector_store


SOURCE = "data/documents/rag_notes.md"

documents = load_markdown(SOURCE)

print(f"Documents loaded: {len(documents)}")

chunks = split_documents(
    documents,
    source_path=SOURCE,
    file_type="markdown",
)

print(f"Chunks created: {len(chunks)}")

for chunk in chunks[:3]:
    print("\n========== CHUNK ==========")
    print(chunk.page_content[:300])
    print("Metadata:", chunk.metadata)


create_vector_store(chunks)

print("Documents embedded and stored successfully.")