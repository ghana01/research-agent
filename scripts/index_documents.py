from app.ingestion import  load_markdown,load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vector_store import create_vector_store


documents = load_markdown("data/documents/rag_notes.md")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(documents)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")

create_vector_store(chunks)

print("Documents embedded and stored successfully.")