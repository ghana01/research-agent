from app.ingestion import  load_markdown,load_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.llm import embeddings
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from app.vector_store import create_vector_store

from app.retrieval import retrieve


#Load documents

documents = load_markdown("data/documents/rag_notes.md")

print(f"Loaded {len(documents)} documents.")

#2 chunking the documents into smaller pieces

splitter =RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks =splitter.split_documents(documents)

print(f"Number of chunks: {len(chunks)}")

#store the chunks in a vector store
vector_store = create_vector_store(chunks)

#query the vector store
query ="What is retrieval-augmented generation?"

results =retrieve(vector_store,query,k=5)

for i, doc in enumerate(results):
    print(f"\n===== RESULT {i + 1} =====")
    print(doc.page_content)
    print("\nMETADATA:")
    print(doc.metadata)