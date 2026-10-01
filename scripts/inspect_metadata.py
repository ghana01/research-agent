from langchain_chroma import Chroma

from app.vector_store import VECTOR_DB_PATH
from app.llm import get_embeddings


vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=get_embeddings(),
)

data = vector_store.get(limit=5)

print("\n========== CHROMA METADATA ==========\n")

for i, metadata in enumerate(data["metadatas"], start=1):
    print(f"Document {i}")
    print(metadata)
    print()