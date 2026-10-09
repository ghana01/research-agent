from langchain_chroma import Chroma
from app.config import settings
from app.llm import get_embeddings


VECTOR_DB_PATH = settings.vector_db_path

def create_vector_store(documents):
    vector_store =Chroma.from_documents(
        persist_directory=VECTOR_DB_PATH,
        documents=documents,
        embedding=get_embeddings())

    return   vector_store 