from langchain_chroma import Chroma

from app.llm import embeddings

VECTOR_DB_PATH = "./data/chroma"

def create_vector_store(documents):
    vector_store =Chroma.from_documents(
        persist_directory=VECTOR_DB_PATH,
        documents=documents,
        embedding=embeddings)

    return   vector_store