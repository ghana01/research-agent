from app.vector_store import create_vector_store

def retrieve(vector_store ,query:str ,k:int=5):
    results =vector_store.similarity_search(query,k=k)
    return results