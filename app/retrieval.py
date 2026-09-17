from app.vector_store import create_vector_store

def retrieve(vector_store ,query:str ,k:int=5):
    results =vector_store.similarity_search_with_relevance_scores(
        query,
        k=k)
    return results

def retrieve_with_mmr(vector_store,query:str,k:int =5):
    result =vector_store.max_marginal_relevance_search(
           query,k=k,fetch_k=20)
    return result