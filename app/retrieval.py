from app.vector_store import create_vector_store
from langsmith import traceable

@traceable(name="retrieval", run_type="chain")
def retrieve(vector_store ,query:str ,k:int=5):
    results =vector_store.similarity_search_with_relevance_scores(
        query,
        k=k)
    return results


@traceable(name="retrieval", run_type="chain")
def retrieve_with_mmr(vector_store,query:str,k:int =5):
    result =vector_store.max_marginal_relevance_search(
           query,k=k,fetch_k=20)
    return result