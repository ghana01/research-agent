from langchain_community.retrievers import BM25Retriever

from app.ingestion import load_markdown, split_documents
from app.retrieval import retrieve


SOURCE = "data/documents/nexora_company_overview.md"


def create_bm25_retriever():
	documents = load_markdown(SOURCE)

	chunks = split_documents(
		documents,
		source_path=SOURCE,
		file_type="markdown",
	)

	bm25 = BM25Retriever.from_documents(chunks)
	bm25.k = 5

	return bm25


def rrf_fusion(rankings, k=60):
	scores = {}

	for ranking in rankings:
		for rank, document_id in enumerate(ranking, start=1):
			score = 1 / (k + rank)

			scores[document_id] = (
				scores.get(document_id, 0) + score
			)

	return sorted(
		scores.items(),
		key=lambda x: x[1],
		reverse=True,
	)


def hybrid_retrieve(
	vector_store,
	bm25,
	question: str,
	k: int = 5,
):
	dense_results = retrieve(
		vector_store,
		question,
		k=k,
	)

	dense_documents = [
		doc
		for doc, _score in dense_results
	]

	dense_ranking = [
		doc.metadata["chunk_id"]
		for doc in dense_documents
	]

	bm25_results = bm25.invoke(question)

	bm25_ranking = [
		doc.metadata["chunk_id"]
		for doc in bm25_results
	]

	fused_results = rrf_fusion(
		[
			dense_ranking,
			bm25_ranking,
		]
	)[:k]

	document_lookup = {}

	for doc in dense_documents:
		document_lookup[doc.metadata["chunk_id"]] = doc

	for doc in bm25_results:
		document_lookup[doc.metadata["chunk_id"]] = doc

	results = []

	for chunk_id, score in fused_results:
		doc = document_lookup.get(chunk_id)

		if doc is not None:
			results.append((doc, score))

	return results
