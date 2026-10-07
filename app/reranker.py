import time

from sentence_transformers import CrossEncoder


RERANKER_MODEL = "BAAI/bge-reranker-base"

_reranker = None


def get_reranker():
    global _reranker

    if _reranker is None:
        start = time.perf_counter()
        _reranker = CrossEncoder(RERANKER_MODEL)
        elapsed = time.perf_counter() - start
        print(f"CrossEncoder initialization: {elapsed:.3f} s")

    return _reranker


def rerank(question, results, top_n=None):
    """
    Rerank retrieved documents using a CrossEncoder.

    Args:
        question: Original user question.
        results: List of (Document, retrieval_score).
        top_n: Number of documents to keep after reranking.

    Returns:
        List of (Document, reranker_score).
    """

    if not results:
        return []

    reranker = get_reranker()

    pairs = [
        (question, doc.page_content)
        for doc, _score in results
    ]

    start = time.perf_counter()
    scores = reranker.predict(pairs)
    elapsed = time.perf_counter() - start
    print(f"CrossEncoder.predict(): {elapsed:.3f} s")

    reranked = list(
        zip(
            [doc for doc, _score in results],
            scores,
        )
    )

    reranked.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    if top_n is not None:
        reranked = reranked[:top_n]

    return reranked