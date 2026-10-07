def rrf_fusion(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            score = 1 / (k + rank)
            scores[document_id] = scores.get(document_id, 0) + score

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )


dense_results = [
    "chunk_A",
    "chunk_B",
    "chunk_C",
    "chunk_D",
    "chunk_E",
]

bm25_results = [
    "chunk_X",
    "chunk_Y",
    "chunk_B",
    "chunk_Z",
    "chunk_E",
]

fused_results = rrf_fusion(
    [dense_results, bm25_results]
)

print("\n========== RRF RESULTS ==========")

for rank, (document_id, score) in enumerate(fused_results, start=1):
    print(
        f"Rank {rank} | "
        f"Chunk: {document_id} | "
        f"RRF score: {score:.6f}"
    )