from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever

from app.ingestion import load_markdown, split_documents
from app.llm import get_embeddings
from app.retrieval import retrieve
from app.vector_store import VECTOR_DB_PATH


SOURCE = "data/documents/nexora_company_overview.md"


# ============================================================
# RRF
# ============================================================

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


# ============================================================
# Evaluation questions
# ============================================================

QUESTIONS = [
    {
        "question": "What is Nexora's monthly meal allowance?",
        "expected": "nexora_company_overview_chunk_005",
    },
    {
        "question": "What is Nexora's annual learning allowance?",
        "expected": "nexora_company_overview_chunk_005",
    },
    {
        "question": (
            "Can Nexora's learning allowance be used "
            "to purchase personal electronic devices?"
        ),
        "expected": "nexora_company_overview_chunk_005",
    },
    {
        "question": "How many days per week can employees work remotely?",
        "expected": "nexora_company_overview_chunk_004",
    },
    {
        "question": "Does Nexora provide free lunch at its offices?",
        "expected": "nexora_company_overview_chunk_005",
    },
]


# ============================================================
# Recall@K
# ============================================================

def recall_at(ranking, expected, k):
    return int(expected in ranking[:k])


# ============================================================
# Load documents for BM25
# ============================================================

documents = load_markdown(SOURCE)

print(f"Documents loaded: {len(documents)}")

chunks = split_documents(
    documents,
    source_path=SOURCE,
    file_type="markdown",
)

print(f"Chunks created: {len(chunks)}")


# ============================================================
# Create BM25 retriever
# ============================================================

bm25 = BM25Retriever.from_documents(chunks)
bm25.k = 5


# ============================================================
# Load existing Chroma vector store
# ============================================================

vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=get_embeddings(),
)


# ============================================================
# Store evaluation results
# ============================================================

dense_scores = {
    1: [],
    3: [],
    5: [],
}

bm25_scores = {
    1: [],
    3: [],
    5: [],
}

rrf_scores = {
    1: [],
    3: [],
    5: [],
}


# ============================================================
# Run evaluation
# ============================================================

for index, item in enumerate(QUESTIONS, start=1):

    question = item["question"]
    expected = item["expected"]

    print("\n" + "=" * 80)
    print(f"QUESTION {index}")
    print(f"Question : {question}")
    print(f"Expected : {expected}")

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_results = retrieve(
        vector_store,
        question,
        k=5,
    )

    dense_ranking = [
        doc.metadata["chunk_id"]
        for doc, _score in dense_results
    ]

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_results = bm25.invoke(question)

    bm25_ranking = [
        doc.metadata["chunk_id"]
        for doc in bm25_results
    ]

    # --------------------------------------------------------
    # RRF fusion
    # --------------------------------------------------------

    fused_results = rrf_fusion(
        [
            dense_ranking,
            bm25_ranking,
        ]
    )

    rrf_ranking = [
        chunk_id
        for chunk_id, _score in fused_results
    ]

    # --------------------------------------------------------
    # Print rankings
    # --------------------------------------------------------

    print("\n---------- DENSE ----------")

    for rank, chunk_id in enumerate(
        dense_ranking,
        start=1,
    ):
        marker = "  <-- EXPECTED" if chunk_id == expected else ""
        print(
            f"Rank {rank} | {chunk_id}{marker}"
        )

    print("\n---------- BM25 ----------")

    for rank, chunk_id in enumerate(
        bm25_ranking,
        start=1,
    ):
        marker = "  <-- EXPECTED" if chunk_id == expected else ""
        print(
            f"Rank {rank} | {chunk_id}{marker}"
        )

    print("\n---------- RRF ----------")

    for rank, (chunk_id, score) in enumerate(
        fused_results,
        start=1,
    ):
        marker = "  <-- EXPECTED" if chunk_id == expected else ""
        print(
            f"Rank {rank} | "
            f"{chunk_id} | "
            f"score={score:.6f}"
            f"{marker}"
        )

    # --------------------------------------------------------
    # Calculate Recall@K for this question
    # --------------------------------------------------------

    for k in [1, 3, 5]:

        dense_scores[k].append(
            recall_at(
                dense_ranking,
                expected,
                k,
            )
        )

        bm25_scores[k].append(
            recall_at(
                bm25_ranking,
                expected,
                k,
            )
        )

        rrf_scores[k].append(
            recall_at(
                rrf_ranking,
                expected,
                k,
            )
        )


# ============================================================
# Final evaluation
# ============================================================

print("\n\n")
print("=" * 80)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 80)


def print_metrics(name, scores):

    print(f"\n{name}")

    for k in [1, 3, 5]:

        recall = sum(scores[k]) / len(scores[k])

        print(
            f"Recall@{k}: "
            f"{recall:.2%} "
            f"({sum(scores[k])}/{len(scores[k])})"
        )


print_metrics("DENSE", dense_scores)
print_metrics("BM25", bm25_scores)
print_metrics("RRF HYBRID", rrf_scores)


# ============================================================
# Side-by-side comparison
# ============================================================

print("\n")
print("=" * 80)
print("COMPARISON")
print("=" * 80)

print(
    f"\n{'Metric':<12}"
    f"{'Dense':<12}"
    f"{'BM25':<12}"
    f"{'RRF':<12}"
)

print("-" * 48)

for k in [1, 3, 5]:

    dense_recall = sum(dense_scores[k]) / len(dense_scores[k])
    bm25_recall = sum(bm25_scores[k]) / len(bm25_scores[k])
    rrf_recall = sum(rrf_scores[k]) / len(rrf_scores[k])

    print(
        f"{'Recall@' + str(k):<12}"
        f"{dense_recall:<12.2%}"
        f"{bm25_recall:<12.2%}"
        f"{rrf_recall:<12.2%}"
    )