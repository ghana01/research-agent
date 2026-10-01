from sentence_transformers import CrossEncoder

from app.vector_store import VECTOR_DB_PATH
from app.retrieval import retrieve
from app.llm import get_embeddings
from langchain_chroma import Chroma


# ---------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------

EVALUATION_DATASET = [
    {
        "question": "What is Nexora's monthly meal allowance?",
        "relevant_chunks": [
            "nexora_company_overview_chunk_005"
        ],
    },
    {
        "question": "What is Nexora's annual learning allowance?",
        "relevant_chunks": [
            "nexora_company_overview_chunk_005"
        ],
    },
    {
        "question": "Can Nexora's learning allowance be used to purchase a smartphone?",
        "relevant_chunks": [
            "nexora_company_overview_chunk_005"
        ],
    },
    {
        "question": "What is Nexora's remote-work limit?",
        "relevant_chunks": [
            "nexora_company_overview_chunk_004"
        ],
    },
    {
        "question": "Does Nexora provide free lunch at its offices?",
        "relevant_chunks": [
            "nexora_company_overview_chunk_005"
        ],
    },
]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_recall(ranked_chunks, relevant_chunks, k):
    top_k = ranked_chunks[:k]

    return int(
        any(
            chunk_id in relevant_chunks
            for chunk_id in top_k
        )
    )


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("Loading vector database...")

    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )

    print("Loading reranker...")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-base"
    )

    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []

    print("\n========== RERANKER EVALUATION ==========")

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1
    ):

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        print(f"\n===== QUESTION {index} =====")

        print(f"Question: {question}")

        print("\nExpected relevant chunks:")

        for chunk_id in relevant_chunks:
            print(f"  - {chunk_id}")

        # -------------------------------------------------
        # Stage 1: Vector retrieval
        # -------------------------------------------------

        results = retrieve(
            vector_store,
            question,
            k=5,
        )

        # -------------------------------------------------
        # Stage 2: Reranking
        # -------------------------------------------------

        pairs = [
            (question, doc.page_content)
            for doc, _score in results
        ]

        scores = reranker.predict(pairs)

        reranked_results = list(
            zip(
                results,
                scores,
            )
        )

        reranked_results.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        ranked_chunks = [
            doc.metadata.get(
                "chunk_id",
                "unknown"
            )
            for (doc, _original_score), _reranker_score
            in reranked_results
        ]

        # -------------------------------------------------
        # Print reranked results
        # -------------------------------------------------

        print("\nReranked chunks:")

        for rank, (
            (doc, original_score),
            reranker_score
        ) in enumerate(
            reranked_results,
            start=1
        ):

            chunk_id = doc.metadata.get(
                "chunk_id",
                "unknown"
            )

            print(
                f"  {rank}. {chunk_id}"
            )

            print(
                f"     Vector score: {original_score}"
            )

            print(
                f"     Reranker score: "
                f"{reranker_score:.4f}"
            )

        # -------------------------------------------------
        # Recall
        # -------------------------------------------------

        r1 = calculate_recall(
            ranked_chunks,
            relevant_chunks,
            1,
        )

        r3 = calculate_recall(
            ranked_chunks,
            relevant_chunks,
            3,
        )

        r5 = calculate_recall(
            ranked_chunks,
            relevant_chunks,
            5,
        )

        recall_at_1.append(r1)
        recall_at_3.append(r3)
        recall_at_5.append(r5)

        print("\nRecall:")

        print(
            f"  Recall@1: {r1} "
            f"({'PASS' if r1 else 'FAIL'})"
        )

        print(
            f"  Recall@3: {r3} "
            f"({'PASS' if r3 else 'FAIL'})"
        )

        print(
            f"  Recall@5: {r5} "
            f"({'PASS' if r5 else 'FAIL'})"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    total = len(EVALUATION_DATASET)

    avg_recall_at_1 = (
        sum(recall_at_1) / total
    )

    avg_recall_at_3 = (
        sum(recall_at_3) / total
    )

    avg_recall_at_5 = (
        sum(recall_at_5) / total
    )

    print("\n========== SUMMARY ==========")

    print(
        f"Recall@1: "
        f"{avg_recall_at_1 * 100:.2f}%"
    )

    print(
        f"Recall@3: "
        f"{avg_recall_at_3 * 100:.2f}%"
    )

    print(
        f"Recall@5: "
        f"{avg_recall_at_5 * 100:.2f}%"
    )


if __name__ == "__main__":
    main()