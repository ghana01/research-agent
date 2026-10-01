from app.vector_store import VECTOR_DB_PATH
from app.retrieval import retrieve
from app.llm import get_embeddings
from langchain_chroma import Chroma


# Ground-truth retrieval dataset
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


def calculate_recall_at_k(
    retrieved_chunk_ids,
    relevant_chunk_ids,
    k,
):
    top_k_ids = retrieved_chunk_ids[:k]

    return int(
        any(
            chunk_id in top_k_ids
            for chunk_id in relevant_chunk_ids
        )
    )


def main():
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )

    k_values = [1, 3, 5]

    totals = {
        k: 0
        for k in k_values
    }

    print("\n========== RETRIEVAL EVALUATION ==========\n")

    for index, item in enumerate(EVALUATION_DATASET, start=1):

        question = item["question"]
        relevant_chunks = item["relevant_chunks"]

        results = retrieve(
            vector_store,
            question,
            k=5,
        )

        retrieved_chunk_ids = [
            doc.metadata.get("chunk_id", "unknown")
            for doc, _score in results
        ]

        print(f"===== QUESTION {index} =====")
        print(f"Question: {question}")

        print("\nExpected relevant chunks:")
        for chunk_id in relevant_chunks:
            print(f"  - {chunk_id}")

        print("\nRetrieved chunks:")
        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            start=1,
        ):
            print(f"  {rank}. {chunk_id}")

        print("\nRecall:")

        for k in k_values:
            recall = calculate_recall_at_k(
                retrieved_chunk_ids,
                relevant_chunks,
                k,
            )

            totals[k] += recall

            status = "PASS" if recall else "FAIL"

            print(
                f"  Recall@{k}: "
                f"{recall} ({status})"
            )

        print()

    total_questions = len(EVALUATION_DATASET)

    print("========== SUMMARY ==========\n")

    for k in k_values:
        recall = totals[k] / total_questions

        print(
            f"Recall@{k}: "
            f"{recall:.2%}"
        )


if __name__ == "__main__":
    main()