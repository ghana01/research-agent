from app.llm import get_llm, get_embeddings
from app.retrieval import retrieve
from app.vector_store import VECTOR_DB_PATH
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate


EVALUATION_QUESTIONS = [
    {
        "question": "What is Nexora's monthly meal allowance?",
        "expected_chunk": "nexora_company_overview_chunk_005",
    },
    {
        "question": "What is Nexora's annual learning allowance?",
        "expected_chunk": "nexora_company_overview_chunk_005",
    },
    {
        "question": "Can Nexora's learning allowance be used to purchase a smartphone?",
        "expected_chunk": "nexora_company_overview_chunk_005",
    },
    {
        "question": "What is Nexora's remote-work limit?",
        "expected_chunk": "nexora_company_overview_chunk_004",
    },
    {
        "question": "Does Nexora provide free lunch at its offices?",
        "expected_chunk": "nexora_company_overview_chunk_005",
    },
]


def transform_query(question: str) -> str:
    llm = get_llm()

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
You are a query transformation system for a RAG pipeline.

Rewrite the user's question into a concise retrieval-friendly search query.

Rules:
1. Preserve the original meaning.
2. Do not answer the question.
3. Do not add facts.
4. Keep important entities, concepts, and constraints.
5. Use terminology likely to appear in the source documents.
6. Return ONLY the rewritten query.

Original question:
{question}

Retrieval query:
"""
    )

    prompt = prompt_template.format(question=question)

    response = llm.invoke(prompt)

    return response.content.strip()


def evaluate_recall(results, expected_chunk, k):
    retrieved_chunks = [
        doc.metadata.get("chunk_id")
        for doc, _score in results[:k]
    ]

    return int(expected_chunk in retrieved_chunks)


def print_results(title, results):
    print(f"\n{title}")

    for rank, (doc, score) in enumerate(results, start=1):
        chunk_id = doc.metadata.get("chunk_id")

        print(
            f"  Rank {rank}: "
            f"{chunk_id} "
            f"(score={score:.4f})"
        )


def main():

    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=get_embeddings(),
    )

    original_recall = {
        1: [],
        3: [],
        5: [],
    }

    transformed_recall = {
        1: [],
        3: [],
        5: [],
    }

    for index, item in enumerate(EVALUATION_QUESTIONS, start=1):

        question = item["question"]
        expected_chunk = item["expected_chunk"]

        print("\n")
        print("=" * 60)
        print(f"QUESTION {index}")
        print("=" * 60)

        print(f"\nQuestion:")
        print(question)

        print(f"\nExpected chunk:")
        print(expected_chunk)

        # -----------------------------------------
        # ORIGINAL QUERY
        # -----------------------------------------

        original_results = retrieve(
            vector_store,
            question,
            k=5,
        )

        print_results(
            "\n--- ORIGINAL QUERY ---",
            original_results,
        )

        # -----------------------------------------
        # TRANSFORM QUERY
        # -----------------------------------------

        transformed_query = transform_query(question)

        print("\nTransformed query:")
        print(transformed_query)

        # -----------------------------------------
        # TRANSFORMED RETRIEVAL
        # -----------------------------------------

        transformed_results = retrieve(
            vector_store,
            transformed_query,
            k=5,
        )

        print_results(
            "\n--- TRANSFORMED QUERY ---",
            transformed_results,
        )

        # -----------------------------------------
        # RECALL
        # -----------------------------------------

        print("\nRecall:")

        for k in [1, 3, 5]:

            original_score = evaluate_recall(
                original_results,
                expected_chunk,
                k,
            )

            transformed_score = evaluate_recall(
                transformed_results,
                expected_chunk,
                k,
            )

            original_recall[k].append(original_score)
            transformed_recall[k].append(transformed_score)

            print(
                f"Recall@{k}: "
                f"Original={original_score} | "
                f"Transformed={transformed_score}"
            )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    print("\n")
    print("=" * 60)
    print("QUERY TRANSFORMATION EVALUATION")
    print("=" * 60)

    for k in [1, 3, 5]:

        original_percentage = (
            sum(original_recall[k])
            / len(original_recall[k])
            * 100
        )

        transformed_percentage = (
            sum(transformed_recall[k])
            / len(transformed_recall[k])
            * 100
        )

        print(
            f"\nRecall@{k}:"
        )

        print(
            f"  Original:    {original_percentage:.2f}%"
        )

        print(
            f"  Transformed: {transformed_percentage:.2f}%"
        )


if __name__ == "__main__":
    main()