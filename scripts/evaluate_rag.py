from langsmith import Client
from dotenv import load_dotenv
load_dotenv()
from app.pipeline import run_question
from scripts.evaluators import correctness_evaluator


client = Client()


def rag_target(inputs: dict) -> dict:

    question = inputs["question"]

    result = run_question(
        question,
        k=5,
        max_attempts=2
    )

    return {
        "answer": result["answer"],
        "decision": result["decision"],
    }


if __name__ == "__main__":

    results = client.evaluate(
        rag_target,
        data="nexora-rag-v1",
        evaluators=[
            correctness_evaluator
        ],
        experiment_prefix="rag-v1-correctness",
    )

    print(results)