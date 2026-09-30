from langsmith import Client
from app.pipeline import run_question

from dotenv import load_dotenv
load_dotenv()
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

    dataset_name = "nexora-rag-v1"

    experiment_results = client.evaluate(
        rag_target,
        data=dataset_name,
        experiment_prefix="rag-v1-baseline",
    )

    print(experiment_results)