from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

dataset = client.create_dataset(
    dataset_name="nexora-rag-v1"
)

examples = [
    {
        "question": "What is Nexora's annual learning allowance?",
        "expected_answer": "INR 18,000 per year."
    },
    {
        "question": "What is Nexora's monthly meal allowance?",
        "expected_answer": "INR 2,400 per month."
    },
    {
        "question": "What is Nexora's remote-work limit?",
        "expected_answer": "Up to 3 days per week."
    },
    {
        "question": "Does Nexora provide free lunch at its offices?",
        "expected_answer": "The provided context does not contain enough information to determine this."
    },
    {
        "question": "Can Nexora's learning allowance be used to purchase a smartphone?",
        "expected_answer": "No. The allowance cannot be used to purchase personal electronic devices."
    }
]

for example in examples:
    client.create_example(
        inputs={
            "question": example["question"]
        },
        outputs={
            "expected_answer": example["expected_answer"]
        },
        dataset_id=dataset.id,
    )

print("Dataset created:", dataset.id)