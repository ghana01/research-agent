from app.pipeline import run_question


def main():
    question = input("\nEnter your question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    result = run_question(question, k=5, max_attempts=2)

    print("\n========== FINAL RESULT ==========")
    print("\nDecision:")
    print(result["decision"])

    print("\nAnswer:")
    print(result["answer"])


if __name__ == "__main__":
    main()