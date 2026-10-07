from app.pipeline import run_question
def main():
    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            print("Please enter a question.")
            continue

        result = run_question(question, k=5, max_attempts=2)

        print("\n========== FINAL RESULT ==========")
        print("\nDecision:")
        print(result["decision"])

        print("\nAnswer:")
        print(result["answer"])

        if "trace" in result:
            result["trace"].print_trace()


if __name__ == "__main__":
    main()