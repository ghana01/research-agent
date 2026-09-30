from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)


def correctness_evaluator(
    inputs,
    outputs,
    reference_outputs
):
    question = inputs["question"]
    actual_answer = outputs["answer"]
    expected_answer = reference_outputs["expected_answer"]

    prompt = f"""
You are evaluating a RAG system.

Determine whether the actual answer correctly answers
the question according to the expected answer.

Question:
{question}

Expected answer:
{expected_answer}

Actual answer:
{actual_answer}

Return exactly this format:

VERDICT: PASS
REASON: <short explanation>

or

VERDICT: FAIL
REASON: <short explanation>

PASS means the actual answer is factually consistent
with the expected answer and answers the question.

FAIL means the actual answer is incorrect, incomplete,
or does not answer the question.

Do not use outside knowledge.
"""

    response = llm.invoke(prompt)

    text = response.content.strip()

    verdict = "PASS" if "VERDICT: PASS" in text else "FAIL"

    reason = ""

    if "REASON:" in text:
        reason = text.split("REASON:", 1)[1].strip()

    return {
        "key": "correctness",
        "score": 1 if verdict == "PASS" else 0,
        "value": verdict,
        "comment": reason,
    }