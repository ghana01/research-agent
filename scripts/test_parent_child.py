from dotenv import load_dotenv
from langsmith import traceable
from langchain_openai import ChatOpenAI

load_dotenv()


# -----------------------------
# CHILD 1: RETRIEVAL
# -----------------------------
@traceable(name="retrieve", run_type="chain")
def retrieve():
    return "RAG retrieves relevant documents and gives them to the LLM as context."


# -----------------------------
# CHILD 2: GENERATION
# -----------------------------
@traceable(name="generate", run_type="chain")
def generate(context):

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
What is RAG?

Answer:
"""

    response = llm.invoke(prompt)

    return response.content


# -----------------------------
# PARENT
# -----------------------------
@traceable(name="rag_pipeline", run_type="chain")
def rag_pipeline():

    context = retrieve()

    answer = generate(context)

    return answer


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    result = rag_pipeline()

    print("\n===== FINAL ANSWER =====")
    print(result)