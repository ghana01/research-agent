
from app.retrieval import retrieve
from langchain_chroma import Chroma
from app.llm import get_embeddings
from app.retrieval import retrieve


EVAL_SET = [
    {
        "question": "What is retrieval augmented generation?",
        "expected_keywords": ["retrieval", "generation"],
    },
    {
        "question": "Why do we use chunking in RAG?",
        "expected_keywords": ["chunk"],
    },
    {
        "question": "What is the difference between similarity search and MMR?",
        "expected_keywords": ["MMR", "similarity"],
    },
    {
        "question": "What causes stale information in a RAG system?",
        "expected_keywords": ["stale", "document"],
    },
    {
        "question": "What role does observability play in a RAG system?",
        "expected_keywords": ["observability"],
    },
]

VECTOR_DB_PATH = "./data/chroma"

vector_store = Chroma(
    persist_directory=VECTOR_DB_PATH,
    embedding_function=get_embeddings(),
)

for eval_index in range(len(EVAL_SET)):
    question =EVAL_SET[eval_index]["question"]
    expected_keywords =EVAL_SET[eval_index]["expected_keywords"]
    
    print(f"\n===== EVALUATION {eval_index + 1} =====")
    print(f"Question: {question}")
    print(f"Expected Keywords: {expected_keywords}")
    
    print("\n========== SIMILARITY SEARCH ==========")

    results = retrieve(
            vector_store,
            question,
            k=5,
        )
    retrieved_text = ""
    for i, (doc, score) in enumerate(results):

        print(f"\n===== RESULT {i + 1} =====")
        print(f"Score: {score}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print(f"Source: {doc.metadata.get('source')}")

        print("\nCONTENT:")
        print(doc.page_content)

    for doc, score in results:
        retrieved_text += doc.page_content.lower()

    hits = 0

    for keyword in expected_keywords:
        if keyword.lower() in retrieved_text:
            hits += 1

    print(f"Matched keywords: {hits}/{len(expected_keywords)}")
    if hits == len(expected_keywords):
        print("Hit@5: 1")
    else:
        print("Hit@5: 0")
