from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")

query = "Does Nexora provide free lunch at its offices?"

documents = [
    "Nexora operates offices across several cities and follows a hybrid work model.",
    "Employees receive a monthly meal allowance of INR 2,400.",
    "Employees may work remotely up to 3 days per week.",
]

pairs = [(query, document) for document in documents]

scores = model.predict(pairs)

ranked = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)

for rank, (document, score) in enumerate(ranked, start=1):
    print(f"\nRank {rank}")
    print(f"Score: {score:.4f}")
    print(f"Document: {document}")