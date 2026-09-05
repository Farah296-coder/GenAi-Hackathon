import chromadb
from sentence_transformers import SentenceTransformer


# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to the existing vector database
client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_collection(
    name="pdf_chunks"
)


def search_question(question, top_k=3):
    # Convert the question into a vector
    question_embedding = model.encode(question)

    # Search for the closest chunks
    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=top_k
    )

    return results


def print_results(results):
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    print("\n========== Relevant Context ==========\n")

    for i, (document, metadata) in enumerate(
        zip(documents, metadatas),
        start=1
    ):
        print(f"Result {i}")
        print(f"Page: {metadata['page']}")
        print(f"Text:\n{document}")
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = search_question(question)

    print_results(results)