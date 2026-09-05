import json
import os
import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load extracted PDF JSON
# -----------------------------

def load_pages(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


# -----------------------------
# 2. Chunking
# -----------------------------

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(pages):
    all_chunks = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        chunks = chunk_text(text)

        for chunk in chunks:

            all_chunks.append({
                "page": page_number,
                "text": chunk
            })

    return all_chunks


# -----------------------------
# 3. Embedding Model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 4. Vector Database
# -----------------------------

client = chromadb.PersistentClient(
    path="vector_db"
)

collection = client.get_or_create_collection(
    name="pdf_chunks"
)


# -----------------------------
# 5. Store Chunks
# -----------------------------

def store_chunks(chunks):

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts)

    for i, chunk in enumerate(chunks):

        collection.upsert(
            ids=[f"chunk_{i}"],

            embeddings=[
                embeddings[i].tolist()
            ],

            documents=[
                chunk["text"]
            ],

            metadatas=[
                {
                    "page": chunk["page"]
                }
            ]
        )


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    json_path = os.path.join(
        "output",
        "extracted_text.json"
    )

    pages = load_pages(json_path)

    print(f"Loaded {len(pages)} pages.")

    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks.")

    store_chunks(chunks)

    print("Embeddings stored successfully!")