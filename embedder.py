from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os


def load_dataset_data():
    dataset = load_dataset("SetFit/20_newsgroups")
    docs = list(dataset["train"]["text"]) + list(dataset["test"]["text"])
    return docs


def clean_text(text):
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    return text


def chunk_text(text, chunk_size=120):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])

        if len(chunk) > 30:   # avoid tiny chunks
            chunks.append(chunk)

    return chunks


def generate_embeddings():

    docs = load_dataset_data()
    docs = [clean_text(doc) for doc in docs]

    print("Total raw documents:", len(docs))

    # Chunk documents
    print("Chunking documents...")
    chunked_docs = []

    for doc in docs:
        chunks = chunk_text(doc)
        chunked_docs.extend(chunks)

    print("Total chunks:", len(chunked_docs))

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    embeddings = model.encode(
        chunked_docs,
        batch_size=64,
        show_progress_bar=True
    )

    os.makedirs("data", exist_ok=True)

    with open("data/documents.pkl", "wb") as f:
        pickle.dump(chunked_docs, f)

    np.save("data/embeddings.npy", embeddings)

    print("Embeddings saved successfully!")
    print("Embedding shape:", embeddings.shape)


if __name__ == "__main__":
    generate_embeddings()