from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os


def load_dataset_data():
    """
    Load the 20 Newsgroups dataset.
    Combine train and test splits.
    """
    dataset = load_dataset("SetFit/20_newsgroups")

    docs = list(dataset["train"]["text"]) + list(dataset["test"]["text"])

    return docs


def clean_text(text):
    """
    Basic text cleaning
    """
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.strip()

    return text


def chunk_text(text, chunk_size=150):
    """
    Split large documents into smaller chunks
    for better semantic embedding.
    """

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(words[i:i + chunk_size])

        # ignore very small chunks
        if len(chunk) > 30:
            chunks.append(chunk)

    return chunks


def generate_embeddings():

    print("Loading dataset...")

    docs = load_dataset_data()

    docs = [clean_text(doc) for doc in docs]

    print("Total raw documents:", len(docs))

    # -------- Chunk documents --------

    print("Chunking documents...")

    chunked_docs = []

    for doc in docs:

        chunks = chunk_text(doc)

        chunked_docs.extend(chunks)

    print("Total chunks:", len(chunked_docs))

    # -------- Load embedding model --------

    print("Loading embedding model...")

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # limit maximum tokens processed
    model.max_seq_length = 512

    # -------- Generate embeddings --------

    print("Generating embeddings...")

    embeddings = model.encode(
        chunked_docs,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embedding shape:", embeddings.shape)

    # -------- Save outputs --------

    os.makedirs("data", exist_ok=True)

    print("Saving documents...")

    with open("data/documents.pkl", "wb") as f:
        pickle.dump(chunked_docs, f)

    print("Saving embeddings...")

    np.save("data/embeddings.npy", embeddings)

    print("Embeddings saved successfully!")


if __name__ == "__main__":
    generate_embeddings()