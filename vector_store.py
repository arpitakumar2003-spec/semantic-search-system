import faiss
import numpy as np

print("Starting vector index creation...")

# Load embeddings
print("Loading embeddings...")
embeddings = np.load("data/embeddings.npy")

print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

print("Adding vectors to index...")
index.add(embeddings)

print("Total vectors in index:", index.ntotal)

# Save index
print("Saving index...")
faiss.write_index(index, "data/vector.index")

print("Vector index saved successfully!")