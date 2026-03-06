from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy as fuzz
import numpy as np

app = FastAPI()

# Sample documents
documents = [
    "Machine learning is a field of AI",
    "Deep learning is part of machine learning",
    "Python is widely used for data science",
    "Neural networks are used in deep learning",
    "Data science uses statistics and machine learning"
]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents).toarray()

# Fuzzy C-Means clustering
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X.T, c=2, m=2, error=0.005, maxiter=1000
)

class Query(BaseModel):
    query: str


@app.post("/query")
def search(query: Query):

    # Convert query to vector
    q_vec = vectorizer.transform([query.query]).toarray()

    # Compute similarity with documents
    similarity = np.dot(X, q_vec.T).flatten()

    # Get best document index
    best_index = np.argmax(similarity)

    return {
        "query": query.query,
        "best_match": documents[best_index],
        "score": float(similarity[best_index])
    }