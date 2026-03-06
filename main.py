from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

from semantic_cache import SemanticCache


app = FastAPI()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("data/vector.index")

print("Loading documents...")
with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Loading clusters...")
with open("data/clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

cluster_probs = cluster_data["cluster_probs"]

print("Initializing semantic cache...")
cache = SemanticCache()

print("System ready!")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_search(request: QueryRequest):

    query = request.query

    # check cache
    matched_query, result, similarity = cache.search_cache(query)

    if matched_query:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": matched_query,
            "similarity_score": similarity,
            "result": result,
            "dominant_cluster": None
        }

    # compute embedding
    query_embedding = model.encode([query])

    # FAISS search
    D, I = index.search(query_embedding, 1)

    idx = int(I[0][0])

    text = documents[idx][:200]
    score = float(D[0][0])

    dominant_cluster = int(np.argmax(cluster_probs[idx]))

    result = {
        "score": score,
        "preview": text
    }

    cache.add_to_cache(query, result)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0.0,
        "result": result,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.get_stats()


@app.delete("/cache")
def clear_cache():

    cache.clear_cache()

    return {
        "message": "Cache cleared successfully"
    }