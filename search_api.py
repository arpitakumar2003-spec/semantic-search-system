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

print("Loading cluster data...")
with open("data/clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

labels = cluster_data["labels"]

print("Initializing semantic cache...")
cache = SemanticCache()

print("System ready!")


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_api(request: QueryRequest):

    query = request.query

    # 1️⃣ Check semantic cache
    matched_query, cached_result, similarity = cache.search_cache(query)

    if matched_query is not None:

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": matched_query,
            "similarity_score": similarity,
            "result": cached_result,
            "dominant_cluster": None
        }

    # 2️⃣ Embed query
    query_embedding = model.encode([query])

    # 3️⃣ Search vector DB
    D, I = index.search(query_embedding, 5)

    results = []

    for i, idx in enumerate(I[0]):

        text = documents[idx][:200]
        score = float(D[0][i])
        cluster = int(labels[idx])

        results.append({
            "score": score,
            "preview": text,
            "cluster": cluster
        })

    dominant_cluster = results[0]["cluster"]

    # 4️⃣ Add to cache
    cache.add_to_cache(query, results)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": dominant_cluster
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.get_stats()


@app.delete("/cache")
def clear_cache():
    cache.clear_cache()
    return {"message": "Cache cleared"}