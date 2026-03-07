from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import time

app = FastAPI(title="Semantic Search System")

# Allow requests from index.html / local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading embedding model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Loading FAISS index...")
index = faiss.read_index("data/vector.index")

print("Loading documents...")
with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print("Loading cluster data...")
with open("data/clusters.pkl", "rb") as f:
    cluster_data = pickle.load(f)

# Fuzzy C-Means output: membership matrix of shape (n_clusters, n_documents)
membership_matrix = cluster_data["membership"]

# ----------------------------
# Semantic Cache
# ----------------------------
cache = {}
hit_count = 0
miss_count = 0
SIMILARITY_THRESHOLD = 0.85


# ----------------------------
# Request Schema
# ----------------------------
class QueryRequest(BaseModel):
    query: str


# ----------------------------
# Helper Functions
# ----------------------------
def get_dominant_cluster(doc_index: int) -> int:
    return int(np.argmax(membership_matrix[:, doc_index]))


def get_cluster_confidence(doc_index: int) -> float:
    return float(np.max(membership_matrix[:, doc_index]))


# ----------------------------
# Root Endpoint
# ----------------------------
@app.get("/")
def home():
    return {"message": "Semantic Search API is running"}


# ----------------------------
# Query Endpoint
# ----------------------------
@app.post("/query")
def query_system(request: QueryRequest):
    global hit_count, miss_count

    start_time = time.time()
    query = request.query.strip()

    if not query:
        return {
            "query": query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": [],
            "dominant_cluster": None,
            "latency_ms": 0.0
        }

    # BGE models work better with a retrieval prefix for queries
    query_text = "Represent this sentence for searching relevant passages: " + query
    query_embedding = model.encode([query_text], normalize_embeddings=True)

    # ----------------------------
    # Cache Check
    # ----------------------------
    best_cached_query = None
    best_cached_result = None
    best_cached_cluster = None
    best_similarity = -1.0

    for cached_query, cached_data in cache.items():
        similarity = float(np.dot(query_embedding, cached_data["embedding"].T)[0][0])

        if similarity > best_similarity:
            best_similarity = similarity
            best_cached_query = cached_query
            best_cached_result = cached_data["result"]
            best_cached_cluster = cached_data["cluster"]

    if best_similarity >= SIMILARITY_THRESHOLD:
        hit_count += 1
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": best_cached_query,
            "similarity_score": round(best_similarity, 4),
            "result": best_cached_result,
            "dominant_cluster": best_cached_cluster,
            "latency_ms": latency_ms
        }

    # ----------------------------
    # Vector Search
    # ----------------------------
    D, I = index.search(query_embedding.astype("float32"), 5)

    results = []
    for rank, idx in enumerate(I[0]):
        idx = int(idx)
        results.append({
            "rank": rank + 1,
            "score": float(D[0][rank]),
            "preview": documents[idx][:300],
            "cluster": get_dominant_cluster(idx),
            "cluster_confidence": round(get_cluster_confidence(idx), 4)
        })

    dominant_cluster = results[0]["cluster"] if results else None

    miss_count += 1

    # ----------------------------
    # Store in Cache
    # ----------------------------
    cache[query] = {
        "embedding": query_embedding,
        "result": results,
        "cluster": dominant_cluster
    }

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": results,
        "dominant_cluster": dominant_cluster,
        "latency_ms": latency_ms
    }


# ----------------------------
# Cache Stats
# ----------------------------
@app.get("/cache/stats")
def cache_stats():
    total_entries = len(cache)
    total_requests = hit_count + miss_count
    hit_rate = round(hit_count / total_requests, 4) if total_requests > 0 else 0.0

    return {
        "total_entries": total_entries,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_rate
    }


# ----------------------------
# Clear Cache
# ----------------------------
@app.delete("/cache")
def clear_cache():
    global cache, hit_count, miss_count

    cache = {}
    hit_count = 0
    miss_count = 0

    return {"message": "Cache cleared successfully"}