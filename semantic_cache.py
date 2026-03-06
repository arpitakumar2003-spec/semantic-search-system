import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.85):

        print("Loading embedding model for cache...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.cache_queries = []
        self.cache_embeddings = []
        self.cache_results = []

        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0

    def search_cache(self, query):

        if len(self.cache_queries) == 0:
            self.miss_count += 1
            return None, None, 0

        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(
            query_embedding,
            np.array(self.cache_embeddings)
        )[0]

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= self.threshold:

            self.hit_count += 1

            return (
                self.cache_queries[best_idx],
                self.cache_results[best_idx],
                float(best_score)
            )

        self.miss_count += 1
        return None, None, float(best_score)

    def add_to_cache(self, query, result):

        embedding = self.model.encode([query])[0]

        self.cache_queries.append(query)
        self.cache_embeddings.append(embedding)
        self.cache_results.append(result)

    def get_stats(self):

        total = len(self.cache_queries)

        hit_rate = 0
        if total > 0:
            hit_rate = self.hit_count / (self.hit_count + self.miss_count)

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear_cache(self):

        self.cache_queries = []
        self.cache_embeddings = []
        self.cache_results = []

        self.hit_count = 0
        self.miss_count = 0