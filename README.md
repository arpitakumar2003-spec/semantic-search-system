Semantic Search System with Fuzzy Clustering & Semantic Cache
AI/ML Engineer Task Submission
This project implements a lightweight semantic search system built on the 20 Newsgroups dataset (~20,000 documents).
The system combines:
•	Vector embeddings
•	Fuzzy clustering
•	Semantic caching
•	FastAPI service
to enable efficient semantic search over noisy textual data.
The architecture is designed to demonstrate real-world ML system design, not just model usage.
________________________________________
System Overview
The search pipeline works as follows:
User Query
   ↓
Query Embedding
   ↓
Semantic Cache Lookup
   ↓ (cache miss)
Vector Search (FAISS)
   ↓
Cluster Analysis
   ↓
Top Results Returned
This design ensures:
•	fast repeated queries
•	semantic understanding of search
•	scalable vector retrieval
________________________________________
Dataset
Dataset used:
20 Newsgroups dataset
~20,000 newsgroup posts across 20 topics
Source:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
Example topics include:
•	computer hardware
•	politics
•	religion
•	sports
•	science
•	firearms
The dataset contains highly noisy raw text, making it suitable for testing semantic retrieval systems.
________________________________________
Part 1 — Embedding & Vector Database
Documents are embedded using:
BAAI/bge-base-en-v1.5
This model was chosen because:
•	high semantic search performance
•	strong performance on retrieval benchmarks
•	efficient inference
•	good balance between quality and speed
Each document is chunked before embedding to improve retrieval quality.
Example pipeline:
Raw document
   ↓
Text cleaning
   ↓
Chunking (~120 words)
   ↓
Embedding generation
Embeddings are stored in a FAISS vector index for fast similarity search.
FAISS enables:
•	approximate nearest neighbor search
•	efficient similarity retrieval
•	scalable vector databases
________________________________________
Part 2 — Fuzzy Clustering
Traditional clustering assigns each document to one cluster only.
However real text topics often overlap.
Example:
Gun legislation discussion
belongs to both:
•	politics
•	firearms
To handle this, the system uses Fuzzy C-Means clustering.
Library used:
scikit-fuzzy
Instead of a single label, each document receives a probability distribution across clusters.
Example:
Document A
Cluster 3 → 0.62
Cluster 6 → 0.31
Cluster 9 → 0.07
This better reflects the true semantic structure of the corpus.
________________________________________
Cluster Interpretation
Cluster inspection shows meaningful semantic groupings.
Examples observed:
Cluster	Topic
3	sports discussions
6	religion debates
10	politics & policy
11	computer hardware
13	firearms discussion
15	technology sales posts
Boundary documents often belong to multiple clusters with similar probabilities, indicating semantic ambiguity.
These cases are important because they show model uncertainty, which is expected in real-world text data.
________________________________________
Part 3 — Semantic Cache
Traditional caching only works if the query is identical.
Example:
"What is machine learning?"
"What is ML?"
These are different strings but semantically identical queries.
This system implements a semantic cache.
Process:
Query embedding
↓
Compare with cached query embeddings
↓
Cosine similarity check
↓
Reuse cached result if similar enough
________________________________________
Cache Similarity Threshold
The cache uses a tunable parameter:
SIMILARITY_THRESHOLD = 0.85
Threshold experiments show:
Threshold	Behaviour
0.70	many incorrect cache hits
0.85	balanced reuse vs accuracy
0.95	almost no cache reuse
0.85 provides a good balance between accuracy and efficiency.
________________________________________
Part 4 — FastAPI Service
The system exposes a REST API using FastAPI.
API endpoints:
________________________________________
POST /query
Accepts a natural language query.
Example request:
{
 "query": "space shuttle launch"
}
Example response:
{
 "query": "space shuttle launch",
 "cache_hit": false,
 "matched_query": null,
 "similarity_score": null,
 "result": [...],
 "dominant_cluster": 11
}
Returned information:
•	whether the query hit the cache
•	similar cached query if available
•	similarity score
•	top search results
•	dominant cluster
________________________________________
GET /cache/stats
Returns cache statistics:
{
 "total_entries": 5,
 "hit_count": 3,
 "miss_count": 2,
 "hit_rate": 0.6
}
________________________________________
DELETE /cache
Clears the semantic cache.
{
 "message": "Cache cleared successfully"
}
________________________________________
Technologies Used
•	Python
•	FastAPI
•	FAISS
•	SentenceTransformers
•	scikit-fuzzy
•	NumPy
•	HuggingFace Datasets
________________________________________
Project Structure
semantic_search_system/

embedder.py
vector_store.py
clustering.py
cluster_analysis.py
semantic_cache.py
main.py
requirements.txt

data/
 embeddings.npy
 vector.index
 documents.pkl
 clusters.pkl
________________________________________
Running the Project
Create virtual environment:
python -m venv venv
Activate environment:
venv\Scripts\activate
Install dependencies:
pip install -r requirements.txt
Run API:
uvicorn main:app --reload
Open API docs:
http://127.0.0.1:8000/docs
________________________________________
Example Query
space shuttle launch
Example result:
•	NASA space shuttle discussions
•	astronaut training descriptions
•	space mission posts
The system successfully retrieves semantically relevant content even when keywords differ.
________________________________________
Design Decisions
Key design decisions made:
1.	Use BGE embeddings for strong semantic retrieval
2.	Use FAISS for scalable vector search
3.	Use fuzzy clustering to capture overlapping topics
4.	Build semantic cache without external caching systems
5.	Provide real-time API with FastAPI
These decisions reflect typical production ML system architecture.
________________________________________
Future Improvements
Potential improvements include:
•	distributed vector search
•	async inference
•	GPU acceleration
•	Redis-based semantic cache
•	hybrid search (keyword + semantic)
________________________________________
Conclusion
This project demonstrates how to build a production-style semantic search system combining:
•	embeddings
•	vector databases
•	fuzzy clustering
•	semantic caching
•	API deployment
The system shows how modern ML techniques can be integrated into a scalable search architecture.
________________________________________
Author
Arpita Kumar
AI/ML Engineer Task Submission
________________________________________
Submission
GitHub Repository:
https://github.com/arpitakumar2003-spec/semantic-search-system
Project API runs using FastAPI and can be tested through the interactive documentation.

