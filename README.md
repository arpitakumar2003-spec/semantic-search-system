# Semantic Search System with Fuzzy Clustering & Semantic Cache

AI/ML Engineer Task Submission

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-Semantic%20Search-purple)

---

## Project Overview

This project implements a **semantic search system** over the **20 Newsgroups dataset (~20,000 documents)**.

The system integrates modern machine learning techniques including:

* Vector embeddings
* Fuzzy clustering
* Semantic caching
* FastAPI-based API service

The goal is to demonstrate a **real-world ML system architecture for semantic information retrieval** rather than simply applying a model.

---

# System Architecture

The search pipeline is structured as follows:

```
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
```

This architecture provides:

* fast repeated query responses
* semantic understanding of natural language queries
* scalable vector retrieval

---

# Dataset

Dataset used:

**20 Newsgroups Dataset**

~20,000 newsgroup posts across 20 discussion categories.

Source:

https://archive.ics.uci.edu/dataset/113/twenty+newsgroups

Example topics include:

* Computer hardware
* Politics
* Religion
* Sports
* Science
* Firearms

The dataset contains **highly noisy real-world text**, making it suitable for testing semantic retrieval systems.

---

# Methodology

## 1. Corpus Preparation

The raw dataset contains headers, quotes, and formatting artifacts.

Preprocessing steps:

* text normalization
* whitespace cleanup
* removal of formatting artifacts

### Document Chunking

Long posts are divided into chunks of approximately **120 words**.

```
Raw Document
   ↓
Text Cleaning
   ↓
Chunking (~120 words)
   ↓
Embedding Generation
```

Chunking improves retrieval quality by allowing embeddings to capture **localized semantic context**.

---

# 2. Embedding Model

Documents are embedded using the model:

**BAAI/bge-base-en-v1.5**

Reasons for choosing this model:

* strong performance on semantic retrieval tasks
* efficient inference
* good trade-off between quality and speed

Each document chunk is mapped into a **768-dimensional embedding space**:

```
f : text → ℝ⁷⁶⁸
```

These vectors encode semantic relationships between documents.

---

# 3. Vector Database

Embeddings are stored in a **FAISS vector index**.

FAISS enables efficient **approximate nearest neighbour search**.

Given a query embedding **q**, the system retrieves the most similar document embeddings:

```
NN_k(q) = argmax similarity(q, d_i)
```

Similarity is computed using **cosine similarity**.

This approach allows scalable search across tens of thousands of vectors.

---

# 4. Fuzzy Clustering

Traditional clustering assigns each document to **one cluster only**.

However real-world topics overlap.

Example:

A document discussing **gun legislation** belongs to both:

* politics
* firearms

To model this, the system uses **Fuzzy C-Means clustering**.

### Membership Matrix

The algorithm produces a membership matrix:

```
U ∈ ℝ^(C × N)
```

Where:

* **C** = number of clusters
* **N** = number of documents
* **Uᵢⱼ** = membership probability of document j in cluster i

Each column satisfies:

```
Σ Uᵢⱼ = 1
```

Example distribution:

| Document | Cluster 3 | Cluster 7 | Cluster 12 |
| -------- | --------- | --------- | ---------- |
| Doc 102  | 0.61      | 0.29      | 0.10       |

This means the document primarily belongs to **Cluster 3**, but also shares semantic similarity with other clusters.

This soft assignment better represents **topic overlap in natural language corpora**.

---

# Cluster Interpretation

Manual inspection of clusters reveals meaningful semantic groupings.

| Cluster | Topic                  |
| ------- | ---------------------- |
| 3       | Sports discussions     |
| 6       | Religion debates       |
| 10      | Politics and policy    |
| 11      | Computer hardware      |
| 13      | Firearms discussions   |
| 15      | Technology sales posts |

Documents near cluster boundaries often exhibit **semantic ambiguity**, which is expected in real-world datasets.

---

# 5. Semantic Cache

Traditional caching only works for **exact query matches**.

Example:

```
"What is machine learning?"
"What is ML?"
```

These queries are semantically identical but lexically different.

To address this, the system implements a **semantic cache**.

Process:

```
Query embedding
↓
Compare with cached query embeddings
↓
Cosine similarity
↓
Reuse cached result if similarity > threshold
```

---

## Cache Threshold Selection

The cache uses:

```
SIMILARITY_THRESHOLD = 0.85
```

Experimental observations:

| Threshold | Behaviour                     |
| --------- | ----------------------------- |
| 0.70      | too many incorrect cache hits |
| 0.85      | balanced accuracy and reuse   |
| 0.95      | almost no cache reuse         |

0.85 provides the best trade-off between **accuracy and efficiency**.

---

# API Service

The system exposes its functionality via **FastAPI**.

### POST /query

Example request:

```
{
 "query": "space shuttle launch"
}
```

Example response:

```
{
 "query": "...",
 "cache_hit": false,
 "result": [...],
 "dominant_cluster": 11
}
```

---

### GET /cache/stats

Returns:

```
{
 "total_entries": 5,
 "hit_count": 3,
 "miss_count": 2,
 "hit_rate": 0.6
}
```

---

### DELETE /cache

Clears the semantic cache.

---

# Project Structure

```
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
```

---

# Running the Project

Create virtual environment

```
python -m venv venv
```

Activate

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

Run API

```
uvicorn main:app --reload
```

Open API documentation

```
http://127.0.0.1:8000/docs
```

---

# Example Query

Example search query:

```
space shuttle launch
```

The system retrieves posts discussing:

* NASA shuttle launches
* astronaut training
* space mission discussions

This demonstrates **semantic retrieval beyond keyword matching**.

---

# Conclusion

This project demonstrates how modern machine learning techniques can be integrated to build a **semantic search engine**.

Key contributions include:

* semantic vector search using embeddings
* fuzzy clustering to capture topic overlap
* semantic caching to reduce redundant computation
* a scalable API layer for real-time query processing

The system reflects **real-world ML system design used in modern search and recommendation systems**.

---

# Author

**Arpita Kumar**

AI/ML Engineer Task Submission

GitHub Repository:

```
https://github.com/arpitakumar2003-spec/semantic-search-system
```

---

# Future Improvements

Possible extensions include:

* distributed vector search
* GPU acceleration
* hybrid search (keyword + semantic)
* Redis-based semantic cache
* evaluation metrics for retrieval quality

---
