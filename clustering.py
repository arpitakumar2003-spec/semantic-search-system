import numpy as np
import pickle
import skfuzzy as fuzz


DATA_DIR = "data"


def load_embeddings():
    """
    Load document embeddings and original documents.
    """
    embeddings = np.load(f"{DATA_DIR}/embeddings.npy")

    with open(f"{DATA_DIR}/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    return embeddings, documents


def perform_clustering(n_clusters=20):
    """
    Perform fuzzy clustering on document embeddings.
    Each document receives a probability distribution across clusters.
    """

    print("Loading embeddings...")
    embeddings, documents = load_embeddings()

    print("Running fuzzy clustering...")

    # scikit-fuzzy expects data in shape (features, samples)
    data = embeddings.T

    cntr, membership, _, _, _, _, _ = fuzz.cluster.cmeans(
        data,
        c=n_clusters,
        m=2.0,
        error=0.005,
        maxiter=1000,
        init=None
    )

    print("Fuzzy clustering complete!")

    cluster_data = {
        "centroids": cntr,
        "membership": membership,
        "n_clusters": n_clusters,
        "n_documents": embeddings.shape[0]
    }

    with open(f"{DATA_DIR}/clusters.pkl", "wb") as f:
        pickle.dump(cluster_data, f)

    print("Clusters saved!")


def load_clusters():
    """
    Load saved cluster data.
    """
    with open(f"{DATA_DIR}/clusters.pkl", "rb") as f:
        return pickle.load(f)


def get_dominant_cluster(doc_index, membership_matrix):
    """
    Returns the dominant cluster for a document.
    """
    return int(np.argmax(membership_matrix[:, doc_index]))


def get_cluster_distribution(doc_index, membership_matrix):
    """
    Returns the probability distribution of a document across clusters.
    """
    return membership_matrix[:, doc_index]


def find_boundary_documents(membership_matrix, threshold=0.6):
    """
    Find documents that lie near cluster boundaries.
    These are documents without strong membership in any single cluster.
    """
    boundary_docs = []

    for i in range(membership_matrix.shape[1]):

        probs = membership_matrix[:, i]

        if np.max(probs) < threshold:
            boundary_docs.append(i)

    return boundary_docs


def print_cluster_samples(n_samples=3):
    """
    Print example documents from each cluster to interpret cluster meaning.
    """

    embeddings, documents = load_embeddings()
    cluster_data = load_clusters()

    membership = cluster_data["membership"]
    n_clusters = cluster_data["n_clusters"]

    print("\nCluster Interpretation\n")

    for cluster_id in range(n_clusters):

        print(f"\n===== Cluster {cluster_id} =====")

        # documents with highest membership
        doc_indices = np.argsort(membership[cluster_id])[::-1][:n_samples]

        for idx in doc_indices:
            print("\nDoc snippet:")
            print(documents[idx][:200])


if __name__ == "__main__":
    perform_clustering()