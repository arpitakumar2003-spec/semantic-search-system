import pickle
import numpy as np


DATA_DIR = "data"


def load_documents():
    with open(f"{DATA_DIR}/documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return documents


def load_clusters():
    with open(f"{DATA_DIR}/clusters.pkl", "rb") as f:
        cluster_data = pickle.load(f)
    return cluster_data


def show_cluster_samples(samples_per_cluster=3):

    documents = load_documents()
    cluster_data = load_clusters()

    membership = cluster_data["membership"]
    n_clusters = cluster_data["n_clusters"]

    print("\nCluster Interpretation\n")

    for cluster_id in range(n_clusters):

        print(f"\n===== Cluster {cluster_id} =====")

        # sort documents by membership probability
        doc_indices = np.argsort(membership[cluster_id])[::-1][:samples_per_cluster]

        for idx in doc_indices:
            print("\nDocument snippet:")
            print(documents[idx][:200])


def show_boundary_documents(threshold=0.6, limit=10):

    documents = load_documents()
    cluster_data = load_clusters()

    membership = cluster_data["membership"]

    print("\nBoundary Documents\n")

    count = 0

    for i in range(membership.shape[1]):

        probs = membership[:, i]

        if np.max(probs) < threshold:

            print("\nDocument:")
            print(documents[i][:200])

            print("Cluster probabilities:")
            print(probs[:5])

            count += 1

            if count >= limit:
                break


if __name__ == "__main__":

    show_cluster_samples()
    show_boundary_documents()