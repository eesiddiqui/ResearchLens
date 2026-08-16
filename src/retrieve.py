import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

CHUNKED_DATA_DIR = Path("data/chunked")
EMBEDDINGS_DATA_DIR = Path("data/embeddings")


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded!")


# --------------------------------------------------
# Load all chunk datasets
# --------------------------------------------------

def load_datasets():

    datasets = []

    chunk_files = sorted(CHUNKED_DATA_DIR.glob("*.json"))

    for chunk_file in chunk_files:

        embedding_file = (
            EMBEDDINGS_DATA_DIR /
            f"{chunk_file.stem}_embeddings.npy"
        )

        if not embedding_file.exists():
            print(f"Skipping {chunk_file.name}: embedding missing")
            continue

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        embeddings = np.load(embedding_file)

        datasets.append({
            "name": chunk_file.stem,
            "chunks": chunks,
            "embeddings": embeddings
        })

    return datasets


# --------------------------------------------------
# Retrieve top-k results
# --------------------------------------------------

def retrieve(query, datasets, top_k=5):

    # Convert query to embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = []

    for dataset in datasets:

        chunks = dataset["chunks"]
        embeddings = dataset["embeddings"]

        # Because embeddings are normalized,
        # dot product = cosine similarity
        similarities = embeddings @ query_embedding

        for i, score in enumerate(similarities):

            results.append({
                "dataset": dataset["name"],
                "chunk_id": chunks[i]["chunk_id"],
                "page": chunks[i]["page"],
                "score": float(score),
                "text": chunks[i]["text"]
            })

    # Sort by similarity
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# Main
# --------------------------------------------------

datasets = load_datasets()

print(f"\nLoaded {len(datasets)} datasets.")


query = input("\nEnter your research question: ")

results = retrieve(
    query,
    datasets,
    top_k=5
)


print("\n" + "=" * 80)
print("TOP 5 RETRIEVED RESULTS")
print("=" * 80)


for rank, result in enumerate(results, start=1):

    print(f"\nRank {rank}")
    print(f"Dataset: {result['dataset']}")
    print(f"Page: {result['page']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Similarity: {result['score']:.4f}")

    print("\nText:")
    print(result["text"][:1000])

    print("-" * 80)