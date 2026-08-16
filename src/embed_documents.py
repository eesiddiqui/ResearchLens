import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Folders
# --------------------------------------------------

CHUNKED_DATA_DIR = Path("data/chunked")
EMBEDDINGS_DATA_DIR = Path("data/embeddings")

EMBEDDINGS_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")


# --------------------------------------------------
# Process every chunked dataset
# --------------------------------------------------

chunk_files = sorted(CHUNKED_DATA_DIR.glob("*.json"))

print(f"\nFound {len(chunk_files)} chunk files.")


for chunk_file in chunk_files:

    print(f"\nProcessing: {chunk_file.name}")

    # Load chunks
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Extract text
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Output filename
    output_file = (
        EMBEDDINGS_DATA_DIR /
        f"{chunk_file.stem}_embeddings.npy"
    )

    # Save embeddings
    np.save(output_file, embeddings)

    print(
        f"Saved {len(embeddings)} embeddings "
        f"with shape {embeddings.shape}"
    )


print("\nAll embeddings generated successfully!")