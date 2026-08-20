import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# Folders
# ============================================================

CHUNKED_DATA_DIR = Path("data/chunked")
EMBEDDINGS_DATA_DIR = Path("data/embeddings")


# ============================================================
# Experimental chunk sizes
# These are WORD counts, not tokenizer tokens.
# ============================================================

CHUNK_SIZES = [200, 400, 800]


# ============================================================
# Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")


# ============================================================
# Load datasets for one chunk size
# ============================================================

def load_datasets(chunk_size):
    """
    Load all papers belonging to one chunk-size condition.

    Example:
        chunk_size = 200

    loads:
        p1_rag_200.json
        p2_dpr_200.json
        ...
        p8_colbert_200.json

    and their corresponding embedding files.
    """

    datasets = []

    chunk_files = sorted(
        CHUNKED_DATA_DIR.glob(
            f"*_{chunk_size}.json"
        )
    )

    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk files found for "
            f"{chunk_size}-word condition."
        )

    for chunk_file in chunk_files:

        # ----------------------------------------------------
        # Load chunks
        # ----------------------------------------------------

        with open(
            chunk_file,
            "r",
            encoding="utf-8"
        ) as f:

            chunks = json.load(f)

        # ----------------------------------------------------
        # Corresponding embedding file
        # ----------------------------------------------------

        embedding_file = (
            EMBEDDINGS_DATA_DIR /
            f"{chunk_file.stem}_embeddings.npy"
        )

        if not embedding_file.exists():

            raise FileNotFoundError(
                f"Embedding file not found:\n"
                f"{embedding_file}"
            )

        embeddings = np.load(
            embedding_file
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(chunks) != len(embeddings):

            raise ValueError(
                f"Chunk/embedding mismatch for "
                f"{chunk_file.name}: "
                f"{len(chunks)} chunks vs "
                f"{len(embeddings)} embeddings"
            )

        # ----------------------------------------------------
        # Paper ID
        #
        # Example:
        # p1_rag_200.json
        #
        # becomes:
        # p1_rag
        # ----------------------------------------------------

        paper_id = chunk_file.stem

        suffix = f"_{chunk_size}"

        if paper_id.endswith(suffix):
            paper_id = paper_id[:-len(suffix)]

        datasets.append({

            "name": chunk_file.stem,

            "paper_id": paper_id,

            "chunks": chunks,

            "embeddings": embeddings

        })

    return datasets


# ============================================================
# Retrieve top-k results
# ============================================================

def retrieve(
    query,
    datasets,
    top_k=5
):
    """
    Retrieve the top-k most semantically similar chunks.

    The query is embedded using the same
    SentenceTransformer model used for documents.

    Because both query and document embeddings are
    normalized, their dot product is cosine similarity.
    """

    # --------------------------------------------------------
    # Encode query
    # --------------------------------------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = []

    # --------------------------------------------------------
    # Compare query against every chunk
    # --------------------------------------------------------

    for dataset in datasets:

        chunks = dataset["chunks"]

        embeddings = dataset["embeddings"]

        similarities = (
            embeddings @ query_embedding
        )

        for i, score in enumerate(
            similarities
        ):

            chunk = chunks[i]

            # ------------------------------------------------
            # Support our current chunk format
            # ------------------------------------------------

            page_start = chunk.get(
                "page_start",
                chunk.get("start_page")
            )

            page_end = chunk.get(
                "page_end",
                chunk.get("end_page")
            )

            word_count = chunk.get(
                "word_count",
                chunk.get("token_count")
            )

            results.append({

                "dataset": dataset["name"],

                "paper_id": dataset["paper_id"],

                "chunk_id": chunk["chunk_id"],

                "page_start": page_start,

                "page_end": page_end,

                "word_count": word_count,

                "score": float(score),

                "text": chunk["text"]

            })

    # --------------------------------------------------------
    # Highest similarity first
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# Display retrieval results
# ============================================================

def display_results(
    results,
    chunk_size,
    query
):

    print("\n" + "=" * 80)

    print(
        f"TOP {len(results)} RESULTS "
        f"— {chunk_size}-WORD CHUNKS"
    )

    print("=" * 80)

    print(
        f"\nQuery: {query}"
    )

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            "\n" + "-" * 80
        )

        print(
            f"Rank: {rank}"
        )

        print(
            f"Paper: "
            f"{result['paper_id']}"
        )

        print(
            f"Dataset: "
            f"{result['dataset']}"
        )

        print(
            f"Chunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Pages: "
            f"{result['page_start']}"
            f"-"
            f"{result['page_end']}"
        )

        print(
            f"Word count: "
            f"{result['word_count']}"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print("\nText:")

        print(
            result["text"][:1000]
        )

    print(
        "\n" + "=" * 80
    )


# ============================================================
# Interactive mode
# ============================================================

def interactive_mode():

    print(
        "\nResearchLens Retrieval Engine"
    )

    print(
        "=" * 80
    )

    print(
        "\nAvailable chunk-size conditions:"
    )

    print(
        "1. 200 words"
    )

    print(
        "2. 400 words"
    )

    print(
        "3. 800 words"
    )

    # --------------------------------------------------------
    # Select chunk size
    # --------------------------------------------------------

    while True:

        choice = input(
            "\nSelect chunk size (1/2/3): "
        ).strip()

        if choice == "1":

            selected_chunk_size = 200
            break

        elif choice == "2":

            selected_chunk_size = 400
            break

        elif choice == "3":

            selected_chunk_size = 800
            break

        else:

            print(
                "Invalid choice. "
                "Please enter 1, 2, or 3."
            )

    # --------------------------------------------------------
    # Load selected condition
    # --------------------------------------------------------

    print(
        f"\nLoading "
        f"{selected_chunk_size}-word datasets..."
    )

    datasets = load_datasets(
        selected_chunk_size
    )

    total_chunks = sum(
        len(dataset["chunks"])
        for dataset in datasets
    )

    print(
        f"Loaded {len(datasets)} paper datasets."
    )

    print(
        f"Total chunks in this condition: "
        f"{total_chunks}"
    )

    # --------------------------------------------------------
    # Enter query
    # --------------------------------------------------------

    query = input(
        "\nEnter your research question: "
    ).strip()

    if not query:

        raise ValueError(
            "Query cannot be empty."
        )

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    results = retrieve(
        query,
        datasets,
        top_k=5
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        results,
        selected_chunk_size,
        query
    )


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    interactive_mode()