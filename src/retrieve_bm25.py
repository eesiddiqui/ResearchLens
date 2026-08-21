"""
retrieve_bm25.py
=================

Sparse-retrieval counterpart to retrieve.py. Loads the SAME chunk
files used by the dense pipeline (data/chunked/*_{chunk_size}.json)
so the BM25 baseline is directly comparable: identical chunk
boundaries, identical questions, identical ground truth -- the only
thing that changes is how a chunk gets scored against a query.

One BM25 index is built per chunk-size condition, pooled across all
papers (matching how retrieve.py pools embeddings across all papers
for a given chunk size before ranking).
"""

import json
from pathlib import Path

from bm25 import BM25Okapi


CHUNKED_DATA_DIR = Path("data/chunked")


def load_bm25_index(chunk_size):
    """
    Build one pooled BM25 index for a chunk-size condition.

    Returns:
        chunk_records: list of dicts (paper_id, dataset, chunk fields)
                       in the same order as the BM25 index's doc ids
        index: BM25Okapi instance
    """

    chunk_files = sorted(CHUNKED_DATA_DIR.glob(f"*_{chunk_size}.json"))

    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk files found for {chunk_size}-word condition."
        )

    chunk_records = []
    texts = []

    for chunk_file in chunk_files:

        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        paper_id = chunk_file.stem
        suffix = f"_{chunk_size}"

        if paper_id.endswith(suffix):
            paper_id = paper_id[: -len(suffix)]

        for chunk in chunks:

            page_start = chunk.get("page_start", chunk.get("start_page"))
            page_end = chunk.get("page_end", chunk.get("end_page"))
            word_count = chunk.get("word_count", chunk.get("token_count"))

            chunk_records.append({
                "dataset": chunk_file.stem,
                "paper_id": paper_id,
                "chunk_id": chunk["chunk_id"],
                "page_start": page_start,
                "page_end": page_end,
                "word_count": word_count,
                "text": chunk["text"],
            })

            texts.append(chunk["text"])

    index = BM25Okapi(texts)

    return chunk_records, index


def retrieve(query, chunk_records, index, top_k=5):

    top = index.top_k(query, k=top_k)

    results = []

    for doc_id, score in top:

        record = chunk_records[doc_id]

        results.append({
            "dataset": record["dataset"],
            "paper_id": record["paper_id"],
            "chunk_id": record["chunk_id"],
            "page_start": record["page_start"],
            "page_end": record["page_end"],
            "word_count": record["word_count"],
            "score": float(score),
            "text": record["text"],
        })

    return results
