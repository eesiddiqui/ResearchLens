"""
reevaluate_page_aware.py
=========================

Re-scores an EXISTING retrieval_results.csv (already produced by
src/evaluate.py) against the new page-aware ground truth, without
needing to re-run embeddings/retrieval.

Purpose: sanity-check whether the current dense-retrieval evaluation
numbers (paper-level recall/MRR) are actually valid, or whether they
were inflated by "right paper, wrong part of the paper" matches.

For each retrieved chunk we compute two relevance flags:
    is_relevant_paper -> the original, coarse check
    is_relevant_page  -> the chunk's page span overlaps the annotated
                         ground-truth pages for that paper

We then recompute Recall@1 / Recall@3 / Recall@5 / MRR under BOTH
definitions, per chunk size, so the two can be compared side by side.

Usage:
    python src/reevaluate_page_aware.py
"""

import csv
from pathlib import Path
from collections import defaultdict

from ground_truth import (
    load_ground_truth,
    load_ground_truth_pages,
    validate_pages_against_papers,
    is_paper_relevant,
    is_page_relevant,
)


RESULTS_DIR = Path("data/evaluation")
DENSE_RESULTS_FILE = RESULTS_DIR / "retrieval_results.csv"
OUTPUT_FILE = RESULTS_DIR / "evaluation_summary_page_aware.csv"

CHUNK_SIZES = [200, 400, 800]
TOP_K = 5


def load_dense_results():
    """
    Load the existing per-chunk-size, per-question, ranked retrieval
    results already saved to disk by src/evaluate.py.

    Returns:
        { (question_id, chunk_size): [row, row, ...] }  (rank order)
    """

    grouped = defaultdict(list)

    with open(DENSE_RESULTS_FILE, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            row["chunk_size"] = int(row["chunk_size"])
            row["rank"] = int(row["rank"])
            row["page_start"] = (
                int(row["page_start"]) if row["page_start"] not in ("", None) else None
            )
            row["page_end"] = (
                int(row["page_end"]) if row["page_end"] not in ("", None) else None
            )

            key = (row["question_id"], row["chunk_size"])
            grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda r: r["rank"])

    return grouped


def recall_at_k(ranked_rows, relevance_fn, k):

    for row in ranked_rows[:k]:
        if relevance_fn(row):
            return 1

    return 0


def reciprocal_rank(ranked_rows, relevance_fn):

    for i, row in enumerate(ranked_rows, start=1):
        if relevance_fn(row):
            return 1.0 / i

    return 0.0


def main():

    print("=" * 80)
    print("Re-scoring existing dense retrieval results: paper-level vs page-aware")
    print("=" * 80)

    ground_truth = load_ground_truth()
    ground_truth_pages = load_ground_truth_pages()

    problems = validate_pages_against_papers(ground_truth, ground_truth_pages)

    if problems:
        print("\nWARNING: ground_truth.csv and ground_truth_pages.csv disagree:")
        for p in problems:
            print(" -", p)
        print()

    dense_results = load_dense_results()

    if not dense_results:
        raise FileNotFoundError(
            f"No existing results found at {DENSE_RESULTS_FILE}. "
            "Run src/evaluate.py first."
        )

    question_ids = sorted({q for q, _ in dense_results.keys()})

    summary_rows = []

    print(
        f"\n{'Chunk':>6} | {'Recall@1':>18} | {'Recall@3':>18} | "
        f"{'Recall@5':>18} | {'MRR':>18}"
    )
    print(
        f"{'Size':>6} | {'paper   page':>18} | {'paper   page':>18} | "
        f"{'paper   page':>18} | {'paper   page':>18}"
    )
    print("-" * 100)

    for chunk_size in CHUNK_SIZES:

        r1_paper, r3_paper, r5_paper, mrr_paper = [], [], [], []
        r1_page, r3_page, r5_page, mrr_page = [], [], [], []

        for question_id in question_ids:

            ranked_rows = dense_results.get((question_id, chunk_size))

            if not ranked_rows:
                continue

            acceptable_papers = ground_truth[question_id]["acceptable_papers"]
            page_gt = ground_truth_pages.get(question_id, {})

            paper_fn = lambda row: is_paper_relevant(row, acceptable_papers)
            page_fn = lambda row: is_page_relevant(row, page_gt)

            r1_paper.append(recall_at_k(ranked_rows, paper_fn, 1))
            r3_paper.append(recall_at_k(ranked_rows, paper_fn, 3))
            r5_paper.append(recall_at_k(ranked_rows, paper_fn, 5))
            mrr_paper.append(reciprocal_rank(ranked_rows, paper_fn))

            r1_page.append(recall_at_k(ranked_rows, page_fn, 1))
            r3_page.append(recall_at_k(ranked_rows, page_fn, 3))
            r5_page.append(recall_at_k(ranked_rows, page_fn, 5))
            mrr_page.append(reciprocal_rank(ranked_rows, page_fn))

        n = len(r1_paper)

        def avg(values):
            return sum(values) / len(values) if values else 0.0

        row = {
            "chunk_size": chunk_size,
            "num_questions": n,
            "recall_at_1_paper": f"{avg(r1_paper):.4f}",
            "recall_at_1_page": f"{avg(r1_page):.4f}",
            "recall_at_3_paper": f"{avg(r3_paper):.4f}",
            "recall_at_3_page": f"{avg(r3_page):.4f}",
            "recall_at_5_paper": f"{avg(r5_paper):.4f}",
            "recall_at_5_page": f"{avg(r5_page):.4f}",
            "mrr_paper": f"{avg(mrr_paper):.4f}",
            "mrr_page": f"{avg(mrr_page):.4f}",
        }

        summary_rows.append(row)

        print(
            f"{chunk_size:>6} | "
            f"{avg(r1_paper):>8.4f} {avg(r1_page):>8.4f} | "
            f"{avg(r3_paper):>8.4f} {avg(r3_page):>8.4f} | "
            f"{avg(r5_paper):>8.4f} {avg(r5_page):>8.4f} | "
            f"{avg(mrr_paper):>8.4f} {avg(mrr_page):>8.4f}"
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:

        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
