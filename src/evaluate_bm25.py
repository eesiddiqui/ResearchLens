"""
evaluate_bm25.py
=================

Runs the exact same 15-question, page-aware evaluation as evaluate.py,
but against the BM25 sparse baseline instead of dense embeddings.

Same questions, same ground truth (paper-level AND page-aware), same
chunk boundaries, same Recall@1/3/5 + MRR metrics -> the only variable
is retriever type. This makes the dense-vs-sparse and chunk-size
comparisons apples-to-apples.

Output:
    data/evaluation/bm25_retrieval_results.csv
    data/evaluation/bm25_evaluation_summary.csv
"""

import csv
from pathlib import Path

from retrieve_bm25 import load_bm25_index, retrieve
from ground_truth import (
    load_ground_truth,
    load_ground_truth_pages,
    validate_pages_against_papers,
    is_paper_relevant,
    is_page_relevant,
)


QUESTIONS_FILE = Path("data/questions.csv")
GROUND_TRUTH_FILE = Path("data/ground_truth.csv")
GROUND_TRUTH_PAGES_FILE = Path("data/ground_truth_pages.csv")

RESULTS_DIR = Path("data/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_RESULTS_FILE = RESULTS_DIR / "bm25_retrieval_results.csv"
SUMMARY_RESULTS_FILE = RESULTS_DIR / "bm25_evaluation_summary.csv"

CHUNK_SIZES = [200, 400, 800]
TOP_K = 5


def load_questions():

    questions = []

    with open(QUESTIONS_FILE, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:
            questions.append({
                "question_id": row["question_id"].strip(),
                "question": row["question"].strip()
            })

    return questions


def calculate_recall(results, relevance_fn, k):

    for result in results[:k]:
        if relevance_fn(result):
            return 1

    return 0


def calculate_reciprocal_rank(results, relevance_fn):

    for rank, result in enumerate(results, start=1):
        if relevance_fn(result):
            return 1.0 / rank

    return 0.0


def main():

    print("\n" + "=" * 80)
    print("ResearchLens Retrieval Evaluation — BM25 baseline")
    print("=" * 80)

    questions = load_questions()
    print(f"\nLoaded {len(questions)} questions.")

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    ground_truth_pages = load_ground_truth_pages(GROUND_TRUTH_PAGES_FILE)

    problems = validate_pages_against_papers(ground_truth, ground_truth_pages)
    if problems:
        print("\nWARNING: ground_truth.csv and ground_truth_pages.csv are out of sync:")
        for p in problems:
            print(" -", p)

    all_summary_rows = []
    all_detailed_rows = []

    for chunk_size in CHUNK_SIZES:

        print("\n" + "=" * 80)
        print(f"EVALUATING {chunk_size}-WORD CHUNKS (BM25)")
        print("=" * 80)

        chunk_records, index = load_bm25_index(chunk_size)

        print(f"Indexed {len(chunk_records)} chunks "
              f"across {len(set(r['paper_id'] for r in chunk_records))} papers.")

        per_question_metrics = []

        for question in questions:

            question_id = question["question_id"]
            query = question["question"]

            acceptable_papers = ground_truth[question_id]["acceptable_papers"]
            page_gt = ground_truth_pages.get(question_id, {})

            results = retrieve(query, chunk_records, index, top_k=TOP_K)

            paper_fn = lambda r: is_paper_relevant(r, acceptable_papers)
            page_fn = lambda r: is_page_relevant(r, page_gt)

            metrics = {
                "recall_at_1_paper": calculate_recall(results, paper_fn, 1),
                "recall_at_1_page": calculate_recall(results, page_fn, 1),
                "recall_at_3_paper": calculate_recall(results, paper_fn, 3),
                "recall_at_3_page": calculate_recall(results, page_fn, 3),
                "recall_at_5_paper": calculate_recall(results, paper_fn, 5),
                "recall_at_5_page": calculate_recall(results, page_fn, 5),
                "mrr_paper": calculate_reciprocal_rank(results, paper_fn),
                "mrr_page": calculate_reciprocal_rank(results, page_fn),
            }

            per_question_metrics.append(metrics)

            print(
                f"{question_id}: "
                f"R@1(p/pg)={metrics['recall_at_1_paper']}/{metrics['recall_at_1_page']}  "
                f"R@3(p/pg)={metrics['recall_at_3_paper']}/{metrics['recall_at_3_page']}  "
                f"R@5(p/pg)={metrics['recall_at_5_paper']}/{metrics['recall_at_5_page']}  "
                f"MRR(p/pg)={metrics['mrr_paper']:.3f}/{metrics['mrr_page']:.3f}"
            )

            for rank, retrieved in enumerate(results, start=1):

                all_detailed_rows.append({
                    "question_id": question_id,
                    "chunk_size": chunk_size,
                    "rank": rank,
                    "paper_id": retrieved["paper_id"],
                    "dataset": retrieved["dataset"],
                    "chunk_id": retrieved["chunk_id"],
                    "page_start": retrieved["page_start"],
                    "page_end": retrieved["page_end"],
                    "word_count": retrieved["word_count"],
                    "bm25_score": f"{retrieved['score']:.6f}",
                    "is_relevant_paper": paper_fn(retrieved),
                    "is_relevant_page": page_fn(retrieved),
                    "text": retrieved["text"],
                })

        num_questions = len(per_question_metrics)

        def avg(key):
            return sum(m[key] for m in per_question_metrics) / num_questions

        summary_row = {
            "chunk_size": chunk_size,
            "num_questions": num_questions,
            "recall_at_1_paper": f"{avg('recall_at_1_paper'):.4f}",
            "recall_at_1_page": f"{avg('recall_at_1_page'):.4f}",
            "recall_at_3_paper": f"{avg('recall_at_3_paper'):.4f}",
            "recall_at_3_page": f"{avg('recall_at_3_page'):.4f}",
            "recall_at_5_paper": f"{avg('recall_at_5_paper'):.4f}",
            "recall_at_5_page": f"{avg('recall_at_5_page'):.4f}",
            "mrr_paper": f"{avg('mrr_paper'):.4f}",
            "mrr_page": f"{avg('mrr_page'):.4f}",
        }

        all_summary_rows.append(summary_row)

        print("\n" + "-" * 80)
        print(f"SUMMARY — {chunk_size}-WORD CHUNKS (BM25)")
        print("-" * 80)
        print(f"Recall@1: paper={summary_row['recall_at_1_paper']}  page={summary_row['recall_at_1_page']}")
        print(f"Recall@3: paper={summary_row['recall_at_3_paper']}  page={summary_row['recall_at_3_page']}")
        print(f"Recall@5: paper={summary_row['recall_at_5_paper']}  page={summary_row['recall_at_5_page']}")
        print(f"MRR:      paper={summary_row['mrr_paper']}  page={summary_row['mrr_page']}")

    with open(DETAILED_RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(all_detailed_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_detailed_rows)

    with open(SUMMARY_RESULTS_FILE, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(all_summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)

    print("\n" + "=" * 80)
    print("FINAL CHUNK-SIZE COMPARISON — BM25 (paper-level / page-aware)")
    print("=" * 80)
    print("\nChunk Size | Recall@1 (p/pg) | Recall@3 (p/pg) | Recall@5 (p/pg) | MRR (p/pg)")
    print("-" * 90)
    for row in all_summary_rows:
        print(
            f"{row['chunk_size']:>10} | "
            f"{row['recall_at_1_paper']}/{row['recall_at_1_page']} | "
            f"{row['recall_at_3_paper']}/{row['recall_at_3_page']} | "
            f"{row['recall_at_5_paper']}/{row['recall_at_5_page']} | "
            f"{row['mrr_paper']}/{row['mrr_page']}"
        )

    print(f"\nDetailed results saved to:\n  {DETAILED_RESULTS_FILE}")
    print(f"\nSummary results saved to:\n  {SUMMARY_RESULTS_FILE}")
    print("\nBM25 evaluation completed successfully!")


if __name__ == "__main__":
    main()
