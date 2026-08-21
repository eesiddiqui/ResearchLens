import csv
from pathlib import Path

from retrieve import load_datasets, retrieve
from ground_truth import (
    load_ground_truth,
    load_ground_truth_pages,
    validate_pages_against_papers,
    is_paper_relevant,
    is_page_relevant,
)


# ============================================================
# Paths
# ============================================================

QUESTIONS_FILE = Path("data/questions.csv")
GROUND_TRUTH_FILE = Path("data/ground_truth.csv")
GROUND_TRUTH_PAGES_FILE = Path("data/ground_truth_pages.csv")

RESULTS_DIR = Path("data/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_RESULTS_FILE = RESULTS_DIR / "retrieval_results.csv"
SUMMARY_RESULTS_FILE = RESULTS_DIR / "evaluation_summary.csv"


# ============================================================
# Experiment configuration
# ============================================================

CHUNK_SIZES = [200, 400, 800]

TOP_K = 5


# ============================================================
# Load questions
# ============================================================

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


# ============================================================
# Check ground truth
# ============================================================

def validate_inputs(questions, ground_truth):

    question_ids = {q["question_id"] for q in questions}

    ground_truth_ids = set(ground_truth.keys())

    missing_ground_truth = question_ids - ground_truth_ids

    if missing_ground_truth:

        raise ValueError(
            "Missing ground truth for: "
            f"{sorted(missing_ground_truth)}"
        )

    extra_ground_truth = ground_truth_ids - question_ids

    if extra_ground_truth:

        print(
            "Warning: ground_truth.csv contains "
            "questions not present in questions.csv: "
            f"{sorted(extra_ground_truth)}"
        )


# ============================================================
# Metric helpers
#
# Every metric is computed TWICE per question:
#
#   *_paper -> the original, coarse notion of relevance: did we
#              retrieve a chunk from the right PAPER?
#
#   *_page  -> the stricter, page-aware notion of relevance: did we
#              retrieve a chunk from the right paper AND does that
#              chunk's page span overlap the annotated answer pages?
#
# Page-aware metrics are the ones that should drive the chunk-size
# conclusions; paper-level metrics are kept alongside them so we can
# see how much the coarse metric was overstating retrieval quality.
# ============================================================

def calculate_recall(results, relevance_fn, k):

    top_results = results[:k]

    for result in top_results:

        if relevance_fn(result):
            return 1

    return 0


def calculate_reciprocal_rank(results, relevance_fn):

    for rank, result in enumerate(results, start=1):

        if relevance_fn(result):
            return 1.0 / rank

    return 0.0


def first_relevant_rank(results, relevance_fn):

    for rank, result in enumerate(results, start=1):

        if relevance_fn(result):
            return rank

    return ""


# ============================================================
# Evaluate one question
# ============================================================

def evaluate_question(question, datasets, ground_truth, ground_truth_pages):

    question_id = question["question_id"]

    query = question["question"]

    acceptable_papers = ground_truth[question_id]["acceptable_papers"]

    page_gt = ground_truth_pages.get(question_id, {})

    results = retrieve(query, datasets, top_k=TOP_K)

    paper_fn = lambda r: is_paper_relevant(r, acceptable_papers)
    page_fn = lambda r: is_page_relevant(r, page_gt)

    metrics = {}

    for suffix, relevance_fn in (("paper", paper_fn), ("page", page_fn)):

        metrics[f"recall_at_1_{suffix}"] = calculate_recall(results, relevance_fn, 1)
        metrics[f"recall_at_3_{suffix}"] = calculate_recall(results, relevance_fn, 3)
        metrics[f"recall_at_5_{suffix}"] = calculate_recall(results, relevance_fn, 5)
        metrics[f"mrr_{suffix}"] = calculate_reciprocal_rank(results, relevance_fn)
        metrics[f"first_relevant_rank_{suffix}"] = first_relevant_rank(results, relevance_fn)

    return {
        "question_id": question_id,
        "question": query,
        "results": results,
        "metrics": metrics,
        "paper_fn": paper_fn,
        "page_fn": page_fn,
    }


# ============================================================
# Save detailed retrieval results
# ============================================================

def save_detailed_results(detailed_rows):

    fieldnames = [
        "question_id",
        "chunk_size",
        "rank",
        "paper_id",
        "dataset",
        "chunk_id",
        "page_start",
        "page_end",
        "word_count",
        "similarity",
        "is_relevant_paper",
        "is_relevant_page",
        "text"
    ]

    with open(DETAILED_RESULTS_FILE, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(detailed_rows)


# ============================================================
# Save summary results
# ============================================================

def save_summary_results(summary_rows):

    fieldnames = [
        "chunk_size",
        "num_questions",
        "recall_at_1_paper",
        "recall_at_1_page",
        "recall_at_3_paper",
        "recall_at_3_page",
        "recall_at_5_paper",
        "recall_at_5_page",
        "mrr_paper",
        "mrr_page",
    ]

    with open(SUMMARY_RESULTS_FILE, "w", encoding="utf-8", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(summary_rows)


# ============================================================
# Print individual question result
# ============================================================

def print_question_result(question_id, chunk_size, metrics):

    print(f"\n{question_id} — {chunk_size}-word chunks")

    print(
        f"Recall@1: paper={metrics['recall_at_1_paper']}  "
        f"page={metrics['recall_at_1_page']}"
    )

    print(
        f"Recall@3: paper={metrics['recall_at_3_paper']}  "
        f"page={metrics['recall_at_3_page']}"
    )

    print(
        f"Recall@5: paper={metrics['recall_at_5_paper']}  "
        f"page={metrics['recall_at_5_page']}"
    )

    print(
        f"MRR: paper={metrics['mrr_paper']:.4f}  "
        f"page={metrics['mrr_page']:.4f}"
    )


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("ResearchLens Retrieval Evaluation (paper-level + page-aware)")
    print("=" * 80)

    # --------------------------------------------------------
    # Load questions and both ground-truth files
    # --------------------------------------------------------

    questions = load_questions()

    print(f"\nLoaded {len(questions)} questions.")

    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)

    print(f"Loaded {len(ground_truth)} paper-level ground-truth entries.")

    ground_truth_pages = load_ground_truth_pages(GROUND_TRUTH_PAGES_FILE)

    print(f"Loaded page-aware ground truth for {len(ground_truth_pages)} questions.")

    validate_inputs(questions, ground_truth)

    consistency_problems = validate_pages_against_papers(
        ground_truth, ground_truth_pages
    )

    if consistency_problems:

        print(
            "\nWARNING: ground_truth.csv and ground_truth_pages.csv "
            "are out of sync:"
        )

        for problem in consistency_problems:
            print(" -", problem)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_summary_rows = []
    all_detailed_rows = []

    # --------------------------------------------------------
    # Run each chunk-size condition
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        print("\n" + "=" * 80)
        print(f"EVALUATING {chunk_size}-WORD CHUNKS")
        print("=" * 80)

        datasets = load_datasets(chunk_size)

        total_chunks = sum(len(dataset["chunks"]) for dataset in datasets)

        print(f"Loaded {len(datasets)} papers.")
        print(f"Total chunks: {total_chunks}")

        per_question_metrics = []

        for question in questions:

            evaluated = evaluate_question(
                question, datasets, ground_truth, ground_truth_pages
            )

            question_id = evaluated["question_id"]
            metrics = evaluated["metrics"]

            per_question_metrics.append(metrics)

            print_question_result(question_id, chunk_size, metrics)

            for rank, retrieved in enumerate(evaluated["results"], start=1):

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
                    "similarity": f"{retrieved['score']:.6f}",
                    "is_relevant_paper": evaluated["paper_fn"](retrieved),
                    "is_relevant_page": evaluated["page_fn"](retrieved),
                    "text": retrieved["text"]

                })

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

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
        print(f"SUMMARY — {chunk_size}-WORD CHUNKS")
        print("-" * 80)
        print(f"Recall@1: paper={summary_row['recall_at_1_paper']}  page={summary_row['recall_at_1_page']}")
        print(f"Recall@3: paper={summary_row['recall_at_3_paper']}  page={summary_row['recall_at_3_page']}")
        print(f"Recall@5: paper={summary_row['recall_at_5_paper']}  page={summary_row['recall_at_5_page']}")
        print(f"MRR:      paper={summary_row['mrr_paper']}  page={summary_row['mrr_page']}")

    # ========================================================
    # Save results
    # ========================================================

    save_detailed_results(all_detailed_rows)
    save_summary_results(all_summary_rows)

    # ========================================================
    # Final comparison
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL CHUNK-SIZE COMPARISON (paper-level / page-aware)")
    print("=" * 80)

    print(
        "\nChunk Size | Recall@1 (p/pg) | Recall@3 (p/pg) | "
        "Recall@5 (p/pg) | MRR (p/pg)"
    )
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
    print("\nEvaluation completed successfully!")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
