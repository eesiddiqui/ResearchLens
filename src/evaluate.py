import csv
import json
from pathlib import Path

import numpy as np

from retrieve import load_datasets, retrieve


# ============================================================
# Paths
# ============================================================

QUESTIONS_FILE = Path("data/questions.csv")
GROUND_TRUTH_FILE = Path("data/ground_truth.csv")

RESULTS_DIR = Path("data/evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETAILED_RESULTS_FILE = (
    RESULTS_DIR / "retrieval_results.csv"
)

SUMMARY_RESULTS_FILE = (
    RESULTS_DIR / "evaluation_summary.csv"
)


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

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            questions.append({
                "question_id": row["question_id"].strip(),
                "question": row["question"].strip()
            })

    return questions


# ============================================================
# Load ground truth
# ============================================================

def load_ground_truth():

    ground_truth = {}

    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "question_id",
            "primary_papers",
            "acceptable_papers"
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:

            raise ValueError(
                "ground_truth.csv is missing "
                f"columns: {sorted(missing)}"
            )

        for row in reader:

            question_id = (
                row["question_id"].strip()
            )

            primary_papers = {
                paper.strip()
                for paper in
                row["primary_papers"].split(";")
                if paper.strip()
            }

            acceptable_papers = {
                paper.strip()
                for paper in
                row["acceptable_papers"].split(";")
                if paper.strip()
            }

            ground_truth[question_id] = {
                "primary_papers": primary_papers,
                "acceptable_papers": acceptable_papers
            }

    return ground_truth


# ============================================================
# Check ground truth
# ============================================================

def validate_inputs(
    questions,
    ground_truth
):

    question_ids = {
        q["question_id"]
        for q in questions
    }

    ground_truth_ids = set(
        ground_truth.keys()
    )

    missing_ground_truth = (
        question_ids - ground_truth_ids
    )

    if missing_ground_truth:

        raise ValueError(
            "Missing ground truth for: "
            f"{sorted(missing_ground_truth)}"
        )

    extra_ground_truth = (
        ground_truth_ids - question_ids
    )

    if extra_ground_truth:

        print(
            "Warning: ground_truth.csv contains "
            "questions not present in questions.csv: "
            f"{sorted(extra_ground_truth)}"
        )


# ============================================================
# Check whether a retrieved result is relevant
# ============================================================

def is_relevant(
    result,
    acceptable_papers
):

    return (
        result["paper_id"]
        in acceptable_papers
    )


# ============================================================
# Calculate Recall@K
# ============================================================

def calculate_recall(
    results,
    acceptable_papers,
    k
):

    top_results = results[:k]

    for result in top_results:

        if is_relevant(
            result,
            acceptable_papers
        ):

            return 1

    return 0


# ============================================================
# Calculate Reciprocal Rank
# ============================================================

def calculate_reciprocal_rank(
    results,
    acceptable_papers
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_relevant(
            result,
            acceptable_papers
        ):

            return 1.0 / rank

    return 0.0


# ============================================================
# Evaluate one question
# ============================================================

def evaluate_question(
    question,
    datasets,
    ground_truth
):

    question_id = question["question_id"]

    query = question["question"]

    acceptable_papers = (
        ground_truth[question_id]
        ["acceptable_papers"]
    )

    results = retrieve(
        query,
        datasets,
        top_k=TOP_K
    )

    recall_at_1 = calculate_recall(
        results,
        acceptable_papers,
        1
    )

    recall_at_3 = calculate_recall(
        results,
        acceptable_papers,
        3
    )

    recall_at_5 = calculate_recall(
        results,
        acceptable_papers,
        5
    )

    reciprocal_rank = (
        calculate_reciprocal_rank(
            results,
            acceptable_papers
        )
    )

    first_relevant_rank = ""

    for rank, result in enumerate(
        results,
        start=1
    ):

        if is_relevant(
            result,
            acceptable_papers
        ):

            first_relevant_rank = rank
            break

    return {
        "question_id": question_id,

        "question": query,

        "recall_at_1": recall_at_1,

        "recall_at_3": recall_at_3,

        "recall_at_5": recall_at_5,

        "mrr": reciprocal_rank,

        "first_relevant_rank":
            first_relevant_rank,

        "results": results
    }


# ============================================================
# Save detailed retrieval results
# ============================================================

def save_detailed_results(
    detailed_rows
):

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
        "is_relevant",
        "text"
    ]

    with open(
        DETAILED_RESULTS_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            detailed_rows
        )


# ============================================================
# Save summary results
# ============================================================

def save_summary_results(
    summary_rows
):

    fieldnames = [
        "chunk_size",
        "num_questions",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr"
    ]

    with open(
        SUMMARY_RESULTS_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


# ============================================================
# Print individual question result
# ============================================================

def print_question_result(
    question_id,
    chunk_size,
    result
):

    print(
        f"\n{question_id} "
        f"— {chunk_size}-word chunks"
    )

    print(
        f"Recall@1: "
        f"{result['recall_at_1']}"
    )

    print(
        f"Recall@3: "
        f"{result['recall_at_3']}"
    )

    print(
        f"Recall@5: "
        f"{result['recall_at_5']}"
    )

    print(
        f"MRR: "
        f"{result['mrr']:.4f}"
    )

    if result["first_relevant_rank"]:

        print(
            "First relevant result: "
            f"Rank {result['first_relevant_rank']}"
        )

    else:

        print(
            "First relevant result: "
            "Not found in top 5"
        )


# ============================================================
# Main evaluation
# ============================================================

def main():

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ResearchLens Retrieval Evaluation"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Load questions
    # --------------------------------------------------------

    questions = load_questions()

    print(
        f"\nLoaded {len(questions)} questions."
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    print(
        f"Loaded {len(ground_truth)} "
        "ground-truth entries."
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_inputs(
        questions,
        ground_truth
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_summary_rows = []

    all_detailed_rows = []

    # --------------------------------------------------------
    # Run each chunk-size condition
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"EVALUATING "
            f"{chunk_size}-WORD CHUNKS"
        )

        print(
            "=" * 80
        )

        # ----------------------------------------------------
        # Load datasets
        # ----------------------------------------------------

        datasets = load_datasets(
            chunk_size
        )

        total_chunks = sum(
            len(dataset["chunks"])
            for dataset in datasets
        )

        print(
            f"Loaded {len(datasets)} papers."
        )

        print(
            f"Total chunks: {total_chunks}"
        )

        # ----------------------------------------------------
        # Per-question metric storage
        # ----------------------------------------------------

        recall_1_values = []
        recall_3_values = []
        recall_5_values = []
        mrr_values = []

        # ----------------------------------------------------
        # Evaluate every question
        # ----------------------------------------------------

        for question in questions:

            result = evaluate_question(
                question,
                datasets,
                ground_truth
            )

            question_id = (
                question["question_id"]
            )

            # ------------------------------------------------
            # Store metrics
            # ------------------------------------------------

            recall_1_values.append(
                result["recall_at_1"]
            )

            recall_3_values.append(
                result["recall_at_3"]
            )

            recall_5_values.append(
                result["recall_at_5"]
            )

            mrr_values.append(
                result["mrr"]
            )

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print_question_result(
                question_id,
                chunk_size,
                result
            )

            # ------------------------------------------------
            # Store detailed retrieval results
            # ------------------------------------------------

            acceptable_papers = (
                ground_truth[
                    question_id
                ]["acceptable_papers"]
            )

            for rank, retrieved in enumerate(
                result["results"],
                start=1
            ):

                all_detailed_rows.append({

                    "question_id":
                        question_id,

                    "chunk_size":
                        chunk_size,

                    "rank":
                        rank,

                    "paper_id":
                        retrieved["paper_id"],

                    "dataset":
                        retrieved["dataset"],

                    "chunk_id":
                        retrieved["chunk_id"],

                    "page_start":
                        retrieved["page_start"],

                    "page_end":
                        retrieved["page_end"],

                    "word_count":
                        retrieved["word_count"],

                    "similarity":
                        f"{retrieved['score']:.6f}",

                    "is_relevant":
                        is_relevant(
                            retrieved,
                            acceptable_papers
                        ),

                    "text":
                        retrieved["text"]

                })

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

        num_questions = len(
            questions
        )

        mean_recall_1 = (
            sum(recall_1_values)
            / num_questions
        )

        mean_recall_3 = (
            sum(recall_3_values)
            / num_questions
        )

        mean_recall_5 = (
            sum(recall_5_values)
            / num_questions
        )

        mean_mrr = (
            sum(mrr_values)
            / num_questions
        )

        summary_row = {

            "chunk_size":
                chunk_size,

            "num_questions":
                num_questions,

            "recall_at_1":
                f"{mean_recall_1:.4f}",

            "recall_at_3":
                f"{mean_recall_3:.4f}",

            "recall_at_5":
                f"{mean_recall_5:.4f}",

            "mrr":
                f"{mean_mrr:.4f}"

        }

        all_summary_rows.append(
            summary_row
        )

        # ----------------------------------------------------
        # Print aggregate results
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"SUMMARY — "
            f"{chunk_size}-WORD CHUNKS"
        )

        print(
            "-" * 80
        )

        print(
            f"Recall@1: "
            f"{mean_recall_1:.4f}"
        )

        print(
            f"Recall@3: "
            f"{mean_recall_3:.4f}"
        )

        print(
            f"Recall@5: "
            f"{mean_recall_5:.4f}"
        )

        print(
            f"MRR: "
            f"{mean_mrr:.4f}"
        )

    # ========================================================
    # Save results
    # ========================================================

    save_detailed_results(
        all_detailed_rows
    )

    save_summary_results(
        all_summary_rows
    )

    # ========================================================
    # Final comparison
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL CHUNK-SIZE COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        "\nChunk Size | Recall@1 | Recall@3 | "
        "Recall@5 | MRR"
    )

    print(
        "-" * 65
    )

    for row in all_summary_rows:

        print(
            f"{row['chunk_size']:>10} | "
            f"{row['recall_at_1']:>8} | "
            f"{row['recall_at_3']:>8} | "
            f"{row['recall_at_5']:>8} | "
            f"{row['mrr']:>7}"
        )

    print(
        "\nDetailed results saved to:"
    )

    print(
        f"  {DETAILED_RESULTS_FILE}"
    )

    print(
        "\nSummary results saved to:"
    )

    print(
        f"  {SUMMARY_RESULTS_FILE}"
    )

    print(
        "\nEvaluation completed successfully!"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()