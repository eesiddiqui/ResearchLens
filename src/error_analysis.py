import csv
from pathlib import Path

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

DENSE_RESULTS_FILE = Path(
    "data/evaluation/retrieval_results.csv"
)

BM25_RESULTS_FILE = Path(
    "data/evaluation/bm25_retrieval_results.csv"
)

GROUND_TRUTH_FILE = Path(
    "data/ground_truth.csv"
)

GROUND_TRUTH_PAGES_FILE = Path(
    "data/ground_truth_pages.csv"
)

OUTPUT_DIR = Path(
    "data/evaluation"
)

ERROR_DETAILS_FILE = (
    OUTPUT_DIR / "error_analysis.csv"
)

ERROR_SUMMARY_FILE = (
    OUTPUT_DIR / "error_analysis_summary.csv"
)

CONSISTENCY_FILE = (
    OUTPUT_DIR / "error_analysis_consistency.csv"
)


# ============================================================
# Retriever configuration
# ============================================================

RETRIEVERS = {
    "dense": DENSE_RESULTS_FILE,
    "bm25": BM25_RESULTS_FILE,
}


# ============================================================
# Load retrieval results
# ============================================================

def load_results(file_path):
    """
    Load retrieval results produced by evaluate.py
    or evaluate_bm25.py.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"Results file not found:\n"
            f"  {file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:

            raise ValueError(
                f"No CSV header found in "
                f"{file_path}"
            )

        required_columns = {
            "question_id",
            "chunk_size",
            "rank",
            "paper_id",
            "page_start",
            "page_end",
        }

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:

            raise ValueError(
                f"{file_path} is missing "
                f"required columns: "
                f"{sorted(missing)}"
            )

        return list(reader)


# ============================================================
# Convert CSV value to integer
# ============================================================

def to_int(value, field_name):

    try:

        return int(value)

    except (TypeError, ValueError):

        raise ValueError(
            f"Invalid integer value for "
            f"{field_name}: {value!r}"
        )


# ============================================================
# Convert CSV boolean
# ============================================================

def to_bool(value):

    if isinstance(value, bool):

        return value

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes"
    }:

        return True

    if normalized in {
        "false",
        "0",
        "no",
        ""
    }:

        return False

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


# ============================================================
# Classify one retrieval result
# ============================================================

def classify_result(
    result,
    ground_truth,
    ground_truth_pages
):
    """
    Independently calculate paper/page relevance and
    classify the retrieval result.

    Categories:

        CORRECT_PAGE
            Correct paper AND correct answer page.

        CORRECT_PAPER_WRONG_PAGE
            Correct paper BUT wrong page.

        WRONG_PAPER
            Retrieved paper is not acceptable.
    """

    question_id = (
        result["question_id"]
    )

    if question_id not in ground_truth:

        raise ValueError(
            f"No paper-level ground truth "
            f"for {question_id}"
        )

    if question_id not in ground_truth_pages:

        raise ValueError(
            f"No page-level ground truth "
            f"for {question_id}"
        )

    # --------------------------------------------------------
    # Make a result dictionary with numeric page values.
    #
    # This matches the interface expected by the existing
    # ground_truth.py functions.
    # --------------------------------------------------------

    normalized_result = dict(result)

    normalized_result["page_start"] = to_int(
        result["page_start"],
        "page_start"
    )

    normalized_result["page_end"] = to_int(
        result["page_end"],
        "page_end"
    )

    # --------------------------------------------------------
    # Paper-level relevance
    #
    # Existing function signature:
    #
    # is_paper_relevant(result, acceptable_papers)
    # --------------------------------------------------------

    acceptable_papers = (
        ground_truth[question_id]
        ["acceptable_papers"]
    )

    paper_relevant = is_paper_relevant(
        normalized_result,
        acceptable_papers
    )

    # --------------------------------------------------------
    # Page-level relevance
    #
    # Existing function signature:
    #
    # is_page_relevant(result, question_page_gt)
    # --------------------------------------------------------

    question_page_gt = (
        ground_truth_pages[question_id]
    )

    page_relevant = is_page_relevant(
        normalized_result,
        question_page_gt
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if page_relevant:

        error_type = "CORRECT_PAGE"

    elif paper_relevant:

        error_type = (
            "CORRECT_PAPER_WRONG_PAGE"
        )

    else:

        error_type = "WRONG_PAPER"

    return (
        error_type,
        paper_relevant,
        page_relevant
    )


# ============================================================
# Analyze one retriever
# ============================================================

def analyze_retriever(
    retriever_name,
    results,
    ground_truth,
    ground_truth_pages
):
    """
    Analyze all retrieved top-k results for one retriever.
    """

    detailed_rows = []

    summary_counts = {}

    consistency_rows = []

    for result in results:

        (
            error_type,
            calculated_paper_relevant,
            calculated_page_relevant
        ) = classify_result(
            result,
            ground_truth,
            ground_truth_pages
        )

        question_id = (
            result["question_id"]
        )

        chunk_size = to_int(
            result["chunk_size"],
            "chunk_size"
        )

        rank = to_int(
            result["rank"],
            "rank"
        )

        page_start = to_int(
            result["page_start"],
            "page_start"
        )

        page_end = to_int(
            result["page_end"],
            "page_end"
        )

        # ----------------------------------------------------
        # Check flags already stored by evaluate.py
        # ----------------------------------------------------

        stored_paper_value = result.get(
            "is_relevant_paper",
            ""
        )

        stored_page_value = result.get(
            "is_relevant_page",
            ""
        )

        stored_paper_relevant = None
        stored_page_relevant = None

        paper_flag_matches = None
        page_flag_matches = None

        if stored_paper_value != "":

            stored_paper_relevant = to_bool(
                stored_paper_value
            )

            paper_flag_matches = (
                stored_paper_relevant
                == calculated_paper_relevant
            )

        if stored_page_value != "":

            stored_page_relevant = to_bool(
                stored_page_value
            )

            page_flag_matches = (
                stored_page_relevant
                == calculated_page_relevant
            )

        # ----------------------------------------------------
        # Record consistency
        # ----------------------------------------------------

        consistency_rows.append({

            "retriever":
                retriever_name,

            "question_id":
                question_id,

            "chunk_size":
                chunk_size,

            "rank":
                rank,

            "stored_paper_relevance":
                stored_paper_relevant,

            "calculated_paper_relevance":
                calculated_paper_relevant,

            "paper_flag_matches":
                paper_flag_matches,

            "stored_page_relevance":
                stored_page_relevant,

            "calculated_page_relevance":
                calculated_page_relevant,

            "page_flag_matches":
                page_flag_matches,

        })

        # ----------------------------------------------------
        # Summary bucket
        # ----------------------------------------------------

        key = (
            retriever_name,
            chunk_size
        )

        if key not in summary_counts:

            summary_counts[key] = {

                "total_results": 0,

                "correct_page": 0,

                "correct_paper_wrong_page": 0,

                "wrong_paper": 0,

                "rank_1_correct_page": 0,

                "rank_2_correct_page": 0,

                "rank_3_correct_page": 0,

                "rank_4_correct_page": 0,

                "rank_5_correct_page": 0,
            }

        summary_counts[key][
            "total_results"
        ] += 1

        # ----------------------------------------------------
        # Count classification
        # ----------------------------------------------------

        if error_type == "CORRECT_PAGE":

            summary_counts[key][
                "correct_page"
            ] += 1

            if rank == 1:

                summary_counts[key][
                    "rank_1_correct_page"
                ] += 1

            elif rank == 2:

                summary_counts[key][
                    "rank_2_correct_page"
                ] += 1

            elif rank == 3:

                summary_counts[key][
                    "rank_3_correct_page"
                ] += 1

            elif rank == 4:

                summary_counts[key][
                    "rank_4_correct_page"
                ] += 1

            elif rank == 5:

                summary_counts[key][
                    "rank_5_correct_page"
                ] += 1

        elif (
            error_type
            == "CORRECT_PAPER_WRONG_PAGE"
        ):

            summary_counts[key][
                "correct_paper_wrong_page"
            ] += 1

        elif error_type == "WRONG_PAPER":

            summary_counts[key][
                "wrong_paper"
            ] += 1

        # ----------------------------------------------------
        # Detailed row
        # ----------------------------------------------------

        similarity = result.get(
            "similarity",
            result.get(
                "score",
                ""
            )
        )

        detailed_rows.append({

            "retriever":
                retriever_name,

            "question_id":
                question_id,

            "chunk_size":
                chunk_size,

            "rank":
                rank,

            "paper_id":
                result["paper_id"],

            "page_start":
                page_start,

            "page_end":
                page_end,

            "error_type":
                error_type,

            "is_relevant_paper":
                calculated_paper_relevant,

            "is_relevant_page":
                calculated_page_relevant,

            "stored_is_relevant_paper":
                stored_paper_relevant,

            "stored_is_relevant_page":
                stored_page_relevant,

            "paper_flag_matches":
                paper_flag_matches,

            "page_flag_matches":
                page_flag_matches,

            "similarity":
                similarity,

            "text":
                result.get(
                    "text",
                    ""
                ),
        })

    return (
        detailed_rows,
        summary_counts,
        consistency_rows
    )


# ============================================================
# Save detailed analysis
# ============================================================

def save_detailed_results(rows):

    fieldnames = [

        "retriever",

        "question_id",

        "chunk_size",

        "rank",

        "paper_id",

        "page_start",

        "page_end",

        "error_type",

        "is_relevant_paper",

        "is_relevant_page",

        "stored_is_relevant_paper",

        "stored_is_relevant_page",

        "paper_flag_matches",

        "page_flag_matches",

        "similarity",

        "text",
    ]

    with open(
        ERROR_DETAILS_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Save summary
# ============================================================

def save_summary(summary_counts):

    rows = []

    for (
        retriever_name,
        chunk_size
    ), counts in sorted(
        summary_counts.items()
    ):

        total = (
            counts["total_results"]
        )

        correct_page_rate = (
            counts["correct_page"]
            / total
            if total
            else 0.0
        )

        correct_paper_wrong_page_rate = (
            counts[
                "correct_paper_wrong_page"
            ]
            / total
            if total
            else 0.0
        )

        wrong_paper_rate = (
            counts["wrong_paper"]
            / total
            if total
            else 0.0
        )

        rows.append({

            "retriever":
                retriever_name,

            "chunk_size":
                chunk_size,

            "total_results":
                total,

            "correct_page":
                counts["correct_page"],

            "correct_page_rate":
                f"{correct_page_rate:.6f}",

            "correct_paper_wrong_page":
                counts[
                    "correct_paper_wrong_page"
                ],

            "correct_paper_wrong_page_rate":
                f"{correct_paper_wrong_page_rate:.6f}",

            "wrong_paper":
                counts["wrong_paper"],

            "wrong_paper_rate":
                f"{wrong_paper_rate:.6f}",

            "rank_1_correct_page":
                counts[
                    "rank_1_correct_page"
                ],

            "rank_2_correct_page":
                counts[
                    "rank_2_correct_page"
                ],

            "rank_3_correct_page":
                counts[
                    "rank_3_correct_page"
                ],

            "rank_4_correct_page":
                counts[
                    "rank_4_correct_page"
                ],

            "rank_5_correct_page":
                counts[
                    "rank_5_correct_page"
                ],
        })

    fieldnames = [

        "retriever",

        "chunk_size",

        "total_results",

        "correct_page",

        "correct_page_rate",

        "correct_paper_wrong_page",

        "correct_paper_wrong_page_rate",

        "wrong_paper",

        "wrong_paper_rate",

        "rank_1_correct_page",

        "rank_2_correct_page",

        "rank_3_correct_page",

        "rank_4_correct_page",

        "rank_5_correct_page",
    ]

    with open(
        ERROR_SUMMARY_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Save consistency checks
# ============================================================

def save_consistency_results(rows):

    fieldnames = [

        "retriever",

        "question_id",

        "chunk_size",

        "rank",

        "stored_paper_relevance",

        "calculated_paper_relevance",

        "paper_flag_matches",

        "stored_page_relevance",

        "calculated_page_relevance",

        "page_flag_matches",
    ]

    with open(
        CONSISTENCY_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# Print summary
# ============================================================

def print_summary(summary_counts):

    print(
        "\n"
        + "=" * 90
    )

    print(
        "RETRIEVAL ERROR ANALYSIS"
    )

    print(
        "=" * 90
    )

    print(
        "\nClassification:"
    )

    print(
        "  CORRECT_PAGE"
        "              = correct paper + correct answer page"
    )

    print(
        "  CORRECT_PAPER_WRONG_PAGE"
        " = correct paper + wrong page"
    )

    print(
        "  WRONG_PAPER"
        "                 = incorrect paper"
    )

    for (
        retriever_name,
        chunk_size
    ), counts in sorted(
        summary_counts.items()
    ):

        total = (
            counts["total_results"]
        )

        correct_page_rate = (
            counts["correct_page"]
            / total
            if total
            else 0.0
        )

        wrong_page_rate = (
            counts[
                "correct_paper_wrong_page"
            ]
            / total
            if total
            else 0.0
        )

        wrong_paper_rate = (
            counts["wrong_paper"]
            / total
            if total
            else 0.0
        )

        print(
            "\n"
            + "-" * 90
        )

        print(
            f"{retriever_name.upper()} "
            f"— {chunk_size}-WORD CHUNKS"
        )

        print(
            "-" * 90
        )

        print(
            f"Total retrieved results: "
            f"{total}"
        )

        print(
            f"Correct page: "
            f"{counts['correct_page']} "
            f"({correct_page_rate:.2%})"
        )

        print(
            f"Correct paper, wrong page: "
            f"{counts['correct_paper_wrong_page']} "
            f"({wrong_page_rate:.2%})"
        )

        print(
            f"Wrong paper: "
            f"{counts['wrong_paper']} "
            f"({wrong_paper_rate:.2%})"
        )

        print(
            "\nCorrect-page hits by rank:"
        )

        print(
            f"  Rank 1: "
            f"{counts['rank_1_correct_page']}"
        )

        print(
            f"  Rank 2: "
            f"{counts['rank_2_correct_page']}"
        )

        print(
            f"  Rank 3: "
            f"{counts['rank_3_correct_page']}"
        )

        print(
            f"  Rank 4: "
            f"{counts['rank_4_correct_page']}"
        )

        print(
            f"  Rank 5: "
            f"{counts['rank_5_correct_page']}"
        )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "ResearchLens Error Analysis"
    )

    print(
        "=" * 90
    )

    # ========================================================
    # Load paper-level ground truth
    # ========================================================

    print(
        "\nLoading paper-level ground truth..."
    )

    ground_truth = load_ground_truth(
        GROUND_TRUTH_FILE
    )

    print(
        f"Loaded {len(ground_truth)} "
        "paper-level ground-truth questions."
    )

    # ========================================================
    # Load page-level ground truth
    # ========================================================

    print(
        "\nLoading page-level ground truth..."
    )

    ground_truth_pages = (
        load_ground_truth_pages(
            GROUND_TRUTH_PAGES_FILE
        )
    )

    print(
        f"Loaded {len(ground_truth_pages)} "
        "page-level ground-truth questions."
    )

    # ========================================================
    # Validate paper/page ground truth consistency
    # ========================================================

    print(
        "\nValidating ground-truth consistency..."
    )

    problems = (
        validate_pages_against_papers(
            ground_truth,
            ground_truth_pages
        )
    )

    if problems:

        print(
            "\nGROUND-TRUTH VALIDATION FAILED"
        )

        for problem in problems:

            print(
                f"  - {problem}"
            )

        raise ValueError(
            "Paper-level and page-level "
            "ground truth are inconsistent. "
            "Fix them before continuing."
        )

    print(
        "Ground-truth consistency check passed."
    )

    # ========================================================
    # Storage
    # ========================================================

    all_detailed_rows = []

    all_consistency_rows = []

    combined_summary = {}

    # ========================================================
    # Analyze each retriever
    # ========================================================

    for (
        retriever_name,
        results_file
    ) in RETRIEVERS.items():

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"ANALYZING "
            f"{retriever_name.upper()} RETRIEVAL"
        )

        print(
            "=" * 90
        )

        results = load_results(
            results_file
        )

        print(
            f"Loaded {len(results)} "
            f"retrieval results."
        )

        (
            detailed_rows,
            summary_counts,
            consistency_rows
        ) = analyze_retriever(
            retriever_name,
            results,
            ground_truth,
            ground_truth_pages
        )

        all_detailed_rows.extend(
            detailed_rows
        )

        all_consistency_rows.extend(
            consistency_rows
        )

        combined_summary.update(
            summary_counts
        )

    # ========================================================
    # Save outputs
    # ========================================================

    save_detailed_results(
        all_detailed_rows
    )

    save_summary(
        combined_summary
    )

    save_consistency_results(
        all_consistency_rows
    )

    # ========================================================
    # Print summary
    # ========================================================

    print_summary(
        combined_summary
    )

    # ========================================================
    # Check stored evaluation flags
    # ========================================================

    paper_mismatches = 0
    page_mismatches = 0

    for row in all_consistency_rows:

        if (
            row["paper_flag_matches"]
            is False
        ):

            paper_mismatches += 1

        if (
            row["page_flag_matches"]
            is False
        ):

            page_mismatches += 1

    print(
        "\n"
        + "=" * 90
    )

    print(
        "EVALUATION CONSISTENCY CHECK"
    )

    print(
        "=" * 90
    )

    print(
        f"Paper relevance mismatches: "
        f"{paper_mismatches}"
    )

    print(
        f"Page relevance mismatches: "
        f"{page_mismatches}"
    )

    if (
        paper_mismatches == 0
        and page_mismatches == 0
    ):

        print(
            "\nStored evaluation relevance flags "
            "match the independently recalculated "
            "ground truth."
        )

    else:

        print(
            "\nWARNING: Stored evaluation flags "
            "do not completely match the "
            "independent ground-truth calculation."
        )

    # ========================================================
    # Output locations
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "Analysis files saved:"
    )

    print(
        f"  {ERROR_DETAILS_FILE}"
    )

    print(
        f"  {ERROR_SUMMARY_FILE}"
    )

    print(
        f"  {CONSISTENCY_FILE}"
    )

    print(
        "\nError analysis completed successfully!"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()