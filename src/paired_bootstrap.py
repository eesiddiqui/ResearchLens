import csv
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

DENSE_RESULTS_FILE = Path(
    "data/evaluation/retrieval_results.csv"
)

BM25_RESULTS_FILE = Path(
    "data/evaluation/bm25_retrieval_results.csv"
)

OUTPUT_DIR = Path(
    "data/evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "paired_bootstrap.csv"
)


# ============================================================
# Configuration
# ============================================================

RETRIEVERS = [
    "dense",
    "bm25",
]

CHUNK_SIZES = [
    200,
    400,
    800,
]

METRICS = [
    "recall@1",
    "recall@3",
    "recall@5",
    "mrr",
]

RELEVANCE_LEVELS = [
    "paper",
    "page",
]

N_BOOTSTRAPS = 1000

RANDOM_SEED = 42

CONFIDENCE_LEVEL = 0.95


# ============================================================
# Load CSV
# ============================================================

def load_results(file_path):

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

        required_columns = {
            "question_id",
            "chunk_size",
            "rank",
            "is_relevant_paper",
            "is_relevant_page",
        }

        missing = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                f"{file_path} is missing "
                f"required columns: "
                f"{sorted(missing)}"
            )

        return list(reader)


# ============================================================
# Convert values
# ============================================================

def to_bool(value):

    normalized = (
        str(value)
        .strip()
        .lower()
    )

    if normalized in {
        "true",
        "1",
        "yes",
    }:

        return True

    if normalized in {
        "false",
        "0",
        "no",
        "",
    }:

        return False

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


def to_int(value, field_name):

    try:

        return int(value)

    except (TypeError, ValueError):

        raise ValueError(
            f"Invalid integer for "
            f"{field_name}: {value!r}"
        )


# ============================================================
# Prepare results
# ============================================================

def prepare_results(
    rows,
    chunk_size,
    relevance_field
):
    """
    Return:

        {
            question_id: metric values
        }

    where each question has its ranked
    top-5 results.
    """

    grouped = {}

    for row in rows:

        current_chunk_size = to_int(
            row["chunk_size"],
            "chunk_size"
        )

        if current_chunk_size != chunk_size:
            continue

        question_id = row[
            "question_id"
        ]

        if question_id not in grouped:

            grouped[question_id] = []

        grouped[question_id].append({

            "rank":
                to_int(
                    row["rank"],
                    "rank"
                ),

            "relevant":
                to_bool(
                    row[relevance_field]
                ),
        })

    # --------------------------------------------------------
    # Sort and validate ranks
    # --------------------------------------------------------

    for question_id in grouped:

        grouped[question_id].sort(
            key=lambda x: x["rank"]
        )

        ranks = [
            result["rank"]
            for result in grouped[question_id]
        ]

        expected = list(
            range(
                1,
                len(ranks) + 1
            )
        )

        if ranks != expected:

            raise ValueError(
                f"Invalid ranks for "
                f"{question_id}, "
                f"chunk size {chunk_size}: "
                f"{ranks}"
            )

    return grouped


# ============================================================
# Calculate question-level metric
# ============================================================

def calculate_metric(
    results,
    metric
):

    if metric == "recall@1":

        return float(
            results[0]["relevant"]
        )

    if metric == "recall@3":

        return float(
            any(
                result["relevant"]
                for result in results[:3]
            )
        )

    if metric == "recall@5":

        return float(
            any(
                result["relevant"]
                for result in results[:5]
            )
        )

    if metric == "mrr":

        for rank, result in enumerate(
            results,
            start=1
        ):

            if result["relevant"]:

                return 1.0 / rank

        return 0.0

    raise ValueError(
        f"Unknown metric: {metric}"
    )


# ============================================================
# Build question-level metric dictionary
# ============================================================

def build_metric_values(
    rows,
    chunk_size,
    relevance_field,
    metric
):

    grouped = prepare_results(
        rows,
        chunk_size,
        relevance_field
    )

    values = {}

    for question_id, results in grouped.items():

        values[question_id] = (
            calculate_metric(
                results,
                metric
            )
        )

    return values


# ============================================================
# Paired bootstrap
# ============================================================

def paired_bootstrap(
    values_a,
    values_b,
    question_ids,
    rng
):
    """
    Bootstrap the paired differences:

        B - A

    using the same question IDs in each sample.
    """

    differences = np.array(
        [
            values_b[qid]
            - values_a[qid]
            for qid in question_ids
        ],
        dtype=float
    )

    observed_difference = (
        np.mean(differences)
    )

    n = len(differences)

    bootstrap_means = np.empty(
        N_BOOTSTRAPS,
        dtype=float
    )

    for i in range(
        N_BOOTSTRAPS
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        sample = (
            differences[indices]
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    lower = np.percentile(
        bootstrap_means,
        100 * alpha / 2
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    return (
        observed_difference,
        lower,
        upper
    )


# ============================================================
# Determine interpretation
# ============================================================

def interpret_difference(
    lower,
    upper
):

    if lower > 0:

        return (
            "B_HIGHER_SUPPORTED"
        )

    if upper < 0:

        return (
            "A_HIGHER_SUPPORTED"
        )

    return (
        "NO_CLEAR_DIFFERENCE"
    )


# ============================================================
# Run one comparison
# ============================================================

def run_comparison(
    rows_a,
    rows_b,
    label_a,
    label_b,
    retriever,
    relevance_level,
    relevance_field,
    chunk_a,
    chunk_b,
    metric,
    rng
):

    values_a = build_metric_values(
        rows_a,
        chunk_a,
        relevance_field,
        metric
    )

    values_b = build_metric_values(
        rows_b,
        chunk_b,
        relevance_field,
        metric
    )

    common_questions = sorted(
        set(values_a)
        & set(values_b)
    )

    if len(common_questions) < 2:

        raise ValueError(
            f"Not enough common questions "
            f"for comparison: "
            f"{label_a} vs {label_b}"
        )

    (
        observed,
        lower,
        upper
    ) = paired_bootstrap(
        values_a,
        values_b,
        common_questions,
        rng
    )

    return {

        "retriever":
            retriever,

        "relevance_level":
            relevance_level,

        "metric":
            metric,

        "comparison":
            f"{label_b} vs {label_a}",

        "condition_a":
            label_a,

        "condition_b":
            label_b,

        "num_questions":
            len(common_questions),

        "observed_difference":
            f"{observed:.6f}",

        "ci_lower":
            f"{lower:.6f}",

        "ci_upper":
            f"{upper:.6f}",

        "ci_width":
            f"{upper - lower:.6f}",

        "interpretation":
            interpret_difference(
                lower,
                upper
            ),
    }


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
        "ResearchLens Paired Bootstrap Analysis"
    )

    print(
        "=" * 90
    )

    print(
        f"\nBootstrap samples: "
        f"{N_BOOTSTRAPS}"
    )

    print(
        f"Confidence level: "
        f"{CONFIDENCE_LEVEL:.0%}"
    )

    print(
        f"Random seed: "
        f"{RANDOM_SEED}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # ========================================================
    # Load retrieval results
    # ========================================================

    print(
        "\nLoading dense results..."
    )

    dense_rows = load_results(
        DENSE_RESULTS_FILE
    )

    print(
        f"Loaded {len(dense_rows)} "
        "dense results."
    )

    print(
        "\nLoading BM25 results..."
    )

    bm25_rows = load_results(
        BM25_RESULTS_FILE
    )

    print(
        f"Loaded {len(bm25_rows)} "
        "BM25 results."
    )

    results_by_retriever = {

        "dense":
            dense_rows,

        "bm25":
            bm25_rows,
    }

    all_rows = []

    # ========================================================
    # 1. Compare chunk sizes within each retriever
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "CHUNK-SIZE COMPARISONS"
    )

    print(
        "=" * 90
    )

    chunk_comparisons = [
        (200, 400),
        (400, 800),
        (200, 800),
    ]

    for retriever in RETRIEVERS:

        rows = (
            results_by_retriever[
                retriever
            ]
        )

        for (
            relevance_level,
            relevance_field
        ) in {

            "paper":
                "is_relevant_paper",

            "page":
                "is_relevant_page",

        }.items():

            for metric in METRICS:

                for (
                    chunk_a,
                    chunk_b
                ) in chunk_comparisons:

                    label_a = (
                        f"{chunk_a}-word"
                    )

                    label_b = (
                        f"{chunk_b}-word"
                    )

                    row = run_comparison(
                        rows,
                        rows,
                        label_a,
                        label_b,
                        retriever,
                        relevance_level,
                        relevance_field,
                        chunk_a,
                        chunk_b,
                        metric,
                        rng
                    )

                    all_rows.append(row)

                    print(
                        f"{retriever.upper():5s} "
                        f"{relevance_level:5s} "
                        f"{metric:9s} "
                        f"{label_b} vs "
                        f"{label_a}: "
                        f"{float(row['observed_difference']):+.4f} "
                        f"["
                        f"{float(row['ci_lower']):+.4f}, "
                        f"{float(row['ci_upper']):+.4f}"
                        f"] "
                        f"{row['interpretation']}"
                    )

    # ========================================================
    # 2. Compare Dense vs BM25 at same chunk size
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DENSE VS BM25 COMPARISONS"
    )

    print(
        "=" * 90
    )

    for chunk_size in CHUNK_SIZES:

        for (
            relevance_level,
            relevance_field
        ) in {

            "paper":
                "is_relevant_paper",

            "page":
                "is_relevant_page",

        }.items():

            for metric in METRICS:

                row = run_comparison(
                    bm25_rows,
                    dense_rows,
                    "BM25",
                    "Dense",
                    "dense_vs_bm25",
                    relevance_level,
                    relevance_field,
                    chunk_size,
                    chunk_size,
                    metric,
                    rng
                )

                all_rows.append(row)

                print(
                    f"{relevance_level:5s} "
                    f"{metric:9s} "
                    f"Dense vs BM25 "
                    f"({chunk_size}): "
                    f"{float(row['observed_difference']):+.4f} "
                    f"["
                    f"{float(row['ci_lower']):+.4f}, "
                    f"{float(row['ci_upper']):+.4f}"
                    f"] "
                    f"{row['interpretation']}"
                )

    # ========================================================
    # Save results
    # ========================================================

    fieldnames = [

        "retriever",

        "relevance_level",

        "metric",

        "comparison",

        "condition_a",

        "condition_b",

        "num_questions",

        "observed_difference",

        "ci_lower",

        "ci_upper",

        "ci_width",

        "interpretation",
    ]

    with open(
        OUTPUT_FILE,
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
            all_rows
        )

    # ========================================================
    # Final
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "PAIRED BOOTSTRAP COMPLETED"
    )

    print(
        "=" * 90
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  B_HIGHER_SUPPORTED"
        "  = 95% CI is entirely above zero"
    )

    print(
        "  A_HIGHER_SUPPORTED"
        "  = 95% CI is entirely below zero"
    )

    print(
        "  NO_CLEAR_DIFFERENCE"
        " = CI includes zero"
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {OUTPUT_FILE}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()