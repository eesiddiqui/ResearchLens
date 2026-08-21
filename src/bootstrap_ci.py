import csv
import random
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
    OUTPUT_DIR / "bootstrap_ci.csv"
)


# ============================================================
# Configuration
# ============================================================

RETRIEVERS = {
    "dense": DENSE_RESULTS_FILE,
    "bm25": BM25_RESULTS_FILE,
}

CHUNK_SIZES = [200, 400, 800]

TOP_K = 5

N_BOOTSTRAPS = 1000

RANDOM_SEED = 42

CONFIDENCE_LEVEL = 0.95


# ============================================================
# Load retrieval results
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

        rows = list(reader)

    return rows


# ============================================================
# Convert boolean CSV value
# ============================================================

def to_bool(value):

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
        "no"
    }:

        return False

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


# ============================================================
# Convert integer
# ============================================================

def to_int(value, field_name):

    try:

        return int(value)

    except (TypeError, ValueError):

        raise ValueError(
            f"Invalid integer for "
            f"{field_name}: {value!r}"
        )


# ============================================================
# Group results by question
# ============================================================

def group_by_question(
    rows,
    retriever,
    chunk_size
):

    selected = []

    for row in rows:

        row_retriever = retriever

        row_chunk_size = to_int(
            row["chunk_size"],
            "chunk_size"
        )

        if row_chunk_size != chunk_size:

            continue

        row = dict(row)

        row["rank"] = to_int(
            row["rank"],
            "rank"
        )

        row["is_relevant_paper"] = (
            to_bool(
                row["is_relevant_paper"]
            )
        )

        row["is_relevant_page"] = (
            to_bool(
                row["is_relevant_page"]
            )
        )

        selected.append(row)

    grouped = {}

    for row in selected:

        question_id = row["question_id"]

        if question_id not in grouped:

            grouped[question_id] = []

        grouped[question_id].append(row)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    for question_id, question_results in grouped.items():

        question_results.sort(
            key=lambda x: x["rank"]
        )

        ranks = [
            row["rank"]
            for row in question_results
        ]

        expected_ranks = list(
            range(
                1,
                len(question_results) + 1
            )
        )

        if ranks != expected_ranks:

            raise ValueError(
                f"Unexpected ranks for "
                f"{retriever}, {chunk_size}, "
                f"{question_id}: "
                f"{ranks}"
            )

    return grouped


# ============================================================
# Calculate one question metric
# ============================================================

def question_metric(
    question_results,
    relevance_field,
    metric
):

    if metric == "recall@1":

        return int(
            question_results[0][
                relevance_field
            ]
        )

    if metric == "recall@3":

        top_results = (
            question_results[:3]
        )

        return int(
            any(
                row[relevance_field]
                for row in top_results
            )
        )

    if metric == "recall@5":

        top_results = (
            question_results[:5]
        )

        return int(
            any(
                row[relevance_field]
                for row in top_results
            )
        )

    if metric == "mrr":

        for rank, row in enumerate(
            question_results,
            start=1
        ):

            if row[relevance_field]:

                return 1.0 / rank

        return 0.0

    raise ValueError(
        f"Unknown metric: {metric}"
    )


# ============================================================
# Calculate per-question metric values
# ============================================================

def calculate_question_values(
    grouped_results,
    relevance_field,
    metric
):

    question_ids = sorted(
        grouped_results.keys()
    )

    values = []

    for question_id in question_ids:

        value = question_metric(
            grouped_results[question_id],
            relevance_field,
            metric
        )

        values.append(
            float(value)
        )

    return values


# ============================================================
# Bootstrap confidence interval
# ============================================================

def bootstrap_ci(
    values,
    n_bootstraps,
    rng,
    confidence_level
):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:

        raise ValueError(
            "Cannot bootstrap an empty "
            "set of values."
        )

    bootstrap_means = np.empty(
        n_bootstraps,
        dtype=float
    )

    n = len(values)

    for i in range(
        n_bootstraps
    ):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower_percentile = (
        100.0 * alpha / 2.0
    )

    upper_percentile = (
        100.0
        * (1.0 - alpha / 2.0)
    )

    lower = np.percentile(
        bootstrap_means,
        lower_percentile
    )

    upper = np.percentile(
        bootstrap_means,
        upper_percentile
    )

    observed = np.mean(values)

    return (
        observed,
        lower,
        upper
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
        "ResearchLens Bootstrap Confidence Intervals"
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

    all_rows = []

    metrics = [
        "recall@1",
        "recall@3",
        "recall@5",
        "mrr",
    ]

    relevance_fields = {
        "paper": "is_relevant_paper",
        "page": "is_relevant_page",
    }

    # ========================================================
    # Each retriever
    # ========================================================

    for (
        retriever,
        results_file
    ) in RETRIEVERS.items():

        print(
            "\n"
            + "=" * 90
        )

        print(
            f"Loading {retriever.upper()} results"
        )

        print(
            "=" * 90
        )

        rows = load_results(
            results_file
        )

        print(
            f"Loaded {len(rows)} results."
        )

        # ----------------------------------------------------
        # Each chunk size
        # ----------------------------------------------------

        for chunk_size in CHUNK_SIZES:

            grouped = group_by_question(
                rows,
                retriever,
                chunk_size
            )

            print(
                f"\n{retriever.upper()} "
                f"{chunk_size}-word chunks:"
            )

            print(
                f"  Questions: "
                f"{len(grouped)}"
            )

            if len(grouped) < 2:

                raise ValueError(
                    "At least two questions "
                    "are required for bootstrap "
                    "analysis."
                )

            # ------------------------------------------------
            # Paper and page metrics
            # ------------------------------------------------

            for (
                relevance_level,
                relevance_field
            ) in relevance_fields.items():

                for metric in metrics:

                    values = (
                        calculate_question_values(
                            grouped,
                            relevance_field,
                            metric
                        )
                    )

                    (
                        observed,
                        lower,
                        upper
                    ) = bootstrap_ci(
                        values,
                        N_BOOTSTRAPS,
                        rng,
                        CONFIDENCE_LEVEL
                    )

                    all_rows.append({

                        "retriever":
                            retriever,

                        "chunk_size":
                            chunk_size,

                        "relevance_level":
                            relevance_level,

                        "metric":
                            metric,

                        "num_questions":
                            len(values),

                        "observed":
                            f"{observed:.6f}",

                        "ci_lower":
                            f"{lower:.6f}",

                        "ci_upper":
                            f"{upper:.6f}",

                        "ci_width":
                            f"{upper - lower:.6f}",

                    })

                    print(
                        f"  {relevance_level:5s} "
                        f"{metric:9s} "
                        f"{observed:.4f} "
                        f"[{lower:.4f}, "
                        f"{upper:.4f}]"
                    )

    # ========================================================
    # Save results
    # ========================================================

    fieldnames = [

        "retriever",

        "chunk_size",

        "relevance_level",

        "metric",

        "num_questions",

        "observed",

        "ci_lower",

        "ci_upper",

        "ci_width",
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
    # Final message
    # ========================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "BOOTSTRAP ANALYSIS COMPLETED"
    )

    print(
        "=" * 90
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