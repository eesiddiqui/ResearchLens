import csv
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
REPORT_DIR = PROJECT_ROOT / "report"

DENSE_FILE = EVALUATION_DIR / "evaluation_summary.csv"
BM25_FILE = EVALUATION_DIR / "bm25_evaluation_summary.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load evaluation summary
# ============================================================

def load_summary(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append({
                "chunk_size": int(row["chunk_size"]),
                "recall_at_5_page": float(
                    row["recall_at_5_page"]
                ),
                "mrr_page": float(
                    row["mrr_page"]
                )
            })

    return sorted(
        rows,
        key=lambda x: x["chunk_size"]
    )


# ============================================================
# Load data
# ============================================================

dense = load_summary(DENSE_FILE)
bm25 = load_summary(BM25_FILE)

dense_by_size = {
    row["chunk_size"]: row
    for row in dense
}

bm25_by_size = {
    row["chunk_size"]: row
    for row in bm25
}

chunk_sizes = [200, 400, 800]


# ============================================================
# Safety checks
# ============================================================

if set(dense_by_size) != set(chunk_sizes):
    raise ValueError(
        "Dense evaluation does not contain exactly "
        "200, 400, and 800 word conditions."
    )

if set(bm25_by_size) != set(chunk_sizes):
    raise ValueError(
        "BM25 evaluation does not contain exactly "
        "200, 400, and 800 word conditions."
    )


# ============================================================
# Plot 1: Page Recall@5
# ============================================================

dense_recall5 = [
    dense_by_size[size]["recall_at_5_page"]
    for size in chunk_sizes
]

bm25_recall5 = [
    bm25_by_size[size]["recall_at_5_page"]
    for size in chunk_sizes
]

plt.figure(figsize=(8, 5))

plt.plot(
    chunk_sizes,
    dense_recall5,
    marker="o",
    linewidth=2,
    label="Dense"
)

plt.plot(
    chunk_sizes,
    bm25_recall5,
    marker="o",
    linewidth=2,
    label="BM25"
)

plt.xlabel("Chunk Size (words)")
plt.ylabel("Page Recall@5")
plt.title("Page Recall@5 vs Chunk Size")
plt.xticks(chunk_sizes)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

recall_path = REPORT_DIR / "recall_at_5_page.png"

plt.savefig(
    recall_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Plot 2: Page MRR
# ============================================================

dense_mrr = [
    dense_by_size[size]["mrr_page"]
    for size in chunk_sizes
]

bm25_mrr = [
    bm25_by_size[size]["mrr_page"]
    for size in chunk_sizes
]

plt.figure(figsize=(8, 5))

plt.plot(
    chunk_sizes,
    dense_mrr,
    marker="o",
    linewidth=2,
    label="Dense"
)

plt.plot(
    chunk_sizes,
    bm25_mrr,
    marker="o",
    linewidth=2,
    label="BM25"
)

plt.xlabel("Chunk Size (words)")
plt.ylabel("Page MRR")
plt.title("Page MRR vs Chunk Size")
plt.xticks(chunk_sizes)
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

mrr_path = REPORT_DIR / "mrr_page.png"

plt.savefig(
    mrr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Print results
# ============================================================

print("=" * 70)
print("ResearchLens Result Visualization")
print("=" * 70)

print("\nPage Recall@5:")
for size in chunk_sizes:
    print(
        f"  {size:>3} words | "
        f"Dense: {dense_by_size[size]['recall_at_5_page']:.4f} | "
        f"BM25: {bm25_by_size[size]['recall_at_5_page']:.4f}"
    )

print("\nPage MRR:")
for size in chunk_sizes:
    print(
        f"  {size:>3} words | "
        f"Dense: {dense_by_size[size]['mrr_page']:.4f} | "
        f"BM25: {bm25_by_size[size]['mrr_page']:.4f}"
    )

print("\nFigures saved:")
print(f"  {recall_path}")
print(f"  {mrr_path}")

print("\nPlotting completed successfully!")