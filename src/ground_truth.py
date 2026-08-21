"""
ground_truth.py
================

Shared utilities for loading the two ground-truth files used across
ResearchLens evaluation scripts:

  data/ground_truth.csv        -> paper-level ground truth (original)
  data/ground_truth_pages.csv  -> page-aware ground truth (new)

Paper-level ground truth answers the question:
    "Did we retrieve a chunk from the RIGHT PAPER?"

Page-aware ground truth answers the stricter question:
    "Did we retrieve a chunk from the right paper AND from the part of
     that paper that actually contains the answer?"

A chunk can span multiple pages (page_start..page_end). A chunk is
considered page-relevant to a question if its page span OVERLAPS any
of the annotated ground-truth page ranges for an acceptable paper.

Why this matters for chunk-size experiments:
    Paper-level recall can look artificially high (or flat) because
    a paper about RAG tends to mention RAG-ish vocabulary on nearly
    every page. A chunk can match the right PAPER by accident while
    containing none of the actual answer. Page-aware recall is a much
    stricter, more valid signal for comparing 200 vs 400 vs 800-word
    chunking, because it also rewards chunk boundaries that keep the
    real answer intact and penalizes retrieving "paper-adjacent" noise.
"""

import csv
from pathlib import Path


GROUND_TRUTH_FILE = Path("data/ground_truth.csv")
GROUND_TRUTH_PAGES_FILE = Path("data/ground_truth_pages.csv")


# ============================================================
# Paper-level ground truth (original)
# ============================================================

def load_ground_truth(path=GROUND_TRUTH_FILE):

    ground_truth = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        required_columns = {
            "question_id",
            "primary_papers",
            "acceptable_papers"
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in reader:

            question_id = row["question_id"].strip()

            primary_papers = {
                paper.strip()
                for paper in row["primary_papers"].split(";")
                if paper.strip()
            }

            acceptable_papers = {
                paper.strip()
                for paper in row["acceptable_papers"].split(";")
                if paper.strip()
            }

            ground_truth[question_id] = {
                "primary_papers": primary_papers,
                "acceptable_papers": acceptable_papers
            }

    return ground_truth


# ============================================================
# Page-aware ground truth (new)
# ============================================================

def _parse_page_ranges(raw):
    """
    Parse a page_ranges cell like "2-4" or "1;5-7" or "16" into a
    list of (start, end) inclusive integer tuples.
    """

    ranges = []

    for part in raw.split(";"):

        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)

        else:
            start = end = int(part)

        ranges.append((start, end))

    return ranges


def load_ground_truth_pages(path=GROUND_TRUTH_PAGES_FILE):
    """
    Returns:
        {
            question_id: {
                paper_id: {
                    "relevance": "primary" | "acceptable",
                    "page_ranges": [(start, end), ...]
                },
                ...
            },
            ...
        }
    """

    ground_truth_pages = {}

    with open(path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        required_columns = {
            "question_id",
            "paper_id",
            "relevance",
            "page_ranges"
        }

        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        for row in reader:

            question_id = row["question_id"].strip()
            paper_id = row["paper_id"].strip()
            relevance = row["relevance"].strip()
            page_ranges = _parse_page_ranges(row["page_ranges"])

            ground_truth_pages.setdefault(question_id, {})

            ground_truth_pages[question_id][paper_id] = {
                "relevance": relevance,
                "page_ranges": page_ranges
            }

    return ground_truth_pages


def validate_pages_against_papers(ground_truth, ground_truth_pages):
    """
    Sanity check: every (question, paper) pair marked acceptable in
    ground_truth.csv should have a matching entry in
    ground_truth_pages.csv, and vice versa. Mismatches usually mean
    one file was edited without updating the other.
    """

    problems = []

    for question_id, gt in ground_truth.items():

        acceptable_papers = gt["acceptable_papers"]
        page_entry = ground_truth_pages.get(question_id, {})
        page_papers = set(page_entry.keys())

        missing_pages = acceptable_papers - page_papers
        extra_pages = page_papers - acceptable_papers

        if missing_pages:
            problems.append(
                f"{question_id}: acceptable in ground_truth.csv but "
                f"missing page ranges: {sorted(missing_pages)}"
            )

        if extra_pages:
            problems.append(
                f"{question_id}: has page ranges but not listed as "
                f"acceptable in ground_truth.csv: {sorted(extra_pages)}"
            )

    return problems


# ============================================================
# Relevance checks used by evaluation scripts
# ============================================================

def is_paper_relevant(result, acceptable_papers):
    """Original, coarse check: did we hit the right paper at all?"""

    return result["paper_id"] in acceptable_papers


def is_page_relevant(result, question_page_gt):
    """
    Strict check: did we hit the right paper AND does the retrieved
    chunk's page span overlap the annotated answer pages?

    question_page_gt: ground_truth_pages[question_id], i.e.
        { paper_id: {"relevance": ..., "page_ranges": [...] }, ... }
    """

    paper_entry = question_page_gt.get(result["paper_id"])

    if paper_entry is None:
        return False

    chunk_start = result["page_start"]
    chunk_end = result["page_end"]

    if chunk_start is None or chunk_end is None:
        return False

    for gt_start, gt_end in paper_entry["page_ranges"]:

        # Standard inclusive integer range overlap test
        if chunk_start <= gt_end and gt_start <= chunk_end:
            return True

    return False
