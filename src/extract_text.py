import json
import re
from pathlib import Path

import fitz


# ============================================================
# FOLDERS
# ============================================================

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PAPERS
# ============================================================

PDF_FILES = [
    "p1_rag.pdf",
    "p2_dpr.pdf",
    "p3_lost_middle.pdf",
    "p4_never_lost.pdf",
    "p5_rag_eval.pdf",
    "p6_rag_survey.pdf",
    "p7_realm.pdf",
    "p8_colbert.pdf"
]


# ============================================================
# PAGE NUMBER CLEANING
# ============================================================

def remove_page_number_artifacts(text, page_number):
    """
    Remove likely PDF page-number artifacts while trying
    to preserve legitimate scientific numbers.

    We do NOT remove every occurrence of the page number.

    A number is removed only when it appears as an isolated
    token and is likely functioning as a printed page number.
    """

    lines = text.splitlines()

    cleaned_lines = []

    page_number_str = str(page_number)

    for line in lines:

        original_line = line
        stripped = line.strip()

        # ----------------------------------------------------
        # Empty lines
        # ----------------------------------------------------

        if not stripped:
            cleaned_lines.append("")
            continue

        # ----------------------------------------------------
        # Case 1:
        # The entire extracted line is just the page number.
        #
        # Example:
        #
        # 17
        #
        # This is highly likely to be a page-number artifact.
        # ----------------------------------------------------

        if stripped == page_number_str:
            continue

        # ----------------------------------------------------
        # Case 2:
        # Remove page number when surrounded by whitespace
        # AND it appears in a suspicious position.
        #
        # We are deliberately conservative here.
        # ----------------------------------------------------

        pattern = rf"(?<!\w){re.escape(page_number_str)}(?!\w)"

        matches = list(
            re.finditer(
                pattern,
                original_line
            )
        )

        if matches:

            # Remove only if the number is likely isolated
            # formatting rather than part of scientific content.

            words_before = original_line[:matches[0].start()].strip()
            words_after = original_line[matches[0].end():].strip()

            # ------------------------------------------------
            # If there is substantial text on both sides,
            # don't automatically remove it.
            #
            # Example:
            #
            # "model achieved 17% improvement"
            #
            # should remain untouched.
            # ------------------------------------------------

            if words_before and words_after:

                # Check whether number has punctuation around it.
                before_char = original_line[
                    matches[0].start() - 1
                ] if matches[0].start() > 0 else ""

                after_char = original_line[
                    matches[0].end()
                ] if matches[0].end() < len(original_line) else ""

                # A plain whitespace-surrounded number in the
                # middle of a sentence is ambiguous.
                #
                # Therefore keep it.
                continue

        cleaned_lines.append(original_line)

    return "\n".join(cleaned_lines)


# ============================================================
# ADDITIONAL CLEANING
# ============================================================

def clean_page_text(text, page_number):
    """
    Apply conservative cleaning to one PDF page.
    """

    # --------------------------------------------------------
    # Normalize excessive whitespace
    # --------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # --------------------------------------------------------
    # Remove obvious standalone page-number lines
    # --------------------------------------------------------

    text = remove_page_number_artifacts(
        text,
        page_number
    )

    # --------------------------------------------------------
    # Remove excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT PDF
# ============================================================

def extract_pdf(pdf_path):
    """
    Extract text page-by-page from a PDF.

    Returns:

        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_index, page in enumerate(
        document,
        start=1
    ):

        raw_text = page.get_text(
            "text"
        )

        cleaned_text = clean_page_text(
            raw_text,
            page_index
        )

        pages.append({
            "page": page_index,
            "text": cleaned_text
        })

    document.close()

    return pages


# ============================================================
# PROCESS PAPERS
# ============================================================

for pdf_filename in PDF_FILES:

    pdf_path = (
        RAW_DATA_DIR /
        pdf_filename
    )

    print(
        f"\nProcessing: {pdf_filename}"
    )

    if not pdf_path.exists():

        print(
            f"ERROR: File not found: {pdf_path}"
        )

        continue

    pages = extract_pdf(
        pdf_path
    )

    output_filename = (
        pdf_path.stem +
        ".json"
    )

    output_path = (
        PROCESSED_DATA_DIR /
        output_filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pages,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Pages: {len(pages)}"
    )

    print(
        "-" * 60
    )


print(
    "\nAll papers processed successfully!"
)