import json
import re
from pathlib import Path


# ============================================================
# Folders
# ============================================================

PROCESSED_DATA_DIR = Path("data/processed")
CHUNKED_DATA_DIR = Path("data/chunked")

CHUNKED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Chunk sizes for our experiment
# ============================================================

CHUNK_SIZES = [200, 400, 800]


# ============================================================
# Reference section headings
# ============================================================

REFERENCE_HEADINGS = [
    "REFERENCES",
    "REFERENCE",
    "BIBLIOGRAPHY",
    "REFERENCES AND NOTES",
    "REFERENCES AND BIBLIOGRAPHY",
]


# ============================================================
# Load processed paper
# ============================================================

def load_paper(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Detect reference heading inside a page
# ============================================================

def find_reference_heading(text):

    """
    Find an explicit reference-section heading.

    We deliberately look for headings rather than individual
    citation patterns such as [1], [2], etc.

    This preserves normal in-text citations in the paper.
    """

    lines = text.splitlines()

    for line_index, line in enumerate(lines):

        cleaned = line.strip()

        if not cleaned:
            continue

        # Remove common formatting characters
        normalized = re.sub(
            r"[^A-Za-z0-9& ]",
            " ",
            cleaned
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized
        ).strip().upper()

        for heading in REFERENCE_HEADINGS:

            if normalized == heading:
                return line_index

    return None


# ============================================================
# Remove reference section
# ============================================================

def remove_reference_section(pages):

    """
    Remove everything beginning with an explicit reference
    heading.

    Important behavior:

    Example:

        Page 17
        ----------------
        VII. CONCLUSION
        conclusion text...

        REFERENCES

        [1] Paper...
        [2] Paper...

    becomes:

        Page 17
        ----------------
        VII. CONCLUSION
        conclusion text...

    All pages after page 17 are discarded.
    """

    cleaned_pages = []

    reference_found = False

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        # Once REFERENCES has been found, ignore all later pages
        if reference_found:
            break

        heading_line = find_reference_heading(text)

        # ----------------------------------------------------
        # No reference heading on this page
        # ----------------------------------------------------

        if heading_line is None:

            cleaned_pages.append({
                "page": page_number,
                "text": text
            })

            continue

        # ----------------------------------------------------
        # Reference heading found
        # ----------------------------------------------------

        lines = text.splitlines()

        # Keep everything before the reference heading
        content_before_references = "\n".join(
            lines[:heading_line]
        ).strip()

        if content_before_references:

            cleaned_pages.append({
                "page": page_number,
                "text": content_before_references
            })

        print(
            f"    Reference section detected "
            f"starting at page {page_number}"
        )

        reference_found = True

        # Do NOT process any later pages

        break

    return cleaned_pages


# ============================================================
# Create continuous chunks
# ============================================================

def create_chunks(pages, chunk_size):

    """
    Create fixed-size word chunks across the ENTIRE document.

    Important:

    Chunks are NOT restarted at every page.

    This means if a page ends with 120 words and the next page
    begins with 100 words, a 200-word chunk can contain:

        120 words from page N
        +
        80 words from page N+1

    This preserves continuity across page boundaries.
    """

    chunks = []

    chunk_id = 0

    # --------------------------------------------------------
    # Combine all cleaned pages into one continuous word stream
    # --------------------------------------------------------

    all_words = []

    word_pages = []

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        words = text.split()

        for word in words:

            all_words.append(word)
            word_pages.append(page_number)

    # --------------------------------------------------------
    # Create fixed-size chunks
    # --------------------------------------------------------

    for start in range(0, len(all_words), chunk_size):

        chunk_words = all_words[
            start:start + chunk_size
        ]

        if not chunk_words:
            continue

        chunk_text = " ".join(chunk_words)

        start_page = word_pages[start]
        end_page = word_pages[
            start + len(chunk_words) - 1
        ]

        chunks.append({
            "chunk_id": chunk_id,
            "page_start": start_page,
            "page_end": end_page,
            "text": chunk_text,
            "token_count": len(chunk_words)
        })

        chunk_id += 1

    return chunks


# ============================================================
# Process all papers
# ============================================================

json_files = sorted(
    PROCESSED_DATA_DIR.glob("*.json")
)

print(
    f"Found {len(json_files)} processed papers."
)


for json_file in json_files:

    paper_id = json_file.stem

    print(
        f"\nProcessing: {paper_id}"
    )

    # --------------------------------------------------------
    # Load original extracted paper
    # --------------------------------------------------------

    pages = load_paper(json_file)

    print(
        f"  Original pages: {len(pages)}"
    )

    # --------------------------------------------------------
    # Remove reference section
    # --------------------------------------------------------

    retrieval_pages = remove_reference_section(
        pages
    )

    print(
        f"  Retrieval pages: "
        f"{len(retrieval_pages)}"
    )

    # --------------------------------------------------------
    # Create each chunk-size condition
    # --------------------------------------------------------

    for chunk_size in CHUNK_SIZES:

        chunks = create_chunks(
            retrieval_pages,
            chunk_size
        )

        output_file = (
            CHUNKED_DATA_DIR /
            f"{paper_id}_{chunk_size}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"  {chunk_size}-word chunks: "
            f"{len(chunks)} chunks"
        )


print(
    "\nChunking completed successfully!"
)