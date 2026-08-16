import fitz
import json
from pathlib import Path


# --------------------------------------------------
# Folders
# --------------------------------------------------

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Papers
# --------------------------------------------------

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


# --------------------------------------------------
# Extract PDF
# --------------------------------------------------

def extract_pdf(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


# --------------------------------------------------
# Process all papers
# --------------------------------------------------

for pdf_file in PDF_FILES:

    pdf_path = RAW_DATA_DIR / pdf_file

    print(f"Processing: {pdf_file}")

    pages = extract_pdf(pdf_path)

    # Create a filename such as:
    # p1_rag.json
    output_name = Path(pdf_file).stem + ".json"

    output_path = PROCESSED_DATA_DIR / output_name

    # Save extracted text
    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            pages,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved: {output_path}")
    print(f"Pages: {len(pages)}")
    print("-" * 60)


print("\nAll papers processed successfully!")