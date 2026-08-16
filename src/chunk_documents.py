import json
from pathlib import Path


# --------------------------------------------------
# Folders
# --------------------------------------------------

PROCESSED_DATA_DIR = Path("data/processed")
CHUNKED_DATA_DIR = Path("data/chunked")

CHUNKED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Chunk sizes for our experiment
# --------------------------------------------------

CHUNK_SIZES = [200, 400, 800]


# --------------------------------------------------
# Load extracted paper
# --------------------------------------------------

def load_paper(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Create chunks
# --------------------------------------------------

def create_chunks(pages, chunk_size):

    chunks = []

    chunk_id = 0

    for page in pages:

        page_number = page["page"]
        text = page["text"]

        # Split text into words
        words = text.split()

        # Create fixed-size chunks
        for start in range(0, len(words), chunk_size):

            chunk_words = words[start:start + chunk_size]

            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words)

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "text": chunk_text,
                "token_count": len(chunk_words)
            })

            chunk_id += 1

    return chunks


# --------------------------------------------------
# Process all papers
# --------------------------------------------------

json_files = sorted(PROCESSED_DATA_DIR.glob("*.json"))


for json_file in json_files:

    paper_id = json_file.stem

    print(f"\nProcessing: {paper_id}")

    pages = load_paper(json_file)

    for chunk_size in CHUNK_SIZES:

        chunks = create_chunks(
            pages,
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


print("\nChunking completed successfully!")