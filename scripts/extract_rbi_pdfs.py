
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "dart" / "raw"
OUTPUT_DIR = ROOT / "data" / "dart" / "extracted"

PDF_FILES = [
    "rbi_kyc_master_direction_2016.pdf",
    "rbi_kyc_amendment_2024.pdf",
    "rbi_kyc_amendment_2025.pdf",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in PDF_FILES:
        pdf_path = PDF_DIR / filename

        if not pdf_path.exists():
            print(f"[MISSING] {pdf_path}")
            continue

        with pymupdf.open(pdf_path) as pdf:
            extracted_pages = []

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                extracted_pages.append(
                    f"\n--- PAGE {page_number} ---\n{text}"
                )

        output_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(
            "\n".join(extracted_pages),
            encoding="utf-8"
        )

        full_text = "\n".join(extracted_pages)
        print(
            f"[OK] {filename}: {len(extracted_pages)} pages, "
            f"{len(full_text)} characters"
        )

        if len(full_text.strip()) < 100:
            print("  WARNING: Very little text extracted; OCR may be needed.")

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
