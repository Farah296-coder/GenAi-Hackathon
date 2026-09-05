import pymupdf
import json

def extract_pdf_text(pdf_path):
    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = r"data\L22_New_Generic Class and Methods.pdf"

    pages = extract_pdf_text(pdf_path)

    output_path = r"output\extracted_text.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=4)

    print(f"Done! Output saved to: {output_path}")