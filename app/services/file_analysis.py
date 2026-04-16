import os

def analyze_file(file_path):
    return {
        "ext": ".pdf",
        "size": os.path.getsize(file_path),
        "pages": get_pdf_pages(file_path),
        "has_text_layer": detect_pdf_text_layer(file_path)
    }

def sample_content(file_path):
    text = extract_first_n_pages(file_path, n=3)

    return {
        "has_toc": "table of content" in text.lower(),
        "has_sections": detect_section_pattern(text),
        "has_tables": detect_table_pattern(text),
        "is_faq_like": detect_qa_pattern(text)
    }

def classify_doc(meta, sample):

    if not meta["has_text_layer"]:
        return "scanned_doc"

    if sample["has_sections"]:
        return "structured_doc"

    if sample["is_faq_like"]:
        return "faq_doc"

    if sample["has_tables"]:
        return "table_doc"

    return "general_text"