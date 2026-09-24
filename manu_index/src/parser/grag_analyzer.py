from pymupdf4llm.helpers.image_analyzer import BaseImageAnalyzer
import pymupdf
import pymupdf4llm

def image_analyzer(document: str | bytes, analyzer: BaseImageAnalyzer=None):
    if isinstance(document, bytes):
        pdf_document = pymupdf.open(stream=document, filetype="pdf")
    else:
        pdf_document = pymupdf.open(document)

    with pdf_document as document:
        parsed_doc = pymupdf4llm.to_markdown(
                document,
                header=False,
                footer=False,
                force_text=False,
                dpi=300,
                analyze_image=analyzer,
        )

    return parsed_doc
