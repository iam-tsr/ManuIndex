from manu_index.src.summary import DocumentSummary
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

def test_document_summary():
    document = """# Introduction
    This is a sample document for testing the DocumentSummary class.
    
    ## Section 1
    This section discusses the first topic in detail.
    
    ## Section 2
    This section covers the second topic and provides examples.
    
    # Conclusion
    The document concludes with a summary of the key points discussed."""
    
    doc_summary = DocumentSummary(document, client, MODEL_NAME)
    summary = doc_summary.summarize()
    print(summary)
    
    assert isinstance(summary, str)
    assert len(summary.split()) > 0  # Ensure the summary is not empty