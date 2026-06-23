import os
import dotenv
from pymupdf4llm.helpers.image_analyzer import GroqImageAnalyzer
import pymupdf
import pymupdf4llm

dotenv.load_dotenv()

analyzer = GroqImageAnalyzer(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct",
    temperature = 0.7,
    max_output_tokens = 2048
)

def test_image_analyzer(document: str | bytes):
    path_export = os.path.splitext(document)[0] + "_parsed.md"
    
    with pymupdf.open(document) as document:
        parsed_doc = pymupdf4llm.to_markdown(
                document,
                header=False,
                footer=False,
                force_text=False,
                dpi=300,
                analyze_image=analyzer,
            )

    with open(path_export, 'w', encoding='utf8') as f:
        f.write(parsed_doc)

if __name__ == "__main__":
    test_image_analyzer("./tests/examples/output.pdf")