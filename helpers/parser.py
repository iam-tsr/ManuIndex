import os
import dotenv
from pymupdf4llm.helpers.image_analyzer import OpenAIImageAnalyzer
import pymupdf
import pymupdf4llm

dotenv.load_dotenv()

analyzer = OpenAIImageAnalyzer(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("OPENAI_MODEL_NAME")
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
