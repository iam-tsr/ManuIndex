import os
import uuid
from dotenv import load_dotenv

from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL"),
)

MODEL_DIR = 'onnx_models/embeddinggemma_300m/onnx/model_q4.onnx'
TOKENIZER_DIR = 'onnx_models/embeddinggemma_300m'
MAX_LENGTH = 768

embeddings = ONNXEmbedder(MODEL_DIR, TOKENIZER_DIR, MAX_LENGTH)
db = ManuIndex(embeddings=embeddings, client=client)

def test_add_document():
    document = "./tests/examples/sample.md"
    
    db.add_document(
        documents=document,
        chunk_size = 120,
        chunk_overlap = 0,
        threshold = 0.7
    )

def test_clear_index():
    db.clear()

def test_search():
    query = "What happens to nerve cells in a baby's brain during early development?"
    doc_list = db.search(
        query=query,
        top_k=2,
        lambda_mult=0.5,
        alpha=0.7,
    )
    print("Number of documents retrieved:", len(doc_list))
    return doc_list

# Subsequent tests can be added here to test retrieval and other functionalities of the ManuIndex.
def test_summary_creation():
    with open("./tests/examples/sample.md", "r") as f:
        document = f.read()

    summary = db._create_summary(document=document, doc_id=str(uuid.uuid4().hex[:6]))
    assert summary is None