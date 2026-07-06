import os
from dotenv import load_dotenv

from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder, ONNXReranker

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

EMB_MODEL = 'onnx_models/qwen3_embedding_0.6b/onnx/model.onnx'
EMD_TOKENIZER = 'onnx_models/qwen3_embedding_0.6b'
RRNK_MODEL = 'onnx_models/bge_reranker_v2_m3/onnx/model_q4.onnx'
RRNK_TOKENIZER = 'onnx_models/bge_reranker_v2_m3'
MAX_LENGTH = 1024

embeddings = ONNXEmbedder(EMB_MODEL, EMD_TOKENIZER, MAX_LENGTH, device="cpu")
reranker = ONNXReranker(RRNK_MODEL, RRNK_TOKENIZER, MAX_LENGTH, device="cpu")
db = ManuIndex(client, MODEL_NAME, embeddings, persist_directory="test_db")

def test_add_document(document):
    db.add_document(
        documents=document,
    )

def test_search():
    query = "What are the three elements of the Fraud Triangle theory of corruption?"
    doc_list = db.search(
        query=query,
        reranker=reranker,
    )
    print("Number of documents retrieved:", len(doc_list))
    return doc_list