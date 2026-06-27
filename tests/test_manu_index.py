import os
import uuid
from dotenv import load_dotenv

from openai import OpenAI
from manu_index import ManuIndex, ONNXEmbedder, ONNXReranker

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

EMB_MODEL = 'onnx_models/bge_m3/onnx/model_q4.onnx'
EMD_TOKENIZER = 'onnx_models/bge_m3'
RRNK_MODEL = 'onnx_models/bge_reranker_v2_m3/onnx/model_q4.onnx'
RRNK_TOKENIZER = 'onnx_models/bge_reranker_v2_m3'
MAX_LENGTH = 1024

embeddings = ONNXEmbedder(EMB_MODEL, EMD_TOKENIZER, MAX_LENGTH, device="cpu")
reranker = ONNXReranker(RRNK_MODEL, RRNK_TOKENIZER, MAX_LENGTH, device="cuda")
db = ManuIndex(client, MODEL_NAME, embeddings)

def test_add_document():
    document = "./tests/examples/sample4.md"
    
    db.add_document(
        documents=document
    )

def test_search():
    query = "What are the three elements of the Fraud Triangle theory of corruption?"
    doc_list = db.search(
        query=query,
        reranker=reranker,
    )
    print("Number of documents retrieved:", len(doc_list))
    return doc_list

if __name__ == "__main__":
    # test_add_document()

    docs = test_search()
    print(docs)
    # query = "What are the three elements of the Fraud Triangle theory of corruption?"
    # context_block = "\n\n".join(docs)
    # response = client.chat.completions.create(
    #     model=MODEL_NAME,
    #     messages=[
    #         {"role": "system", "content": "Answer the question using only the provided context. Be concise — no markdown, no headers, no bullet points. If the context does not contain the answer, say 'I cannot answer based on the given context.'."},
    #         {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
    #     ],
    #     temperature=0,
    # )
    # print(response.choices[0].message.content.strip())