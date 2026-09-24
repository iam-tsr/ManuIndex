import os

from dotenv import load_dotenv

from manuindex import ManuIndex, MongoDBHandler
from manu_embed import ONNXEmbedder

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL_NAME")
base_url = os.getenv("OPENAI_API_BASE_URL")

embeddings = ONNXEmbedder(                  # Refer to https://iam-tsr.github.io/blog/cutting-edge-onnx-embedding-models/
    model="bge_m3/onnx/model_fp16.onnx",
    tokenizer="bge_m3",
    max_length=1024,
    normalize=True,
    device="cuda",
)

def test_manuindex():
    # Create a MongoDBHandler instance
    mongo_handler = MongoDBHandler(os.getenv("MONGODB_URI"), os.getenv("MONGODB_NAME"))

    # Create a ManuIndex instance with the MongoDBHandler
    manu_index = ManuIndex(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        embeddings=embeddings,
        mongo_handler=mongo_handler
    )

    # Test adding a document to the index
    document = "This is a test document."
    metadata = {"id": "doc1", "title": "Test Document"}
    manu_index.add_document(document, metadata)

    # Test retrieving the document from the index
    retrieved_doc = manu_index.search(document)
    assert retrieved_doc[0] == document, "Retrieved document does not match the original."

    # Test clearing the index
    manu_index.clear()


if __name__ == "__main__":
    test_manuindex()
    print("All tests passed.")