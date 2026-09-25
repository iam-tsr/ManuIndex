from manu_index import LocalDataStore, ManuIndex
from manu_embed import ONNXEmbedder

api_key = "rui_tachibana"
model_name = "rui_tachibana"
base_url = "http://localhost:8888/v1"

embeddings = ONNXEmbedder(                  # Refer to https://iam-tsr.github.io/blog/cutting-edge-onnx-embedding-models/
    model="bge_m3/onnx/model_fp16.onnx",
    tokenizer="bge_m3",
    max_length=1024,
    normalize=True,
    device="cuda",
)

def test_manuindex():
    datastore = LocalDataStore("manu_index_test.sqlite3")
    try:
        # Create a ManuIndex instance with local SQLite storage.
        manu_index = ManuIndex(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            embeddings=embeddings,
            datastore=datastore,
        )

        # Test adding a document to the index.
        document = "This is a test document."
        metadata = {"id": "doc1", "title": "Test Document"}
        manu_index.add_document(document, metadata)

        # Test retrieving the document from the index.
        retrieved_doc = manu_index.search(document)
        assert retrieved_doc[0] == document, "Retrieved document does not match the original."

        # Test clearing the index.
        manu_index.clear()
    finally:
        datastore.close()


if __name__ == "__main__":
    test_manuindex()
    print("All tests passed.")