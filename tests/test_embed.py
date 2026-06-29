import numpy as np
from manu_index import ONNXEmbedder

def test_onnx_embedder():
    MODEL_DIR = 'onnx_models/bge_m3/onnx/model_q4.onnx'
    TOKENIZER_DIR = 'onnx_models/bge_m3'
    MAX_LENGTH = 1024

    sentences = [
        "FastEmbed is lighter than Transformers & Sentence-Transformers.",
        "FastEmbed is supported and maintained by Qdrant.",
        "My name is Tushar and I am a software developer.",
    ]

    embedder = ONNXEmbedder(MODEL_DIR, TOKENIZER_DIR, MAX_LENGTH, device="cpu")
    embeddings = embedder.embed_documents(sentences)

    print("Shape  :", len(embeddings))
    print("Sample :", embeddings[0][:5])

    # Compute similarity between first two sentences
    sim = np.dot(embeddings[0], embeddings[1])
    print(f"Similarity between sentence 1 and 2: {sim:.4f}")