from manu_index import ONNXReranker

def test_reranker():
    # Qwen decoder example:
    # MODEL_DIR = 'onnx_models/qwen3_reranker_0dot6b/onnx/model_q4.onnx'
    # TOKENIZER_DIR = 'onnx_models/qwen3_reranker_0dot6b'
    # RERANKER_TYPE = 'qwen_decoder'

    # BGE classifier example (bge-reranker-v2-m3):
    MODEL_DIR = 'onnx_models/bge_reranker_v2_m3/onnx/model_q4.onnx'
    TOKENIZER_DIR = 'onnx_models/bge_reranker_v2_m3'
    RERANKER_TYPE = 'bge_classifier'

    # BGE decoder examples, e.g. bge-reranker-v2-gemma, should use:
    # RERANKER_TYPE = 'bge_decoder'
    MAX_LENGTH = 1024
    
    user_query = "What is the best way to optimize AI model speed?"
    
    retrieved_docs = [
        "To optimize AI speed, export your model to ONNX runtime or TensorRT.",
        "A healthy diet consists of vegetables, fruits, and lean proteins.",
        "Quantization reduces model weights from FP32 to INT8 for faster inference.",
        "The Eiffel Tower is located in Paris and welcomes millions of tourists.",
        "Pruning removes redundant neural network connections to speed up execution.",
        "Deep learning models require highly specialized GPU architectures to train fast."
    ]
    
    reranker = ONNXReranker(MODEL_DIR, TOKENIZER_DIR, MAX_LENGTH, reranker_type=RERANKER_TYPE)

    reranking = reranker.rerank(
        query=user_query,
        documents=retrieved_docs,
    )
    
    # Print results
    print("\n--- Re-ranked Results ---")
    for doc, score in reranking:
        print(f"[{score:.4f}] -> {doc}")