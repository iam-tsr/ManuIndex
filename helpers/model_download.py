import os
from huggingface_hub import snapshot_download

# SSLM (Super Small Language Models) ONNX models for embedding tasks — under 1B parameters.
EMBEDDING_MODELS = {
    "bge_m3": "iam-tsr/bge-m3-ONNX", # 569M parameters
}

# SSLM (Super Small Language Models) ONNX models for reranking tasks — under 1B parameters.
RERANKER_MODELS = {
    "bge_reranker_v2_m3": "onnx-community/bge-reranker-v2-m3-ONNX" # 600M parameters
}

def download_onnx_models(model: str, export_dir: str):
    if model == "embedding":
        MODELS = EMBEDDING_MODELS
    elif model == "reranker":
        MODELS = RERANKER_MODELS
    else:
        raise ValueError("Invalid model type. Choose 'embedding' or 'reranker'.")

    for model_name, model_id in MODELS.items():
        model_dir = os.path.join(export_dir, model_name)
        # Download tokenizer files to model folder
        print(f"Downloading root files for {model_id} to {model_name} folder...")

        snapshot_download(repo_id=model_id, allow_patterns=["*.json*", "*.onnx*"], local_dir=model_dir)
        print(f"Download completed for {model_name}")

if __name__ == "__main__":
    EXPORT_DIR = "onnx_models"

    download_onnx_models("embedding", EXPORT_DIR)