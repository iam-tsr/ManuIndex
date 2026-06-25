import os
from huggingface_hub import snapshot_download

# SSLM (Super Small Language Models) ONNX models for embedding tasks — under 1B parameters.
MODELS = {
    # "embeddinggemma_300m": "onnx-community/embeddinggemma-300m-ONNX", # 300M parameters
    # "qwen3_embedding_0dot6b": "onnx-community/Qwen3-Embedding-0.6B-ONNX", # 600M parameters
    # "modernBERT_base": "answerdotai/ModernBERT-base", # 100M parameters
    # "modernBERT_large": "answerdotai/ModernBERT-large", # 400M parameters
    # "bge_m3": "iam-tsr/bge-m3-ONNX", # 569M parameters
    # "bge_large_en_v1dot5": "onnx-community/bge-large-en-v1.5-ONNX", # 300M parameters
    # "bge_base_en_v1dot5": "onnx-community/bge-base-en-v1.5-ONNX", # 100M parameters
    # "all_MiniLM_L12_v2": "onnx-community/all-MiniLM-L12-v2-qa-all-ONNX", # 33.4M parameters
    "all_MiniLM_L6_v2": "onnx-community/all-MiniLM-L6-v2-ONNX", # 22.7M parameters
}
EXPORT_DIR = "onnx_models"

def download_onnx_models():
    for model_name, model_id in MODELS.items():
        model_dir = os.path.join(EXPORT_DIR, model_name)
        # Download tokenizer files to model folder
        print(f"Downloading root files for {model_id} to {model_name} folder...")

        snapshot_download(repo_id=model_id, allow_patterns=["*.json*", "*.onnx*"], local_dir=model_dir)
        print(f"Download completed for {model_name}")

if __name__ == "__main__":
    download_onnx_models()