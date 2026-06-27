from __future__ import annotations

from typing import List
import numpy as np
from transformers import AutoTokenizer
from langchain_core.embeddings import Embeddings


class ONNXEmbedder(Embeddings):
    def __init__(
            self, 
            model: str, 
            tokenizer: str, 
            max_length: int,
            batch_size: int = 16,
            normalize: bool = True,
            device: str = "cpu"
    ):
        """Embeddings class that uses an ONNX model for generating embeddings.
        
        Args:
            model: Path to the ONNX model file.
            tokenizer: Hugging Face model name or path for the tokenizer.
            max_length: Maximum sequence length for tokenization.
            batch_size: The number of texts to process in a single batch.
            normalize: Whether to L2-normalize the resulting embeddings.
            device: Device to run inference on, either 'cpu' or 'cuda'. Default is 'cpu'.
        """
        if device == "cuda":
            import torch
            import onnxruntime as ort
            if not 'CUDAExecutionProvider' in ort.get_available_providers():
                raise RuntimeError("""CUDAExecutionProvider is not available in ONNX Runtime.
                                    `pip install onnxruntime-gpu` to enable GPU support.""")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            import onnxruntime as ort
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported device: {device}. Choose 'cpu' or 'cuda'.")

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.session = ort.InferenceSession(model, sess_options=session_options, providers=providers)
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize

    def __call__(self, text: str, batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        return self.encode(text, batch_size=batch_size, normalize=normalize)

    def _get_model_input_names(self) -> set[str]:
        return {inp.name for inp in self.session.get_inputs()}

    def _build_feed(self, inputs: dict) -> dict:
        feed = dict(inputs)
        required = self._get_model_input_names()

        if "position_ids" in required:
            input_ids = feed["input_ids"]
            feed["position_ids"] = np.arange(input_ids.shape[1], dtype=np.int64)[None].repeat(input_ids.shape[0], axis=0)

        kv_inputs = {inp.name: inp for inp in self.session.get_inputs() if inp.name.startswith("past_key_values")}
        for name, spec in kv_inputs.items():
            # Shape is [batch, heads, 0, head_dim] for empty KV cache; use spec shape hints
            shape = [d if isinstance(d, int) and d >= 0 else 0 for d in spec.shape]
            shape[0] = feed["input_ids"].shape[0]  # batch size
            dtype = np.float16 if spec.type == "tensor(float16)" else np.float32
            feed[name] = np.zeros(shape, dtype=dtype)

        return feed

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents and return their embeddings as lists of floats.
        
        Args:
            documents: A list of strings, each representing a document to be embedded.

        Returns:
            A list of lists of floats, each representing the embedding of a document.
        """
        return self.encode(documents).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string and return its embedding as a list of floats.
        
        Args:
            text: A string representing the query to be embedded.

        Returns:
            A list of floats representing the embedding of the query.
        """
        return self.encode([text])[0].tolist()
    
    def encode(
        self,
        texts: list[str]
    ) -> np.ndarray:
        """Encode a list of texts into their corresponding embeddings using the ONNX model.
        
        Args:
            texts: A list of strings to be embedded.
            
        Returns:
            A NumPy array of shape (num_texts, embedding_dim) containing the embeddings.
        """
        all_embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = self._build_feed(dict(inputs))
            outputs = self.session.run(None, feed)
            emb = outputs[0]  # (batch, seq_len, hidden) — last hidden state
            # Mean pool over sequence dimension if 3D (batch, seq_len, hidden)
            if emb.ndim == 3:
                mask = feed["attention_mask"].astype(np.float32)[..., None]
                emb = (emb * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)

            if self.normalize:
                emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

            all_embeddings.append(emb)

        return np.concatenate(all_embeddings, axis=0)