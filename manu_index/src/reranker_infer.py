from __future__ import annotations

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class ONNXReranker:
    def __init__(
            self, 
            model: str, 
            tokenizer: str, 
            max_length: int,
            batch_size: int = 16,
            normalize: bool = True,
            device: str = "cpu"
    ):
        """Reranker class supports Qwen ONNX reranker model only for reranking documents based on a query.
        
        Args:
            model: Path to the ONNX model file.
            tokenizer: Hugging Face model name or path for the tokenizer.
            max_length: Maximum sequence length for tokenization.
            batch_size: The number of documents to process in a single batch.
            normalize: Whether to normalize the logits.
            device: Device to run inference on, either 'cpu' or 'cuda'. Default is 'cpu'.
        """
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"Unsupported device: {device}. Choose 'cpu' or 'cuda'.")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer,
            padding_side="left",
            fix_mistral_regex=True,
        )
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.session = ort.InferenceSession(model, providers=providers)
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def __call__(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        return self.rerank(query, documents)

    def _get_model_input_names(self) -> set[str]:
        return {inp.name for inp in self.session.get_inputs()}

    def _build_feed(self, inputs: dict) -> dict:
        feed = dict(inputs)
        required = self._get_model_input_names()

        if "position_ids" in required:
            attention_mask = feed["attention_mask"]
            position_ids = np.cumsum(attention_mask, axis=1) - 1
            feed["position_ids"] = np.where(attention_mask, position_ids, 0).astype(np.int64)

        kv_inputs = {inp.name: inp for inp in self.session.get_inputs() if inp.name.startswith("past_key_values")}
        for name, spec in kv_inputs.items():
            # Shape is [batch, heads, 0, head_dim] for empty KV cache; use spec shape hints
            shape = [d if isinstance(d, int) and d >= 0 else 0 for d in spec.shape]
            shape[0] = feed["input_ids"].shape[0]  # batch size
            dtype = np.float16 if spec.type == "tensor(float16)" else np.float32
            feed[name] = np.zeros(shape, dtype=dtype)

        return feed

    def _format_instruction(self, query: str, doc: str) -> str:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _tokenize_pairs(self, query: str, documents: list[str]) -> dict:
        pairs = [self._format_instruction(query, doc) for doc in documents]
        max_pair_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_pair_length <= 0:
            raise ValueError("max_length is too small for the Qwen3 reranker prompt template.")

        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=max_pair_length,
        )
        for i, input_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + input_ids + self.suffix_tokens
        return self.tokenizer.pad(inputs, padding=True, return_tensors="np")

    def _scores_from_logits(self, logits: np.ndarray) -> list[float]:
        final_token_logits = logits[:, -1, :]
        yes_logits = final_token_logits[:, self.token_true_id]
        no_logits = final_token_logits[:, self.token_false_id]

        if not self.normalize:
            return [float(score) for score in yes_logits - no_logits]

        max_logits = np.maximum(yes_logits, no_logits)
        yes_exp = np.exp(yes_logits - max_logits)
        no_exp = np.exp(no_logits - max_logits)
        return [float(score) for score in yes_exp / (yes_exp + no_exp)]
    
    def score(
            self, 
            query: str, 
            documents: list[str]
    ) -> list[float]:
        """
        Scores a list of documents against a query using ONNX Runtime in batches.
        """
        
        all_scores = []
        
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            inputs = self._tokenize_pairs(query, batch_docs)
            
            # Map tokenizer inputs to ONNX expected inputs
            feed = self._build_feed(dict(inputs))
            outputs = self.session.run(None, feed)
            all_scores.extend(self._scores_from_logits(np.asarray(outputs[0])))

        return all_scores

    def rerank(
            self, 
            query: str, 
            documents: list[str]
    ) -> list[tuple[str, float]]:
        """
        Reranks a list of documents against a query using ONNX Runtime in batches.
        """
        all_scores = self.score(query, documents)
        scored_results = list(zip(documents, all_scores))
        
        return sorted(scored_results, key=lambda x: x[1], reverse=True)

    def rerank_batched(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        return self.rerank(query, documents)





if __name__ == "__main__":
    MODEL_DIR = 'onnx_models/qwen3_reranker_0dot6b/onnx/model_q4.onnx'
    TOKENIZER_DIR = 'onnx_models/qwen3_reranker_0dot6b'
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
    
    reranker = ONNXReranker(MODEL_DIR, TOKENIZER_DIR, MAX_LENGTH)

    reranking = reranker.rerank(
        query=user_query,
        documents=retrieved_docs,
    )
    
    # Print results
    print("\n--- Re-ranked Results ---")
    for doc, score in reranking:
        print(f"[{score:.4f}] -> {doc}")