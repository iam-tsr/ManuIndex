from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

class ONNXReranker:
    def __init__(
            self, 
            model: str, 
            tokenizer: str, 
            max_length: int,
            batch_size: int = 16,
            normalize: bool = True,
            device: str = "cpu",
            reranker_type: str = "auto",
    ):
        """Reranker class for ONNX reranking models.

        Supports Qwen decoder rerankers, BGE decoder rerankers, and BGE
        sequence-classification rerankers such as bge-reranker-v2-m3.
        
        Args:
            model: Path to the ONNX model file.
            tokenizer: Hugging Face model name or path for the tokenizer.
            max_length: Maximum sequence length for tokenization.
            batch_size: The number of documents to process in a single batch.
            normalize: Whether to normalize the logits.
            device: Device to run inference on, either 'cpu' or 'cuda'. Default is 'cpu'.
            reranker_type: One of 'auto', 'qwen_decoder', 'bge_decoder', or 'bge_classifier'.
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

        self.reranker_type = self._resolve_reranker_type(reranker_type, tokenizer)
        tokenizer_kwargs = {}
        if self._is_decoder_reranker():
            tokenizer_kwargs["padding_side"] = "left"
            tokenizer_kwargs["fix_mistral_regex"] = True

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 4

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_kwargs)
        if hasattr(self.tokenizer, "deprecation_warnings"):
            self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.session = ort.InferenceSession(model, sess_options=session_options, providers=providers)
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.token_false_id: int | None = None
        self.token_true_id: int | None = None
        self.prefix = ""
        self.suffix = ""
        self.prefix_tokens: list[int] = []
        self.suffix_tokens: list[int] = []

        if self._is_decoder_reranker():
            self._configure_decoder_prompt()

    def __call__(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        return self.rerank(query, documents)

    def _get_model_input_names(self) -> set[str]:
        return {inp.name for inp in self.session.get_inputs()}

    def _resolve_reranker_type(self, reranker_type: str, tokenizer: str) -> str:
        supported = {"auto", "qwen_decoder", "bge_decoder", "bge_classifier"}
        if reranker_type not in supported:
            raise ValueError(f"Unsupported reranker_type: {reranker_type}. Choose one of {sorted(supported)}.")

        if reranker_type != "auto":
            return reranker_type

        config = self._load_model_config(tokenizer)
        model_type = str(config.get("model_type", "")).lower()
        architectures = " ".join(config.get("architectures", [])).lower()
        tokenizer_name = str(tokenizer).lower()
        model_family = " ".join([model_type, architectures, tokenizer_name])

        if "qwen" in model_family:
            return "qwen_decoder"
        if "forsequenceclassification" in architectures or "xlm-roberta" in model_family:
            return "bge_classifier"
        if "bge" in model_family and any(name in model_family for name in ("gemma", "minicpm", "llm")):
            return "bge_decoder"

        raise ValueError(
            "Could not infer reranker_type. Pass one of: "
            "'qwen_decoder', 'bge_decoder', or 'bge_classifier'."
        )

    def _load_model_config(self, tokenizer: str) -> dict:
        config_path = Path(tokenizer) / "config.json"
        if not config_path.exists():
            return {}

        try:
            return json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return {}

    def _is_decoder_reranker(self) -> bool:
        return self.reranker_type in {"qwen_decoder", "bge_decoder"}

    def _configure_decoder_prompt(self) -> None:
        if self.reranker_type == "qwen_decoder":
            true_token = "yes"
            false_token = "no"
            self.prefix = (
                "<|im_start|>system\n"
                "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                "Note that the answer can only be \"yes\" or \"no\"."
                "<|im_end|>\n<|im_start|>user\n"
            )
            self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        else:
            true_token = "Yes"
            false_token = "No"
            self.prefix = (
                "Given a query A and a passage B, determine whether the passage contains an answer "
                "to the query by providing a prediction of either 'Yes' or 'No'.\n"
            )
            self.suffix = "\nAnswer:"

        self.token_true_id = self._single_token_id(true_token)
        self.token_false_id = self._single_token_id(false_token)
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _single_token_id(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            return token_id

        encoded = self.tokenizer.encode(token, add_special_tokens=False)
        if not encoded:
            raise ValueError(f"Could not encode reranker label token: {token!r}")
        return int(encoded[-1])

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

    def _format_qwen_instruction(self, query: str, doc: str) -> str:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def _format_bge_decoder_instruction(self, query: str, doc: str) -> str:
        return f"A: {query}\nB: {doc}"

    def _format_decoder_instruction(self, query: str, doc: str) -> str:
        if self.reranker_type == "qwen_decoder":
            return self._format_qwen_instruction(query, doc)
        return self._format_bge_decoder_instruction(query, doc)

    def _tokenize_decoder_pairs(self, query: str, documents: list[str]) -> dict:
        pairs = [self._format_decoder_instruction(query, doc) for doc in documents]
        max_pair_length = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if max_pair_length <= 0:
            raise ValueError(f"max_length is too small for the {self.reranker_type} prompt template.")

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

    def _tokenize_classifier_pairs(self, query: str, documents: list[str]) -> dict:
        pairs = [[query, doc] for doc in documents]
        return self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )

    def _tokenize_pairs(self, query: str, documents: list[str]) -> dict:
        if self._is_decoder_reranker():
            return self._tokenize_decoder_pairs(query, documents)
        return self._tokenize_classifier_pairs(query, documents)

    def _scores_from_logits(self, logits: np.ndarray) -> list[float]:
        if self.reranker_type == "bge_classifier":
            return self._classifier_scores_from_logits(logits)

        if self.token_true_id is None or self.token_false_id is None:
            raise ValueError("Decoder reranker label token ids are not configured.")

        final_token_logits = logits[:, -1, :]
        yes_logits = final_token_logits[:, self.token_true_id]
        no_logits = final_token_logits[:, self.token_false_id]

        if not self.normalize:
            return [float(score) for score in yes_logits - no_logits]

        max_logits = np.maximum(yes_logits, no_logits)
        yes_exp = np.exp(yes_logits - max_logits)
        no_exp = np.exp(no_logits - max_logits)
        return [float(score) for score in yes_exp / (yes_exp + no_exp)]

    def _classifier_scores_from_logits(self, logits: np.ndarray) -> list[float]:
        scores = np.asarray(logits).reshape(-1)
        if not self.normalize:
            return [float(score) for score in scores]

        scores = np.clip(scores, -50, 50)
        return [float(score) for score in 1 / (1 + np.exp(-scores))]
    
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