import json

from langchain_community.vectorstores import FAISS

from ._shared import select_top_k, split_documents, tokenize_keywords


class QueryRewriteRAG:
    def __init__(
        self,
        embeddings,
        top_k: int = 3,
        chunk_size: int = 150,
        rewrite_count: int = 3,
        client=None,
        model_name: str | None = None,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.rewrite_count = rewrite_count
        self.client = client
        self.model_name = model_name
        self._last_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated": False,
        }

    def get_chunks(self, document: str):
        return split_documents(document, chunk_size=self.chunk_size, chunk_overlap=30)

    def _estimate_text_tokens(self, text: str) -> int:
        # Rough approximation for accounting when provider usage is unavailable.
        return max(1, (len(text) + 3) // 4) if text else 0

    def _set_last_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        estimated: bool = False,
    ) -> None:
        computed_total = input_tokens + output_tokens if total_tokens is None else total_tokens
        self._last_token_usage = {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "total_tokens": int(computed_total),
            "estimated": estimated,
        }

    def consume_token_usage(self) -> dict[str, int | bool]:
        usage = dict(self._last_token_usage)
        self._set_last_token_usage()
        return usage

    def rewrite_query(self, user_question: str) -> list[str]:
        self._set_last_token_usage()
        if self.client is not None and self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the user's question into short retrieval-oriented alternatives. Return JSON only with the shape {\"queries\": [\"...\"]}.",
                    },
                    {
                        "role": "user",
                        "content": f"Question: {user_question}\nReturn at most {self.rewrite_count} rewrites.",
                    },
                ],
                temperature=0,
                max_tokens=256,
            )
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                total_tokens = int(
                    getattr(usage, "total_tokens", prompt_tokens + completion_tokens)
                    or (prompt_tokens + completion_tokens)
                )
                self._set_last_token_usage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated=False,
                )
            else:
                prompt_text = (
                    "Rewrite the user's question into short retrieval-oriented alternatives. "
                    "Return JSON only with the shape {\"queries\": [\"...\"]}.\n"
                    f"Question: {user_question}\nReturn at most {self.rewrite_count} rewrites."
                )
                self._set_last_token_usage(
                    input_tokens=self._estimate_text_tokens(prompt_text),
                    output_tokens=self._estimate_text_tokens(content),
                    estimated=True,
                )
            try:
                parsed = json.loads(content)
                queries = [query.strip() for query in parsed.get("queries", []) if isinstance(query, str) and query.strip()]
                if queries:
                    return [user_question, *queries][: self.rewrite_count + 1]
            except json.JSONDecodeError:
                pass

        keywords = tokenize_keywords(user_question)
        keyword_query = " ".join(keywords[: min(len(keywords), 8)]).strip()
        rewrites = [
            user_question,
            f"Find the passage that answers: {user_question}",
            f"Relevant details about {keyword_query or user_question}",
            f"{user_question} key facts",
        ]
        unique_rewrites: list[str] = []
        for rewrite in rewrites:
            if rewrite not in unique_rewrites:
                unique_rewrites.append(rewrite)
        return unique_rewrites[: self.rewrite_count + 1]

    def main(self, document: str, user_question: str) -> list[str]:
        chunks = self.get_chunks(document)
        vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        queries = self.rewrite_query(user_question)

        texts: list[str] = []
        for query in queries:
            docs = vector_store.similarity_search(query, k=max(self.top_k, 2))
            texts.extend(doc.page_content for doc in docs)

        return select_top_k(texts, self.top_k)
