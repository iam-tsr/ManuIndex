import json

from langchain_community.vectorstores import FAISS

from ._shared import join_top_k, split_documents, tokenize_keywords


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

    def get_chunks(self, document: str):
        return split_documents(document, chunk_size=self.chunk_size)

    def rewrite_query(self, user_question: str) -> list[str]:
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

    def main(self, document: str, user_question: str) -> str:
        chunks = self.get_chunks(document)
        vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
        queries = self.rewrite_query(user_question)

        texts: list[str] = []
        for query in queries:
            docs = vector_store.similarity_search(query, k=max(self.top_k, 2))
            texts.extend(doc.page_content for doc in docs)

        return join_top_k(texts, self.top_k)
