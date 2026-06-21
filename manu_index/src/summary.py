import re
from typing import Any

SYSTEM_PROMPT = """You are a document summarization expert. Your task is to produce a short, dense, narrative summary that captures the document's core topics, key arguments, and domain — in 3 to 5 sentences. The summary will be used as a routing vector in a retrieval system, so it must be semantically rich and representative of the full document."""

JUDGE_PROMPT = """You are evaluating whether a list of section headings provides sufficient context to write a meaningful summary of a document.

Respond with only "true" or "false".

- "true"  → the headings clearly convey the document's topics, domain, and structure
- "false" → the headings are too sparse, generic, or ambiguous to summarize the document reliably

Headings:
{titles}"""


class DocumentSummary:
    def __init__(
            self,
            document: str,
            client: Any,
            model_name: str = "gpt-4o-mini",
        ):
        self.client = client
        self.document = document
        self.model_name = model_name

    def summarize(self) -> str:
        titles = self._extract_titles(self.document)
        content = "\n".join(titles) if titles and self._titles_sufficient(titles) else self.document

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content

    def _extract_titles(self, document: str) -> list[str]:
        seen = set()
        titles = []
        for line in document.split('\n'):
            if m := re.match(r'^#{1,6} (.+)', line):
                title = m.group(1).rstrip()
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        return titles

    def _titles_sufficient(self, titles: list[str]) -> bool:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": JUDGE_PROMPT.format(titles="\n".join(titles))},
            ],
            temperature=0,
            max_tokens=5,
        )
        print("Judge response:", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip().lower() == "true"
