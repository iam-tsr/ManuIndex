import re
from typing import Any

SYSTEM_PROMPT = """You are a document summarization expert. Your task is to produce a short, dense, narrative summary that captures the document's core topics, key arguments, and domain — in 4 to 5 sentences. The summary will be used as a routing vector in a retrieval system, so it must be semantically rich and representative of the full document."""

JUDGE_PROMPT = """You are evaluating whether a list of section headings provides enough document-wide context to write a meaningful summary of the full document.

Respond with only "true" or "false".

Return "true" only when the headings, by themselves, clearly reveal all of the following:
- the document's domain or subject area
- multiple specific topics or subtopics
- enough structure to infer the document's overall scope

Return "false" when any of these apply:
- there are fewer than 4 distinct substantive headings
- the list is mostly a document title plus 1-2 section headings
- the headings are generic labels such as Introduction, Overview, Background, Method, Results, Conclusion, References, or Appendix
- the headings are sparse, ambiguous, repetitive, or lack domain-specific terms
- the headings would force you to guess the document's core content

When uncertain, return "false" so the full document is summarized instead.

Headings:
{titles}"""


class DocumentSummary:
    def __init__(
            self,
            document: str,
            client: Any,
            model_name: str,
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
            max_tokens=1024,
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
            max_tokens=256, # Some models may have a lower max token limit, so keeping it at 256 to be safe
        )
        content = response.choices[0].message.content
        return content.strip().lower() == "true"