import re
from typing import Any

SYSTEM_PROMPT = """
Your job is to create a concise summary in continuous narrative of the content provided in the document.

Document:
{document}

The summary should be short and concise and capture the main themes of the document as best as possible, even if the document is not well-structured or lacks clear content.
"""


class DocumentSummary:
    def __init__(
            self, 
            document: str, 
            client: Any,
            model_name: str = "gpt-4o-mini",
        ):
        """
        Initialize the DocumentSummary class with the document and model.

        Parameters:
        - document (str): The document to be summarized.
        - client (OpenAI): The OpenAI client to use for summarization.
        - model_name (str): The name of the OpenAI model to use for summarization. Default is "gpt-4o-mini".
        """

        self.client = client
        self.document = document
        self.model_name = model_name

    def summarize(
        self,
    ) -> str:
        """
        Generate a summary of the document using the OpenAI model.

        Parameters:
        - document (str): The document to be summarized.
        Returns:
        - str: The generated summary.
        """

        prompt = self._generate_summary_prompt(self.document)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def _generate_summary_prompt(self, document: str) -> list:
        """
        For best case scenarios, pre-process the document to extract titles and generate a summary prompt based on the titles. If no titles are found, use a generic prompt.

        Parameters:
        - document (str): The document to be summarized.
        Returns:
        - str: The generated summary prompt.
        """

        seen = set()
        titles = []
        for line in document.split('\n'):
            if m := re.match(r'^#{1,6} (.+)', line):
                title = m.group(1).rstrip()
                if title not in seen:
                    seen.add(title)
                    titles.append(title)

        if titles:
            document = "\n".join(titles)
        return SYSTEM_PROMPT.format(document=document)