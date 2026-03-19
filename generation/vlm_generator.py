"""VLM-based answer generation using Azure OpenAI.

Takes retrieved document page images + query, generates an answer
using GPT-4o vision capabilities.
"""

import base64
import os
from pathlib import Path

from openai import AzureOpenAI


SYSTEM_PROMPT = """You are a helpful technical assistant for industrial equipment operators.
You are given document pages from equipment manuals and a user's question.
Answer the question based ONLY on the information visible in the provided document pages.
If the answer is not found in the pages, say so clearly.
Be concise and practical — operators need quick, actionable answers."""


class VLMGenerator:
    def __init__(
        self,
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-10-21",
        deployment: str = "gpt-4o",
    ):
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
        )
        self.deployment = deployment

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def generate(
        self,
        query: str,
        retrieved_pages: list[dict],
        image_dir: str = "data/images",
        max_pages: int = 3,
    ) -> str:
        """Generate answer using retrieved page images + query.

        Args:
            query: User's question text
            retrieved_pages: List of dicts from retriever (with image_path, score, etc.)
            image_dir: Base directory for page images
            max_pages: Max number of pages to send to VLM

        Returns:
            Generated answer text
        """
        # Build message content with images
        content = [{"type": "text", "text": f"User question: {query}\n\nRelevant manual pages:"}]

        pages_used = 0
        for page in retrieved_pages[:max_pages]:
            img_path = os.path.join(image_dir, page["image_path"])
            if not os.path.exists(img_path):
                continue

            b64_image = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_image}",
                    "detail": "high",
                },
            })
            pages_used += 1

        if pages_used == 0:
            return "No relevant manual pages found for your question."

        content.append({
            "type": "text",
            "text": "Based on the manual pages above, please answer the user's question.",
        })

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        return response.choices[0].message.content
