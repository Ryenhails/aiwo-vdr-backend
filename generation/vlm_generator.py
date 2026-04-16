"""VLM-based answer generation using Azure OpenAI.

Takes retrieved document page images + query, generates an answer
using GPT-4o vision capabilities.
"""

import base64
import os
from pathlib import Path

from openai import AzureOpenAI


SYSTEM_PROMPT = """You are an expert technical assistant for operators of industrial forestry machines. You are shown page images from the equipment's operator manual and the prior conversation.

Answering rules:
1. Ground every claim in what is visible on the pages or already established earlier in the conversation. Never invent part numbers, torque values, service intervals, or part locations.
2. For procedures, answer as a short numbered list of concrete steps (at most 8 steps).
3. When the pages label parts or figures (e.g. "bolt (5)", "Picture 3"), refer to them by the same labels.
4. Surface any safety warnings (DANGER / WARNING / CAUTION) and any relevant torque, pressure, temperature, or interval values that appear on the pages.
5. If the answer is not on the provided pages and was not established earlier, say so clearly and name the manual section that likely covers it.
6. Answer in the same language the operator used in their question.
7. Keep answers tight — operators need quick, practical guidance, not an essay."""


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
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _last_user_content(
        self, query: str, retrieved_pages: list[dict], image_dir: str, max_pages: int
    ) -> list[dict]:
        content: list[dict] = []
        pages_used = 0
        for page in retrieved_pages[:max_pages]:
            img_path = os.path.join(image_dir, page["image_path"])
            if not os.path.exists(img_path):
                continue
            b64 = self._encode_image(img_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            })
            pages_used += 1

        if pages_used == 0:
            content.append({"type": "text", "text": query})
        else:
            content.insert(0, {"type": "text", "text": "Relevant manual pages:"})
            content.append({
                "type": "text",
                "text": f"User question: {query}\n\n"
                        "Answer based on the pages above and what was discussed earlier.",
            })
        return content

    def generate(
        self,
        query: str,
        retrieved_pages: list[dict],
        image_dir: str = "data/images",
        max_pages: int = 3,
    ) -> str:
        return self.generate_chat(
            messages=[{"role": "user", "content": query}],
            retrieved_pages=retrieved_pages,
            image_dir=image_dir,
            max_pages=max_pages,
        )

    def generate_chat(
        self,
        messages: list[dict],
        retrieved_pages: list[dict],
        image_dir: str = "data/images",
        max_pages: int = 3,
    ) -> str:
        oai_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages[:-1]:
            if m["role"] not in ("user", "assistant"):
                continue
            oai_messages.append({"role": m["role"], "content": m["content"]})

        last = messages[-1]
        oai_messages.append({
            "role": "user",
            "content": self._last_user_content(
                query=last["content"],
                retrieved_pages=retrieved_pages,
                image_dir=image_dir,
                max_pages=max_pages,
            ),
        })

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=oai_messages,
            max_tokens=1024,
            temperature=0.3,
        )
        return response.choices[0].message.content

    def generate_text(self, prompt: str, max_new_tokens: int = 32) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()
