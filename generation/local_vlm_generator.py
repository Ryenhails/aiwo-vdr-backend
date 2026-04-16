"""Local VLM answer generation using a Qwen2.5-VL-Instruct model.

Kept dependency-light: imports torch/transformers lazily in __init__ so that
`from generation.local_vlm_generator import LocalVLMGenerator` never fails
when the backend is configured to use Azure or Claude instead.
"""

import os
from pathlib import Path


SYSTEM_PROMPT = """You are an expert technical assistant for operators of industrial forestry machines. You are shown page images from the equipment's operator manual and the prior conversation.

Answering rules:
1. Ground every claim in what is visible on the pages or already established earlier in the conversation. Never invent part numbers, torque values, service intervals, or part locations.
2. For procedures, answer as a short numbered list of concrete steps (at most 8 steps).
3. When the pages label parts or figures (e.g. "bolt (5)", "Picture 3"), refer to them by the same labels.
4. Surface any safety warnings (DANGER / WARNING / CAUTION) and any relevant torque, pressure, temperature, or interval values that appear on the pages.
5. If the answer is not on the provided pages and was not established earlier, say so clearly and name the manual section that likely covers it.
6. Answer in the same language the operator used in their question.
7. Keep answers tight — operators need quick, practical guidance, not an essay."""


class LocalVLMGenerator:
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        torch_dtype: str = "float16",
        max_new_tokens: int = 1024,
    ):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.model_name = model_name or os.environ.get(
            "LOCAL_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"
        )
        self.device = device or os.environ.get("LOCAL_VLM_DEVICE", "cuda")
        self.max_new_tokens = max_new_tokens

        dtype = getattr(torch, torch_dtype)
        print(f"Loading local VLM: {self.model_name} on {self.device} ({torch_dtype})")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        self._torch = torch

    # ── internal helpers ──────────────────────────────────────────────
    def _run(self, vlm_messages: list[dict], max_new_tokens: int) -> str:
        """Tokenise `vlm_messages`, call .generate(), return decoded text."""
        text = self.processor.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs = None
        video_inputs = None
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(vlm_messages)
        except Exception:
            pass

        proc_kwargs = dict(text=[text], padding=True, return_tensors="pt")
        if image_inputs:
            proc_kwargs["images"] = image_inputs
        if video_inputs:
            proc_kwargs["videos"] = video_inputs
        inputs = self.processor(**proc_kwargs).to(self.device)

        with self._torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )

        trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
        out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out[0].strip() if out else ""

    # ── public API ────────────────────────────────────────────────────
    def generate(
        self,
        query: str,
        retrieved_pages: list[dict],
        image_dir: str = "data/images",
        max_pages: int = 3,
    ) -> str:
        """Single-shot generation (legacy /query endpoint)."""
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
        """Multi-turn generation.

        History turns are replayed text-only; the images from
        `retrieved_pages` are attached to the LAST user turn, where the VLM
        is about to answer. The model sees the full transcript plus the
        current pages in one prompt.
        """
        vlm_messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        ]

        for m in messages[:-1]:
            if m["role"] not in ("user", "assistant"):
                continue
            vlm_messages.append({
                "role": m["role"],
                "content": [{"type": "text", "text": m["content"]}],
            })

        last = messages[-1]
        assert last["role"] == "user", "last message must be a user turn"

        user_content: list[dict] = []
        pages_used = 0
        for page in retrieved_pages[:max_pages]:
            img_path = os.path.join(image_dir, page["image_path"])
            if not os.path.exists(img_path):
                continue
            user_content.append({"type": "image", "image": img_path})
            pages_used += 1

        if pages_used == 0:
            user_content.append({"type": "text", "text": last["content"]})
        else:
            user_content.insert(0, {"type": "text", "text": "Relevant manual pages:"})
            user_content.append({
                "type": "text",
                "text": f"User question: {last['content']}\n\n"
                        "Answer based on the pages above and what was discussed earlier.",
            })

        vlm_messages.append({"role": "user", "content": user_content})

        return self._run(vlm_messages, self.max_new_tokens)

    def generate_text(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Lightweight text-only inference, used for routing / classification."""
        vlm_messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        return self._run(vlm_messages, max_new_tokens)
