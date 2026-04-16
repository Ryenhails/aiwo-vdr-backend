"""Backend selection for answer generation.

Picks one of {azure, claude, local} at startup based on `VLM_BACKEND` env
var. Default is `azure` to preserve the original behaviour before the
TUNI-integration refactor.
"""

import os


def build_generator():
    backend = os.environ.get("VLM_BACKEND", "azure").lower().strip()

    if backend == "azure":
        from generation.vlm_generator import VLMGenerator
        deployment = os.environ.get("VLM_DEPLOYMENT", "gpt-4o")
        return VLMGenerator(deployment=deployment), "azure"

    if backend == "claude":
        from generation.claude_generator import ClaudeGenerator
        return ClaudeGenerator(), "claude"

    if backend == "local":
        from generation.local_vlm_generator import LocalVLMGenerator
        return LocalVLMGenerator(), "local"

    raise ValueError(
        f"Unknown VLM_BACKEND={backend!r}. Expected one of: azure, claude, local."
    )
