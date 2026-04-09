from __future__ import annotations

from ghost_ux.config import ModelConfig
from ghost_ux.llm.base import VisionModelClient
from ghost_ux.llm.mock_client import MockReplayVisionClient
from ghost_ux.llm.openai_client import OpenAIVisionClient


PROVIDER_ALIASES = {
    "openai",
    "openai_compatible",
    "qwen",
    "gemini",
    "dashscope",
    "openrouter",
    "deepseek",
}


def build_vision_client(model_config: ModelConfig) -> VisionModelClient:
    if model_config.provider.lower() == "mock":
        return MockReplayVisionClient(model_config)
    if model_config.provider.lower() in PROVIDER_ALIASES:
        return OpenAIVisionClient(model_config)
    raise ValueError(
        f"Unsupported provider `{model_config.provider}`. "
        "Add a new adapter in ghost_ux.llm.factory."
    )
