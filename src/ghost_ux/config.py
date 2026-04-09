from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


def _load_local_dotenv() -> None:
    dotenv_path = Path(".env")
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_dotenv()


class BrowserConfig(BaseModel):
    headless: bool = True
    viewport_width: int = 1440
    viewport_height: int = 960
    navigation_timeout_ms: int = 20_000
    action_timeout_ms: int = 8_000
    max_dom_elements: int = 40
    screenshot_full_page: bool = False
    locale: str = "en-US"
    timezone_id: str = "Asia/Shanghai"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )


class ModelConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    language: Literal["en", "zh"] = "en"
    replay_path: Path | None = None
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_output_tokens: int = 900
    retry_limit: int = 2
    retry_backoff_seconds: float = 2.0
    max_retry_wait_seconds: float = 45.0
    include_url_in_prompt: bool = True
    include_title_in_prompt: bool = True
    image_detail: Literal["low", "high", "auto"] = "auto"

    @property
    def resolved_base_url(self) -> str | None:
        if self.base_url:
            return self.base_url
        if self.provider.lower() == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return None

    @property
    def resolved_model_name(self) -> str:
        if self.provider.lower() == "gemini" and "/" in self.model:
            return self.model.split("/")[-1]
        return self.model

    @property
    def resolved_api_key(self) -> str:
        if self.provider.lower() == "mock":
            return self.api_key or "mock-api-key-not-required"
        api_key = self.api_key or os.getenv(self.api_key_env)
        if not api_key:
            raise ValueError(f"Missing API key. Set `{self.api_key_env}` or provide `api_key` in config.")
        return api_key


class AgentConfig(BaseModel):
    max_steps: int = 15
    min_action_delay_ms: int = 350
    max_action_delay_ms: int = 1200
    low_confidence_threshold: float = 0.45
    retry_limit_per_action: int = 2
    stall_detection_window: int = 4
    repeat_action_limit: int = 2
    persona: str
    goal: str


class SensoryConfig(BaseModel):
    auto_from_persona: bool = True
    filters: list[str] = Field(default_factory=list)
    blur_radius: float = 3.0
    blurry_vision_severity: Literal["auto", "mild", "moderate", "severe"] = "auto"
    colorblind_mode: Literal["achromatopsia", "protanopia"] = "protanopia"
    tunnel_visible_width_ratio: float = 0.5
    tunnel_visible_height_ratio: float = 0.5
    tunnel_darkness: int = 230
    tunnel_blur_radius: float = 36.0
    tunnel_min_visible_ratio: float = 0.55
    tunnel_safety_inset_ratio: float = 0.08
    low_patience_max_text_length: int = 48
    low_patience_max_y_ratio: float = 0.68
    symbol_mask_padding_px: int = 6
    symbol_nearby_text_radius_px: int = 48
    symbol_dom_strategy: Literal["remove", "placeholder_only"] = "remove"
    symbol_mask_style: Literal["blackout", "mosaic"] = "blackout"
    cognitive_enabled_domains: list[str] = Field(default_factory=lambda: ["general", "b2b_saas", "ai", "web3"])
    cognitive_custom_terms: list[str] = Field(default_factory=list)
    cognitive_placeholder: str = "[unfamiliar jargon]"
    cognitive_case_sensitive: bool = False
    cognitive_visual_scrub: bool = True
    cognitive_visual_scrub_strength: Literal["off", "light"] = "light"
    cognitive_phrase_density_threshold: int = 2
    cognitive_visual_max_text_length: int = 30


class MotorConfig(BaseModel):
    auto_from_persona: bool = True
    filters: list[str] = Field(default_factory=list)
    profile: Literal[
        "none",
        "tipsy",
        "drunk",
        "subway_one_hand",
        "tremor",
        "parkinson_light",
        "parkinson_strong",
    ] = "none"
    click_offset_px: int = 0
    scroll_noise_px: int = 0
    disable_precise_click_fallback: bool = True


class ReportConfig(BaseModel):
    output_dir: Path = Path("artifacts")
    keep_screenshots: bool = True


class DebugConfig(BaseModel):
    capture_raw_observation: bool = True
    capture_filtered_observation: bool = True
    capture_prompt_payload: bool = True
    leak_probe_mode: bool = True


class SessionConfig(BaseModel):
    start_url: str
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig
    sensory: SensoryConfig = Field(default_factory=SensoryConfig)
    motor: MotorConfig = Field(default_factory=MotorConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @field_validator("start_url", mode="before")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("start_url must be a string.")
        normalized = value.strip()
        parsed = urlparse(normalized)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return normalized
        if parsed.scheme == "file" and parsed.path:
            return normalized
        raise ValueError("start_url must be an absolute http(s) URL or an absolute file:// URL.")

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SessionConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
