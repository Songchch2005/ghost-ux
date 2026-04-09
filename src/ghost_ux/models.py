from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL_DOWN = "scroll_down"
    SCROLL_UP = "scroll_up"
    FINISH = "finish"
    FAIL = "fail"


class UIAction(BaseModel):
    thought: str = Field(
        description="Persona driven inner monologue that explains why this next step makes sense."
    )
    action_type: ActionType = Field(
        description="One of click, type, scroll_down, scroll_up, finish, fail."
    )
    target_element_id: str | None = Field(
        default=None,
        description="Custom DOM element id for click or type actions.",
    )
    input_text: str | None = Field(default=None, description="Text to enter for type actions.")
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0.",
    )

    @model_validator(mode="after")
    def _validate_targets(self) -> "UIAction":
        if self.action_type in {ActionType.CLICK, ActionType.TYPE} and not self.target_element_id:
            raise ValueError("target_element_id is required for click/type actions.")
        if self.action_type == ActionType.TYPE and not self.input_text:
            raise ValueError("input_text is required for type actions.")
        return self


class FilterEffect(BaseModel):
    filter_name: str
    removed_count: int = 0
    modified_count: int = 0
    reasons: dict[str, int] = Field(default_factory=dict)


class PromptContextFlags(BaseModel):
    include_url: bool = True
    include_title: bool = True
    include_dom: bool = True
    include_image: bool = True
    image_detail: str = "auto"


class LeakProbeResult(BaseModel):
    probe_name: str
    visible_tokens: list[str] = Field(default_factory=list)
    rationale: str = ""


class LeakDiagnosis(BaseModel):
    classification: str = "inconclusive"
    suspect_tokens: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    probe_results: list[LeakProbeResult] = Field(default_factory=list)


class DOMElement(BaseModel):
    element_id: str
    tag: str
    is_interactive: bool = True
    role: str | None = None
    text: str | None = None
    visible_text: str | None = None
    nearby_visible_text: str | None = None
    labelledby_text: str | None = None
    title_text: str | None = None
    placeholder: str | None = None
    alt: str | None = None
    aria_label: str | None = None
    title: str | None = None
    name: str | None = None
    href: str | None = None
    value: str | None = None
    input_type: str | None = None
    font_size: float | None = None
    text_color: str | None = None
    background_color: str | None = None
    disabled: bool = False
    css_classes: list[str] = Field(default_factory=list)
    child_tags: list[str] = Field(default_factory=list)
    has_svg_child: bool = False
    has_img_child: bool = False
    has_icon_like_class: bool = False
    icon_x: float | None = None
    icon_y: float | None = None
    icon_width: float | None = None
    icon_height: float | None = None
    unreadable: bool = False
    scrub_reasons: list[str] = Field(default_factory=list)
    scrubbed_fields: list[str] = Field(default_factory=list)
    scrubbed_terms: list[str] = Field(default_factory=list)
    x: float
    y: float
    width: float
    height: float

    def as_prompt_line(self) -> str:
        parts = [f"[ID:{self.element_id}]" if self.is_interactive else "[READONLY]", self.tag]
        if self.role:
            parts.append(f"role={self.role}")
        if self.text:
            parts.append(f"text={self.text!r}")
        if self.placeholder:
            parts.append(f"placeholder={self.placeholder!r}")
        if self.alt:
            parts.append(f"alt={self.alt!r}")
        if self.aria_label:
            parts.append(f"aria={self.aria_label!r}")
        if self.title:
            parts.append(f"title={self.title!r}")
        if self.name:
            parts.append(f"name={self.name!r}")
        if self.href:
            parts.append(f"href={self.href!r}")
        if self.value:
            parts.append(f"value={self.value!r}")
        if self.input_type:
            parts.append(f"type={self.input_type!r}")
        if self.font_size is not None:
            parts.append(f"font_size={self.font_size}")
        if self.disabled:
            parts.append("disabled=true")
        parts.append(
            f"bbox=({round(self.x, 1)}, {round(self.y, 1)}, {round(self.width, 1)}, {round(self.height, 1)})"
        )
        return " | ".join(parts)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + (self.width / 2), self.y + (self.height / 2))


class ExecutableAction(BaseModel):
    action_type: ActionType
    target_element_id: str | None = None
    intended_x: float | None = None
    intended_y: float | None = None
    actual_x: float | None = None
    actual_y: float | None = None
    offset_x: float = 0.0
    offset_y: float = 0.0
    scroll_delta_x: float = 0.0
    scroll_delta_y: float = 0.0
    noise_applied: bool = False
    noise_profile: str | None = None
    pre_hit_summary: str | None = None
    post_hit_summary: str | None = None
    executed_selector: str | None = None
    disable_precise_click_fallback: bool = False


class Observation(BaseModel):
    step_index: int
    url: str
    title: str
    screenshot_bytes: bytes
    screenshot_base64: str
    screenshot_path: Path
    elements: list[DOMElement]
    raw_elements: list[DOMElement] = Field(default_factory=list)
    filtered_elements: list[DOMElement] = Field(default_factory=list)
    last_error: str | None = None
    viewport_width: int
    viewport_height: int
    filter_effects: list[FilterEffect] = Field(default_factory=list)
    dom_removed_count: int = 0
    dom_modified_count: int = 0
    prompt_context_flags: PromptContextFlags = Field(default_factory=PromptContextFlags)
    suspect_tokens: list[str] = Field(default_factory=list)
    cognitive_terms: list[str] = Field(default_factory=list)

    @property
    def interactive_elements(self) -> list[DOMElement]:
        return [element for element in self.elements if element.is_interactive]

    @property
    def raw_interactive_elements(self) -> list[DOMElement]:
        return [element for element in self.raw_elements if element.is_interactive]

    @property
    def filtered_interactive_elements(self) -> list[DOMElement]:
        return [element for element in self.filtered_elements if element.is_interactive]

    @property
    def dom_prompt(self) -> str:
        prompt_elements = self.interactive_elements
        if not prompt_elements:
            return "No visible interactive elements detected in the current viewport."
        return "\n".join(element.as_prompt_line() for element in prompt_elements)

    @property
    def raw_dom_prompt(self) -> str:
        prompt_elements = self.raw_interactive_elements
        if not prompt_elements:
            return "No visible interactive elements detected in the current viewport."
        return "\n".join(element.as_prompt_line() for element in prompt_elements)

    @property
    def filtered_dom_prompt(self) -> str:
        prompt_elements = self.filtered_interactive_elements
        if not prompt_elements:
            return "No visible interactive elements detected in the current viewport."
        return "\n".join(element.as_prompt_line() for element in prompt_elements)

    @property
    def fingerprint(self) -> str:
        compact_elements = [
            "|".join(
                filter(
                    None,
                    [
                        element.tag,
                        element.role or "",
                        element.text or "",
                        element.placeholder or "",
                        element.aria_label or "",
                        element.href or "",
                    ],
                )
            )
            for element in self.elements[:10]
        ]
        return f"{self.url}::{self.title}::" + "||".join(compact_elements)


class StepRecord(BaseModel):
    step_index: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    observation_url: str
    observation_title: str
    screenshot_path: Path
    action: UIAction
    execution_success: bool
    execution_detail: str
    visible_elements: int
    observation_fingerprint: str
    active_filters: list[str] = Field(default_factory=list)
    filter_effects: list[FilterEffect] = Field(default_factory=list)
    dom_removed_count: int = 0
    dom_modified_count: int = 0
    raw_visible_elements: int = 0
    raw_dom_prompt: str = ""
    filtered_dom_prompt: str = ""
    prompt_context_flags: PromptContextFlags = Field(default_factory=PromptContextFlags)
    suspect_tokens: list[str] = Field(default_factory=list)
    cognitive_terms: list[str] = Field(default_factory=list)
    leak_attribution: str | None = None
    leak_evidence: list[str] = Field(default_factory=list)
    noise_applied: bool = False
    noise_profile: str | None = None
    intended_point: tuple[float, float] | None = None
    actual_point: tuple[float, float] | None = None
    offset: tuple[float, float] | None = None
    actual_hit_summary: str | None = None
    misfire: bool = False
    scroll_delta: tuple[float, float] | None = None


class RunArtifacts(BaseModel):
    session_id: str
    session_dir: Path
    report_path: Path
    playback_path: Path
    active_filters: list[str]
    steps: list[StepRecord]
    final_status: str
    started_at: datetime
    finished_at: datetime


class ActionResult(BaseModel):
    success: bool
    detail: str
    executed_selector: str | None = None
    noise_applied: bool = False
    noise_profile: str | None = None
    intended_point: tuple[float, float] | None = None
    actual_point: tuple[float, float] | None = None
    offset: tuple[float, float] | None = None
    actual_hit_summary: str | None = None
    misfire: bool = False
    scroll_delta: tuple[float, float] | None = None
