from __future__ import annotations

import json
import re
from pathlib import Path

from ghost_ux.models import DOMElement, LeakDiagnosis, LeakProbeResult, Observation


TOKEN_PATTERN = re.compile(r"[a-z0-9-]{4,}")
STOP_TOKENS = {
    "http",
    "https",
    "www",
    "page",
    "quote",
    "quotes",
    "href",
    "link",
    "button",
    "value",
    "title",
    "aria",
    "name",
    "text",
    "from",
    "with",
}


def _tokenize(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token
        for token in TOKEN_PATTERN.findall(value.lower())
        if token not in STOP_TOKENS
    }


def _element_tokens(element: DOMElement) -> set[str]:
    tokens: set[str] = set()
    for value in (
        element.text,
        element.aria_label,
        element.title,
        element.name,
        element.placeholder,
        element.value,
        element.href,
    ):
        tokens.update(_tokenize(value))
    return tokens


def extract_suspect_tokens(raw_elements: list[DOMElement], filtered_elements: list[DOMElement]) -> list[str]:
    raw_tokens = set().union(*(_element_tokens(element) for element in raw_elements)) if raw_elements else set()
    filtered_tokens = set().union(*(_element_tokens(element) for element in filtered_elements)) if filtered_elements else set()
    return sorted(raw_tokens - filtered_tokens)


def observation_snapshot_dict(observation: Observation, *, raw: bool) -> dict[str, object]:
    elements = observation.raw_elements if raw else observation.filtered_elements
    return {
        "step_index": observation.step_index,
        "url": observation.url,
        "title": observation.title,
        "viewport_width": observation.viewport_width,
        "viewport_height": observation.viewport_height,
        "suspect_tokens": observation.suspect_tokens,
        "elements": [element.model_dump(mode="json") for element in elements],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_probe_results(probe_results: list[LeakProbeResult]) -> LeakDiagnosis:
    result_by_name = {result.probe_name: result for result in probe_results}
    full_tokens = set(result_by_name.get("full", LeakProbeResult(probe_name="full")).visible_tokens)
    dom_tokens = set(result_by_name.get("dom_only", LeakProbeResult(probe_name="dom_only")).visible_tokens)
    image_tokens = set(result_by_name.get("image_only", LeakProbeResult(probe_name="image_only")).visible_tokens)
    blind_tokens = set(result_by_name.get("blind_context", LeakProbeResult(probe_name="blind_context")).visible_tokens)

    evidence: list[str] = []
    classification = "inconclusive"

    if dom_tokens:
        classification = "dom_leak_suspected"
        evidence.append(f"Tokens visible in DOM-only probe: {', '.join(sorted(dom_tokens))}.")
    elif image_tokens:
        classification = "image_ocr_leak_suspected"
        evidence.append(f"Tokens visible in image-only probe: {', '.join(sorted(image_tokens))}.")
    elif full_tokens and not blind_tokens:
        classification = "prior_knowledge_leak_suspected"
        evidence.append(
            "Tokens appeared only when URL/title context was present, but disappeared in blind-context probing."
        )
    elif sum(bool(tokens) for tokens in (full_tokens, dom_tokens, image_tokens, blind_tokens)) >= 2:
        classification = "mixed_signal"
        evidence.append("Multiple probe types produced suspicious tokens, suggesting a mixed leakage source.")
    else:
        evidence.append("No probe produced strong enough evidence to confidently attribute the leakage source.")

    suspect_tokens = sorted(full_tokens | dom_tokens | image_tokens | blind_tokens)
    if blind_tokens and full_tokens and full_tokens != blind_tokens:
        evidence.append(
            f"Blind-context difference: full={sorted(full_tokens)}, blind_context={sorted(blind_tokens)}."
        )
    return LeakDiagnosis(
        classification=classification,
        suspect_tokens=suspect_tokens,
        evidence=evidence,
        probe_results=probe_results,
    )
