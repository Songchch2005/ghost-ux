from pathlib import Path

from ghost_ux.config import AgentConfig, ModelConfig
from ghost_ux.leak_detection import classify_probe_results, extract_suspect_tokens
from ghost_ux.llm.openai_client import OpenAIVisionClient
from ghost_ux.models import DOMElement, LeakProbeResult, Observation, PromptContextFlags


def _sample_observation() -> Observation:
    raw_elements = [
        DOMElement(
            element_id="1",
            tag="a",
            text="change",
            href="/tag/change/page/1/",
            x=10,
            y=10,
            width=40,
            height=20,
        ),
        DOMElement(
            element_id="2",
            tag="button",
            text="Explore quotes thinking",
            x=10,
            y=40,
            width=120,
            height=30,
        ),
    ]
    filtered_elements = [
        DOMElement(
            element_id="2",
            tag="button",
            text="Explore quotes",
            x=10,
            y=40,
            width=120,
            height=30,
        )
    ]
    return Observation(
        step_index=1,
        url="https://quotes.toscrape.com/",
        title="Quotes to Scrape",
        screenshot_bytes=b"fake-image",
        screenshot_base64="ZmFrZS1pbWFnZQ==",
        screenshot_path=Path("step_01.png"),
        elements=filtered_elements,
        raw_elements=raw_elements,
        filtered_elements=filtered_elements,
        viewport_width=1280,
        viewport_height=720,
    )


def test_extract_suspect_tokens_uses_raw_minus_filtered_delta() -> None:
    observation = _sample_observation()
    tokens = extract_suspect_tokens(observation.raw_elements, observation.filtered_elements)
    assert "change" in tokens
    assert "thinking" in tokens
    assert "explore" not in tokens


def test_classify_probe_results_flags_dom_leak() -> None:
    diagnosis = classify_probe_results(
        [
            LeakProbeResult(probe_name="full", visible_tokens=["change"], rationale="I can see it."),
            LeakProbeResult(probe_name="dom_only", visible_tokens=["change"], rationale="Still present."),
            LeakProbeResult(probe_name="image_only", visible_tokens=[], rationale="No OCR hit."),
            LeakProbeResult(probe_name="blind_context", visible_tokens=["change"], rationale="Still present."),
        ]
    )
    assert diagnosis.classification == "dom_leak_suspected"


def test_classify_probe_results_flags_prior_knowledge() -> None:
    diagnosis = classify_probe_results(
        [
            LeakProbeResult(probe_name="full", visible_tokens=["world"], rationale="Likely from context."),
            LeakProbeResult(probe_name="dom_only", visible_tokens=[], rationale="No DOM signal."),
            LeakProbeResult(probe_name="image_only", visible_tokens=[], rationale="No image signal."),
            LeakProbeResult(probe_name="blind_context", visible_tokens=[], rationale="Gone without identity."),
        ]
    )
    assert diagnosis.classification == "prior_knowledge_leak_suspected"


def test_openai_payload_respects_prompt_context_switches() -> None:
    client = OpenAIVisionClient(
        ModelConfig(provider="openai", model="gpt-4o", api_key="test-key")
    )
    observation = _sample_observation()
    payload = client.build_decision_payload(
        observation,
        AgentConfig(persona="A careful user.", goal="Inspect the page."),
        history=[],
        prompt_context=PromptContextFlags(
            include_url=False,
            include_title=False,
            include_dom=True,
            include_image=False,
            image_detail="low",
        ),
    )
    user_content = payload["messages"][1]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 1
    user_text = user_content[0]["text"]
    assert "- URL:" not in user_text
    assert "- Title:" not in user_text
    assert "Visible Interactive Elements" in user_text


def test_openai_leak_probe_payload_can_strip_dom_or_image() -> None:
    client = OpenAIVisionClient(
        ModelConfig(provider="openai", model="gpt-4o", api_key="test-key")
    )
    observation = _sample_observation()
    payload = client.build_leak_probe_payload(
        observation,
        AgentConfig(persona="A careful user.", goal="Inspect the page."),
        history=[],
        suspect_tokens=["change", "thinking"],
        probe_name="image_only",
        prompt_context=PromptContextFlags(
            include_url=False,
            include_title=False,
            include_dom=False,
            include_image=True,
            image_detail="low",
        ),
    )
    user_content = payload["messages"][1]["content"]
    assert isinstance(user_content, list)
    assert any(item["type"] == "image_url" for item in user_content)
    assert "DOM Snapshot:\nDOM intentionally withheld." in user_content[0]["text"]
