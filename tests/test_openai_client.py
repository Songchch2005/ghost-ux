import pytest
from openai import NotFoundError, RateLimitError

from ghost_ux.config import ModelConfig
from ghost_ux.llm.openai_client import OpenAIVisionClient
from ghost_ux.models import ActionType, FilterEffect, Observation, StepRecord, UIAction


class _FakeResponse:
    request = None
    status_code = 404
    headers = {}

    def json(self):
        return {"error": {"message": "not found"}}

    @property
    def text(self):
        return "not found"


class _FakeRateLimitResponse:
    request = None
    status_code = 429
    headers = {}

    def json(self):
        return {"error": {"message": "rate limit"}}

    @property
    def text(self):
        return "Please retry in 1s. retryDelay: 1s"


@pytest.mark.asyncio
async def test_gemini_not_found_error_is_rewritten(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    client = OpenAIVisionClient(
        ModelConfig(provider="gemini", model="gemini-1.5-pro", api_key_env="GEMINI_API_KEY")
    )

    async def _raise_not_found(**kwargs):
        raise NotFoundError("not found", response=_FakeResponse(), body=None)

    client.client.chat.completions.create = _raise_not_found

    observation = Observation(
        step_index=1,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"fake",
        screenshot_base64="ZmFrZQ==",
        screenshot_path="step.png",
        elements=[],
        viewport_width=1280,
        viewport_height=720,
    )

    with pytest.raises(RuntimeError, match="Gemini model not found"):
        await client.decide(
            observation=observation,
            agent_config=type("AgentConfigLike", (), {"persona": "P", "goal": "G"})(),
            history=[],
        )


@pytest.mark.asyncio
async def test_rate_limit_is_retried_once(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIVisionClient(
        ModelConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            retry_limit=1,
            retry_backoff_seconds=0.01,
            max_retry_wait_seconds=0.01,
        )
    )
    calls = {"count": 0}

    async def _rate_limit_then_succeed(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RateLimitError("Please retry in 1s.", response=_FakeRateLimitResponse(), body=None)
        return type(
            "ResponseLike",
            (),
            {
                "choices": [
                    type(
                        "ChoiceLike",
                        (),
                        {
                            "message": type(
                                "MessageLike",
                                (),
                                {
                                    "content": '{"thought":"done","action_type":"finish","target_element_id":null,"input_text":null,"confidence_score":1.0}'
                                },
                            )()
                        },
                    )()
                ]
            },
        )()

    client.client.chat.completions.create = _rate_limit_then_succeed

    observation = Observation(
        step_index=1,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"fake",
        screenshot_base64="ZmFrZQ==",
        screenshot_path="step.png",
        elements=[],
        viewport_width=1280,
        viewport_height=720,
    )

    action = await client.decide(
        observation=observation,
        agent_config=type("AgentConfigLike", (), {"persona": "P", "goal": "G"})(),
        history=[],
    )

    assert calls["count"] == 2
    assert action.action_type.value == "finish"


def test_build_decision_payload_includes_background_misfire_feedback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIVisionClient(
        ModelConfig(provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")
    )
    observation = Observation(
        step_index=2,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"fake",
        screenshot_base64="ZmFrZQ==",
        screenshot_path="step.png",
        elements=[],
        viewport_width=1280,
        viewport_height=720,
    )
    history = [
        StepRecord(
            step_index=1,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path="step.png",
            action=UIAction(
                thought="我点一下登录按钮。",
                action_type=ActionType.CLICK,
                target_element_id="4",
                confidence_score=0.8,
            ),
            execution_success=True,
            execution_detail="Clicked with motor noise.",
            visible_elements=3,
            observation_fingerprint="fp",
            noise_applied=True,
            noise_profile="drunk",
            intended_point=(200.0, 80.0),
            actual_point=(231.0, 109.0),
            offset=(31.0, 29.0),
            actual_hit_summary="div .example text=Welcome",
            misfire=True,
        )
    ]

    payload = client.build_decision_payload(
        observation=observation,
        agent_config=type("AgentConfigLike", (), {"persona": "P", "goal": "G"})(),
        history=history,
    )

    prompt_text = payload["messages"][1]["content"][0]["text"]
    assert "Previous Action Tactile Feedback" in prompt_text
    assert "drunk" in prompt_text
    assert "unresponsive background area" in prompt_text
    assert "[ID:4]" in prompt_text


def test_build_decision_payload_includes_neighbor_mistouch_feedback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIVisionClient(
        ModelConfig(provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")
    )
    observation = Observation(
        step_index=2,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"fake",
        screenshot_base64="ZmFrZQ==",
        screenshot_path="step.png",
        elements=[],
        viewport_width=1280,
        viewport_height=720,
    )
    history = [
        StepRecord(
            step_index=1,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path="step.png",
            action=UIAction(
                thought="我点一下继续按钮。",
                action_type=ActionType.CLICK,
                target_element_id="7",
                confidence_score=0.9,
            ),
            execution_success=True,
            execution_detail="Clicked with motor noise.",
            visible_elements=3,
            observation_fingerprint="fp",
            noise_applied=True,
            noise_profile="tremor",
            intended_point=(120.0, 44.0),
            actual_point=(138.0, 41.0),
            offset=(18.0, -3.0),
            actual_hit_summary="button .cancel text=Cancel",
            misfire=True,
        )
    ]

    payload = client.build_decision_payload(
        observation=observation,
        agent_config=type("AgentConfigLike", (), {"persona": "P", "goal": "G"})(),
        history=history,
    )

    prompt_text = payload["messages"][1]["content"][0]["text"]
    assert "hit another element instead" in prompt_text
    assert "button .cancel text=Cancel" in prompt_text
    assert "background area" not in prompt_text


def test_build_decision_payload_includes_edge_hit_feedback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIVisionClient(
        ModelConfig(provider="openai", model="gpt-4o", api_key_env="OPENAI_API_KEY")
    )
    observation = Observation(
        step_index=2,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"fake",
        screenshot_base64="ZmFrZQ==",
        screenshot_path="step.png",
        elements=[],
        viewport_width=1280,
        viewport_height=720,
    )
    history = [
        StepRecord(
            step_index=1,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path="step.png",
            action=UIAction(
                thought="我点一下主要按钮。",
                action_type=ActionType.CLICK,
                target_element_id="2",
                confidence_score=0.7,
            ),
            execution_success=True,
            execution_detail="Clicked with motor noise.",
            visible_elements=3,
            observation_fingerprint="fp",
            noise_applied=True,
            noise_profile="tipsy",
            intended_point=(80.0, 30.0),
            actual_point=(85.0, 33.0),
            offset=(5.0, 3.0),
            actual_hit_summary="button #2 text=Continue",
            misfire=False,
        )
    ]

    payload = client.build_decision_payload(
        observation=observation,
        agent_config=type("AgentConfigLike", (), {"persona": "P", "goal": "G"})(),
        history=history,
    )

    prompt_text = payload["messages"][1]["content"][0]["text"]
    assert "barely landed near the intended target" in prompt_text
