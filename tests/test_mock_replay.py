from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ghost_ux.agent import GhostUXAgent
from ghost_ux.config import AgentConfig, BrowserConfig, ModelConfig, ReportConfig, SessionConfig
from ghost_ux.llm.factory import build_vision_client
from ghost_ux.models import ActionResult, DOMElement, Observation, StepRecord, UIAction


class FakeBrowser:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.performed_actions: list[str] = []

    async def start(self) -> None:
        return None

    async def goto(self, url: str) -> None:
        return None

    async def observe(self, step_index: int, last_error: str | None = None) -> Observation:
        screenshot_path = self.session_dir / f"fake_step_{step_index:02d}.png"
        screenshot_path.write_bytes(b"fake")
        return Observation(
            step_index=step_index,
            url="https://example.com",
            title="Example Domain",
            screenshot_bytes=b"fake",
            screenshot_base64="ZmFrZQ==",
            screenshot_path=screenshot_path,
            elements=[
                DOMElement(
                    element_id="1",
                    tag="button",
                    text="Start Trial",
                    x=10,
                    y=10,
                    width=100,
                    height=40,
                )
            ],
            last_error=last_error,
            viewport_width=1280,
            viewport_height=720,
        )

    async def perform(self, action, observation=None) -> ActionResult:
        self.performed_actions.append(action.action_type.value)
        return ActionResult(success=True, detail=f"Performed {action.action_type.value}.")

    async def close(self) -> None:
        return None


def _build_config(tmp_path: Path, replay_path: Path) -> SessionConfig:
    return SessionConfig(
        start_url="https://example.com",
        browser=BrowserConfig(),
        model=ModelConfig(provider="mock", model="mock-replay", replay_path=replay_path),
        agent=AgentConfig(
            max_steps=5,
            min_action_delay_ms=0,
            max_action_delay_ms=0,
            persona="A cautious first-time user.",
            goal="Find the trial flow.",
        ),
        report=ReportConfig(output_dir=tmp_path, keep_screenshots=False),
    )


def _workspace_temp_dir() -> Path:
    path = Path("tests/.tmp") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.asyncio
async def test_mock_provider_replays_scripted_actions() -> None:
    tmp_path = _workspace_temp_dir()
    replay_path = Path("tests/fixtures/mock_replay.json")
    client = build_vision_client(ModelConfig(provider="mock", model="mock-replay", replay_path=replay_path))
    fake_browser = FakeBrowser(tmp_path / "browser")
    fake_browser.session_dir.mkdir(parents=True, exist_ok=True)
    observation = await fake_browser.observe(step_index=1)

    first = await client.decide(observation, _build_config(tmp_path, replay_path).agent, history=[])
    second = await client.decide(observation, _build_config(tmp_path, replay_path).agent, history=[])
    third = await client.decide(observation, _build_config(tmp_path, replay_path).agent, history=[])

    assert first.action_type.value == "click"
    assert second.action_type.value == "finish"
    assert third.action_type.value == "finish"


@pytest.mark.asyncio
async def test_agent_can_run_with_fake_browser_and_mock_model() -> None:
    tmp_path = _workspace_temp_dir()
    replay_path = Path("tests/fixtures/mock_replay.json")
    config = _build_config(tmp_path, replay_path)
    fake_browser = FakeBrowser(tmp_path / "browser")
    fake_browser.session_dir.mkdir(parents=True, exist_ok=True)
    model = build_vision_client(config.model)

    agent = GhostUXAgent(config=config, browser=fake_browser, model=model)
    result = await agent.run()

    assert result.final_status == "finish"
    assert [step.action.action_type.value for step in result.steps] == ["click", "finish"]
    assert fake_browser.performed_actions == ["click", "finish"]
    assert result.report_path.exists()
    assert result.playback_path.exists()
    assert result.active_filters == []


def test_agent_detects_stall_on_repeated_same_page_actions() -> None:
    tmp_path = _workspace_temp_dir()
    config = SessionConfig(
        start_url="https://example.com",
        browser=BrowserConfig(),
        model=ModelConfig(provider="mock", model="mock-replay", replay_path=Path("tests/fixtures/mock_replay.json")),
        agent=AgentConfig(
            max_steps=8,
            persona="A cautious first-time user.",
            goal="Find the trial flow.",
            stall_detection_window=4,
            repeat_action_limit=2,
        ),
        report=ReportConfig(output_dir=tmp_path, keep_screenshots=False),
    )
    agent = GhostUXAgent(config=config, browser=FakeBrowser(tmp_path / "browser"), model=build_vision_client(config.model))
    records = [
        StepRecord(
            step_index=index,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path=tmp_path / f"step_{index}.png",
            action=UIAction(
                thought="Still looking for the same CTA.",
                action_type="click" if index % 2 == 0 else "scroll_down",
                target_element_id="4" if index % 2 == 0 else None,
                confidence_score=0.9,
            ),
            execution_success=True,
            execution_detail="No meaningful navigation happened.",
            visible_elements=12,
            observation_fingerprint="same-page-fingerprint",
        )
        for index in range(1, 5)
    ]
    assert agent._detect_stall(records) is not None
