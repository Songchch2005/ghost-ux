from __future__ import annotations

import json
from pathlib import Path

from ghost_ux.config import AgentConfig, ModelConfig
from ghost_ux.llm.base import ModelRequestPayload, VisionModelClient
from ghost_ux.models import LeakProbeResult, Observation, PromptContextFlags, StepRecord, UIAction


class MockReplayVisionClient(VisionModelClient):
    def __init__(self, model_config: ModelConfig):
        replay_path = model_config.replay_path
        if not replay_path:
            raise ValueError("Mock provider requires `model.replay_path`.")
        self.replay_path = Path(replay_path)
        payload = json.loads(self.replay_path.read_text(encoding="utf-8"))
        self.actions = [UIAction.model_validate(item) for item in payload]
        if not self.actions:
            raise ValueError("Replay fixture must contain at least one action.")
        self.cursor = 0

    async def decide(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
    ) -> UIAction:
        if self.cursor >= len(self.actions):
            return UIAction(
                thought="I have reached the end of the scripted replay and should stop here.",
                action_type="finish",
                confidence_score=1.0,
            )
        action = self.actions[self.cursor]
        self.cursor += 1
        return action

    def build_decision_payload(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        prompt_context: PromptContextFlags | None = None,
    ) -> ModelRequestPayload:
        flags = prompt_context or observation.prompt_context_flags
        return ModelRequestPayload(
            provider="mock",
            model="mock-replay",
            prompt_context=flags.model_dump(),
            url=observation.url if flags.include_url else None,
            title=observation.title if flags.include_title else None,
            dom_prompt=observation.dom_prompt if flags.include_dom else None,
            image_included=flags.include_image,
        )

    def build_leak_probe_payload(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        suspect_tokens: list[str],
        probe_name: str,
        prompt_context: PromptContextFlags,
    ) -> ModelRequestPayload:
        return ModelRequestPayload(
            provider="mock",
            model="mock-replay",
            probe_name=probe_name,
            suspect_tokens=suspect_tokens,
            prompt_context=prompt_context.model_dump(),
        )

    async def probe_visible_tokens(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        suspect_tokens: list[str],
        probe_name: str,
        prompt_context: PromptContextFlags,
    ) -> LeakProbeResult:
        return LeakProbeResult(
            probe_name=probe_name,
            visible_tokens=[],
            rationale="Mock replay client does not perform leak attribution probes.",
        )
