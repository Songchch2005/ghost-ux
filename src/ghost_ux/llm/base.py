from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ghost_ux.config import AgentConfig
from ghost_ux.models import LeakProbeResult, Observation, PromptContextFlags, StepRecord, UIAction


class ModelRequestPayload(dict[str, Any]):
    pass


class VisionModelClient(ABC):
    @abstractmethod
    async def decide(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
    ) -> UIAction:
        raise NotImplementedError

    @abstractmethod
    def build_decision_payload(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        prompt_context: PromptContextFlags | None = None,
    ) -> ModelRequestPayload:
        raise NotImplementedError

    @abstractmethod
    def build_leak_probe_payload(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        suspect_tokens: list[str],
        probe_name: str,
        prompt_context: PromptContextFlags,
    ) -> ModelRequestPayload:
        raise NotImplementedError

    @abstractmethod
    async def probe_visible_tokens(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        suspect_tokens: list[str],
        probe_name: str,
        prompt_context: PromptContextFlags,
    ) -> LeakProbeResult:
        raise NotImplementedError
