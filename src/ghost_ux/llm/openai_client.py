from __future__ import annotations

import asyncio
import json
import re

from openai import AsyncOpenAI, AuthenticationError, NotFoundError, RateLimitError

from ghost_ux.config import AgentConfig, ModelConfig
from ghost_ux.llm.base import ModelRequestPayload, VisionModelClient
from ghost_ux.llm.tactile_feedback import latest_tactile_feedback
from ghost_ux.models import (
    ActionType,
    LeakProbeResult,
    Observation,
    PromptContextFlags,
    StepRecord,
    UIAction,
)


def _recent_history_summary(history: list[StepRecord], limit: int = 6) -> str:
    if not history:
        return "No prior steps yet."
    lines = []
    for item in history[-limit:]:
        lines.append(
            f"Step {item.step_index}: action={item.action.action_type.value}, "
            f"success={item.execution_success}, thought={item.action.thought!r}, "
            f"detail={item.execution_detail!r}"
        )
    return "\n".join(lines)


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    return raw


def _extract_content(payload: object) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(payload or "")


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    message = str(exc)
    for pattern in (
        r"retry in ([\d.]+)s",
        r"'retryDelay': '(\d+)s'",
        r'"retryDelay":\s*"(\d+)s"',
    ):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


class OpenAIVisionClient(VisionModelClient):
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.client = AsyncOpenAI(
            api_key=model_config.resolved_api_key,
            base_url=model_config.resolved_base_url,
        )

    async def decide(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
    ) -> UIAction:
        payload = self.build_decision_payload(observation, agent_config, history)
        for attempt in range(self.model_config.retry_limit + 1):
            try:
                response = await self.client.chat.completions.create(**payload)
                content = _extract_content(response.choices[0].message.content)
                return UIAction.model_validate_json(_strip_code_fences(content))
            except AuthenticationError as exc:
                self._raise_authentication_error(exc)
            except NotFoundError as exc:
                self._raise_not_found_error(exc)
            except RateLimitError as exc:
                await self._handle_rate_limit(exc, attempt)

        raise RuntimeError("Model decision loop exhausted without returning a valid action.")

    def build_decision_payload(
        self,
        observation: Observation,
        agent_config: AgentConfig,
        history: list[StepRecord],
        prompt_context: PromptContextFlags | None = None,
    ) -> ModelRequestPayload:
        flags = prompt_context or PromptContextFlags(
            include_url=self.model_config.include_url_in_prompt,
            include_title=self.model_config.include_title_in_prompt,
            include_dom=True,
            include_image=True,
            image_detail=self.model_config.image_detail,
        )
        active_filter_names = {effect.filter_name for effect in observation.filter_effects}
        if "blurry_vision" in active_filter_names:
            blurry_effect = next(
                (effect for effect in observation.filter_effects if effect.filter_name == "blurry_vision"),
                None,
            )
            severity = "moderate"
            if blurry_effect:
                if "severity_severe" in blurry_effect.reasons:
                    severity = "severe"
                elif "severity_mild" in blurry_effect.reasons:
                    severity = "mild"
            flags_update = {"include_url": False, "include_title": False}
            if severity in {"moderate", "severe"}:
                flags_update["image_detail"] = "low"
            flags = flags.model_copy(update=flags_update)
        observation.prompt_context_flags = flags
        schema = json.dumps(UIAction.model_json_schema(), ensure_ascii=False, indent=2)
        sensory_warning = ""
        if observation.filter_effects:
            sensory_warning = (
                "You are receiving sensory-limited inputs. "
                "Do not infer missing text from prior knowledge, brand familiarity, or likely dataset contents. "
                "If text is not visible in the screenshot or absent from the provided DOM, treat it as unknown.\n"
            )
        system_prompt = (
            "You are Ghost-UX, an autonomous UX tester. "
            "You must act like the provided persona, stay focused on the goal, "
            "and return only valid JSON that matches the schema. "
            "Prefer small, reversible steps. "
            "If the page is blocked, confusing, or broken, return fail with a clear thought. "
            "If the goal is truly complete, return finish. "
            "Do not rely on prior knowledge of the site, brand, or likely dataset contents. "
            "If text is not visible or not present in the provided DOM, treat it as unknown."
        )
        page_lines = [
            f"- Viewport: {observation.viewport_width}x{observation.viewport_height}",
            f"- Last execution error: {observation.last_error or 'None'}",
        ]
        if flags.include_url:
            page_lines.insert(0, f"- URL: {observation.url}")
        if flags.include_title:
            page_lines.insert(1 if flags.include_url else 0, f"- Title: {observation.title}")

        dom_prompt = observation.filtered_dom_prompt if observation.filtered_elements else observation.dom_prompt
        if not flags.include_dom:
            dom_prompt = "DOM intentionally withheld for leak attribution diagnostics."
        tactile_feedback = latest_tactile_feedback(history, language=self.model_config.language)

        user_prompt = f"""
Persona:
{agent_config.persona}

Task Goal:
{agent_config.goal}

Current Page:
{chr(10).join(page_lines)}

Previous Action Tactile Feedback:
{tactile_feedback}

Recent History:
{_recent_history_summary(history)}

Sensory Guardrails:
{sensory_warning or "No extra sensory guardrails for this step."}

Visible Interactive Elements:
{dom_prompt}

Output JSON schema:
{schema}

Rules:
- Return JSON only. No markdown.
- Keep thought vivid and persona-driven, but concise.
- Use target_element_id only from the provided [ID:x] list.
- If action_type is type, include input_text.
- If no useful next step is visible, scroll before failing.
"""
        content = [{"type": "text", "text": user_prompt}]
        if flags.include_image:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{observation.screenshot_base64}",
                        "detail": flags.image_detail,
                    },
                }
            )
        return ModelRequestPayload(
            model=self.model_config.resolved_model_name,
            temperature=self.model_config.temperature,
            max_tokens=self.model_config.max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
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
        page_lines = [f"- Viewport: {observation.viewport_width}x{observation.viewport_height}"]
        if prompt_context.include_url:
            page_lines.insert(0, f"- URL: {observation.url}")
        if prompt_context.include_title:
            page_lines.insert(1 if prompt_context.include_url else 0, f"- Title: {observation.title}")
        dom_prompt = observation.filtered_dom_prompt if prompt_context.include_dom else "DOM intentionally withheld."
        token_list = ", ".join(suspect_tokens) if suspect_tokens else "none"
        system_prompt = (
            "You are diagnosing sensory leakage in a UX testing agent. "
            "Do not use prior knowledge of the site. "
            "Only report tokens you can genuinely read from the provided sensory inputs."
        )
        user_prompt = f"""
Probe Mode: {probe_name}
Persona:
{agent_config.persona}

Task Goal:
{agent_config.goal}

Current Page:
{chr(10).join(page_lines)}

Recent History:
{_recent_history_summary(history)}

DOM Snapshot:
{dom_prompt}

Candidate suspicious tokens:
{token_list}

Return JSON with this exact shape:
{{
  "visible_tokens": ["token-a"],
  "rationale": "short explanation"
}}

Rules:
- Only include tokens from the candidate list.
- If none are safely visible or inferable from the provided sensory input, return an empty list.
- Do not guess from site identity or prior familiarity.
"""
        content = [{"type": "text", "text": user_prompt}]
        if prompt_context.include_image:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{observation.screenshot_base64}",
                        "detail": prompt_context.image_detail,
                    },
                }
            )
        return ModelRequestPayload(
            model=self.model_config.resolved_model_name,
            temperature=0.0,
            max_tokens=min(self.model_config.max_output_tokens, 400),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
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
        payload = self.build_leak_probe_payload(
            observation,
            agent_config,
            history,
            suspect_tokens,
            probe_name,
            prompt_context,
        )
        for attempt in range(self.model_config.retry_limit + 1):
            try:
                response = await self.client.chat.completions.create(**payload)
                content = _extract_content(response.choices[0].message.content)
                parsed = json.loads(_strip_code_fences(content))
                visible_tokens = parsed.get("visible_tokens") or []
                if not isinstance(visible_tokens, list):
                    visible_tokens = []
                filtered_tokens = [
                    token for token in visible_tokens
                    if isinstance(token, str) and token in suspect_tokens
                ]
                return LeakProbeResult(
                    probe_name=probe_name,
                    visible_tokens=filtered_tokens,
                    rationale=str(parsed.get("rationale", "")),
                )
            except AuthenticationError as exc:
                self._raise_authentication_error(exc)
            except NotFoundError as exc:
                self._raise_not_found_error(exc)
            except RateLimitError as exc:
                await self._handle_rate_limit(exc, attempt)

        raise RuntimeError("Leak probe loop exhausted without returning a valid result.")

    def _raise_authentication_error(self, exc: AuthenticationError) -> None:
        if self.model_config.provider.lower() == "gemini":
            raise RuntimeError(
                "Gemini authentication failed. "
                "Ghost-UX now expects Gemini's OpenAI-compatible endpoint "
                f"({self.model_config.resolved_base_url}) and a valid Google Gemini API key "
                f"from `{self.model_config.api_key_env}`. "
                "If you are using an official Gemini key that starts with `AIza`, do not send it to the default OpenAI endpoint."
            ) from exc
        raise exc

    def _raise_not_found_error(self, exc: NotFoundError) -> None:
        if self.model_config.provider.lower() == "gemini":
            raise RuntimeError(
                "Gemini model not found. "
                f"`{self.model_config.resolved_model_name}` is not available on Gemini's OpenAI-compatible endpoint "
                f"({self.model_config.resolved_base_url}). "
                "Try a currently supported Gemini model such as `gemini-2.5-flash` or `gemini-2.5-pro`. "
                "Older names like `gemini-1.5-pro` may no longer be available for this API surface."
            ) from exc
        raise exc

    async def _handle_rate_limit(self, exc: RateLimitError, attempt: int) -> None:
        if attempt >= self.model_config.retry_limit:
            raise RuntimeError(
                "Model rate limit exceeded and retry budget exhausted. "
                "Try again later, lower max_steps, or use a model / quota tier with more requests."
            ) from exc
        retry_after = _extract_retry_after_seconds(exc)
        wait_seconds = retry_after or (self.model_config.retry_backoff_seconds * (attempt + 1))
        wait_seconds = min(wait_seconds, self.model_config.max_retry_wait_seconds)
        await asyncio.sleep(wait_seconds)
