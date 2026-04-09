from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from loguru import logger

from ghost_ux.browser import BrowserController
from ghost_ux.config import SessionConfig
from ghost_ux.llm.base import VisionModelClient
from ghost_ux.llm import build_vision_client
from ghost_ux.logging_utils import print_banner, print_step_summary, print_thought, setup_logging
from ghost_ux.models import ActionType, RunArtifacts, StepRecord
from ghost_ux.reporting import build_markdown_report, build_playback_report, build_run_artifacts
from ghost_ux.leak_detection import extract_suspect_tokens
from ghost_ux.sensory import build_sensory_pipeline


class GhostUXAgent:
    def __init__(
        self,
        config: SessionConfig,
        browser: BrowserController | None = None,
        model: VisionModelClient | None = None,
    ):
        self.config = config
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
        self.session_dir = config.report.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.session_dir / "session.log")
        print_banner(
            session_id=self.session_id,
            url=str(config.start_url),
            persona=config.agent.persona,
            goal=config.agent.goal,
        )
        self.browser = browser or BrowserController(config.browser, config.agent, config.sensory, config.motor, self.session_dir)
        self.model = model or build_vision_client(config.model)
        self.sensory_pipeline = build_sensory_pipeline(config.agent.persona, config.sensory)
        if self.sensory_pipeline.active_filter_names:
            logger.info("Active sensory filters: {}", ", ".join(self.sensory_pipeline.active_filter_names))

    async def run(self) -> RunArtifacts:
        started_at = datetime.utcnow()
        steps: list[StepRecord] = []
        last_error: str | None = None
        final_status = "incomplete"

        await self.browser.start()
        try:
            await self.browser.goto(str(self.config.start_url))
            for step_index in range(1, self.config.agent.max_steps + 1):
                observation = await self.browser.observe(step_index, last_error=last_error)
                observation = self._apply_sensory_filters(observation)
                action = await self.model.decide(observation, self.config.agent, steps)
                print_thought(step_index, action.thought, action.confidence_score)
                logger.info(
                    "Step {} decided action={} target={} confidence={:.2f}",
                    step_index,
                    action.action_type.value,
                    action.target_element_id,
                    action.confidence_score,
                )

                action_result = await self.browser.perform(action, observation)
                last_error = None if action_result.success else action_result.detail
                if action_result.noise_applied:
                    logger.info(
                        "MotorNoise profile={} intended={} actual={} offset={} hit={} misfire={}",
                        action_result.noise_profile,
                        action_result.intended_point,
                        action_result.actual_point,
                        action_result.offset,
                        action_result.actual_hit_summary,
                        action_result.misfire,
                    )
                steps.append(
                    StepRecord(
                        step_index=step_index,
                        observation_url=observation.url,
                        observation_title=observation.title,
                        screenshot_path=observation.screenshot_path,
                        action=action,
                        execution_success=action_result.success,
                        execution_detail=action_result.detail,
                        visible_elements=len(observation.filtered_interactive_elements),
                        observation_fingerprint=observation.fingerprint,
                        active_filters=self.sensory_pipeline.active_filter_names,
                        filter_effects=observation.filter_effects,
                        dom_removed_count=observation.dom_removed_count,
                        dom_modified_count=observation.dom_modified_count,
                        raw_visible_elements=len(observation.raw_interactive_elements),
                        raw_dom_prompt=observation.raw_dom_prompt,
                        filtered_dom_prompt=observation.filtered_dom_prompt,
                        prompt_context_flags=observation.prompt_context_flags,
                        suspect_tokens=observation.suspect_tokens,
                        cognitive_terms=observation.cognitive_terms,
                        noise_applied=action_result.noise_applied,
                        noise_profile=action_result.noise_profile,
                        intended_point=action_result.intended_point,
                        actual_point=action_result.actual_point,
                        offset=action_result.offset,
                        actual_hit_summary=action_result.actual_hit_summary,
                        misfire=action_result.misfire,
                        scroll_delta=action_result.scroll_delta,
                    )
                )

                stall_reason = self._detect_stall(steps)
                if stall_reason:
                    logger.warning("Agent stalled at step {}: {}", step_index, stall_reason)
                    final_status = "stalled"
                    break

                if action.action_type == ActionType.FINISH:
                    final_status = "finish"
                    break
                if action.action_type == ActionType.FAIL:
                    final_status = "fail"
                    break
            else:
                final_status = "max_steps_reached"
        finally:
            await self.browser.close()

        finished_at = datetime.utcnow()
        report_path = build_markdown_report(
            config=self.config,
            session_id=self.session_id,
            steps=steps,
            output_path=self.session_dir / "report.md",
            started_at=started_at,
            finished_at=finished_at,
            final_status=final_status,
            active_filters=self.sensory_pipeline.active_filter_names,
        )
        playback_path = build_playback_report(
            config=self.config,
            session_id=self.session_id,
            steps=steps,
            output_path=self.session_dir / "playback.html",
            started_at=started_at,
            finished_at=finished_at,
            final_status=final_status,
            active_filters=self.sensory_pipeline.active_filter_names,
        )
        self._cleanup_screenshots(steps)
        print_step_summary(steps)
        return build_run_artifacts(
            session_id=self.session_id,
            session_dir=self.session_dir,
            report_path=report_path,
            playback_path=playback_path,
            active_filters=self.sensory_pipeline.active_filter_names,
            steps=steps,
            final_status=final_status,
            started_at=started_at,
            finished_at=finished_at,
        )

    def _cleanup_screenshots(self, steps: list[StepRecord]) -> None:
        if self.config.report.keep_screenshots:
            return
        for step in steps:
            path = Path(step.screenshot_path)
            if path.exists():
                path.unlink(missing_ok=True)

    def _detect_stall(self, steps: list[StepRecord]) -> str | None:
        window = self.config.agent.stall_detection_window
        if len(steps) < window:
            return None
        recent_steps = steps[-window:]
        if len({step.observation_fingerprint for step in recent_steps}) != 1:
            return None

        signatures = [
            f"{step.action.action_type.value}:{step.action.target_element_id or '-'}"
            for step in recent_steps
        ]
        repeated_signatures = {
            signature
            for signature in signatures
            if signatures.count(signature) >= self.config.agent.repeat_action_limit
        }
        repeated_scrolls = sum(
            1
            for step in recent_steps
            if step.action.action_type in {ActionType.SCROLL_DOWN, ActionType.SCROLL_UP}
        )
        click_targets = [
            step.action.target_element_id
            for step in recent_steps
            if step.action.action_type == ActionType.CLICK and step.action.target_element_id
        ]
        repeated_click_targets = {
            target
            for target in click_targets
            if click_targets.count(target) >= self.config.agent.repeat_action_limit
        }

        if repeated_signatures or repeated_scrolls >= self.config.agent.repeat_action_limit or repeated_click_targets:
            return (
                "The agent stayed on the same page fingerprint and kept repeating the same action pattern. "
                "Stopping early to avoid quota burn and infinite loops."
            )
        return None

    def _apply_sensory_filters(self, observation):
        if not observation.raw_elements:
            observation.raw_elements = [element.model_copy(deep=True) for element in observation.elements]
        if not self.sensory_pipeline.active_filter_names:
            observation.filtered_elements = [element.model_copy(deep=True) for element in observation.elements]
            observation.suspect_tokens = extract_suspect_tokens(observation.raw_elements, observation.filtered_elements)
            observation.cognitive_terms = []
            return observation
        filtered_bytes, filtered_elements, filter_effects = self.sensory_pipeline.apply_with_trace(
            observation.screenshot_bytes,
            observation.elements,
        )
        observation.screenshot_bytes = filtered_bytes
        observation.screenshot_base64 = base64.b64encode(filtered_bytes).decode("utf-8")
        observation.elements = filtered_elements
        observation.filtered_elements = [element.model_copy(deep=True) for element in filtered_elements]
        observation.filter_effects = filter_effects
        observation.dom_removed_count = sum(effect.removed_count for effect in filter_effects)
        observation.dom_modified_count = sum(effect.modified_count for effect in filter_effects)
        observation.suspect_tokens = extract_suspect_tokens(observation.raw_elements, observation.filtered_elements)
        observation.cognitive_terms = sorted(
            {
                term
                for element in observation.filtered_elements
                for term in element.scrubbed_terms
            }
        )
        observation.screenshot_path.write_bytes(filtered_bytes)
        return observation
