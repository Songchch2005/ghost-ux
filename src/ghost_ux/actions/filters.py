from __future__ import annotations

import random

from ghost_ux.actions.base import BaseActionFilter
from ghost_ux.models import ActionType, ExecutableAction, Observation, UIAction


PROFILE_DEFAULTS = {
    "tipsy": {"click_offset_px": 10, "scroll_noise_px": 80},
    "drunk": {"click_offset_px": 40, "scroll_noise_px": 280},
    "subway_one_hand": {"click_offset_px": 18, "scroll_noise_px": 140},
    "tremor": {"click_offset_px": 12, "scroll_noise_px": 90},
    "parkinson_light": {"click_offset_px": 12, "scroll_noise_px": 110},
    "parkinson_strong": {"click_offset_px": 28, "scroll_noise_px": 220},
}


class MotorNoiseFilter(BaseActionFilter):
    name = "motor_noise"

    def __init__(
        self,
        *,
        profile: str,
        click_offset_px: int,
        scroll_noise_px: int,
        disable_precise_click_fallback: bool = True,
        rng: random.Random | None = None,
    ):
        defaults = PROFILE_DEFAULTS.get(profile, {})
        self.profile = profile
        self.click_offset_px = click_offset_px or defaults.get("click_offset_px", 0)
        self.scroll_noise_px = scroll_noise_px or defaults.get("scroll_noise_px", 0)
        self.disable_precise_click_fallback = disable_precise_click_fallback
        self.rng = rng or random.Random()

    def apply(
        self,
        action: UIAction,
        observation: Observation,
        plan: ExecutableAction,
    ) -> ExecutableAction:
        updated = plan.model_copy(deep=True)
        updated.noise_profile = self.profile
        updated.disable_precise_click_fallback = self.disable_precise_click_fallback

        if action.action_type == ActionType.CLICK and updated.intended_x is not None and updated.intended_y is not None:
            offset_x = self.rng.randint(-self.click_offset_px, self.click_offset_px) if self.click_offset_px else 0
            offset_y = self.rng.randint(-self.click_offset_px, self.click_offset_px) if self.click_offset_px else 0
            updated.offset_x = float(offset_x)
            updated.offset_y = float(offset_y)
            updated.actual_x = float(min(max(updated.intended_x + offset_x, 0.0), observation.viewport_width - 1))
            updated.actual_y = float(min(max(updated.intended_y + offset_y, 0.0), observation.viewport_height - 1))
            updated.noise_applied = bool(offset_x or offset_y)
            return updated

        if action.action_type in {ActionType.SCROLL_DOWN, ActionType.SCROLL_UP}:
            noise_y = self.rng.randint(-self.scroll_noise_px, self.scroll_noise_px) if self.scroll_noise_px else 0
            base_delta = updated.scroll_delta_y
            updated.scroll_delta_y = float(base_delta + noise_y)
            updated.offset_y = float(noise_y)
            updated.noise_applied = bool(noise_y)
            return updated

        return updated
