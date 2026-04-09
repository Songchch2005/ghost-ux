from __future__ import annotations

from ghost_ux.actions.filters import MotorNoiseFilter
from ghost_ux.actions.pipeline import ActionFilterPipeline
from ghost_ux.config import MotorConfig


FILTER_REGISTRY = {
    "motor_noise": lambda config: MotorNoiseFilter(
        profile=config.profile,
        click_offset_px=config.click_offset_px,
        scroll_noise_px=config.scroll_noise_px,
        disable_precise_click_fallback=config.disable_precise_click_fallback,
    )
}


def build_action_pipeline(persona: str, config: MotorConfig) -> ActionFilterPipeline:
    filter_names = list(config.filters)
    if config.auto_from_persona:
        inferred = infer_motor_profile_from_persona(persona, config)
        if inferred and "motor_noise" not in filter_names:
            filter_names.append("motor_noise")
    filters = [FILTER_REGISTRY[name](config) for name in filter_names if name in FILTER_REGISTRY]
    return ActionFilterPipeline(filters)


def infer_motor_profile_from_persona(persona: str, config: MotorConfig) -> str | None:
    lowered = persona.lower()
    if any(keyword in lowered for keyword in ("喝醉", "大醉", "drunk")):
        config.profile = "drunk"
        return "drunk"
    if any(keyword in lowered for keyword in ("微醺", "tipsy")):
        config.profile = "tipsy"
        return "tipsy"
    if any(keyword in lowered for keyword in ("单手操作", "地铁", "拥挤", "subway", "one-handed")):
        config.profile = "subway_one_hand"
        return "subway_one_hand"
    if any(keyword in lowered for keyword in ("帕金森", "parkinson strong")):
        config.profile = "parkinson_strong"
        return "parkinson_strong"
    if any(keyword in lowered for keyword in ("手抖", "颤抖", "tremor", "parkinson")):
        config.profile = "parkinson_light" if "parkinson" in lowered else "tremor"
        return config.profile
    return None
