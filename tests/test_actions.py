from random import Random
from pathlib import Path

from ghost_ux.actions.filters import MotorNoiseFilter
from ghost_ux.actions.registry import build_action_pipeline
from ghost_ux.config import MotorConfig
from ghost_ux.models import ActionType, ExecutableAction, Observation, UIAction


def _observation(width: int = 100, height: int = 80) -> Observation:
    return Observation(
        step_index=1,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"",
        screenshot_base64="",
        screenshot_path=Path("tests/.tmp/noop.png"),
        elements=[],
        viewport_width=width,
        viewport_height=height,
    )


def test_motor_noise_click_offset_stays_in_bounds_and_clamps() -> None:
    filt = MotorNoiseFilter(
        profile="drunk",
        click_offset_px=40,
        scroll_noise_px=0,
        rng=Random(7),
    )
    plan = ExecutableAction(
        action_type=ActionType.CLICK,
        target_element_id="4",
        intended_x=95.0,
        intended_y=4.0,
        actual_x=95.0,
        actual_y=4.0,
    )
    result = filt.apply(
        UIAction(thought="click", action_type=ActionType.CLICK, target_element_id="4", confidence_score=0.9),
        _observation(),
        plan,
    )
    assert abs(result.offset_x) <= 40
    assert abs(result.offset_y) <= 40
    assert 0 <= result.actual_x <= 99
    assert 0 <= result.actual_y <= 79
    assert result.noise_applied is True


def test_motor_noise_scroll_changes_delta_within_bounds() -> None:
    filt = MotorNoiseFilter(
        profile="tipsy",
        click_offset_px=0,
        scroll_noise_px=80,
        rng=Random(3),
    )
    plan = ExecutableAction(action_type=ActionType.SCROLL_DOWN, scroll_delta_y=720.0)
    result = filt.apply(
        UIAction(thought="scroll", action_type=ActionType.SCROLL_DOWN, confidence_score=0.8),
        _observation(),
        plan,
    )
    assert 640 <= result.scroll_delta_y <= 800
    assert abs(result.offset_y) <= 80
    assert result.noise_applied is True


def test_motor_pipeline_auto_enables_from_persona() -> None:
    config = MotorConfig(auto_from_persona=True)
    pipeline = build_action_pipeline("一个在拥挤地铁上单手操作手机的用户。", config)
    assert pipeline.active_filter_names == ["motor_noise"]
    assert config.profile == "subway_one_hand"


def test_motor_pipeline_leaves_action_unchanged_when_disabled() -> None:
    pipeline = build_action_pipeline("A calm user.", MotorConfig(auto_from_persona=False))
    plan = ExecutableAction(action_type=ActionType.SCROLL_DOWN, scroll_delta_y=720.0)
    result = pipeline.apply(
        UIAction(thought="scroll", action_type=ActionType.SCROLL_DOWN, confidence_score=0.5),
        _observation(),
        plan,
    )
    assert result.scroll_delta_y == 720.0
    assert result.noise_applied is False
