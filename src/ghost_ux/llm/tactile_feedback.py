from __future__ import annotations

from ghost_ux.models import ActionType, StepRecord


def looks_like_background_hit(hit_summary: str | None) -> bool:
    if not hit_summary:
        return True
    lowered = hit_summary.lower()
    if any(token in lowered for token in ("button", "a ", " a.", "input", "select", "textarea")):
        return False
    return any(
        token in lowered
        for token in ("div", "main", "section", "article", "body", "html", "background", "container", "panel")
    )


def format_point(point: tuple[float, float] | None) -> str:
    if not point:
        return "unknown"
    return f"({round(point[0], 1)}, {round(point[1], 1)})"


def latest_tactile_feedback(history: list[StepRecord], language: str = "en") -> str:
    if not history:
        return "No tactile feedback yet."

    latest = history[-1]
    if not latest.noise_applied:
        return "No tactile anomalies were detected on the previous action."

    profile = latest.noise_profile or "motor_noise"
    target = latest.action.target_element_id or "unknown"
    intended = format_point(latest.intended_point)
    actual = format_point(latest.actual_point)
    hit = latest.actual_hit_summary or "unknown surface"

    if latest.action.action_type == ActionType.CLICK:
        if latest.misfire:
            if looks_like_background_hit(latest.actual_hit_summary):
                if language == "zh":
                    return (
                        "[系统触觉反馈]: 你刚才试图点击目标，但由于你的状态 "
                        f"({profile})，你的手明显偏离了目标点。你本来想点 [ID:{target}]，"
                        f"目标中心大约在 {intended}，实际却落在了 {actual}，点到了没有明显反应的背景区域 "
                        f"(hit: {hit})。这次点击大概率没有生效。"
                    )
                return (
                    "[System tactile feedback]: You tried to click the target, but your "
                    f"current state ({profile}) caused your hand to drift away from the intended point. "
                    f"You meant to click [ID:{target}] near {intended}, but actually landed at {actual} "
                    f"on an unresponsive background area (hit: {hit}). This click was probably ineffective."
                )
            if language == "zh":
                return (
                    "[系统触觉反馈]: 你刚才试图点击目标，但由于你的状态 "
                    f"({profile})，你的手抖动后偏到了旁边。你本来想点 [ID:{target}]，"
                    f"目标中心大约在 {intended}，实际却落在了 {actual}，误触了另一个元素 "
                    f"(hit: {hit})。请根据页面的真实反馈判断刚才发生了什么。"
                )
            return (
                "[System tactile feedback]: You tried to click the target, but your "
                f"current state ({profile}) made your hand slip sideways. "
                f"You meant to click [ID:{target}] near {intended}, but actually landed at {actual} "
                f"and hit another element instead (hit: {hit}). Use the page's real response to infer what happened."
            )
        if language == "zh":
            return (
                "[系统触觉反馈]: 你刚才点击时虽然受到了 "
                f"{profile} 的动作干扰，但这次仍然勉强点中了目标附近区域。"
            )
        return (
            "[System tactile feedback]: Your movement was affected by "
            f"{profile}, but you still barely landed near the intended target."
        )

    if latest.action.action_type in {ActionType.SCROLL_DOWN, ActionType.SCROLL_UP} and latest.scroll_delta:
        intended_delta = 720 if latest.action.action_type == ActionType.SCROLL_DOWN else -720
        actual_delta = round(latest.scroll_delta[1], 1)
        if abs(actual_delta - intended_delta) >= 80:
            if language == "zh":
                return (
                    "[系统触觉反馈]: 你刚才滚动时手势不太稳。"
                    f"原本想滚动大约 {intended_delta}px，但实际滚动了 {actual_delta}px。"
                    "请留意页面是否被你滑过头了，或停在了意料之外的位置。"
                )
            return (
                "[System tactile feedback]: Your scrolling gesture was unstable. "
                f"You intended to scroll about {intended_delta}px, but actually scrolled {actual_delta}px. "
                "Check whether you overshot the page or stopped somewhere unexpected."
            )
        if language == "zh":
            return "[系统触觉反馈]: 你刚才滚动时有轻微抖动，但整体仍接近预期。"
        return "[System tactile feedback]: Your scroll had a slight wobble, but it still stayed close to the intended movement."

    return "No tactile anomalies were detected on the previous action."
