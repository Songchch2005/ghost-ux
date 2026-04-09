from ghost_ux.models import ActionType, UIAction


def test_ui_action_requires_target_for_click() -> None:
    try:
        UIAction(
            thought="I should press the CTA.",
            action_type=ActionType.CLICK,
            confidence_score=0.7,
        )
    except ValueError:
        return
    raise AssertionError("UIAction should reject click without target_element_id.")


def test_ui_action_accepts_type_action() -> None:
    action = UIAction(
        thought="I can fill the email field now.",
        action_type=ActionType.TYPE,
        target_element_id="7",
        input_text="hello@example.com",
        confidence_score=0.91,
    )
    assert action.action_type == ActionType.TYPE
    assert action.target_element_id == "7"
