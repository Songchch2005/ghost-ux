from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ghost_ux.config import AgentConfig, ReportConfig, SessionConfig
from ghost_ux.models import ActionType, FilterEffect, PromptContextFlags, StepRecord, UIAction
from ghost_ux.reporting import build_markdown_report, build_playback_report, infer_pain_points


def _workspace_temp_dir() -> Path:
    path = Path("tests/.tmp") / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_infer_pain_points_flags_failure_and_low_confidence() -> None:
    steps = [
        StepRecord(
            step_index=1,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path=Path("step_01.png"),
            action=UIAction(
                thought="I am confused by the layout.",
                action_type=ActionType.SCROLL_DOWN,
                confidence_score=0.2,
            ),
            execution_success=False,
            execution_detail="Element was covered by a modal.",
            visible_elements=10,
            observation_fingerprint="https://example.com::Example::button|Buy",
            timestamp=datetime.utcnow(),
        )
    ]
    issues = infer_pain_points(steps, 0.45)
    assert any("failed interactions" in issue for issue in issues)
    assert any("low confidence" in issue for issue in issues)


def test_build_playback_report_embeds_image_data() -> None:
    tmp_path = _workspace_temp_dir()
    screenshot_path = tmp_path / "step_01.png"
    screenshot_path.write_bytes(b"fake-image")
    screenshot_path_2 = tmp_path / "step_02.png"
    screenshot_path_2.write_bytes(b"fake-image-2")
    steps = [
        StepRecord(
            step_index=1,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path=screenshot_path,
            action=UIAction(
                thought="The big button is impossible to miss.",
                action_type=ActionType.CLICK,
                target_element_id="1",
                confidence_score=0.91,
            ),
            execution_success=True,
            execution_detail="Clicked the button.",
            visible_elements=3,
            observation_fingerprint="https://example.com::Example::button|Start",
            timestamp=datetime.utcnow(),
            active_filters=["blurry_vision", "low_patience"],
            filter_effects=[
                FilterEffect(
                    filter_name="blurry_vision",
                    removed_count=1,
                    modified_count=2,
                    reasons={"font_size_below_threshold": 1, "descendant_text_scrubbed": 1, "severity_severe": 1},
                )
            ],
            dom_removed_count=1,
            dom_modified_count=2,
            raw_visible_elements=5,
            raw_dom_prompt="[ID:1] button | text='Explore quotes thinking'",
            filtered_dom_prompt="[ID:1] button | text='Explore quotes'",
            prompt_context_flags=PromptContextFlags(include_url=False, include_title=False, include_dom=True, include_image=True, image_detail="low"),
            suspect_tokens=["thinking", "change"],
            cognitive_terms=["SaaS", "API"],
            noise_applied=True,
            noise_profile="drunk",
            intended_point=(120.0, 44.0),
            actual_point=(132.0, 36.0),
            offset=(12.0, -8.0),
            actual_hit_summary="button.secondary text=Learn more",
            misfire=True,
        ),
        StepRecord(
            step_index=2,
            observation_url="https://example.com/pricing",
            observation_title="Pricing",
            screenshot_path=screenshot_path_2,
            action=UIAction(
                thought="This page is cluttered and I am no longer sure which CTA is safe.",
                action_type=ActionType.SCROLL_DOWN,
                confidence_score=0.21,
            ),
            execution_success=False,
            execution_detail="The CTA was obscured by a sticky banner.",
            visible_elements=9,
            observation_fingerprint="https://example.com/pricing::Pricing::button|Buy",
            timestamp=datetime.utcnow(),
            active_filters=["blurry_vision", "low_patience"],
            filter_effects=[
                FilterEffect(
                    filter_name="low_patience",
                    removed_count=3,
                    modified_count=0,
                    reasons={"below_patience_window": 3},
                )
            ],
            dom_removed_count=3,
            dom_modified_count=0,
            raw_visible_elements=11,
            raw_dom_prompt="[ID:2] a | text='deep-thoughts'",
            filtered_dom_prompt="No visible interactive elements detected in the current viewport.",
            prompt_context_flags=PromptContextFlags(),
            suspect_tokens=["deep-thoughts"],
            cognitive_terms=["赋能", "闭环"],
        ),
    ]
    config = SessionConfig(
        start_url="https://example.com",
        agent=AgentConfig(persona="A first-time visitor.", goal="Start the signup flow."),
        report=ReportConfig(output_dir=tmp_path, keep_screenshots=False),
    )

    output_path = build_playback_report(
        config=config,
        session_id="demo-session",
        steps=steps,
        output_path=tmp_path / "playback.html",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        final_status="finish",
        active_filters=["blurry_vision", "low_patience"],
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Ghost-UX Playback" in html
    assert "data:image/png;base64," in html
    assert "The big button is impossible to miss." in html
    assert "Active Filters" in html
    assert "blurry_vision, low_patience" in html
    assert "Failed only (1)" in html
    assert "Low confidence only (1)" in html
    assert "Cognitive scrub (2)" in html
    assert 'data-filter="critical"' in html
    assert 'data-filter="cognitive"' in html
    assert 'data-action-filter="all"' in html
    assert 'data-action-filter="scroll_down"' in html
    assert 'id="recent-filter"' in html
    assert "Last 3 steps" not in html
    assert 'data-low-confidence="true"' in html
    assert 'data-cognitive-impact="true"' in html
    assert 'data-success="false"' in html
    assert 'data-action="collapse-all"' in html
    assert 'class="mini-map"' in html
    assert 'href="#step-2"' in html
    assert "Jump to hotspots" in html
    assert "applyAllFilters()" in html
    assert "refreshActiveMiniMapLink()" in html
    assert "DOM Removed By Filters" in html
    assert "DOM Modified By Filters" in html
    assert "Blurry Vision Severity" in html
    assert "status-blurry" in html
    assert ">severe<" in html
    assert "blurry_vision: removed 1, modified 2" in html
    assert "Raw DOM Prompt" in html
    assert "Filtered DOM Prompt" in html
    assert "Prompt Context" in html
    assert "thinking, change" in html
    assert "Noise Applied" in html
    assert "button.secondary text=Learn more" in html
    assert "(120.0, 44.0)" in html
    assert "Tactile Feedback Shown To Model" in html
    assert "unresponsive background area" in html or "hit another element instead" in html
    assert "Physical mis-touch" in html
    assert "tactile-callout tactile-warning" in html
    assert "status-motor-warning" in html
    assert "Jargon Terms Scrubbed" in html
    assert "SaaS, API" in html
    assert "Jargon Scrub Summary" in html
    assert "jargon-callout" in html


def test_markdown_report_lists_active_filters() -> None:
    tmp_path = _workspace_temp_dir()
    steps = [
        StepRecord(
            step_index=1,
            observation_url="https://example.com",
            observation_title="Example",
            screenshot_path=tmp_path / "step.png",
            action=UIAction(
                thought="I can barely read the small copy.",
                action_type=ActionType.SCROLL_DOWN,
                confidence_score=0.4,
            ),
            execution_success=True,
            execution_detail="Scrolled.",
            visible_elements=4,
            observation_fingerprint="fingerprint",
            timestamp=datetime.utcnow(),
            active_filters=["blurry_vision", "low_patience"],
            filter_effects=[
                FilterEffect(
                    filter_name="blurry_vision",
                    removed_count=2,
                    modified_count=3,
                    reasons={"font_size_below_threshold": 2, "severity_moderate": 1},
                )
            ],
            dom_removed_count=2,
            dom_modified_count=3,
            raw_visible_elements=6,
            raw_dom_prompt="[ID:4] a | text='world'",
            filtered_dom_prompt="[ID:4] a | text='[blurred fine print]'",
            prompt_context_flags=PromptContextFlags(include_url=False, include_title=True, include_dom=True, include_image=False, image_detail="auto"),
            suspect_tokens=["world"],
            cognitive_terms=["SaaS"],
            noise_applied=True,
            noise_profile="tipsy",
            intended_point=(200.0, 380.0),
            actual_point=(209.0, 371.0),
            offset=(9.0, -9.0),
            actual_hit_summary="main text=Pricing section",
            misfire=True,
        )
    ]
    config = SessionConfig(
        start_url="https://example.com",
        agent=AgentConfig(persona="A low-vision impatient user.", goal="Find pricing."),
        report=ReportConfig(output_dir=tmp_path, keep_screenshots=False),
    )
    output_path = build_markdown_report(
        config=config,
        session_id="demo-session",
        steps=steps,
        output_path=tmp_path / "report.md",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        final_status="finish",
        active_filters=["blurry_vision", "low_patience"],
    )
    markdown = output_path.read_text(encoding="utf-8")
    assert "Active Sensory Filters" in markdown
    assert "blurry_vision, low_patience" in markdown
    assert "DOM Removed By Filters: `2`" in markdown
    assert "DOM Modified By Filters: `3`" in markdown
    assert "Blurry Vision Severity: `moderate`" in markdown
    assert "blurry_vision: removed 2, modified 3" in markdown
    assert "Raw DOM Elements: `6`" in markdown
    assert "Filtered DOM Elements: `4`" in markdown
    assert "Prompt Context: `URL=off, Title=on, DOM=on, Image=off, Image detail=auto`" in markdown
    assert "Suspect Tokens: `world`" in markdown
    assert "Noise Applied: `True`" in markdown
    assert "Actual Hit: `main text=Pricing section`" in markdown
    assert "Tactile Feedback Shown To Model:" in markdown
    assert "Jargon Terms Scrubbed: `SaaS`" in markdown
    assert "Raw DOM Prompt:" in markdown
    assert "Filtered DOM Prompt:" in markdown
