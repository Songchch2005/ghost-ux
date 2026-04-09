from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path

from ghost_ux.config import SessionConfig
from ghost_ux.llm.tactile_feedback import latest_tactile_feedback
from ghost_ux.models import FilterEffect, RunArtifacts, StepRecord


def infer_pain_points(steps: list[StepRecord], low_conf_threshold: float) -> list[str]:
    issues: list[str] = []
    failed_steps = [step for step in steps if not step.execution_success]
    low_conf = [step for step in steps if step.action.confidence_score < low_conf_threshold]
    action_counter = Counter(step.action.action_type.value for step in steps)

    if failed_steps:
        issues.append(
            f"There were {len(failed_steps)} failed interactions, suggesting unstable affordances or obstructed controls."
        )
    if low_conf:
        issues.append(
            f"The agent reported low confidence on {len(low_conf)} steps, which hints that navigation cues or copy may be unclear."
        )
    if action_counter.get("scroll_down", 0) >= 3:
        issues.append(
            "The agent needed repeated scrolling to make progress, which can indicate weak information scent or poor content hierarchy."
        )
    if not issues:
        issues.append(
            "No major friction was inferred heuristically; review the timeline for subtle copy, trust, or discoverability issues."
        )
    return issues


def format_filter_breakdown(filter_effects: list[FilterEffect]) -> str:
    if not filter_effects:
        return "none"
    parts = []
    for effect in filter_effects:
        parts.append(
            f"{effect.filter_name}: removed {effect.removed_count}, modified {effect.modified_count}"
        )
    return "; ".join(parts)


def blurry_vision_severity(filter_effects: list[FilterEffect]) -> str | None:
    for effect in filter_effects:
        if effect.filter_name != "blurry_vision":
            continue
        for severity in ("severe", "moderate", "mild"):
            if effect.reasons.get(f"severity_{severity}"):
                return severity
    return None


def format_prompt_context(flags) -> str:
    return (
        f"URL={'on' if flags.include_url else 'off'}, "
        f"Title={'on' if flags.include_title else 'off'}, "
        f"DOM={'on' if flags.include_dom else 'off'}, "
        f"Image={'on' if flags.include_image else 'off'}, "
        f"Image detail={flags.image_detail}"
    )


def format_cognitive_terms(terms: list[str]) -> str:
    return ", ".join(terms) if terms else "none"


def tactile_feedback_for_step(step_index: int, steps: list[StepRecord], language: str = "en") -> str:
    if step_index <= 1:
        return "No tactile feedback yet."
    return latest_tactile_feedback([steps[step_index - 2]], language=language)


def tactile_feedback_variant(step_index: int, steps: list[StepRecord]) -> str:
    if step_index <= 1:
        return "neutral"
    previous_step = steps[step_index - 2]
    if previous_step.noise_applied and previous_step.misfire:
        return "warning"
    if previous_step.noise_applied:
        return "info"
    return "neutral"


def build_markdown_report(
    config: SessionConfig,
    session_id: str,
    steps: list[StepRecord],
    output_path: Path,
    started_at: datetime,
    finished_at: datetime,
    final_status: str,
    active_filters: list[str],
) -> Path:
    pain_points = infer_pain_points(steps, config.agent.low_confidence_threshold)
    lines = [
        "# Ghost-UX Report",
        "",
        "## Run Meta",
        f"- Session ID: `{session_id}`",
        f"- Start URL: `{config.start_url}`",
        f"- Persona: {config.agent.persona}",
        f"- Goal: {config.agent.goal}",
        f"- Started At (UTC): `{started_at.isoformat()}`",
        f"- Finished At (UTC): `{finished_at.isoformat()}`",
        f"- Final Status: `{final_status}`",
        f"- Active Sensory Filters: `{', '.join(active_filters) if active_filters else 'none'}`",
        "",
        "## Potential UX Pain Points",
    ]
    for issue in pain_points:
        lines.append(f"- {issue}")

    lines.extend(["", "## Timeline"])
    if not steps:
        lines.append("- No steps were recorded.")
    for step in steps:
        tactile_feedback = tactile_feedback_for_step(step.step_index, steps, language=config.model.language)
        blurry_severity = blurry_vision_severity(step.filter_effects)
        lines.extend(
            [
                "",
                f"### Step {step.step_index}",
                f"- Page: `{step.observation_title}`",
                f"- URL: `{step.observation_url}`",
                f"- Thought: {step.action.thought}",
                f"- Action: `{step.action.action_type.value}`",
                f"- Target: `{step.action.target_element_id or '-'}`",
                f"- Input: `{step.action.input_text or '-'}`",
                f"- Confidence: `{step.action.confidence_score:.2f}`",
                f"- Execution Success: `{step.execution_success}`",
                f"- Execution Detail: {step.execution_detail}",
                f"- Noise Applied: `{step.noise_applied}`",
                f"- Noise Profile: `{step.noise_profile or '-'}`",
                f"- Intended Point: `{step.intended_point or '-'}`",
                f"- Actual Point: `{step.actual_point or '-'}`",
                f"- Offset: `{step.offset or '-'}`",
                f"- Actual Hit: `{step.actual_hit_summary or '-'}`",
                f"- Misfire: `{step.misfire}`",
                f"- Scroll Delta: `{step.scroll_delta or '-'}`",
                f"- Tactile Feedback Shown To Model: {tactile_feedback}",
                f"- Visible Elements: `{step.visible_elements}`",
                f"- Raw DOM Elements: `{step.raw_visible_elements}`",
                f"- Filtered DOM Elements: `{step.visible_elements}`",
                f"- Active Filters: `{', '.join(step.active_filters) if step.active_filters else 'none'}`",
                f"- Blurry Vision Severity: `{blurry_severity or '-'}`",
                f"- DOM Removed By Filters: `{step.dom_removed_count}`",
                f"- DOM Modified By Filters: `{step.dom_modified_count}`",
                f"- Filter Breakdown: {format_filter_breakdown(step.filter_effects)}",
                f"- Prompt Context: `{format_prompt_context(step.prompt_context_flags)}`",
                f"- Suspect Tokens: `{', '.join(step.suspect_tokens) if step.suspect_tokens else 'none'}`",
                f"- Jargon Terms Scrubbed: `{format_cognitive_terms(step.cognitive_terms)}`",
                f"- Screenshot: `{step.screenshot_path.name}`",
                "- Raw DOM Prompt:",
                "```text",
                step.raw_dom_prompt or "No raw DOM prompt recorded.",
                "```",
                "- Filtered DOM Prompt:",
                "```text",
                step.filtered_dom_prompt or "No filtered DOM prompt recorded.",
                "```",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_playback_report(
    config: SessionConfig,
    session_id: str,
    steps: list[StepRecord],
    output_path: Path,
    started_at: datetime,
    finished_at: datetime,
    final_status: str,
    active_filters: list[str],
) -> Path:
    pain_points = infer_pain_points(steps, config.agent.low_confidence_threshold)
    duration_seconds = max((finished_at - started_at).total_seconds(), 0.0)
    failed_count = sum(1 for step in steps if not step.execution_success)
    low_conf_count = sum(
        1 for step in steps if step.action.confidence_score < config.agent.low_confidence_threshold
    )
    cognitive_count = sum(1 for step in steps if step.cognitive_terms)
    critical_count = sum(
        1
        for step in steps
        if (not step.execution_success)
        or step.action.confidence_score < config.agent.low_confidence_threshold
    )
    action_counts = Counter(step.action.action_type.value for step in steps)
    ordered_actions = [action for action in ("click", "type", "scroll_down", "scroll_up", "finish", "fail") if action_counts.get(action)]
    recent_windows = [window for window in (3, 5, 10, 15) if len(steps) > window]

    meta_items = [
        ("Session ID", session_id),
        ("Start URL", str(config.start_url)),
        ("Persona", config.agent.persona),
        ("Goal", config.agent.goal),
        ("Started At (UTC)", started_at.isoformat()),
        ("Finished At (UTC)", finished_at.isoformat()),
        ("Duration (s)", f"{duration_seconds:.1f}"),
        ("Final Status", final_status),
        ("Active Filters", ", ".join(active_filters) if active_filters else "none"),
    ]
    meta_html = "\n".join(
        f"<div class='meta-item'><span class='meta-label'>{escape(label)}</span><span class='meta-value'>{escape(value)}</span></div>"
        for label, value in meta_items
    )
    pain_points_html = "\n".join(f"<li>{escape(item)}</li>" for item in pain_points)
    action_filter_html = "".join(
        f'<button class="action-chip" type="button" data-action-filter="{escape(action)}">{escape(action)} ({action_counts[action]})</button>'
        for action in ordered_actions
    )
    recent_options_html = "\n".join(
        [f'<option value="all">All steps ({len(steps)})</option>']
        + [
            f'<option value="{window}">Last {window} steps</option>'
            for window in recent_windows
        ]
    )
    mini_map_items: list[str] = []

    step_cards: list[str] = []
    for step in steps:
        tactile_feedback = tactile_feedback_for_step(step.step_index, steps, language=config.model.language)
        tactile_variant = tactile_feedback_variant(step.step_index, steps)
        blurry_severity = blurry_vision_severity(step.filter_effects)
        jargon_summary = format_cognitive_terms(step.cognitive_terms)
        jargon_callout = (
            f"""
                    <div class="jargon-callout">
                      <span class="jargon-label">Jargon Scrub Summary</span>
                      <p>{escape(jargon_summary)}</p>
                    </div>
            """
            if step.cognitive_terms
            else ""
        )
        screenshot_src = ""
        if step.screenshot_path.exists():
            encoded = base64.b64encode(step.screenshot_path.read_bytes()).decode("ascii")
            screenshot_src = f"data:image/png;base64,{encoded}"
        is_low_conf = step.action.confidence_score < config.agent.low_confidence_threshold
        status_badge = "Failure" if not step.execution_success else "Success"
        shot_html = (
            f'<img src="{screenshot_src}" alt="Step {step.step_index} screenshot" loading="lazy" />'
            if screenshot_src
            else '<div class="shot-missing">Screenshot unavailable</div>'
        )
        step_cards.append(
            f"""
            <article
              id="step-{step.step_index}"
              class="step-card {'failure' if not step.execution_success else ''}"
              data-success="{str(step.execution_success).lower()}"
              data-low-confidence="{str(is_low_conf).lower()}"
              data-cognitive-impact="{str(bool(step.cognitive_terms)).lower()}"
              data-action="{escape(step.action.action_type.value)}"
              data-step-index="{step.step_index}"
            >
              <details class="step-details" open>
                <summary class="step-summary">
                  <div class="summary-left">
                    <p class="eyebrow">Step {step.step_index}</p>
                    <h3>{escape(step.action.action_type.value.upper())}</h3>
                    <p class="summary-thought">{escape(step.action.thought)}</p>
                  </div>
                  <div class="summary-right">
                    <span class="status-pill {'status-failure' if not step.execution_success else 'status-success'}">{status_badge}</span>
                    {'<span class="status-pill status-low-conf">Low confidence</span>' if is_low_conf else ''}
                    {'<span class="status-pill status-motor-warning">Physical mis-touch</span>' if tactile_variant == 'warning' else ''}
                    {f'<span class="status-pill status-blurry">{escape(blurry_severity)}</span>' if blurry_severity else ''}
                    <div class="confidence">{step.action.confidence_score:.2f}</div>
                  </div>
                </summary>
                <div class="step-grid">
                  <div class="step-shot">
                    {shot_html}
                  </div>
                  <div class="step-copy">
                    <div class="tactile-callout tactile-{tactile_variant}">
                      <span class="tactile-label">{'Physical Mis-touch' if tactile_variant == 'warning' else 'Tactile Feedback'}</span>
                      <p>{escape(tactile_feedback)}</p>
                    </div>
                    {jargon_callout}
                    <p><strong>Page</strong><br />{escape(step.observation_title)}</p>
                    <p><strong>URL</strong><br /><code>{escape(step.observation_url)}</code></p>
                    <p><strong>Thought</strong><br />{escape(step.action.thought)}</p>
                    <p><strong>Target</strong><br /><code>{escape(step.action.target_element_id or '-')}</code></p>
                    <p><strong>Input</strong><br /><code>{escape(step.action.input_text or '-')}</code></p>
                    <p><strong>Execution</strong><br />{escape(step.execution_detail)}</p>
                    <p><strong>Noise Applied</strong><br />{'yes' if step.noise_applied else 'no'}</p>
                    <p><strong>Noise Profile</strong><br />{escape(step.noise_profile or '-')}</p>
                    <p><strong>Intended Point</strong><br /><code>{escape(str(step.intended_point or '-'))}</code></p>
                    <p><strong>Actual Point</strong><br /><code>{escape(str(step.actual_point or '-'))}</code></p>
                    <p><strong>Offset</strong><br /><code>{escape(str(step.offset or '-'))}</code></p>
                    <p><strong>Actual Hit</strong><br />{escape(step.actual_hit_summary or '-')}</p>
                    <p><strong>Misfire</strong><br />{'yes' if step.misfire else 'no'}</p>
                    <p><strong>Scroll Delta</strong><br /><code>{escape(str(step.scroll_delta or '-'))}</code></p>
                    <p><strong>Tactile Feedback Shown To Model</strong><br />{escape(tactile_feedback)}</p>
                    <p><strong>Visible Elements</strong><br />{step.visible_elements}</p>
                    <p><strong>Raw DOM Elements</strong><br />{step.raw_visible_elements}</p>
                    <p><strong>Filtered DOM Elements</strong><br />{step.visible_elements}</p>
                    <p><strong>Active Filters</strong><br />{escape(', '.join(step.active_filters) if step.active_filters else 'none')}</p>
                    <p><strong>Blurry Vision Severity</strong><br />{escape(blurry_severity or '-')}</p>
                    <p><strong>DOM Removed By Filters</strong><br />{step.dom_removed_count}</p>
                    <p><strong>DOM Modified By Filters</strong><br />{step.dom_modified_count}</p>
                    <p><strong>Filter Breakdown</strong><br />{escape(format_filter_breakdown(step.filter_effects))}</p>
                    <p><strong>Prompt Context</strong><br />{escape(format_prompt_context(step.prompt_context_flags))}</p>
                    <p><strong>Suspect Tokens</strong><br />{escape(', '.join(step.suspect_tokens) if step.suspect_tokens else 'none')}</p>
                    <p><strong>Jargon Terms Scrubbed</strong><br />{escape(format_cognitive_terms(step.cognitive_terms))}</p>
                    <p><strong>Success</strong><br />{'yes' if step.execution_success else 'no'}</p>
                    <p><strong>Raw DOM Prompt</strong><br /><pre>{escape(step.raw_dom_prompt or 'No raw DOM prompt recorded.')}</pre></p>
                    <p><strong>Filtered DOM Prompt</strong><br /><pre>{escape(step.filtered_dom_prompt or 'No filtered DOM prompt recorded.')}</pre></p>
                  </div>
                </div>
              </details>
            </article>
            """
        )
        mini_map_items.append(
            f"""
            <a
              class="mini-map-link {'is-failure' if not step.execution_success else ''}"
              href="#step-{step.step_index}"
              data-target="step-{step.step_index}"
            >
              <span class="mini-step-number">{step.step_index}</span>
              <span class="mini-step-copy">
                <strong>{escape(step.action.action_type.value)}</strong>
                <span>{escape(step.observation_title)}</span>
              </span>
              <span class="mini-step-confidence {'mini-low-conf' if is_low_conf else ''}">{step.action.confidence_score:.2f}</span>
            </a>
            """
        )

    if not step_cards:
        step_cards.append(
            """
            <article class="step-card empty">
              <div class="step-copy">
                <h3>No steps recorded</h3>
                <p>The session finished before any timeline entries were created.</p>
              </div>
            </article>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Ghost-UX Playback | {escape(session_id)}</title>
    <style>
      :root {{
        --bg: #0b1020;
        --panel: rgba(15, 23, 42, 0.82);
        --panel-border: rgba(148, 163, 184, 0.18);
        --text: #e5eefb;
        --muted: #98a7c2;
        --accent: #7dd3fc;
        --accent-2: #f9a8d4;
        --danger: #fb7185;
        --success: #86efac;
        --shadow: 0 24px 80px rgba(8, 15, 35, 0.45);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(125, 211, 252, 0.18), transparent 24%),
          radial-gradient(circle at top right, rgba(249, 168, 212, 0.16), transparent 22%),
          linear-gradient(180deg, #09101f 0%, #111a32 100%);
      }}
      .shell {{
        max-width: 1380px;
        margin: 0 auto;
        padding: 32px 20px 64px;
        display: grid;
        grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
        gap: 24px;
        align-items: start;
      }}
      .mini-map {{
        position: sticky;
        top: 22px;
        padding: 18px;
        background: rgba(9, 16, 31, 0.88);
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }}
      .mini-map h2 {{
        margin: 0 0 8px;
        font-size: 1.1rem;
      }}
      .mini-map p {{
        margin: 0;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.45;
      }}
      .mini-map-list {{
        display: grid;
        gap: 10px;
        margin-top: 18px;
      }}
      .mini-map-link {{
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) auto;
        gap: 10px;
        align-items: center;
        padding: 10px 12px;
        text-decoration: none;
        color: var(--text);
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 23, 42, 0.72);
      }}
      .mini-map-link:hover {{
        border-color: rgba(125, 211, 252, 0.38);
      }}
      .mini-map-link.active {{
        border-color: rgba(125, 211, 252, 0.48);
        background: rgba(125, 211, 252, 0.12);
      }}
      .mini-map-link.is-failure {{
        border-color: rgba(251, 113, 133, 0.24);
      }}
      .mini-step-number {{
        display: inline-grid;
        place-items: center;
        width: 30px;
        height: 30px;
        border-radius: 999px;
        background: rgba(125, 211, 252, 0.14);
        color: var(--accent);
        font-weight: 700;
        font-size: 0.86rem;
      }}
      .mini-step-copy {{
        min-width: 0;
        display: grid;
        gap: 2px;
      }}
      .mini-step-copy strong {{
        text-transform: uppercase;
        font-size: 0.82rem;
      }}
      .mini-step-copy span {{
        color: var(--muted);
        font-size: 0.84rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .mini-step-confidence {{
        font-size: 0.8rem;
        color: var(--muted);
      }}
      .mini-low-conf {{
        color: var(--accent-2);
      }}
      .content-shell {{
        min-width: 0;
      }}
      .hero {{
        background: linear-gradient(135deg, rgba(125, 211, 252, 0.16), rgba(17, 24, 39, 0.72));
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        padding: 28px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(12px);
      }}
      .hero h1 {{
        margin: 0 0 8px;
        font-size: clamp(2rem, 4vw, 3.3rem);
        line-height: 1.02;
      }}
      .hero p {{
        margin: 0;
        color: var(--muted);
        max-width: 760px;
      }}
      .eyebrow {{
        margin: 0 0 8px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.75rem;
        color: var(--accent);
      }}
      .meta-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin: 24px 0 0;
      }}
      .meta-item, .pain-panel, .step-card {{
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 20px;
        box-shadow: var(--shadow);
      }}
      .meta-item {{
        padding: 16px;
      }}
      .meta-label {{
        display: block;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 0.82rem;
      }}
      .meta-value {{
        display: block;
        font-size: 0.98rem;
        line-height: 1.45;
        word-break: break-word;
      }}
      .pain-panel {{
        margin: 26px 0 24px;
        padding: 22px 24px;
      }}
      .pain-panel ul {{
        margin: 14px 0 0;
        padding-left: 20px;
      }}
      .timeline {{
        display: grid;
        gap: 18px;
      }}
      .timeline-toolbar {{
        position: sticky;
        top: 14px;
        z-index: 4;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 20px;
        padding: 16px 18px;
        background: rgba(9, 16, 31, 0.86);
        border: 1px solid var(--panel-border);
        border-radius: 18px;
        box-shadow: var(--shadow);
        backdrop-filter: blur(14px);
      }}
      .toolbar-group {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
      }}
      .filter-chip, .toolbar-button, .action-chip, .toolbar-select {{
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.95);
        color: var(--text);
        font: inherit;
      }}
      .filter-chip, .toolbar-button, .action-chip {{
        cursor: pointer;
      }}
      .filter-chip {{
        padding: 10px 14px;
      }}
      .filter-chip.active {{
        border-color: rgba(125, 211, 252, 0.55);
        background: rgba(125, 211, 252, 0.16);
        color: var(--accent);
      }}
      .action-chip {{
        padding: 9px 13px;
        text-transform: uppercase;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
      }}
      .action-chip.active {{
        border-color: rgba(249, 168, 212, 0.55);
        background: rgba(249, 168, 212, 0.14);
        color: var(--accent-2);
      }}
      .toolbar-button {{
        padding: 10px 16px;
      }}
      .toolbar-select {{
        padding: 10px 14px;
        min-width: 155px;
      }}
      .toolbar-caption {{
        color: var(--muted);
        font-size: 0.92rem;
      }}
      .step-card {{
        overflow: hidden;
      }}
      .step-card.failure {{
        border-color: rgba(251, 113, 133, 0.42);
      }}
      .step-details[open] {{
        background: rgba(15, 23, 42, 0.92);
      }}
      .step-details > summary {{
        list-style: none;
      }}
      .step-details > summary::-webkit-details-marker {{
        display: none;
      }}
      .step-summary {{
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 16px;
        padding: 18px;
        cursor: pointer;
      }}
      .step-summary h3 {{
        margin: 0;
      }}
      .summary-left {{
        min-width: 0;
      }}
      .summary-thought {{
        margin: 8px 0 0;
        color: var(--muted);
        line-height: 1.45;
      }}
      .summary-right {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: end;
        gap: 8px;
      }}
      .status-pill {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 8px 11px;
        font-size: 0.82rem;
        font-weight: 700;
      }}
      .status-success {{
        background: rgba(134, 239, 172, 0.12);
        color: var(--success);
      }}
      .status-failure {{
        background: rgba(251, 113, 133, 0.12);
        color: var(--danger);
      }}
      .status-low-conf {{
        background: rgba(249, 168, 212, 0.12);
        color: var(--accent-2);
      }}
      .status-motor-warning {{
        background: rgba(251, 113, 133, 0.14);
        color: #fecdd3;
      }}
      .status-blurry {{
        background: rgba(125, 211, 252, 0.16);
        color: #bae6fd;
        text-transform: capitalize;
      }}
      .confidence {{
        min-width: 72px;
        padding: 10px 12px;
        border-radius: 999px;
        text-align: center;
        font-weight: 700;
        background: rgba(125, 211, 252, 0.12);
        color: var(--accent);
      }}
      .step-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.9fr);
        gap: 18px;
        padding: 0 18px 18px;
      }}
      .step-shot img {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: #020617;
      }}
      .shot-missing {{
        display: grid;
        place-items: center;
        min-height: 240px;
        border-radius: 14px;
        border: 1px dashed rgba(148, 163, 184, 0.3);
        background: rgba(2, 6, 23, 0.72);
        color: var(--muted);
      }}
      .step-copy p {{
        margin: 0 0 14px;
        line-height: 1.5;
        color: var(--muted);
      }}
      .tactile-callout {{
        margin: 0 0 16px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(15, 23, 42, 0.78);
      }}
      .tactile-callout p {{
        margin: 8px 0 0;
        color: var(--text);
      }}
      .tactile-label {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .tactile-warning {{
        border-color: rgba(251, 113, 133, 0.36);
        background: rgba(127, 29, 29, 0.22);
        box-shadow: inset 0 0 0 1px rgba(251, 113, 133, 0.08);
      }}
      .tactile-warning .tactile-label {{
        background: rgba(251, 113, 133, 0.16);
        color: #fecdd3;
      }}
      .tactile-info {{
        border-color: rgba(125, 211, 252, 0.3);
        background: rgba(8, 47, 73, 0.22);
      }}
      .tactile-info .tactile-label {{
        background: rgba(125, 211, 252, 0.16);
        color: #bae6fd;
      }}
      .tactile-neutral {{
        border-style: dashed;
      }}
      .tactile-neutral .tactile-label {{
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
      }}
      .jargon-callout {{
        margin: 0 0 16px;
        padding: 14px 16px;
        border-radius: 16px;
        border: 1px solid rgba(125, 211, 252, 0.25);
        background: rgba(8, 47, 73, 0.18);
      }}
      .jargon-callout p {{
        margin: 8px 0 0;
        color: var(--text);
      }}
      .jargon-label {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        background: rgba(125, 211, 252, 0.16);
        color: #bae6fd;
      }}
      .step-copy strong {{
        color: var(--text);
      }}
      .step-copy pre {{
        margin: 8px 0 0;
        padding: 12px;
        border-radius: 12px;
        overflow: auto;
        white-space: pre-wrap;
        background: rgba(2, 6, 23, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #dbeafe;
        font-family: "Cascadia Code", "SFMono-Regular", monospace;
        font-size: 0.84rem;
      }}
      .empty-state {{
        display: none;
        padding: 28px;
        text-align: center;
        color: var(--muted);
        background: var(--panel);
        border: 1px dashed rgba(148, 163, 184, 0.3);
        border-radius: 20px;
      }}
      code {{
        font-family: "Cascadia Code", "SFMono-Regular", monospace;
        color: #bfdbfe;
      }}
      @media (max-width: 860px) {{
        .shell {{
          padding: 20px 14px 48px;
          grid-template-columns: 1fr;
        }}
        .mini-map {{
          position: static;
          order: 2;
        }}
        .content-shell {{
          order: 1;
        }}
        .hero {{ padding: 22px; }}
        .timeline-toolbar {{ top: 8px; padding: 14px; }}
        .step-summary {{ padding: 16px; }}
        .step-grid {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="shell">
      <aside class="mini-map">
        <p class="eyebrow">Step Map</p>
        <h2>Jump to hotspots</h2>
        <p>Use the mini-map to jump around the journey. It follows your active filters and highlights the current step in view.</p>
        <nav class="mini-map-list" id="mini-map-list">
          {''.join(mini_map_items)}
        </nav>
      </aside>

      <div class="content-shell">
        <section class="hero">
          <p class="eyebrow">Ghost-UX Playback</p>
          <h1>Persona journey replay</h1>
          <p>Every step below pairs the AI tester's inner monologue with the actual screenshot that drove the next action.</p>
          <div class="meta-grid">{meta_html}</div>
        </section>

        <section class="pain-panel">
          <p class="eyebrow">UX Signals</p>
          <h2>Potential pain points</h2>
          <ul>{pain_points_html}</ul>
        </section>

        <section class="timeline">
          <div class="timeline-toolbar">
            <div class="toolbar-group">
              <button class="filter-chip active" type="button" data-filter="all">All steps ({len(steps)})</button>
              <button class="filter-chip" type="button" data-filter="failed">Failed only ({failed_count})</button>
              <button class="filter-chip" type="button" data-filter="low-confidence">Low confidence only ({low_conf_count})</button>
              <button class="filter-chip" type="button" data-filter="cognitive">Cognitive scrub ({cognitive_count})</button>
              <button class="filter-chip" type="button" data-filter="critical">Critical view ({critical_count})</button>
            </div>
            <div class="toolbar-group">
              <span class="toolbar-caption">Action type</span>
              <button class="action-chip active" type="button" data-action-filter="all">All actions</button>
              {action_filter_html}
            </div>
            <div class="toolbar-group">
              <label class="toolbar-caption" for="recent-filter">Recent window</label>
              <select class="toolbar-select" id="recent-filter">
                {recent_options_html}
              </select>
            </div>
            <div class="toolbar-group">
              <span class="toolbar-caption">Low confidence threshold: {config.agent.low_confidence_threshold:.2f}</span>
              <button class="toolbar-button" type="button" data-action="expand-all">Expand all</button>
              <button class="toolbar-button" type="button" data-action="collapse-all">Collapse all</button>
            </div>
          </div>
          <div class="empty-state" id="empty-state">
            No steps match the current filter. Try switching back to All steps.
          </div>
          {''.join(step_cards)}
        </section>
      </div>
    </main>
    <script>
      const cards = Array.from(document.querySelectorAll('.step-card'));
      const chips = Array.from(document.querySelectorAll('.filter-chip'));
      const actionChips = Array.from(document.querySelectorAll('.action-chip'));
      const detailsList = Array.from(document.querySelectorAll('.step-details'));
      const emptyState = document.getElementById('empty-state');
      const recentFilter = document.getElementById('recent-filter');
      const miniMapLinks = Array.from(document.querySelectorAll('.mini-map-link'));
      const maxStepIndex = cards.reduce((maxValue, card) => Math.max(maxValue, Number(card.dataset.stepIndex)), 0);
      const state = {{
        focusFilter: 'all',
        actionFilter: 'all',
        recentFilter: 'all',
      }};

      function matchesFocusFilter(card, filterName) {{
        const isFailure = card.dataset.success === 'false';
        const isLowConfidence = card.dataset.lowConfidence === 'true';
        const hasCognitiveImpact = card.dataset.cognitiveImpact === 'true';
        if (filterName === 'failed') return isFailure;
        if (filterName === 'low-confidence') return isLowConfidence;
        if (filterName === 'cognitive') return hasCognitiveImpact;
        if (filterName === 'critical') return isFailure || isLowConfidence;
        return true;
      }}

      function matchesActionFilter(card, actionFilter) {{
        return actionFilter === 'all' || card.dataset.action === actionFilter;
      }}

      function matchesRecentFilter(card, recentFilterValue) {{
        if (recentFilterValue === 'all') return true;
        const windowSize = Number(recentFilterValue);
        return Number(card.dataset.stepIndex) > maxStepIndex - windowSize;
      }}

      function updateMiniMap() {{
        let firstVisibleLink = null;
        miniMapLinks.forEach((link) => {{
          const target = document.getElementById(link.dataset.target);
          const visible = Boolean(target) && !target.hidden;
          link.hidden = !visible;
          if (visible && !firstVisibleLink) firstVisibleLink = link;
        }});
        if (!miniMapLinks.some((link) => link.classList.contains('active') && !link.hidden) && firstVisibleLink) {{
          miniMapLinks.forEach((link) => link.classList.toggle('active', link === firstVisibleLink));
        }}
      }}

      function applyAllFilters() {{
        let visibleCount = 0;
        cards.forEach((card) => {{
          const visible =
            matchesFocusFilter(card, state.focusFilter) &&
            matchesActionFilter(card, state.actionFilter) &&
            matchesRecentFilter(card, state.recentFilter);
          card.hidden = !visible;
          if (visible) visibleCount += 1;
        }});
        chips.forEach((chip) => chip.classList.toggle('active', chip.dataset.filter === state.focusFilter));
        actionChips.forEach((chip) => chip.classList.toggle('active', chip.dataset.actionFilter === state.actionFilter));
        emptyState.style.display = visibleCount === 0 ? 'block' : 'none';
        updateMiniMap();
      }}

      chips.forEach((chip) => {{
        chip.addEventListener('click', () => {{
          state.focusFilter = chip.dataset.filter;
          applyAllFilters();
        }});
      }});

      actionChips.forEach((chip) => {{
        chip.addEventListener('click', () => {{
          state.actionFilter = chip.dataset.actionFilter;
          applyAllFilters();
        }});
      }});

      recentFilter.addEventListener('change', (event) => {{
        state.recentFilter = event.target.value;
        applyAllFilters();
      }});

      document.querySelector('[data-action="expand-all"]').addEventListener('click', () => {{
        detailsList.forEach((item) => {{
          if (!item.closest('.step-card').hidden) item.open = true;
        }});
      }});

      document.querySelector('[data-action="collapse-all"]').addEventListener('click', () => {{
        detailsList.forEach((item) => {{
          if (!item.closest('.step-card').hidden) item.open = false;
        }});
      }});

      miniMapLinks.forEach((link) => {{
        link.addEventListener('click', () => {{
          miniMapLinks.forEach((item) => item.classList.toggle('active', item === link));
        }});
      }});

      function refreshActiveMiniMapLink() {{
        const visibleCards = cards.filter((card) => !card.hidden);
        if (!visibleCards.length) return;
        let activeCard = visibleCards[0];
        let bestDistance = Infinity;
        visibleCards.forEach((card) => {{
          const distance = Math.abs(card.getBoundingClientRect().top - 140);
          if (distance < bestDistance) {{
            bestDistance = distance;
            activeCard = card;
          }}
        }});
        miniMapLinks.forEach((link) => {{
          link.classList.toggle('active', link.dataset.target === activeCard.id && !link.hidden);
        }});
      }}

      window.addEventListener('scroll', refreshActiveMiniMapLink, {{ passive: true }});

      applyAllFilters();
      refreshActiveMiniMapLink();
    </script>
  </body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def build_run_artifacts(
    session_id: str,
    session_dir: Path,
    report_path: Path,
    playback_path: Path,
    active_filters: list[str],
    steps: list[StepRecord],
    final_status: str,
    started_at: datetime,
    finished_at: datetime,
) -> RunArtifacts:
    return RunArtifacts(
        session_id=session_id,
        session_dir=session_dir,
        report_path=report_path,
        playback_path=playback_path,
        active_filters=active_filters,
        steps=steps,
        final_status=final_status,
        started_at=started_at,
        finished_at=finished_at,
    )
