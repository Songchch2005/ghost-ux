from __future__ import annotations

import asyncio
import base64
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ghost_ux.browser import BrowserController
from ghost_ux.config import SessionConfig
from ghost_ux.leak_detection import classify_probe_results, extract_suspect_tokens, observation_snapshot_dict, write_json
from ghost_ux.llm import build_vision_client
from ghost_ux.logging_utils import setup_logging
from ghost_ux.models import PromptContextFlags
from ghost_ux.sensory import build_sensory_pipeline


console = Console()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


async def _browser_smoke_test() -> tuple[bool, str]:
    try:
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://example.com", wait_until="domcontentloaded")
        title = await page.title()
        await browser.close()
        await playwright.stop()
        return True, f"Chromium launched successfully and opened page title `{title}`."
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"


async def run_diagnostics(api_key_env: str, include_browser: bool) -> int:
    table = Table(title="Ghost-UX Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    table.add_row("Python", "ok", sys.version.split()[0])
    table.add_row("ghost-ux", "ok", _package_version("ghost-ux"))
    table.add_row("playwright", "ok", _package_version("playwright"))
    table.add_row("openai", "ok", _package_version("openai"))
    table.add_row("pydantic", "ok", _package_version("pydantic"))
    table.add_row("rich", "ok", _package_version("rich"))
    table.add_row(
        f"env:{api_key_env}",
        "ok" if os.getenv(api_key_env) else "warn",
        "configured" if os.getenv(api_key_env) else "missing",
    )

    exit_code = 0
    if include_browser:
        success, detail = await _browser_smoke_test()
        table.add_row("browser-launch", "ok" if success else "fail", detail)
        if not success:
            exit_code = 1
    else:
        table.add_row(
            "browser-launch",
            "skip",
            "Skipped. Run `ghost-ux doctor --browser-check` to verify Chromium launch.",
        )

    console.print(table)
    return exit_code


def _probe_contexts(config: SessionConfig) -> dict[str, PromptContextFlags]:
    image_detail = config.model.image_detail
    return {
        "full": PromptContextFlags(
            include_url=config.model.include_url_in_prompt,
            include_title=config.model.include_title_in_prompt,
            include_dom=True,
            include_image=True,
            image_detail=image_detail,
        ),
        "dom_only": PromptContextFlags(
            include_url=False,
            include_title=False,
            include_dom=True,
            include_image=False,
            image_detail=image_detail,
        ),
        "image_only": PromptContextFlags(
            include_url=False,
            include_title=False,
            include_dom=False,
            include_image=True,
            image_detail="low",
        ),
        "blind_context": PromptContextFlags(
            include_url=False,
            include_title=False,
            include_dom=True,
            include_image=True,
            image_detail=image_detail,
        ),
    }


def _build_diagnosis_markdown(
    config: SessionConfig,
    session_id: str,
    diagnosis_path: Path,
    classification: str,
    suspect_tokens: list[str],
    evidence: list[str],
    probe_rows: list[tuple[str, str, str]],
) -> Path:
    lines = [
        "# Ghost-UX Leak Diagnosis",
        "",
        f"- Session ID: `{session_id}`",
        f"- Start URL: `{config.start_url}`",
        f"- Persona: {config.agent.persona}",
        f"- Goal: {config.agent.goal}",
        f"- Classification: `{classification}`",
        f"- Suspect Tokens: `{', '.join(suspect_tokens) if suspect_tokens else 'none'}`",
        "",
        "## Evidence",
    ]
    for item in evidence:
        lines.append(f"- {item}")
    lines.extend(["", "## Probe Results"])
    if not probe_rows:
        lines.append("- No probe results were collected.")
    for probe_name, tokens, rationale in probe_rows:
        lines.extend(
            [
                "",
                f"### {probe_name}",
                f"- Tokens: `{tokens or 'none'}`",
                f"- Rationale: {rationale or 'n/a'}",
            ]
        )
    diagnosis_path.write_text("\n".join(lines), encoding="utf-8")
    return diagnosis_path


async def run_leak_diagnosis(config: SessionConfig) -> tuple[int, Path]:
    session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    session_dir = config.report.output_dir / f"diagnose_{session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(session_dir / "diagnosis.log")

    browser = BrowserController(config.browser, config.agent, config.sensory, config.motor, session_dir)
    model = build_vision_client(config.model)
    pipeline = build_sensory_pipeline(config.agent.persona, config.sensory)

    await browser.start()
    try:
        await browser.goto(str(config.start_url))
        observation = await browser.observe(step_index=1)
        observation.raw_elements = [element.model_copy(deep=True) for element in (observation.raw_elements or observation.elements)]
        if pipeline.active_filter_names:
            filtered_bytes, filtered_elements, filter_effects = pipeline.apply_with_trace(
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
        else:
            observation.filtered_elements = [element.model_copy(deep=True) for element in observation.elements]
            observation.screenshot_base64 = base64.b64encode(observation.screenshot_bytes).decode("utf-8")

        observation.suspect_tokens = extract_suspect_tokens(observation.raw_elements, observation.filtered_elements)

        if config.debug.capture_raw_observation:
            write_json(session_dir / "raw_observation.json", observation_snapshot_dict(observation, raw=True))
        if config.debug.capture_filtered_observation:
            write_json(session_dir / "filtered_observation.json", observation_snapshot_dict(observation, raw=False))

        probe_results = []
        probe_rows: list[tuple[str, str, str]] = []
        for probe_name, prompt_context in _probe_contexts(config).items():
            payload = model.build_leak_probe_payload(
                observation,
                config.agent,
                [],
                observation.suspect_tokens,
                probe_name,
                prompt_context,
            )
            if config.debug.capture_prompt_payload:
                write_json(session_dir / f"llm_payload_{probe_name}.json", dict(payload))
            if config.debug.leak_probe_mode:
                result = await model.probe_visible_tokens(
                    observation,
                    config.agent,
                    [],
                    observation.suspect_tokens,
                    probe_name,
                    prompt_context,
                )
                probe_results.append(result)
                probe_rows.append((probe_name, ", ".join(result.visible_tokens), result.rationale))

        diagnosis = classify_probe_results(probe_results)
        diagnosis_path = _build_diagnosis_markdown(
            config,
            session_id,
            session_dir / "diagnosis.md",
            diagnosis.classification,
            observation.suspect_tokens,
            diagnosis.evidence,
            probe_rows,
        )

        table = Table(title="Ghost-UX Leak Diagnosis")
        table.add_column("Probe")
        table.add_column("Visible Tokens")
        table.add_column("Rationale")
        for probe_name, tokens, rationale in probe_rows:
            table.add_row(probe_name, tokens or "none", rationale or "n/a")
        console.print(table)
        console.print(
            Panel.fit(
                (
                    f"Classification: {diagnosis.classification}\n"
                    f"Suspect tokens: {', '.join(observation.suspect_tokens) or 'none'}\n"
                    f"Artifacts: {session_dir}\n"
                    f"Diagnosis: {diagnosis_path}"
                ),
                title="Leak Attribution Complete",
                border_style="cyan",
            )
        )
        return 0, diagnosis_path
    finally:
        await browser.close()
