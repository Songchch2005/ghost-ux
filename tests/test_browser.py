from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest
from playwright.async_api import async_playwright

from ghost_ux.browser import (
    BLURRY_TEXT_PLACEHOLDER,
    BrowserController,
    COLLECT_INTERACTIVE_ELEMENTS_SCRIPT,
    normalize_font_size_to_px,
)
from ghost_ux.config import AgentConfig, BrowserConfig, MotorConfig, SensoryConfig
from ghost_ux.sensory.registry import build_sensory_pipeline


def test_normalize_font_size_to_px_supports_common_units() -> None:
    assert normalize_font_size_to_px("12px") == 12.0
    assert normalize_font_size_to_px("0.8rem", root_font_size_px=16.0) == 12.8
    assert normalize_font_size_to_px("0.875em", root_font_size_px=16.0, parent_font_size_px=20.0) == 17.5
    assert normalize_font_size_to_px("75%", root_font_size_px=16.0, parent_font_size_px=20.0) == 15.0
    assert normalize_font_size_to_px("nonsense") is None


@pytest.mark.asyncio
async def test_browser_fixture_cognitive_filter_blurs_static_web3_copy_without_polluting_action_space(
) -> None:
    fixture_path = Path("C:/Users/SCC/Desktop/gi/tests/fixtures/ethereum_web3_fixture.html")
    session_dir = Path("C:/Users/SCC/Desktop/gi/tests/.tmp") / uuid4().hex / "browser"
    session_dir.mkdir(parents=True, exist_ok=True)
    browser = BrowserController(
        BrowserConfig(headless=True, max_dom_elements=20),
        AgentConfig(
            persona="一个行业外的小白，看到 Web3 和去中心化黑话就发懵。",
            goal="找到开始按钮并判断首页是否容易理解。",
        ),
        SensoryConfig(
            auto_from_persona=True,
            cognitive_enabled_domains=["web3"],
            cognitive_visual_scrub=True,
            cognitive_visual_max_text_length=30,
        ),
        MotorConfig(),
        session_dir,
    )

    try:
        await browser.start()
        await browser.goto(fixture_path.as_uri())
        observation = await browser.observe(step_index=1)
    except PermissionError as exc:
        pytest.skip(f"Playwright launch is blocked in this sandbox: {exc}")
    finally:
        try:
            await browser.close()
        except Exception:
            pass

    assert any(not element.is_interactive and element.tag == "h1" for element in observation.elements)
    assert any(not element.is_interactive and element.tag == "p" for element in observation.elements)
    assert any(element.is_interactive and element.tag == "a" for element in observation.elements)

    pipeline = build_sensory_pipeline(browser.agent_config.persona, browser.sensory_config)
    filtered_bytes, filtered_elements, effects = pipeline.apply_with_trace(
        observation.screenshot_bytes,
        observation.elements,
    )
    filtered_interactive = [element for element in filtered_elements if element.is_interactive]
    filtered_static = [element for element in filtered_elements if not element.is_interactive]

    assert filtered_bytes != observation.screenshot_bytes
    assert any(element.tag == "h1" and "[unfamiliar jargon]" in (element.text or "") for element in filtered_static)
    assert any("web3" in {term.lower() for term in element.scrubbed_terms} for element in filtered_static)
    assert any("decentralization" in {term.lower() for term in element.scrubbed_terms} for element in filtered_static)
    assert any(element.tag == "a" and element.text == "Learn" for element in filtered_interactive)

    filtered_prompt = "\n".join(element.as_prompt_line() for element in filtered_interactive)
    assert "Web3 for everyone" not in filtered_prompt
    assert "Decentralization for builders." not in filtered_prompt
    assert "[ID:" in filtered_prompt
    assert "[READONLY]" not in filtered_prompt
    assert effects[-1].filter_name == "cognitive"
    assert effects[-1].modified_count >= 2
    assert effects[-1].reasons["visual_jargon_blur_applied"] >= 2


@pytest.mark.asyncio
async def test_collect_script_scrubs_descendant_text_and_unreadable_href() -> None:
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1280, "height": 720})
            await page.set_content(
                """
                <html>
                  <body style="background:#fff; color:#111; font-size:16px;">
                    <button id="quote-cta" style="font-size:16px; padding: 12px;">
                      Explore quotes
                      <span style="font-size:12px; color:#111;">thinking</span>
                    </button>
                    <a id="tiny-tag" href="/tag/change/page/1/" style="font-size:12px; color:#111;">change</a>
                  </body>
                </html>
                """
            )

            snapshot = await page.evaluate(COLLECT_INTERACTIVE_ELEMENTS_SCRIPT, {"maxElements": 10})
            await browser.close()
    except PermissionError as exc:
        pytest.skip(f"Playwright launch is blocked in this sandbox: {exc}")

    elements = {item["element_id"]: item for item in snapshot["elements"]}
    raw_elements = {item["element_id"]: item for item in snapshot["raw_elements"]}
    assert len(elements) == 2
    assert len(raw_elements) == 2

    button = next(item for item in elements.values() if item["tag"] == "button")
    tiny_link = next(item for item in elements.values() if item["tag"] == "a")
    raw_button = next(item for item in raw_elements.values() if item["tag"] == "button")
    raw_tiny_link = next(item for item in raw_elements.values() if item["tag"] == "a")

    assert button["text"] == "Explore quotes"
    assert "thinking" not in button["text"]
    assert "descendant_text_scrubbed" in button["scrub_reasons"]
    assert button["href"] is None
    assert raw_button["text"] == "Explore quotes thinking"

    assert tiny_link["text"] == BLURRY_TEXT_PLACEHOLDER
    assert tiny_link["href"] is None
    assert tiny_link["unreadable"] is True
    assert "font_size_below_threshold" in tiny_link["scrub_reasons"]
    assert raw_tiny_link["text"] == "change"
    assert raw_tiny_link["href"] == "/tag/change/page/1/"


@pytest.mark.asyncio
async def test_collect_script_symbol_labels_only_from_explicit_sources() -> None:
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1280, "height": 720})
            await page.set_content(
                """
                <html>
                  <body style="background:#fff; color:#111; font-size:16px;">
                    <section class="card">
                      <h2>Pure Hamburger Icon</h2>
                      <button id="pure-menu" aria-label="Menu" style="font-size:16px; width:48px; height:48px;">☰</button>
                    </section>
                    <section class="card">
                      <h2>Mixed Content</h2>
                      <button id="mixed-button" style="font-size:16px; padding: 12px;">🔍 搜索</button>
                    </section>
                    <section class="card row" style="display:flex; gap:12px; align-items:center;">
                      <button id="sibling-button" style="font-size:16px; width:48px; height:48px;">☰</button>
                      <span>展开菜单</span>
                    </section>
                    <section class="card row" style="display:flex; gap:12px; align-items:center;">
                      <button id="labelledby-button" aria-labelledby="menu-label" style="font-size:16px; width:48px; height:48px;">☰</button>
                      <span id="menu-label">打开导航</span>
                    </section>
                  </body>
                </html>
                """
            )

            snapshot = await page.evaluate(
                COLLECT_INTERACTIVE_ELEMENTS_SCRIPT,
                {"maxElements": 20, "nearbyTextRadiusPx": 48},
            )
            await browser.close()
    except PermissionError as exc:
        pytest.skip(f"Playwright launch is blocked in this sandbox: {exc}")

    buttons = [item for item in snapshot["elements"] if item["tag"] == "button"]

    pure_menu = next(
        item
        for item in buttons
        if item.get("text") == "☰"
        and item.get("nearby_visible_text") is None
        and item.get("labelledby_text") is None
    )
    mixed_button = next(item for item in buttons if item.get("text") == "🔍 搜索")
    sibling_button = next(item for item in buttons if item.get("nearby_visible_text") == "展开菜单")
    labelledby_button = next(item for item in buttons if item.get("labelledby_text") == "打开导航")

    assert pure_menu["nearby_visible_text"] is None
    assert pure_menu["labelledby_text"] is None
    assert mixed_button["visible_text"] == "🔍 搜索"
    assert sibling_button["nearby_visible_text"] == "展开菜单"
    assert labelledby_button["labelledby_text"] == "打开导航"
