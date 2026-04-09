from pathlib import Path

from loguru import logger
from PIL import Image

from ghost_ux.config import SensoryConfig
from ghost_ux.models import DOMElement, Observation
from ghost_ux.sensory import FilterPipeline
from ghost_ux.sensory.cognitive_terms import build_compiled_jargon_terms, replace_jargon_terms
from ghost_ux.sensory.filters import (
    BlurryVisionFilter,
    CognitiveFilter,
    LowPatienceFilter,
    SYMBOL_TEXT_PLACEHOLDER,
    SymbolCognitionFilter,
    TUNNEL_TEXT_PLACEHOLDER,
    TunnelVisionFilter,
    _explicit_label_source,
    _is_word_like_text,
    _scrub_occluded_element,
    _symbol_mask_bounds,
)
from ghost_ux.sensory.registry import build_sensory_pipeline
from ghost_ux.sensory.utils import save_image_to_png_bytes


def _sample_png_bytes(color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    image = Image.new("RGB", (120, 80), color)
    return save_image_to_png_bytes(image.convert("RGBA"))


def test_blurry_filter_removes_small_or_low_contrast_text() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="button",
            text="Tiny text",
            font_size=12,
            text_color="rgb(10, 10, 10)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=10,
            width=80,
            height=30,
        ),
        DOMElement(
            element_id="2",
            tag="button",
            text="Readable CTA",
            font_size=16,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=50,
            width=80,
            height=30,
        ),
    ]
    blurry_filter = BlurryVisionFilter(radius=2.0)
    _, filtered_elements = blurry_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["2"]
    assert blurry_filter.last_effect is not None
    assert blurry_filter.last_effect.removed_count == 1
    assert blurry_filter.last_effect.modified_count == 1
    assert blurry_filter.last_effect.reasons["font_size_below_threshold"] == 1


def test_blurry_filter_keeps_roomier_mid_size_titles_but_removes_tiny_metadata() -> None:
    dom_elements = [
        DOMElement(
            element_id="title",
            tag="a",
            text="LittleSnitch for Linux",
            visible_text="LittleSnitch for Linux",
            font_size=13.3333,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=10,
            width=140,
            height=24,
        ),
        DOMElement(
            element_id="meta",
            tag="a",
            text="60 comments",
            font_size=9.3333,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=34,
            width=80,
            height=12,
        ),
    ]
    blurry_filter = BlurryVisionFilter(radius=3.0)
    _, filtered_elements = blurry_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["title"]
    assert blurry_filter.last_effect is not None
    assert blurry_filter.last_effect.removed_count == 1
    assert blurry_filter.last_effect.modified_count == 1


def test_blurry_filter_removes_compact_small_link_rows_even_above_base_threshold() -> None:
    dom_elements = [
        DOMElement(
            element_id="hn-title",
            tag="a",
            text="LittleSnitch for Linux",
            visible_text="LittleSnitch for Linux",
            font_size=13.3333,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=10,
            width=160,
            height=16,
        ),
        DOMElement(
            element_id="primary-cta",
            tag="a",
            text="Get started",
            visible_text="Get started",
            font_size=16.0,
            text_color="rgb(255, 255, 255)",
            background_color="rgb(17, 24, 39)",
            x=10,
            y=40,
            width=120,
            height=44,
        ),
    ]
    blurry_filter = BlurryVisionFilter(radius=3.0)
    _, filtered_elements = blurry_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["primary-cta"]
    assert blurry_filter.last_effect is not None
    assert blurry_filter.last_effect.reasons["compact_text_hard_to_read"] == 1


def test_blurry_filter_severity_profiles_change_readability_thresholds() -> None:
    dom_elements = [
        DOMElement(
            element_id="small-nav",
            tag="a",
            text="Pricing",
            visible_text="Pricing",
            font_size=13.2,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=10,
            width=56,
            height=16,
        ),
        DOMElement(
            element_id="cta",
            tag="a",
            text="Start free trial",
            visible_text="Start free trial",
            font_size=16.0,
            text_color="rgb(255, 255, 255)",
            background_color="rgb(17, 24, 39)",
            x=10,
            y=40,
            width=140,
            height=44,
        ),
    ]
    mild_filter = BlurryVisionFilter(radius=3.0, severity="mild")
    severe_filter = BlurryVisionFilter(radius=3.0, severity="severe")

    _, mild_elements = mild_filter.apply(_sample_png_bytes(), dom_elements)
    _, severe_elements = severe_filter.apply(_sample_png_bytes(), dom_elements)

    assert [element.element_id for element in mild_elements] == ["small-nav", "cta"]
    assert [element.element_id for element in severe_elements] == ["cta"]
    assert severe_filter.last_effect is not None
    assert severe_filter.last_effect.reasons["severity_severe"] == 1


def test_blurry_filter_applies_local_occlusion_to_tiny_text_regions() -> None:
    image = Image.new("RGB", (120, 80), (255, 255, 255))
    for x in range(10, 55):
        for y in range(10, 24):
            image.putpixel((x, y), (24, 24, 24))
    base_bytes = save_image_to_png_bytes(image.convert("RGBA"))
    blurry_filter = BlurryVisionFilter(radius=3.0)
    dom_elements = [
        DOMElement(
            element_id="meta",
            tag="a",
            text="9 comments",
            font_size=9.5,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=10,
            width=45,
            height=14,
        )
    ]
    filtered_bytes, filtered_elements = blurry_filter.apply(base_bytes, dom_elements)
    assert filtered_bytes != base_bytes
    assert filtered_elements == []


def test_tunnel_vision_filter_removes_edge_dom_elements() -> None:
    dom_elements = [
        DOMElement(element_id="1", tag="button", text="Center", x=45, y=30, width=20, height=20),
        DOMElement(element_id="2", tag="button", text="Edge", x=100, y=5, width=15, height=15),
    ]
    tunnel_filter = TunnelVisionFilter(visible_width_ratio=0.5, visible_height_ratio=0.5)
    _, filtered_elements = tunnel_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["1"]
    assert tunnel_filter.last_effect is not None
    assert tunnel_filter.last_effect.removed_count == 1
    assert "outside_tunnel_focus" in tunnel_filter.last_effect.reasons


def test_tunnel_vision_filter_removes_boundary_dom_elements_conservatively() -> None:
    dom_elements = [
        DOMElement(element_id="1", tag="button", text="Boundary", x=22, y=20, width=28, height=22),
        DOMElement(element_id="2", tag="button", text="Center", x=48, y=28, width=16, height=16),
    ]
    tunnel_filter = TunnelVisionFilter(
        visible_width_ratio=0.5,
        visible_height_ratio=0.5,
        min_visible_ratio=0.65,
        safety_inset_ratio=0.12,
    )
    _, filtered_elements = tunnel_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["2"]
    assert tunnel_filter.last_effect is not None
    assert tunnel_filter.last_effect.removed_count == 1
    assert "below_tunnel_visibility_threshold" in tunnel_filter.last_effect.reasons


def test_tunnel_scrub_helper_overwrites_semantic_fields() -> None:
    element = DOMElement(
        element_id="7",
        tag="a",
        text="login",
        aria_label="login link",
        title="login",
        href="/login",
        x=0,
        y=0,
        width=20,
        height=10,
    )
    scrubbed = _scrub_occluded_element(
        element,
        placeholder=TUNNEL_TEXT_PLACEHOLDER,
        reasons=["outside_tunnel_focus"],
    )
    assert scrubbed.text == TUNNEL_TEXT_PLACEHOLDER
    assert scrubbed.aria_label == TUNNEL_TEXT_PLACEHOLDER
    assert scrubbed.title == TUNNEL_TEXT_PLACEHOLDER
    assert scrubbed.href is None
    assert "outside_tunnel_focus" in scrubbed.scrub_reasons


def test_tunnel_vision_filter_logs_scrubbed_elements() -> None:
    dom_elements = [
        DOMElement(
            element_id="2",
            tag="a",
            text="login",
            aria_label="login",
            href="/login",
            x=102,
            y=4,
            width=14,
            height=14,
        )
    ]
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        tunnel_filter = TunnelVisionFilter(visible_width_ratio=0.5, visible_height_ratio=0.5)
        _, filtered_elements = tunnel_filter.apply(_sample_png_bytes(), dom_elements)
    finally:
        logger.remove(sink_id)
    assert filtered_elements == []
    assert any("TunnelVision scrubbed element" in message for message in messages)
    assert any("login" in message for message in messages)


def test_symbol_cognition_filter_removes_icon_only_svg_button() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="button",
            text=None,
            visible_text=None,
            nearby_visible_text=None,
            labelledby_text=None,
            aria_label="Search",
            has_svg_child=True,
            child_tags=["svg", "path"],
            icon_x=12,
            icon_y=12,
            icon_width=12,
            icon_height=12,
            x=10,
            y=10,
            width=20,
            height=20,
        ),
        DOMElement(
            element_id="2",
            tag="button",
            text="Search",
            visible_text="Search",
            has_svg_child=True,
            nearby_visible_text=None,
            labelledby_text=None,
            x=40,
            y=10,
            width=28,
            height=20,
        ),
    ]
    symbol_filter = SymbolCognitionFilter()
    _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["2"]
    assert symbol_filter.last_effect is not None
    assert symbol_filter.last_effect.removed_count == 1
    assert symbol_filter.last_effect.modified_count == 1
    assert symbol_filter.last_effect.reasons["icon_only_symbol"] == 1
    assert symbol_filter.last_effect.reasons["missing_visible_label"] == 1


def test_symbol_cognition_filter_removes_icon_font_button_without_label() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="a",
            visible_text=None,
            css_classes=["btn", "fa-search"],
            has_icon_like_class=True,
            child_tags=["i"],
            href="/search",
            x=12,
            y=12,
            width=18,
            height=18,
        ),
    ]
    symbol_filter = SymbolCognitionFilter(mask_style="mosaic")
    _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    assert filtered_elements == []
    assert symbol_filter.last_effect is not None
    assert symbol_filter.last_effect.removed_count == 1
    assert symbol_filter.last_effect.modified_count == 1
    assert symbol_filter.last_effect.reasons["icon_only_symbol"] == 1


def test_symbol_cognition_filter_keeps_mixed_content_button_with_visible_text() -> None:
    dom_elements = [
        DOMElement(
            element_id="mixed",
            tag="button",
            text="🔍 搜索",
            visible_text="🔍 搜索",
            child_tags=["span"],
            x=12,
            y=10,
            width=60,
            height=24,
        )
    ]
    symbol_filter = SymbolCognitionFilter()
    _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["mixed"]
    assert symbol_filter.last_effect is not None
    assert symbol_filter.last_effect.removed_count == 0
    assert _explicit_label_source(dom_elements[0]) == "visible_text"


def test_symbol_cognition_filter_keeps_icon_with_adjacent_text() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="button",
            text=None,
            visible_text=None,
            nearby_visible_text="展开菜单",
            labelledby_text=None,
            has_svg_child=True,
            child_tags=["svg"],
            x=10,
            y=10,
            width=20,
            height=20,
        ),
    ]
    symbol_filter = SymbolCognitionFilter()
    _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["1"]
    assert symbol_filter.last_effect is not None
    assert symbol_filter.last_effect.removed_count == 0


def test_symbol_cognition_filter_keeps_icon_with_labelledby_text() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="button",
            text=None,
            visible_text=None,
            nearby_visible_text=None,
            labelledby_text="打开导航",
            aria_label="Menu",
            has_svg_child=True,
            child_tags=["svg"],
            x=10,
            y=10,
            width=20,
            height=20,
        ),
    ]
    symbol_filter = SymbolCognitionFilter()
    _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["1"]


def test_symbol_cognition_filter_placeholder_mode_keeps_scrubbed_element() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="button",
            text="☰",
            visible_text="☰",
            aria_label="Menu",
            has_svg_child=False,
            has_icon_like_class=False,
            child_tags=[],
            x=8,
            y=8,
            width=16,
            height=16,
        ),
    ]
    symbol_filter = SymbolCognitionFilter(dom_strategy="placeholder_only")
    _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    assert len(filtered_elements) == 1
    assert filtered_elements[0].text == SYMBOL_TEXT_PLACEHOLDER
    assert filtered_elements[0].aria_label == SYMBOL_TEXT_PLACEHOLDER
    assert filtered_elements[0].href is None
    assert "icon_only_symbol" in filtered_elements[0].scrub_reasons
    assert symbol_filter.last_effect is not None
    assert symbol_filter.last_effect.removed_count == 0
    assert symbol_filter.last_effect.modified_count == 1


def test_symbol_cognition_filter_logs_scrubbed_elements() -> None:
    dom_elements = [
        DOMElement(
            element_id="3",
            tag="button",
            aria_label="Search",
            css_classes=["icon-button", "fa-search"],
            has_icon_like_class=True,
            child_tags=["i"],
            x=10,
            y=10,
            width=18,
            height=18,
        )
    ]
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        symbol_filter = SymbolCognitionFilter()
        _, filtered_elements = symbol_filter.apply(_sample_png_bytes(), dom_elements)
    finally:
        logger.remove(sink_id)
    assert filtered_elements == []
    assert any("SymbolCognition scrubbed element" in message for message in messages)


def test_readable_text_regex_accepts_cjk_and_rejects_placeholders() -> None:
    assert _is_word_like_text("🔍 搜索")
    assert _is_word_like_text("打开菜单")
    assert _is_word_like_text("Search")
    assert not _is_word_like_text("☰")
    assert not _is_word_like_text(SYMBOL_TEXT_PLACEHOLDER)


def test_symbol_mask_prefers_icon_bbox_and_is_smaller_than_full_control() -> None:
    element_with_icon = DOMElement(
        element_id="icon",
        tag="button",
        x=10,
        y=10,
        width=60,
        height=30,
        icon_x=18,
        icon_y=14,
        icon_width=14,
        icon_height=14,
    )
    masked = _symbol_mask_bounds(element_with_icon, (120, 80), padding=4)
    assert masked == (14, 10, 36, 32)

    fallback_element = DOMElement(
        element_id="fallback",
        tag="button",
        x=10,
        y=10,
        width=60,
        height=30,
    )
    fallback_masked = _symbol_mask_bounds(fallback_element, (120, 80), padding=4)
    full_bounds = (6, 6, 74, 44)
    assert fallback_masked[0] > full_bounds[0]
    assert fallback_masked[1] > full_bounds[1]
    assert fallback_masked[2] < full_bounds[2]
    assert fallback_masked[3] < full_bounds[3]


def test_persona_keywords_enable_filters_automatically() -> None:
    config = SensoryConfig(auto_from_persona=True)
    pipeline = build_sensory_pipeline("一位老花、色弱、缺乏耐心而且看不懂图标的用户。", config)
    assert pipeline.active_filter_names == [
        "blurry_vision",
        "colorblindness",
        "low_patience",
        "symbol_cognition",
    ]


def test_filter_pipeline_chains_filters() -> None:
    dom_elements = [DOMElement(element_id="1", tag="button", text="CTA", x=50, y=30, width=20, height=20)]
    pipeline = FilterPipeline([TunnelVisionFilter(), BlurryVisionFilter(radius=1.0)])
    filtered_bytes, filtered_elements = pipeline.apply(_sample_png_bytes(), dom_elements)
    assert isinstance(filtered_bytes, bytes)
    assert [element.element_id for element in filtered_elements] == ["1"]


def test_low_patience_filter_truncates_far_or_verbose_elements() -> None:
    dom_elements = [
        DOMElement(element_id="1", tag="button", text="Short CTA", x=20, y=10, width=20, height=20),
        DOMElement(
            element_id="2",
            tag="button",
            text="This call to action includes a very long explanation that impatient users will skip",
            x=20,
            y=15,
            width=30,
            height=20,
        ),
        DOMElement(element_id="3", tag="button", text="Lower CTA", x=20, y=70, width=20, height=10),
    ]
    low_patience_filter = LowPatienceFilter(max_text_length=24, max_y_ratio=0.68)
    _, filtered_elements = low_patience_filter.apply(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["1"]
    assert low_patience_filter.last_effect is not None
    assert low_patience_filter.last_effect.removed_count == 2


def test_filter_pipeline_returns_effect_trace() -> None:
    dom_elements = [
        DOMElement(
            element_id="1",
            tag="button",
            text="Tiny text",
            font_size=12,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=10,
            y=10,
            width=20,
            height=20,
        ),
        DOMElement(
            element_id="2",
            tag="button",
            text="Readable CTA",
            font_size=16,
            text_color="rgb(0, 0, 0)",
            background_color="rgb(255, 255, 255)",
            x=30,
            y=10,
            width=20,
            height=20,
        ),
    ]
    pipeline = FilterPipeline([BlurryVisionFilter(radius=1.0), CognitiveFilter()])
    _, filtered_elements, effects = pipeline.apply_with_trace(_sample_png_bytes(), dom_elements)
    assert [element.element_id for element in filtered_elements] == ["2"]
    assert [effect.filter_name for effect in effects] == ["blurry_vision", "cognitive"]
    assert effects[0].removed_count == 1
    assert effects[0].modified_count == 1
    assert effects[1].removed_count == 0


def test_cognitive_term_replacement_uses_smart_boundaries_for_english_terms() -> None:
    terms = build_compiled_jargon_terms(["ai"], [], case_sensitive=False)
    replaced_api, api_matches, _, _ = replace_jargon_terms(
        "Use the API to call this service.",
        terms,
        placeholder="[unfamiliar jargon]",
        density_threshold=2,
    )
    replaced_capital, capital_matches, _, _ = replace_jargon_terms(
        "capital planning avoids certain pitfalls.",
        terms,
        placeholder="[unfamiliar jargon]",
        density_threshold=2,
    )
    assert api_matches >= 1
    assert "[unfamiliar jargon]" in replaced_api
    assert capital_matches == 0
    assert replaced_capital == "capital planning avoids certain pitfalls."


def test_cognitive_term_replacement_supports_web3_dictionary_terms() -> None:
    terms = build_compiled_jargon_terms(["web3"], [], case_sensitive=False)
    replaced, matches, matched_terms, _ = replace_jargon_terms(
        "Explore Decentralized Web 3.0 tools, DAO governance, and Smart Contract security.",
        terms,
        placeholder="[unfamiliar jargon]",
        density_threshold=99,
    )
    assert matches >= 4
    assert {"Decentralized", "Web 3.0", "DAO", "Smart Contract"} <= set(matched_terms)
    assert replaced.count("[unfamiliar jargon]") >= 4


def test_cognitive_filter_scrubs_alt_text_without_dropping_structure() -> None:
    base_bytes = _sample_png_bytes()
    cognitive_filter = CognitiveFilter(
        SensoryConfig(
            cognitive_enabled_domains=["b2b_saas", "ai"],
            cognitive_visual_scrub=False,
        )
    )
    dom_elements = [
        DOMElement(
            element_id="img-1",
            tag="img",
            alt="SaaS API dashboard",
            x=10,
            y=10,
            width=40,
            height=20,
        )
    ]
    filtered_bytes, filtered_elements = cognitive_filter.apply(base_bytes, dom_elements)
    assert filtered_bytes == base_bytes
    assert len(filtered_elements) == 1
    assert filtered_elements[0].alt == "[unfamiliar jargon]"
    assert set(filtered_elements[0].scrubbed_terms) == {"API", "SaaS"}
    assert filtered_elements[0].element_id == "img-1"
    assert cognitive_filter.last_effect is not None
    assert cognitive_filter.last_effect.modified_count == 1
    assert cognitive_filter.last_effect.reasons["jargon_alt_scrubbed"] == 1


def test_cognitive_filter_does_not_visually_blur_long_text_blocks() -> None:
    base_bytes = _sample_png_bytes()
    cognitive_filter = CognitiveFilter(
        SensoryConfig(
            cognitive_enabled_domains=["b2b_saas", "ai", "general"],
            cognitive_visual_scrub=True,
            cognitive_visual_max_text_length=30,
        )
    )
    long_text = (
        "我们的 SaaS API 平台提供闭环分析和底层逻辑洞察，"
        "帮助团队统一看板与自动化流程。"
    )
    dom_elements = [
        DOMElement(
            element_id="copy",
            tag="p",
            text=long_text,
            visible_text=long_text,
            font_size=14,
            x=5,
            y=5,
            width=110,
            height=60,
        )
    ]
    filtered_bytes, filtered_elements = cognitive_filter.apply(base_bytes, dom_elements)
    assert filtered_bytes == base_bytes
    assert filtered_elements[0].text == "[unfamiliar jargon]"
    assert {"SaaS", "API", "闭环", "底层逻辑"} <= set(filtered_elements[0].scrubbed_terms)
    assert cognitive_filter.last_effect is not None
    assert "visual_jargon_blur_applied" not in cognitive_filter.last_effect.reasons


def test_cognitive_filter_visually_blurs_short_labels() -> None:
    image = Image.new("RGB", (120, 80), (255, 255, 255))
    for x in range(20, 60):
        for y in range(20, 36):
            image.putpixel((x, y), (32, 32, 32))
    base_bytes = save_image_to_png_bytes(image.convert("RGBA"))
    cognitive_filter = CognitiveFilter(
        SensoryConfig(
            cognitive_enabled_domains=["b2b_saas"],
            cognitive_visual_scrub=True,
            cognitive_visual_max_text_length=30,
        )
    )
    dom_elements = [
        DOMElement(
            element_id="cta",
            tag="button",
            text="SaaS 报价",
            visible_text="SaaS 报价",
            x=20,
            y=20,
            width=40,
            height=16,
        )
    ]
    filtered_bytes, filtered_elements = cognitive_filter.apply(base_bytes, dom_elements)
    assert filtered_bytes != base_bytes
    assert "[unfamiliar jargon]" in filtered_elements[0].text
    assert "SaaS" in filtered_elements[0].scrubbed_terms
    assert cognitive_filter.last_effect is not None
    assert cognitive_filter.last_effect.reasons["visual_jargon_blur_applied"] == 1


def test_cognitive_filter_can_blur_static_read_only_text_blocks() -> None:
    image = Image.new("RGB", (120, 80), (255, 255, 255))
    for x in range(15, 105):
        for y in range(12, 34):
            image.putpixel((x, y), (24, 24, 24))
    base_bytes = save_image_to_png_bytes(image.convert("RGBA"))
    cognitive_filter = CognitiveFilter(
        SensoryConfig(
            cognitive_enabled_domains=["web3"],
            cognitive_visual_scrub=True,
            cognitive_visual_max_text_length=30,
        )
    )
    dom_elements = [
        DOMElement(
            element_id="static-1",
            tag="h1",
            is_interactive=False,
            text="Web3 decentralization",
            visible_text="Web3 decentralization",
            x=15,
            y=12,
            width=90,
            height=22,
        )
    ]
    filtered_bytes, filtered_elements = cognitive_filter.apply(base_bytes, dom_elements)
    assert filtered_bytes != base_bytes
    assert filtered_elements[0].is_interactive is False
    assert "[unfamiliar jargon]" in filtered_elements[0].text
    scrubbed_terms = {term.lower() for term in filtered_elements[0].scrubbed_terms}
    assert {"web3", "decentralization"} <= scrubbed_terms


def test_observation_prompts_exclude_read_only_static_elements() -> None:
    observation = Observation(
        step_index=1,
        url="https://example.com",
        title="Example",
        screenshot_bytes=b"demo",
        screenshot_base64="ZGVtbw==",
        screenshot_path=Path("C:/Users/SCC/Desktop/gi/artifacts/demo.png"),
        elements=[
            DOMElement(
                element_id="1",
                tag="button",
                is_interactive=True,
                text="Get started",
                x=10,
                y=10,
                width=40,
                height=20,
            ),
            DOMElement(
                element_id="static-1",
                tag="h1",
                is_interactive=False,
                text="[unfamiliar jargon]",
                x=5,
                y=45,
                width=90,
                height=24,
            ),
        ],
        raw_elements=[],
        filtered_elements=[],
        viewport_width=120,
        viewport_height=80,
    )
    assert "[ID:1]" in observation.dom_prompt
    assert "static-1" not in observation.dom_prompt
    assert "[READONLY]" not in observation.dom_prompt


def test_persona_keywords_enable_cognitive_filter_for_jargon_outsider_profiles() -> None:
    config = SensoryConfig(auto_from_persona=True)
    pipeline = build_sensory_pipeline(
        "这是一个行业外的小白，对黑话和专业术语一窍不通，看不懂英文 jargon。",
        config,
    )
    assert "cognitive" in pipeline.active_filter_names


def test_persona_keywords_escalate_blurry_vision_severity_for_severe_presbyopia() -> None:
    config = SensoryConfig(auto_from_persona=True, blurry_vision_severity="auto")
    pipeline = build_sensory_pipeline(
        "你是一位严重老花眼的 65 岁用户，今天没戴老花镜，近处小字几乎看不清。",
        config,
    )
    assert "blurry_vision" in pipeline.active_filter_names
    assert config.blurry_vision_severity == "severe"


def test_persona_keywords_keep_blurry_vision_mild_for_early_presbyopia() -> None:
    config = SensoryConfig(auto_from_persona=True, blurry_vision_severity="auto")
    pipeline = build_sensory_pipeline(
        "你是一位刚开始老花的用户，偶尔看不清很小的字，但大部分页面还能正常阅读。",
        config,
    )
    assert "blurry_vision" in pipeline.active_filter_names
    assert config.blurry_vision_severity == "mild"
