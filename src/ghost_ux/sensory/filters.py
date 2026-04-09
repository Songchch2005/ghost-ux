from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from loguru import logger

from ghost_ux.browser import BLURRY_TEXT_PLACEHOLDER
from ghost_ux.config import SensoryConfig
from ghost_ux.models import DOMElement
from ghost_ux.sensory.base import BaseSensoryFilter
from ghost_ux.sensory.cognitive_terms import build_compiled_jargon_terms, replace_jargon_terms
from ghost_ux.sensory.utils import contrast_ratio, load_image_from_bytes, save_image_to_png_bytes

TUNNEL_TEXT_PLACEHOLDER = "[text occluded by limited vision]"
SYMBOL_TEXT_PLACEHOLDER = "[unknown symbol]"
READABLE_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9]")
SEMANTIC_TEXT_FIELDS = (
    "text",
    "visible_text",
    "nearby_visible_text",
    "labelledby_text",
    "title_text",
    "placeholder",
    "alt",
    "aria_label",
    "title",
    "name",
    "value",
)
@dataclass(frozen=True)
class PresbyopiaProfile:
    severity: str
    min_readable_font_px: float
    compact_font_px: float
    compact_line_height_px: float
    compact_min_chars: int
    global_blur_radius: float
    local_blur_radius: float
    overlay_alpha: int
    image_detail: str


PRESBYOPIA_PROFILES = {
    "mild": PresbyopiaProfile(
        severity="mild",
        min_readable_font_px=11.5,
        compact_font_px=13.5,
        compact_line_height_px=17.0,
        compact_min_chars=16,
        global_blur_radius=0.8,
        local_blur_radius=5.5,
        overlay_alpha=120,
        image_detail="auto",
    ),
    "moderate": PresbyopiaProfile(
        severity="moderate",
        min_readable_font_px=12.5,
        compact_font_px=14.5,
        compact_line_height_px=18.0,
        compact_min_chars=12,
        global_blur_radius=1.2,
        local_blur_radius=6.8,
        overlay_alpha=150,
        image_detail="low",
    ),
    "severe": PresbyopiaProfile(
        severity="severe",
        min_readable_font_px=14.5,
        compact_font_px=17.0,
        compact_line_height_px=22.0,
        compact_min_chars=8,
        global_blur_radius=1.9,
        local_blur_radius=8.5,
        overlay_alpha=185,
        image_detail="low",
    ),
}


def _scrub_unreadable_element(element: DOMElement, reasons: list[str]) -> DOMElement:
    scrubbed = element.model_copy(deep=True)
    changed_fields: set[str] = set(scrubbed.scrubbed_fields)

    for field_name in ("text", "aria_label", "title", "name", "placeholder", "alt", "value"):
        current_value = getattr(scrubbed, field_name)
        if current_value:
            setattr(scrubbed, field_name, BLURRY_TEXT_PLACEHOLDER)
            changed_fields.add(field_name)

    if scrubbed.href:
        scrubbed.href = None
        changed_fields.add("href")

    scrubbed.unreadable = True
    scrubbed.scrub_reasons = list(dict.fromkeys([*scrubbed.scrub_reasons, *reasons]))
    scrubbed.scrubbed_fields = sorted(changed_fields)
    return scrubbed


def _scrub_occluded_element(
    element: DOMElement,
    *,
    placeholder: str,
    reasons: list[str],
) -> DOMElement:
    scrubbed = element.model_copy(deep=True)
    changed_fields: set[str] = set(scrubbed.scrubbed_fields)

    for field_name in ("text", "aria_label", "title", "name", "placeholder", "alt", "value"):
        current_value = getattr(scrubbed, field_name)
        if current_value:
            setattr(scrubbed, field_name, placeholder)
            changed_fields.add(field_name)

    if scrubbed.href:
        scrubbed.href = None
        changed_fields.add("href")

    scrubbed.unreadable = True
    scrubbed.scrub_reasons = list(dict.fromkeys([*scrubbed.scrub_reasons, *reasons]))
    scrubbed.scrubbed_fields = sorted(changed_fields)
    return scrubbed


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_word_like_text(value: str | None) -> bool:
    if not value:
        return False
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return False
    if normalized in {BLURRY_TEXT_PLACEHOLDER, TUNNEL_TEXT_PLACEHOLDER, SYMBOL_TEXT_PLACEHOLDER}:
        return False
    return bool(READABLE_CHAR_PATTERN.search(normalized))


def _is_symbol_glyph_text(value: str | None) -> bool:
    if not value:
        return False
    normalized = "".join(value.split())
    if not normalized:
        return False
    if any(char.isalnum() or _is_cjk(char) for char in normalized):
        return False
    return len(normalized) <= 4


def _mask_bounds(element: DOMElement, image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(int(round(element.x - padding)), 0)
    top = max(int(round(element.y - padding)), 0)
    right = min(int(round(element.x + element.width + padding)), width)
    bottom = min(int(round(element.y + element.height + padding)), height)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _symbol_mask_bounds(element: DOMElement, image_size: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    width, height = image_size
    if (
        element.icon_x is not None
        and element.icon_y is not None
        and element.icon_width is not None
        and element.icon_height is not None
        and element.icon_width > 0
        and element.icon_height > 0
    ):
        left = max(int(round(element.icon_x - padding)), 0)
        top = max(int(round(element.icon_y - padding)), 0)
        right = min(int(round(element.icon_x + element.icon_width + padding)), width)
        bottom = min(int(round(element.icon_y + element.icon_height + padding)), height)
        return left, top, max(left + 1, right), max(top + 1, bottom)

    inset_x = max(element.width * 0.2, float(padding))
    inset_y = max(element.height * 0.2, float(padding))
    left = max(int(round(element.x + inset_x)), 0)
    top = max(int(round(element.y + inset_y)), 0)
    right = min(int(round(element.x + element.width - inset_x)), width)
    bottom = min(int(round(element.y + element.height - inset_y)), height)
    if right <= left or bottom <= top:
        return _mask_bounds(element, image_size, max(1, padding // 2))
    return left, top, right, bottom


def _apply_blackout_mask(image: Image.Image, bounds: tuple[int, int, int, int]) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bounds, fill=(0, 0, 0, 255))


def _apply_mosaic_mask(image: Image.Image, bounds: tuple[int, int, int, int]) -> None:
    region = image.crop(bounds)
    if region.width <= 1 or region.height <= 1:
        _apply_blackout_mask(image, bounds)
        return
    downsample = max(2, min(region.width, region.height, 8))
    coarse = region.resize(
        (max(1, region.width // downsample), max(1, region.height // downsample)),
        resample=Image.Resampling.BILINEAR,
    )
    pixelated = coarse.resize(region.size, resample=Image.Resampling.NEAREST)
    image.paste(pixelated, bounds)


def _apply_local_blur(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    *,
    radius: float,
) -> None:
    region = image.crop(bounds)
    if region.width <= 1 or region.height <= 1:
        return
    image.paste(region.filter(ImageFilter.GaussianBlur(radius=radius)), bounds)


def _apply_low_vision_occlusion(
    image: Image.Image,
    bounds: tuple[int, int, int, int],
    *,
    blur_radius: float,
    overlay_alpha: int = 150,
) -> None:
    region = image.crop(bounds)
    if region.width <= 1 or region.height <= 1:
        return
    softened = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    veil = Image.new("RGBA", region.size, (248, 248, 244, overlay_alpha))
    image.paste(Image.alpha_composite(softened.convert("RGBA"), veil), bounds)


def _explicit_label_source(element: DOMElement) -> str | None:
    if _is_word_like_text(element.visible_text):
        return "visible_text"
    if _is_word_like_text(element.nearby_visible_text):
        return "nearby_visible_text"
    if _is_word_like_text(element.labelledby_text):
        return "aria_labelledby"
    return None


def _is_icon_only_candidate(element: DOMElement) -> bool:
    return (
        element.has_svg_child
        or element.has_img_child
        or element.has_icon_like_class
        or any(tag in {"svg", "img", "i", "use"} for tag in element.child_tags)
        or _is_symbol_glyph_text(element.visible_text or element.text)
    )


def _infer_symbol_reasons(element: DOMElement) -> list[str]:
    if _explicit_label_source(element):
        return []

    if not _is_icon_only_candidate(element):
        return []

    reasons = ["icon_only_symbol", "missing_visible_label"]
    if element.aria_label and not element.labelledby_text:
        reasons.append("aria_only_not_visible")
    return reasons


def _infer_blurry_reasons(element: DOMElement, profile: PresbyopiaProfile) -> list[str]:
    reasons = list(element.scrub_reasons)
    if element.font_size is not None and element.font_size < profile.min_readable_font_px:
        reasons.append("font_size_below_threshold")
    visible_text = (element.visible_text or element.text or "").strip()
    if (
        element.font_size is not None
        and element.font_size < profile.compact_font_px
        and element.height <= profile.compact_line_height_px
        and len(visible_text) >= profile.compact_min_chars
        and _is_word_like_text(visible_text)
    ):
        reasons.append("compact_text_hard_to_read")
    ratio = contrast_ratio(element.text_color, element.background_color)
    if ratio is not None and ratio < 4.5:
        reasons.append("contrast_below_threshold")
    return list(dict.fromkeys(reasons))


class BlurryVisionFilter(BaseSensoryFilter):
    name = "blurry_vision"

    def __init__(self, radius: float = 3.0, severity: str = "moderate"):
        self.radius = radius
        resolved_severity = severity if severity in PRESBYOPIA_PROFILES else "moderate"
        if resolved_severity == "auto":
            resolved_severity = "moderate"
        self.severity = resolved_severity
        self.profile = PRESBYOPIA_PROFILES[resolved_severity]

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        image = load_image_from_bytes(screenshot_bytes)
        global_radius = max(self.profile.global_blur_radius, min(self.radius * 0.5, self.profile.global_blur_radius))
        blurred = image.filter(ImageFilter.GaussianBlur(radius=global_radius))

        filtered_elements: list[DOMElement] = []
        reasons_counter: Counter[str] = Counter()
        removed_count = 0
        modified_count = 0

        for element in dom_elements_list:
            reasons = _infer_blurry_reasons(element, self.profile)
            if reasons:
                local_radius = max(self.radius * 1.8, self.profile.local_blur_radius)
                _apply_low_vision_occlusion(
                    blurred,
                    _mask_bounds(element, blurred.size, padding=2),
                    blur_radius=local_radius,
                    overlay_alpha=self.profile.overlay_alpha,
                )
                scrubbed = _scrub_unreadable_element(element, reasons)
                removed_count += 1
                modified_count += 1
                reasons_counter.update(reasons)
                continue
            filtered_elements.append(element.model_copy(deep=True))

        reasons_counter[f"severity_{self.severity}"] += 1
        self._record_effect(
            removed_count=removed_count,
            modified_count=modified_count,
            reasons=dict(reasons_counter),
        )
        return save_image_to_png_bytes(blurred), filtered_elements


class ColorblindnessFilter(BaseSensoryFilter):
    name = "colorblindness"
    PROTANOPIA_MATRIX = (
        0.56667, 0.43333, 0.0, 0.0,
        0.55833, 0.44167, 0.0, 0.0,
        0.0, 0.24167, 0.75833, 0.0,
    )

    def __init__(self, mode: str = "protanopia"):
        self.mode = mode

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        image = load_image_from_bytes(screenshot_bytes).convert("RGB")
        if self.mode == "achromatopsia":
            filtered_image = ImageOps.grayscale(image).convert("RGBA")
        else:
            filtered_image = image.convert("RGB", self.PROTANOPIA_MATRIX).convert("RGBA")
        self._record_effect()
        return save_image_to_png_bytes(filtered_image), [element.model_copy(deep=True) for element in dom_elements_list]


class TunnelVisionFilter(BaseSensoryFilter):
    name = "tunnel_vision"

    def __init__(
        self,
        visible_width_ratio: float = 0.5,
        visible_height_ratio: float = 0.5,
        darkness: int = 230,
        blur_radius: float = 36.0,
        min_visible_ratio: float = 0.55,
        safety_inset_ratio: float = 0.08,
    ):
        self.visible_width_ratio = visible_width_ratio
        self.visible_height_ratio = visible_height_ratio
        self.darkness = darkness
        self.blur_radius = blur_radius
        self.min_visible_ratio = min_visible_ratio
        self.safety_inset_ratio = safety_inset_ratio

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        image = load_image_from_bytes(screenshot_bytes)
        width, height = image.size
        focus_box, dom_focus_box = self._focus_geometry(width, height)

        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        alpha_mask = Image.new("L", image.size, color=self.darkness)
        draw = ImageDraw.Draw(alpha_mask)
        draw.ellipse(focus_box, fill=0)
        alpha_mask = alpha_mask.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        overlay.putalpha(alpha_mask)
        filtered_image = Image.alpha_composite(image, overlay)

        filtered_elements: list[DOMElement] = []
        removed_count = 0
        modified_count = 0
        reasons_counter: Counter[str] = Counter()
        for element in dom_elements_list:
            reasons = self._occlusion_reasons(element, dom_focus_box)
            if not reasons:
                filtered_elements.append(element.model_copy(deep=True))
            else:
                scrubbed = _scrub_occluded_element(
                    element,
                    placeholder=TUNNEL_TEXT_PLACEHOLDER,
                    reasons=reasons,
                )
                removed_count += 1
                modified_count += 1
                reasons_counter.update(reasons)
                logger.debug(
                    "TunnelVision scrubbed element id={} tag={} text={!r} aria={!r} href={!r} "
                    "bbox=({}, {}, {}, {}) reasons={}",
                    element.element_id,
                    element.tag,
                    element.text,
                    element.aria_label,
                    element.href,
                    round(element.x, 1),
                    round(element.y, 1),
                    round(element.width, 1),
                    round(element.height, 1),
                    reasons,
                )
                logger.trace("TunnelVision scrubbed placeholder fields: {}", scrubbed.scrubbed_fields)

        self._record_effect(
            removed_count=removed_count,
            modified_count=modified_count,
            reasons=dict(reasons_counter),
        )
        return save_image_to_png_bytes(filtered_image), filtered_elements

    def _focus_geometry(
        self,
        width: int,
        height: int,
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        focus_box = self._focus_box(width, height)
        inset = max(min(width, height) * self.safety_inset_ratio, self.blur_radius * 0.25)
        left, top, right, bottom = focus_box
        dom_focus_box = (
            min(max(left + inset, 0.0), right),
            min(max(top + inset, 0.0), bottom),
            max(min(right - inset, width), left),
            max(min(bottom - inset, height), top),
        )
        return focus_box, dom_focus_box

    def _focus_box(self, width: int, height: int) -> tuple[float, float, float, float]:
        focus_width = width * self.visible_width_ratio
        focus_height = height * self.visible_height_ratio
        left = (width - focus_width) / 2
        top = (height - focus_height) / 2
        return (left, top, left + focus_width, top + focus_height)

    def _occlusion_reasons(
        self,
        element: DOMElement,
        focus_box: tuple[float, float, float, float],
    ) -> list[str]:
        visibility_ratio = self._visible_ratio_in_focus(element, focus_box)
        if visibility_ratio <= 0.0:
            return ["outside_tunnel_focus"]
        if visibility_ratio < self.min_visible_ratio:
            return ["below_tunnel_visibility_threshold"]
        return []

    def _visible_ratio_in_focus(
        self,
        element: DOMElement,
        focus_box: tuple[float, float, float, float],
    ) -> float:
        left, top, right, bottom = focus_box
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        radius_x = max((right - left) / 2, 1e-6)
        radius_y = max((bottom - top) / 2, 1e-6)

        x_samples = self._sample_axis(element.x, element.width)
        y_samples = self._sample_axis(element.y, element.height)
        total = 0
        visible = 0
        for sample_x in x_samples:
            for sample_y in y_samples:
                total += 1
                normalized = (
                    ((sample_x - center_x) / radius_x) ** 2
                    + ((sample_y - center_y) / radius_y) ** 2
                )
                if normalized <= 1.0:
                    visible += 1
        return visible / total if total else 0.0

    def _sample_axis(self, start: float, length: float) -> list[float]:
        if length <= 0:
            return [start]
        fractions = (0.15, 0.5, 0.85)
        return [start + (length * fraction) for fraction in fractions]


class SymbolCognitionFilter(BaseSensoryFilter):
    name = "symbol_cognition"

    def __init__(
        self,
        mask_padding_px: int = 6,
        dom_strategy: str = "remove",
        mask_style: str = "blackout",
    ):
        self.mask_padding_px = mask_padding_px
        self.dom_strategy = dom_strategy
        self.mask_style = mask_style

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        image = load_image_from_bytes(screenshot_bytes)
        filtered_image = image.copy()
        filtered_elements: list[DOMElement] = []
        reasons_counter: Counter[str] = Counter()
        removed_count = 0
        modified_count = 0

        for element in dom_elements_list:
            allowed_label_source = _explicit_label_source(element)
            if allowed_label_source:
                logger.debug(
                    "SymbolCognition allowed element id={} tag={} via={} text={!r} nearby={!r} labelledby={!r}",
                    element.element_id,
                    element.tag,
                    allowed_label_source,
                    element.visible_text,
                    element.nearby_visible_text,
                    element.labelledby_text,
                )
                filtered_elements.append(element.model_copy(deep=True))
                continue

            reasons = _infer_symbol_reasons(element)
            if not reasons:
                filtered_elements.append(element.model_copy(deep=True))
                continue

            bounds = _symbol_mask_bounds(element, filtered_image.size, self.mask_padding_px)
            if self.mask_style == "mosaic":
                _apply_mosaic_mask(filtered_image, bounds)
            else:
                _apply_blackout_mask(filtered_image, bounds)

            scrubbed = _scrub_occluded_element(
                element,
                placeholder=SYMBOL_TEXT_PLACEHOLDER,
                reasons=reasons,
            )
            modified_count += 1
            reasons_counter.update(reasons)
            logger.debug(
                "SymbolCognition scrubbed element id={} tag={} text={!r} nearby={!r} labelledby={!r} "
                "aria={!r} classes={} child_tags={} reasons={}",
                element.element_id,
                element.tag,
                element.text,
                element.nearby_visible_text,
                element.labelledby_text,
                element.aria_label,
                element.css_classes,
                element.child_tags,
                reasons,
            )
            if self.dom_strategy == "placeholder_only":
                filtered_elements.append(scrubbed)
            else:
                removed_count += 1

        self._record_effect(
            removed_count=removed_count,
            modified_count=modified_count,
            reasons=dict(reasons_counter),
        )
        return save_image_to_png_bytes(filtered_image), filtered_elements


class LowPatienceFilter(BaseSensoryFilter):
    name = "low_patience"

    def __init__(self, max_text_length: int = 48, max_y_ratio: float = 0.68):
        self.max_text_length = max_text_length
        self.max_y_ratio = max_y_ratio

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        image = load_image_from_bytes(screenshot_bytes)
        _, height = image.size
        cutoff_y = height * self.max_y_ratio
        filtered_image = self._fade_lower_region(image, cutoff_y)

        filtered_elements: list[DOMElement] = []
        reasons_counter: Counter[str] = Counter()
        removed_count = 0
        for element in dom_elements_list:
            _, center_y = element.center
            if center_y > cutoff_y:
                removed_count += 1
                reasons_counter.update(["below_patience_window"])
                continue
            if element.text and len(element.text) > self.max_text_length:
                removed_count += 1
                reasons_counter.update(["verbose_text"])
                continue
            if element.placeholder and len(element.placeholder) > self.max_text_length:
                removed_count += 1
                reasons_counter.update(["verbose_placeholder"])
                continue
            if element.aria_label and len(element.aria_label) > self.max_text_length:
                removed_count += 1
                reasons_counter.update(["verbose_aria_label"])
                continue
            filtered_elements.append(element.model_copy(deep=True))

        self._record_effect(
            removed_count=removed_count,
            reasons=dict(reasons_counter),
        )
        return save_image_to_png_bytes(filtered_image), filtered_elements

    def _fade_lower_region(self, image: Image.Image, cutoff_y: float) -> Image.Image:
        width, height = image.size
        overlay = Image.new("RGBA", image.size, (12, 18, 30, 0))
        alpha = Image.new("L", image.size, color=0)
        draw = ImageDraw.Draw(alpha)
        draw.rectangle((0, cutoff_y, width, height), fill=180)
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=18))
        overlay.putalpha(alpha)
        return Image.alpha_composite(image, overlay)


class CognitiveFilter(BaseSensoryFilter):
    name = "cognitive"

    def __init__(self, config: SensoryConfig | None = None):
        config = config or SensoryConfig()
        self.placeholder = config.cognitive_placeholder
        self.visual_scrub = config.cognitive_visual_scrub
        self.visual_scrub_strength = config.cognitive_visual_scrub_strength
        self.phrase_density_threshold = config.cognitive_phrase_density_threshold
        self.visual_max_text_length = config.cognitive_visual_max_text_length
        self.compiled_terms = build_compiled_jargon_terms(
            config.cognitive_enabled_domains,
            config.cognitive_custom_terms,
            case_sensitive=config.cognitive_case_sensitive,
        )

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        if not self.compiled_terms:
            self._record_effect()
            return screenshot_bytes, [element.model_copy(deep=True) for element in dom_elements_list]

        filtered_image = load_image_from_bytes(screenshot_bytes)
        filtered_elements: list[DOMElement] = []
        reasons_counter: Counter[str] = Counter()
        modified_count = 0
        visual_changed = False

        for element in dom_elements_list:
            scrubbed = element.model_copy(deep=True)
            changed_fields = set(scrubbed.scrubbed_fields)
            element_reasons: list[str] = []
            matched_terms: list[str] = []

            for field_name in SEMANTIC_TEXT_FIELDS:
                current_value = getattr(scrubbed, field_name, None)
                replaced, match_count, field_matches, dense = replace_jargon_terms(
                    current_value,
                    self.compiled_terms,
                    placeholder=self.placeholder,
                    density_threshold=self.phrase_density_threshold,
                )
                if match_count <= 0:
                    continue
                setattr(scrubbed, field_name, replaced)
                changed_fields.add(field_name)
                matched_terms.extend(field_matches)
                element_reasons.append("jargon_dense_phrase_scrubbed" if dense else "jargon_term_scrubbed")
                if field_name == "alt":
                    element_reasons.append("jargon_alt_scrubbed")

            replaced_href, href_matches, href_terms, _ = replace_jargon_terms(
                scrubbed.href,
                self.compiled_terms,
                placeholder="redacted-jargon-link",
                density_threshold=self.phrase_density_threshold,
            )
            if href_matches > 0:
                scrubbed.href = replaced_href
                changed_fields.add("href")
                matched_terms.extend(href_terms)
                element_reasons.append("jargon_href_scrubbed")

            if changed_fields:
                modified_count += 1
                scrubbed.unreadable = True
                scrubbed.scrubbed_fields = sorted(changed_fields)
                scrubbed.scrubbed_terms = sorted(set(matched_terms))
                scrubbed.scrub_reasons = list(dict.fromkeys([*scrubbed.scrub_reasons, *element_reasons]))
                reasons_counter.update(element_reasons)
                if self._should_visually_scrub(element):
                    _apply_local_blur(filtered_image, _mask_bounds(element, filtered_image.size, 3), radius=4.0)
                    reasons_counter.update(["visual_jargon_blur_applied"])
                    visual_changed = True
                logger.debug(
                    "CognitiveFilter scrubbed element id={} tag={} fields={} terms={}",
                    element.element_id,
                    element.tag,
                    scrubbed.scrubbed_fields,
                    sorted(set(matched_terms)),
                )
            filtered_elements.append(scrubbed)

        self._record_effect(
            removed_count=0,
            modified_count=modified_count,
            reasons=dict(reasons_counter),
        )
        if visual_changed:
            return save_image_to_png_bytes(filtered_image), filtered_elements
        return screenshot_bytes, filtered_elements

    def _should_visually_scrub(self, element: DOMElement) -> bool:
        if not self.visual_scrub or self.visual_scrub_strength == "off":
            return False
        candidate_lengths = [
            len(value)
            for value in (element.text, element.visible_text, element.alt, element.aria_label, element.title)
            if value
        ]
        if not candidate_lengths:
            return False
        if max(candidate_lengths) > self.visual_max_text_length:
            return False
        if element.width * element.height > 60_000 and (element.font_size or 0) <= 16:
            return False
        return True
