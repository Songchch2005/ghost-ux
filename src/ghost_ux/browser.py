from __future__ import annotations

import asyncio
import base64
import random
from pathlib import Path

from loguru import logger
from playwright.async_api import Browser, BrowserContext, Error, Page, Playwright, TimeoutError, async_playwright

from ghost_ux.actions import build_action_pipeline
from ghost_ux.config import AgentConfig, BrowserConfig, MotorConfig, SensoryConfig
from ghost_ux.models import ActionResult, ActionType, DOMElement, ExecutableAction, Observation, UIAction


MIN_READABLE_FONT_SIZE_PX = 12.5
BLURRY_TEXT_PLACEHOLDER = "[blurred fine print]"

def normalize_font_size_to_px(
    value: str | None,
    *,
    root_font_size_px: float = 16.0,
    parent_font_size_px: float | None = None,
) -> float | None:
    if not value:
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    try:
        if raw.endswith("px"):
            return float(raw[:-2].strip())
        if raw.endswith("rem"):
            return float(raw[:-3].strip()) * root_font_size_px
        if raw.endswith("em"):
            base = parent_font_size_px if parent_font_size_px is not None else root_font_size_px
            return float(raw[:-2].strip()) * base
        if raw.endswith("%"):
            base = parent_font_size_px if parent_font_size_px is not None else root_font_size_px
            return (float(raw[:-1].strip()) / 100.0) * base
        return float(raw)
    except ValueError:
        return None


STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32' });
window.chrome = window.chrome || { runtime: {} };
Object.defineProperty(navigator, 'plugins', {
  get: () => [{ name: 'Chrome PDF Plugin' }, { name: 'Chrome PDF Viewer' }]
});
"""


COLLECT_INTERACTIVE_ELEMENTS_SCRIPT = """
({ maxElements, maxStaticElements, nearbyTextRadiusPx }) => {
  const interactiveSelectors = [
    'a[href]',
    'button',
    'input',
    'textarea',
    'select',
    '[role="button"]',
    '[role="link"]',
    '[role="textbox"]',
    '[contenteditable="true"]',
    '[tabindex]'
  ];
  const staticTextSelectors = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'blockquote'];
  const interactiveSelectorString = interactiveSelectors.join(', ');
  const maxStaticTextLength = 240;
  const minStaticTextLength = 20;

  const textPlaceholder = "__TEXT_PLACEHOLDER__";
  const minReadableFontSizePx = __MIN_FONT__;
  const iconClassPattern = /(fa[srlbd]?|icon|icons|lucide|heroicon|material-icons|mdi|glyph|hamburger|menu|search|magnifier|navicon)/i;
  const readableCharPattern = /[\u4e00-\u9fa5a-zA-Z0-9]/;

  const clampText = (value, maxLength = 180) => {
    if (!value) return null;
    return value.replace(/\\s+/g, ' ').trim().slice(0, maxLength) || null;
  };

  const hasReadableChars = (value) => {
    if (!value) return false;
    return readableCharPattern.test(clampText(value) || '');
  };

  const isInteractiveElement = (el) => {
    if (!(el instanceof Element)) return false;
    try {
      return el.matches(interactiveSelectorString);
    } catch (error) {
      return false;
    }
  };

  const parseColor = (value) => {
    if (!value) return null;
    const normalized = value.trim().toLowerCase();
    if (normalized.startsWith('rgb(') || normalized.startsWith('rgba(')) {
      const numbers = normalized.slice(normalized.indexOf('(') + 1, normalized.lastIndexOf(')')).split(',');
      if (numbers.length < 3) return null;
      const rgb = numbers.slice(0, 3).map((part) => Number.parseFloat(part.trim()));
      if (rgb.some((channel) => Number.isNaN(channel))) return null;
      return rgb.map((channel) => channel / 255);
    }
    if (normalized.startsWith('#') && (normalized.length === 4 || normalized.length === 7)) {
      const expanded = normalized.length === 4
        ? `#${normalized[1]}${normalized[1]}${normalized[2]}${normalized[2]}${normalized[3]}${normalized[3]}`
        : normalized;
      return [1, 3, 5].map((index) => Number.parseInt(expanded.slice(index, index + 2), 16) / 255);
    }
    return null;
  };

  const relativeLuminance = (rgb) => {
    if (!rgb) return null;
    const transform = (channel) => (
      channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4
    );
    const [red, green, blue] = rgb.map(transform);
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
  };

  const contrastRatio = (textColor, backgroundColor) => {
    const foreground = parseColor(textColor);
    const background = parseColor(backgroundColor);
    if (!foreground || !background) return null;
    const lighter = Math.max(relativeLuminance(foreground), relativeLuminance(background));
    const darker = Math.min(relativeLuminance(foreground), relativeLuminance(background));
    return (lighter + 0.05) / (darker + 0.05);
  };

  const isVisible = (el) => {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    if (rect.width < 8 || rect.height < 8) return false;
    return (
      rect.bottom >= 0 &&
      rect.right >= 0 &&
      rect.top <= window.innerHeight &&
      rect.left <= window.innerWidth
    );
  };

  const resolveBackgroundColor = (el) => {
    let node = el;
    while (node && node !== document.documentElement) {
      const color = window.getComputedStyle(node).backgroundColor;
      if (color && color !== 'rgba(0, 0, 0, 0)' && color !== 'transparent') {
        return color;
      }
      node = node.parentElement;
    }
    const bodyColor = window.getComputedStyle(document.body).backgroundColor;
    return bodyColor || 'rgb(255, 255, 255)';
  };

  const normalizeFontSizePx = (fontSize, el) => {
    if (!fontSize) return null;
    const raw = String(fontSize).trim().toLowerCase();
    if (!raw) return null;
    const rootFontSize = Number.parseFloat(window.getComputedStyle(document.documentElement).fontSize || '16') || 16;
    const parentFontSize = el && el.parentElement
      ? Number.parseFloat(window.getComputedStyle(el.parentElement).fontSize || `${rootFontSize}`) || rootFontSize
      : rootFontSize;
    const parsed = Number.parseFloat(raw);
    if (Number.isNaN(parsed)) return null;
    if (raw.endsWith('px')) return parsed;
    if (raw.endsWith('rem')) return parsed * rootFontSize;
    if (raw.endsWith('em')) return parsed * parentFontSize;
    if (raw.endsWith('%')) return (parsed / 100) * parentFontSize;
    return parsed;
  };

  const describeReadability = (el) => {
    const style = window.getComputedStyle(el);
    const fontSizePx = normalizeFontSizePx(style.fontSize, el);
    const textColor = clampText(style.color);
    const backgroundColor = clampText(resolveBackgroundColor(el));
    const reasons = [];
    if (fontSizePx !== null && fontSizePx < minReadableFontSizePx) {
      reasons.push('font_size_below_threshold');
    }
    const ratio = contrastRatio(textColor, backgroundColor);
    if (ratio !== null && ratio < 4.5) {
      reasons.push('contrast_below_threshold');
    }
    return {
      fontSizePx,
      textColor,
      backgroundColor,
      contrast: ratio,
      unreadable: reasons.length > 0,
      reasons,
    };
  };

  const tokenize = (value) => {
    if (!value) return [];
    return Array.from(
      new Set(
        value
          .toLowerCase()
          .replace(/[^a-z0-9-]+/g, ' ')
          .split(/\\s+/)
          .filter((token) => token.length >= 4)
      )
    );
  };

  const collectClassTokens = (el) => {
    const tokens = new Set();
    if (el.classList) {
      for (const className of el.classList) {
        const normalized = clampText(className);
        if (normalized) tokens.add(normalized);
      }
    }
    for (const node of el.querySelectorAll('[class]')) {
      if (!(node instanceof Element)) continue;
      for (const className of node.classList) {
        const normalized = clampText(className);
        if (normalized) tokens.add(normalized);
        if (tokens.size >= 24) break;
      }
      if (tokens.size >= 24) break;
    }
    return Array.from(tokens).slice(0, 24);
  };

  const collectChildTags = (el) => {
    const tags = new Set();
    for (const node of el.querySelectorAll('*')) {
      if (!(node instanceof Element)) continue;
      tags.add(node.tagName.toLowerCase());
      if (tags.size >= 12) break;
    }
    return Array.from(tags);
  };

  const collectSanitizedText = (root, rootReadability) => {
    const rawText = clampText(root.innerText || root.textContent);
    if (rootReadability.unreadable) {
      return {
        text: null,
        descendantTextRemoved: false,
        hiddenTokens: tokenize(rawText || ''),
      };
    }
    const pieces = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const textValue = clampText(node.textContent);
      if (!textValue) continue;
      const parent = node.parentElement;
      if (!parent || !isVisible(parent)) continue;
      if (parent !== root) {
        const parentReadability = describeReadability(parent);
        if (parentReadability.unreadable) continue;
      }
      pieces.push(textValue);
    }
    const sanitizedText = clampText(pieces.join(' '));
    const rawTokens = tokenize(rawText || '');
    const sanitizedTokens = tokenize(sanitizedText || '');
    return {
      text: sanitizedText,
      descendantTextRemoved: rawText !== sanitizedText,
      hiddenTokens: rawTokens.filter((token) => !sanitizedTokens.includes(token)),
    };
  };

  const collectReadableElementText = (el) => {
    if (!el || !isVisible(el)) return null;
    const readability = describeReadability(el);
    const text = clampText(collectSanitizedText(el, readability).text);
    return hasReadableChars(text) ? text : null;
  };

  const isMeaningfulStaticElement = (el) => {
    if (!(el instanceof Element) || !isVisible(el) || isInteractiveElement(el)) return false;
    const tag = el.tagName.toLowerCase();
    if (['script', 'style', 'noscript'].includes(tag)) return false;
    if (el.closest('button, a[href], label, input, textarea, select, [role="button"], [role="link"], [role="textbox"]')) {
      return false;
    }

    const readability = describeReadability(el);
    const sanitized = collectSanitizedText(el, readability);
    const text = clampText(sanitized.text, maxStaticTextLength);
    if (!hasReadableChars(text)) return false;

    const isNamedStaticTag = staticTextSelectors.includes(tag);
    if (!isNamedStaticTag && (text?.length || 0) < minStaticTextLength) {
      return false;
    }

    if (!isNamedStaticTag) {
      const hasMeaningfulChildBlock = Array.from(el.children).some((child) => {
        if (!(child instanceof Element) || !isVisible(child) || isInteractiveElement(child)) return false;
        const childTag = child.tagName.toLowerCase();
        if (staticTextSelectors.includes(childTag)) return true;
        const childText = clampText(child.innerText || child.textContent, maxStaticTextLength);
        return Boolean(childText && childText.length >= minStaticTextLength);
      });
      if (hasMeaningfulChildBlock) return false;
    }

    return true;
  };

  const resolveLabelledByText = (el) => {
    const labelledBy = clampText(el.getAttribute('aria-labelledby'));
    if (!labelledBy) return null;
    const pieces = [];
    for (const rawId of labelledBy.split(/\\s+/)) {
      const id = rawId.trim();
      if (!id) continue;
      const target = document.getElementById(id);
      const text = collectReadableElementText(target);
      if (text) pieces.push(text);
    }
    return clampText(pieces.join(' '));
  };

  const resolveExplicitLabelForText = (el) => {
    if (!(el instanceof Element) || !el.id) return null;
    const pieces = [];
    for (const label of document.querySelectorAll(`label[for="${el.id}"]`)) {
      const text = collectReadableElementText(label);
      if (text) pieces.push(text);
    }
    const resolved = clampText(Array.from(new Set(pieces)).join(' '));
    return hasReadableChars(resolved) ? resolved : null;
  };

  const isNearRect = (sourceRect, candidateRect, radius) => {
    const horizontalGap = Math.max(0, Math.max(sourceRect.left - candidateRect.right, candidateRect.left - sourceRect.right));
    const verticalGap = Math.max(0, Math.max(sourceRect.top - candidateRect.bottom, candidateRect.top - sourceRect.bottom));
    const verticalOverlap = Math.min(sourceRect.bottom, candidateRect.bottom) - Math.max(sourceRect.top, candidateRect.top);
    const horizontalOverlap = Math.min(sourceRect.right, candidateRect.right) - Math.max(sourceRect.left, candidateRect.left);
    const alignedRow = horizontalGap <= radius && verticalOverlap >= -6;
    const alignedColumn = verticalGap <= radius && horizontalOverlap >= -6;
    return alignedRow || alignedColumn;
  };

  const isInlineSiblingLabel = (sourceRect, candidateRect, radius) => {
    const horizontalGap = Math.max(0, Math.max(sourceRect.left - candidateRect.right, candidateRect.left - sourceRect.right));
    const verticalOverlap = Math.min(sourceRect.bottom, candidateRect.bottom) - Math.max(sourceRect.top, candidateRect.top);
    const minExpectedOverlap = Math.min(sourceRect.height, candidateRect.height) * 0.35;
    return horizontalGap <= radius && verticalOverlap >= minExpectedOverlap;
  };

  const collectNearbyVisibleText = (el, radius) => {
    const sourceRect = el.getBoundingClientRect();
    const pieces = [];

    const maybeAddImmediateSibling = (node) => {
      if (!node || !(node instanceof Element) || node === el) return;
      if (!isVisible(node)) return;
      if (!isInlineSiblingLabel(sourceRect, node.getBoundingClientRect(), radius)) return;
      const text = collectReadableElementText(node);
      if (text && hasReadableChars(text)) {
        pieces.push(text);
      }
    };

    maybeAddImmediateSibling(el.previousElementSibling);
    maybeAddImmediateSibling(el.nextElementSibling);

    const nearbyText = clampText(Array.from(new Set(pieces)).join(' '));
    return hasReadableChars(nearbyText) ? nearbyText : null;
  };

  const firstVisibleIconRect = (el) => {
    const candidates = [];
    for (const node of el.querySelectorAll('svg, img, i, use, [class*="icon"], [class*="Icon"], [class*="fa-"], [class*="mdi"]')) {
      if (!(node instanceof Element) || !isVisible(node)) continue;
      const rect = node.getBoundingClientRect();
      if (rect.width < 4 || rect.height < 4) continue;
      candidates.push(rect);
    }
    if (!candidates.length) return null;
    candidates.sort((left, right) => (left.width * left.height) - (right.width * right.height));
    const rect = candidates[0];
    return {
      x: rect.x,
      y: rect.y,
      width: rect.width,
      height: rect.height,
    };
  };

  const buildIconSignals = (el, sanitizedVisibleText) => {
    const cssClasses = collectClassTokens(el);
    const childTags = collectChildTags(el);
    const hasSvgChild = Boolean(el.querySelector('svg, use'));
    const hasImgChild = Boolean(el.querySelector('img'));
    const hasIconLikeClass = cssClasses.some((className) => iconClassPattern.test(className));
    const titleText = clampText(el.getAttribute('title'));
    const labelledbyText = resolveLabelledByText(el);
    const nearbyVisibleText = clampText(
      [
        collectNearbyVisibleText(el, nearbyTextRadiusPx),
        resolveExplicitLabelForText(el),
      ].filter(Boolean).join(' ')
    );
    const iconRect = firstVisibleIconRect(el);
    const visibleText = clampText(sanitizedVisibleText || el.innerText || el.textContent);
    return {
      cssClasses,
      childTags,
      hasSvgChild,
      hasImgChild,
      hasIconLikeClass,
      nearbyVisibleText,
      labelledbyText,
      titleText,
      visibleText,
      iconRect,
    };
  };

  const scrubSemanticField = (fieldValue, hiddenTokens, modifiedFields, reasonSet, fieldName) => {
    if (!fieldValue) return fieldValue;
    const lowered = fieldValue.toLowerCase();
    if (!hiddenTokens.some((token) => lowered.includes(token))) {
      return fieldValue;
    }
    modifiedFields.push(fieldName);
    reasonSet.add('semantic_field_scrubbed');
    return fieldName === 'href' ? null : textPlaceholder;
  };

  const buildRawSnapshot = (el, iconSignals) => ({
    text: clampText(el.innerText || el.textContent),
    visible_text: clampText(el.innerText || el.textContent),
    nearby_visible_text: iconSignals.nearbyVisibleText,
    labelledby_text: iconSignals.labelledbyText,
    title_text: iconSignals.titleText,
    placeholder: clampText(el.getAttribute('placeholder')),
    alt: clampText(el.getAttribute('alt')),
    aria_label: clampText(el.getAttribute('aria-label')),
    title: clampText(el.getAttribute('title')),
    name: clampText(el.getAttribute('name')),
    href: clampText(el.getAttribute('href')),
    value: clampText(el.value),
  });

  const sanitizeSnapshotFields = (el, readability) => {
    const modifiedFields = [];
    const reasonSet = new Set(readability.reasons);
    const sanitizedText = collectSanitizedText(el, readability);
    const iconSignals = buildIconSignals(el, sanitizedText.text);
    const snapshot = buildRawSnapshot(el, iconSignals);
    snapshot.text = sanitizedText.text;
    snapshot.visible_text = iconSignals.visibleText;

    if (sanitizedText.descendantTextRemoved) {
      modifiedFields.push('text');
      reasonSet.add('descendant_text_scrubbed');
      snapshot.href = null;
      modifiedFields.push('href');
      snapshot.aria_label = scrubSemanticField(snapshot.aria_label, sanitizedText.hiddenTokens, modifiedFields, reasonSet, 'aria_label');
      snapshot.title = scrubSemanticField(snapshot.title, sanitizedText.hiddenTokens, modifiedFields, reasonSet, 'title');
      snapshot.name = scrubSemanticField(snapshot.name, sanitizedText.hiddenTokens, modifiedFields, reasonSet, 'name');
      snapshot.placeholder = scrubSemanticField(snapshot.placeholder, sanitizedText.hiddenTokens, modifiedFields, reasonSet, 'placeholder');
      snapshot.alt = scrubSemanticField(snapshot.alt, sanitizedText.hiddenTokens, modifiedFields, reasonSet, 'alt');
      snapshot.value = scrubSemanticField(snapshot.value, sanitizedText.hiddenTokens, modifiedFields, reasonSet, 'value');
    }

    if (readability.unreadable) {
      for (const key of ['text', 'placeholder', 'alt', 'aria_label', 'title', 'name', 'value']) {
        if (snapshot[key]) {
          snapshot[key] = textPlaceholder;
          modifiedFields.push(key);
        }
      }
      if (snapshot.href) {
        snapshot.href = null;
        modifiedFields.push('href');
      }
    }

    return {
      ...snapshot,
      css_classes: iconSignals.cssClasses,
      child_tags: iconSignals.childTags,
      has_svg_child: iconSignals.hasSvgChild,
      has_img_child: iconSignals.hasImgChild,
      has_icon_like_class: iconSignals.hasIconLikeClass,
      icon_x: iconSignals.iconRect ? iconSignals.iconRect.x : null,
      icon_y: iconSignals.iconRect ? iconSignals.iconRect.y : null,
      icon_width: iconSignals.iconRect ? iconSignals.iconRect.width : null,
      icon_height: iconSignals.iconRect ? iconSignals.iconRect.height : null,
      unreadable: readability.unreadable,
      scrub_reasons: Array.from(reasonSet),
      scrubbed_fields: Array.from(new Set(modifiedFields)),
    };
  };

  const seen = new Set();
  const nodes = [];
  const rawNodes = [];
  let interactiveCount = 0;
  let staticCount = 0;

  const pushSnapshot = (el, isInteractive) => {
      const rect = el.getBoundingClientRect();
      const readability = describeReadability(el);
      const iconSignals = buildIconSignals(el, null);
      const rawSnapshot = buildRawSnapshot(el, iconSignals);
      const sanitized = sanitizeSnapshotFields(el, readability);
      const id = isInteractive ? String(interactiveCount + 1) : `static-${staticCount + 1}`;
      if (isInteractive) {
        interactiveCount += 1;
        el.setAttribute('data-ghost-ux-id', id);
      } else {
        staticCount += 1;
      }
      const baseFields = {
        element_id: id,
        is_interactive: isInteractive,
        tag: el.tagName.toLowerCase(),
        role: el.getAttribute('role'),
        input_type: clampText(el.getAttribute('type')),
        font_size: readability.fontSizePx,
        text_color: readability.textColor,
        background_color: readability.backgroundColor,
        disabled: isInteractive ? el.disabled === true || el.getAttribute('aria-disabled') === 'true' : false,
        css_classes: iconSignals.cssClasses,
        child_tags: iconSignals.childTags,
        has_svg_child: iconSignals.hasSvgChild,
        has_img_child: iconSignals.hasImgChild,
        has_icon_like_class: iconSignals.hasIconLikeClass,
        icon_x: iconSignals.iconRect ? iconSignals.iconRect.x : null,
        icon_y: iconSignals.iconRect ? iconSignals.iconRect.y : null,
        icon_width: iconSignals.iconRect ? iconSignals.iconRect.width : null,
        icon_height: iconSignals.iconRect ? iconSignals.iconRect.height : null,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
      };

      rawNodes.push({
        ...baseFields,
        text: rawSnapshot.text,
        visible_text: rawSnapshot.visible_text,
        nearby_visible_text: rawSnapshot.nearby_visible_text,
        labelledby_text: rawSnapshot.labelledby_text,
        title_text: rawSnapshot.title_text,
        placeholder: rawSnapshot.placeholder,
        alt: rawSnapshot.alt,
        aria_label: rawSnapshot.aria_label,
        title: rawSnapshot.title,
        name: rawSnapshot.name,
        href: rawSnapshot.href,
        value: rawSnapshot.value,
        unreadable: readability.unreadable,
        scrub_reasons: Array.from(readability.reasons),
        scrubbed_fields: [],
      });

      nodes.push({
        ...baseFields,
        text: sanitized.text,
        visible_text: sanitized.visible_text,
        nearby_visible_text: sanitized.nearby_visible_text,
        labelledby_text: sanitized.labelledby_text,
        title_text: sanitized.title_text,
        placeholder: sanitized.placeholder,
        alt: sanitized.alt,
        aria_label: sanitized.aria_label,
        title: sanitized.title,
        name: sanitized.name,
        href: sanitized.href,
        value: sanitized.value,
        unreadable: sanitized.unreadable,
        scrub_reasons: sanitized.scrub_reasons,
        scrubbed_fields: sanitized.scrubbed_fields,
      });
  };

  for (const selector of interactiveSelectors) {
    for (const el of document.querySelectorAll(selector)) {
      if (interactiveCount >= maxElements) break;
      if (seen.has(el)) continue;
      seen.add(el);
      if (!isVisible(el)) continue;
      pushSnapshot(el, true);
    }
  }

  for (const selector of staticTextSelectors) {
    for (const el of document.querySelectorAll(selector)) {
      if (staticCount >= maxStaticElements) break;
      if (seen.has(el)) continue;
      if (!isMeaningfulStaticElement(el)) continue;
      seen.add(el);
      pushSnapshot(el, false);
    }
  }

  if (staticCount < maxStaticElements) {
    for (const el of document.querySelectorAll('div, section, article, span')) {
      if (staticCount >= maxStaticElements) break;
      if (seen.has(el)) continue;
      if (!isMeaningfulStaticElement(el)) continue;
      seen.add(el);
      pushSnapshot(el, false);
    }
  }

  return {
    raw_elements: rawNodes,
    elements: nodes,
    viewport_width: window.innerWidth,
    viewport_height: window.innerHeight
  };
}
""".replace("__TEXT_PLACEHOLDER__", BLURRY_TEXT_PLACEHOLDER).replace("__MIN_FONT__", str(MIN_READABLE_FONT_SIZE_PX))


class BrowserController:
    def __init__(
        self,
        config: BrowserConfig,
        agent_config: AgentConfig,
        sensory_config: SensoryConfig,
        motor_config: MotorConfig,
        session_dir: Path,
    ):
        self.config = config
        self.agent_config = agent_config
        self.sensory_config = sensory_config
        self.motor_config = motor_config
        self.session_dir = session_dir
        self.action_pipeline = build_action_pipeline(agent_config.persona, motor_config)
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    async def start(self) -> None:
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.config.headless)
        self.context = await self.browser.new_context(
            viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
            user_agent=self.config.user_agent,
            locale=self.config.locale,
            timezone_id=self.config.timezone_id,
            ignore_https_errors=True,
        )
        await self.context.add_init_script(STEALTH_INIT_SCRIPT)
        self.page = await self.context.new_page()
        self.page.set_default_timeout(self.config.action_timeout_ms)
        self.page.set_default_navigation_timeout(self.config.navigation_timeout_ms)

    async def goto(self, url: str) -> None:
        assert self.page is not None
        logger.info("Navigating to {}", url)
        await self.page.goto(url, wait_until="domcontentloaded")
        try:
            await self.page.wait_for_load_state("networkidle", timeout=self.config.navigation_timeout_ms)
        except TimeoutError:
            logger.warning("networkidle timeout reached for {}; continuing with current DOM.", url)
            await self._human_delay()

    async def observe(self, step_index: int, last_error: str | None = None) -> Observation:
        assert self.page is not None
        screenshot_path = self.session_dir / f"step_{step_index:02d}.png"
        await self.page.screenshot(
            path=str(screenshot_path),
            type="png",
            full_page=self.config.screenshot_full_page,
            animations="disabled",
        )
        raw = screenshot_path.read_bytes()
        snapshot = await self.page.evaluate(
            COLLECT_INTERACTIVE_ELEMENTS_SCRIPT,
            {
                "maxElements": self.config.max_dom_elements,
                "maxStaticElements": max(self.config.max_dom_elements, 24),
                "nearbyTextRadiusPx": self.sensory_config.symbol_nearby_text_radius_px,
            },
        )
        raw_elements = [DOMElement.model_validate(item) for item in snapshot.get("raw_elements", [])]
        filtered_elements = [DOMElement.model_validate(item) for item in snapshot["elements"]]
        return Observation(
            step_index=step_index,
            url=self.page.url,
            title=await self.page.title(),
            screenshot_bytes=raw,
            screenshot_base64=base64.b64encode(raw).decode("utf-8"),
            screenshot_path=screenshot_path,
            elements=[element.model_copy(deep=True) for element in filtered_elements],
            raw_elements=raw_elements,
            filtered_elements=filtered_elements,
            last_error=last_error,
            viewport_width=snapshot["viewport_width"],
            viewport_height=snapshot["viewport_height"],
        )

    async def perform(self, action: UIAction, observation: Observation | None = None) -> ActionResult:
        assert self.page is not None
        if action.action_type == ActionType.CLICK:
            return await self._click(action, observation)
        if action.action_type == ActionType.TYPE:
            return await self._type(action.target_element_id or "", action.input_text or "")
        if action.action_type == ActionType.SCROLL_DOWN:
            plan = await self._resolve_action_plan(action, observation)
            plan = self.action_pipeline.apply(action, observation or self._empty_observation(), plan)
            await self.page.mouse.wheel(plan.scroll_delta_x, plan.scroll_delta_y)
            await self._human_delay()
            return ActionResult(
                success=True,
                detail=f"Scrolled down by {int(plan.scroll_delta_y)}px.",
                noise_applied=plan.noise_applied,
                noise_profile=plan.noise_profile,
                offset=(plan.offset_x, plan.offset_y),
                scroll_delta=(plan.scroll_delta_x, plan.scroll_delta_y),
            )
        if action.action_type == ActionType.SCROLL_UP:
            plan = await self._resolve_action_plan(action, observation)
            plan = self.action_pipeline.apply(action, observation or self._empty_observation(), plan)
            await self.page.mouse.wheel(plan.scroll_delta_x, plan.scroll_delta_y)
            await self._human_delay()
            return ActionResult(
                success=True,
                detail=f"Scrolled up by {int(abs(plan.scroll_delta_y))}px.",
                noise_applied=plan.noise_applied,
                noise_profile=plan.noise_profile,
                offset=(plan.offset_x, plan.offset_y),
                scroll_delta=(plan.scroll_delta_x, plan.scroll_delta_y),
            )
        if action.action_type == ActionType.FINISH:
            return ActionResult(success=True, detail="Model marked task as finished.")
        return ActionResult(success=False, detail="Model declared failure.")

    async def close(self) -> None:
        if self.context is not None:
            await self.context.close()
        if self.browser is not None:
            await self.browser.close()
        if self.playwright is not None:
            await self.playwright.stop()

    async def _click(self, action: UIAction, observation: Observation | None) -> ActionResult:
        element_id = action.target_element_id or ""
        selector = f'[data-ghost-ux-id="{element_id}"]'
        assert self.page is not None
        last_error = ""
        plan = await self._resolve_action_plan(action, observation)
        plan = self.action_pipeline.apply(action, observation or self._empty_observation(), plan)
        for attempt in range(1, self.agent_config.retry_limit_per_action + 1):
            try:
                locator = self.page.locator(selector).first
                await locator.wait_for(state="visible", timeout=self.config.action_timeout_ms)
                await locator.scroll_into_view_if_needed(timeout=self.config.action_timeout_ms)
                await self._human_delay()
                if plan.actual_x is not None and plan.actual_y is not None:
                    await self.page.mouse.move(plan.actual_x, plan.actual_y)
                    plan.pre_hit_summary = await self._element_summary_at_point(plan.actual_x, plan.actual_y)
                    await self._human_delay()
                    await self.page.mouse.click(plan.actual_x, plan.actual_y)
                    plan.post_hit_summary = await self._element_summary_at_point(plan.actual_x, plan.actual_y)
                else:
                    await locator.hover(timeout=self.config.action_timeout_ms)
                    await self._human_delay()
                    await locator.click(timeout=self.config.action_timeout_ms)
                await self._settle_after_action()
                await self._human_delay()
                misfire = bool(
                    plan.pre_hit_summary
                    and element_id not in plan.pre_hit_summary
                    and selector not in plan.pre_hit_summary
                )
                return ActionResult(
                    success=True,
                    detail=f"Clicked {selector} on attempt {attempt}.",
                    executed_selector=selector,
                    noise_applied=plan.noise_applied,
                    noise_profile=plan.noise_profile,
                    intended_point=self._point_tuple(plan.intended_x, plan.intended_y),
                    actual_point=self._point_tuple(plan.actual_x, plan.actual_y),
                    offset=(plan.offset_x, plan.offset_y),
                    actual_hit_summary=plan.pre_hit_summary or plan.post_hit_summary,
                    misfire=misfire,
                )
            except (TimeoutError, Error) as exc:
                last_error = str(exc)
                logger.warning("Click attempt {} failed for {}: {}", attempt, selector, exc)
                if not plan.disable_precise_click_fallback:
                    try:
                        await self.page.locator(selector).evaluate("(el) => el.click()")
                        await self._settle_after_action()
                        await self._human_delay()
                        return ActionResult(
                            success=True,
                            detail=f"Fallback DOM click succeeded for {selector}.",
                            executed_selector=selector,
                            noise_applied=plan.noise_applied,
                            noise_profile=plan.noise_profile,
                            intended_point=self._point_tuple(plan.intended_x, plan.intended_y),
                            actual_point=self._point_tuple(plan.actual_x, plan.actual_y),
                            offset=(plan.offset_x, plan.offset_y),
                            actual_hit_summary=plan.pre_hit_summary,
                            misfire=False,
                        )
                    except (TimeoutError, Error):
                        await self._human_delay()
                else:
                    await self._human_delay()
        return ActionResult(
            success=False,
            detail=f"Click failed for {selector}: {last_error}",
            executed_selector=selector,
            noise_applied=plan.noise_applied,
            noise_profile=plan.noise_profile,
            intended_point=self._point_tuple(plan.intended_x, plan.intended_y),
            actual_point=self._point_tuple(plan.actual_x, plan.actual_y),
            offset=(plan.offset_x, plan.offset_y),
            actual_hit_summary=plan.pre_hit_summary,
            misfire=True,
        )

    async def _type(self, element_id: str, text: str) -> ActionResult:
        selector = f'[data-ghost-ux-id="{element_id}"]'
        assert self.page is not None
        last_error = ""
        for attempt in range(1, self.agent_config.retry_limit_per_action + 1):
            try:
                locator = self.page.locator(selector).first
                await locator.wait_for(state="visible", timeout=self.config.action_timeout_ms)
                await locator.scroll_into_view_if_needed(timeout=self.config.action_timeout_ms)
                await self._human_delay()
                await locator.click(timeout=self.config.action_timeout_ms)
                await self._human_delay()
                await locator.fill(text, timeout=self.config.action_timeout_ms)
                await self._settle_after_action()
                await self._human_delay()
                return ActionResult(
                    success=True,
                    detail=f"Typed into {selector} on attempt {attempt}.",
                    executed_selector=selector,
                )
            except (TimeoutError, Error) as exc:
                last_error = str(exc)
                logger.warning("Type attempt {} failed for {}: {}", attempt, selector, exc)
                await self._human_delay()
        return ActionResult(
            success=False,
            detail=f"Type failed for {selector}: {last_error}",
            executed_selector=selector,
        )

    async def _human_delay(self) -> None:
        delay_ms = random.randint(
            self.agent_config.min_action_delay_ms,
            self.agent_config.max_action_delay_ms,
        )
        await asyncio.sleep(delay_ms / 1000)

    async def _settle_after_action(self) -> None:
        assert self.page is not None
        try:
            await self.page.wait_for_load_state("domcontentloaded", timeout=2_500)
        except TimeoutError:
            pass
        try:
            await self.page.wait_for_load_state("networkidle", timeout=2_500)
        except TimeoutError:
            pass

    async def _resolve_action_plan(self, action: UIAction, observation: Observation | None) -> ExecutableAction:
        plan = ExecutableAction(
            action_type=action.action_type,
            target_element_id=action.target_element_id,
            scroll_delta_x=0.0,
            scroll_delta_y=0.0,
        )
        if action.action_type == ActionType.CLICK and action.target_element_id:
            element = None
            if observation:
                element = next((item for item in observation.elements if item.element_id == action.target_element_id), None)
            if element:
                intended_x, intended_y = element.center
                plan.intended_x = intended_x
                plan.intended_y = intended_y
                plan.actual_x = intended_x
                plan.actual_y = intended_y
            plan.executed_selector = f'[data-ghost-ux-id="{action.target_element_id}"]'
        elif action.action_type == ActionType.SCROLL_DOWN:
            plan.scroll_delta_y = 720.0
        elif action.action_type == ActionType.SCROLL_UP:
            plan.scroll_delta_y = -720.0
        return plan

    async def _element_summary_at_point(self, x: float, y: float) -> str | None:
        assert self.page is not None
        result = await self.page.evaluate(
            """([px, py]) => {
                const el = document.elementFromPoint(px, py);
                if (!el) return null;
                const tag = el.tagName ? el.tagName.toLowerCase() : 'unknown';
                const id = el.getAttribute('data-ghost-ux-id') || el.id || '';
                const classes = (el.className && typeof el.className === 'string') ? el.className.trim().split(/\\s+/).slice(0, 3).join('.') : '';
                const text = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 40);
                return [tag, id && `#${id}`, classes && `.${classes}`, text && `text=${text}`].filter(Boolean).join(' ');
            }""",
            [x, y],
        )
        return result

    def _empty_observation(self) -> Observation:
        return Observation(
            step_index=0,
            url="about:blank",
            title="",
            screenshot_bytes=b"",
            screenshot_base64="",
            screenshot_path=self.session_dir / "_noop.png",
            elements=[],
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
        )

    def _point_tuple(self, x: float | None, y: float | None) -> tuple[float, float] | None:
        if x is None or y is None:
            return None
        return (round(x, 1), round(y, 1))
