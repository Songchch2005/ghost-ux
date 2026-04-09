from __future__ import annotations

from ghost_ux.config import SensoryConfig
from ghost_ux.sensory.filters import (
    BlurryVisionFilter,
    CognitiveFilter,
    ColorblindnessFilter,
    LowPatienceFilter,
    SymbolCognitionFilter,
    TunnelVisionFilter,
)
from ghost_ux.sensory.pipeline import FilterPipeline


FILTER_REGISTRY = {
    "blurry_vision": lambda config: BlurryVisionFilter(
        radius=config.blur_radius,
        severity=config.blurry_vision_severity,
    ),
    "colorblindness": lambda config: ColorblindnessFilter(mode=config.colorblind_mode),
    "tunnel_vision": lambda config: TunnelVisionFilter(
        visible_width_ratio=config.tunnel_visible_width_ratio,
        visible_height_ratio=config.tunnel_visible_height_ratio,
        darkness=config.tunnel_darkness,
        blur_radius=config.tunnel_blur_radius,
        min_visible_ratio=config.tunnel_min_visible_ratio,
        safety_inset_ratio=config.tunnel_safety_inset_ratio,
    ),
    "low_patience": lambda config: LowPatienceFilter(
        max_text_length=config.low_patience_max_text_length,
        max_y_ratio=config.low_patience_max_y_ratio,
    ),
    "symbol_cognition": lambda config: SymbolCognitionFilter(
        mask_padding_px=config.symbol_mask_padding_px,
        dom_strategy=config.symbol_dom_strategy,
        mask_style=config.symbol_mask_style,
    ),
    "cognitive": lambda config: CognitiveFilter(config),
}


def build_sensory_pipeline(persona: str, config: SensoryConfig) -> FilterPipeline:
    filter_names = list(config.filters)
    if config.auto_from_persona:
        for inferred in infer_filters_from_persona(persona, config):
            if inferred not in filter_names:
                filter_names.append(inferred)
    filters = [FILTER_REGISTRY[name](config) for name in filter_names if name in FILTER_REGISTRY]
    return FilterPipeline(filters)


def infer_filters_from_persona(persona: str, config: SensoryConfig) -> list[str]:
    lowered = persona.lower()
    inferred: list[str] = []

    severe_blurry_keywords = (
        "严重老花",
        "重度老花",
        "严重视力衰退",
        "没戴老花镜",
        "没戴眼镜",
        "看近处几乎看不清",
        "severe presbyopia",
        "without glasses",
    )
    mild_blurry_keywords = (
        "轻度老花",
        "刚开始老花",
        "偶尔看不清小字",
        "mild presbyopia",
    )
    blurry_keywords = (
        "老花",
        "老花眼",
        "视力衰退",
        "视力不好",
        "看不清",
        "低视力",
        "blur",
        "blurry",
        "low vision",
    )
    colorblind_keywords = (
        "色盲",
        "色弱",
        "红绿色盲",
        "全色盲",
        "colorblind",
        "protanopia",
        "achromatopsia",
    )
    tunnel_keywords = ("管状视野", "视野狭窄", "焦躁", "tunnel vision")
    low_patience_keywords = ("缺乏耐心", "没耐心", "impatient", "short attention span", "急躁")
    symbol_keywords = (
        "技术符号认知障碍",
        "看不懂图标",
        "看不懂三道杠",
        "看不懂放大镜",
        "icon blindness",
        "symbol cognition",
    )
    cognitive_keywords = (
        "专业术语",
        "黑话",
        "洋词汇",
        "一窍不通",
        "文化程度不高",
        "看不懂英文",
        "看不懂专业术语",
        "术语过载",
        "行业黑话",
        "行业外",
        "行业外的小白",
        "小白",
        "外行",
        "jargon",
        "outsider",
        "non-expert",
        "layperson",
        "novice",
        "cognitive",
    )

    if any(keyword in lowered for keyword in blurry_keywords):
        inferred.append("blurry_vision")
        if config.blurry_vision_severity == "auto":
            if any(keyword in lowered for keyword in severe_blurry_keywords):
                config.blurry_vision_severity = "severe"
            elif any(keyword in lowered for keyword in mild_blurry_keywords):
                config.blurry_vision_severity = "mild"
            else:
                config.blurry_vision_severity = "moderate"
    if any(keyword in lowered for keyword in colorblind_keywords):
        if "全色盲" in lowered or "achromatopsia" in lowered:
            config.colorblind_mode = "achromatopsia"
        inferred.append("colorblindness")
    if any(keyword in lowered for keyword in tunnel_keywords):
        inferred.append("tunnel_vision")
    if any(keyword in lowered for keyword in low_patience_keywords):
        inferred.append("low_patience")
    if any(keyword in lowered for keyword in symbol_keywords):
        inferred.append("symbol_cognition")
    if any(keyword in lowered for keyword in cognitive_keywords):
        inferred.append("cognitive")
        for keyword, domain in (
            ("b2b", "b2b_saas"),
            ("saas", "b2b_saas"),
            ("crm", "b2b_saas"),
            ("erp", "b2b_saas"),
            ("ai", "ai"),
            ("agent", "ai"),
            ("api", "ai"),
            ("web3", "web3"),
            ("web 3.0", "web3"),
            ("blockchain", "web3"),
            ("crypto", "web3"),
            ("dao", "web3"),
            ("以太坊", "web3"),
            ("去中心化", "web3"),
        ):
            if keyword in lowered and domain not in config.cognitive_enabled_domains:
                config.cognitive_enabled_domains.append(domain)
    return inferred
