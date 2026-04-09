from __future__ import annotations

import re
from dataclasses import dataclass


BUILTIN_JARGON_TERMS: dict[str, list[str]] = {
    "general": [
        "赋能",
        "颗粒度",
        "闭环",
        "底层逻辑",
        "抓手",
        "矩阵",
        "生态位",
        "范式",
        "去中心化",
    ],
    "b2b_saas": [
        "SaaS",
        "PaaS",
        "CRM",
        "ERP",
        "私有化部署",
        "线索培育",
        "续费率",
        "客户成功",
    ],
    "ai": [
        "AI",
        "API",
        "Agent",
        "RAG",
        "Embedding",
        "Inference",
        "Token",
        "多模态",
        "工作流编排",
    ],
    "web3": [
        "Web3",
        "Web 3.0",
        "Decentralized",
        "Decentralization",
        "DAO",
        "DAOs",
        "Token",
        "Smart Contract",
        "Blockchain",
        "Crypto",
        "去中心化",
        "区块链",
        "链上",
        "钱包",
        "Gas",
        "智能合约",
        "DEX",
        "NFT",
    ],
}

ASCII_BOUNDARY_CHARS = r"A-Za-z0-9"
ASCII_WORD_PATTERN = re.compile(rf"[{ASCII_BOUNDARY_CHARS}]")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class CompiledJargonTerm:
    term: str
    pattern: re.Pattern[str]


def _contains_cjk(value: str) -> bool:
    return bool(CJK_PATTERN.search(value))


def _contains_ascii_word(value: str) -> bool:
    return bool(ASCII_WORD_PATTERN.search(value))


def _compile_term(term: str, *, case_sensitive: bool) -> CompiledJargonTerm:
    escaped = re.escape(term)
    flags = 0 if case_sensitive else re.IGNORECASE
    if _contains_cjk(term):
        pattern = re.compile(escaped, flags)
    elif _contains_ascii_word(term):
        pattern = re.compile(
            rf"(?<![{ASCII_BOUNDARY_CHARS}]){escaped}(?![{ASCII_BOUNDARY_CHARS}])",
            flags,
        )
    else:
        pattern = re.compile(escaped, flags)
    return CompiledJargonTerm(term=term, pattern=pattern)


def build_compiled_jargon_terms(
    domains: list[str],
    custom_terms: list[str],
    *,
    case_sensitive: bool,
) -> list[CompiledJargonTerm]:
    ordered_terms: list[str] = []
    seen: set[str] = set()
    for domain in domains:
        for term in BUILTIN_JARGON_TERMS.get(domain, []):
            normalized = term.strip()
            if normalized and normalized.lower() not in seen:
                ordered_terms.append(normalized)
                seen.add(normalized.lower())
    for term in custom_terms:
        normalized = term.strip()
        if normalized and normalized.lower() not in seen:
            ordered_terms.append(normalized)
            seen.add(normalized.lower())
    ordered_terms.sort(key=len, reverse=True)
    return [_compile_term(term, case_sensitive=case_sensitive) for term in ordered_terms]


def replace_jargon_terms(
    value: str | None,
    compiled_terms: list[CompiledJargonTerm],
    *,
    placeholder: str,
    density_threshold: int,
) -> tuple[str | None, int, list[str], bool]:
    if not value:
        return value, 0, [], False
    if value == placeholder:
        return value, 0, [], False

    replaced = value
    total_matches = 0
    matched_terms: list[str] = []
    for compiled in compiled_terms:
        replaced, count = compiled.pattern.subn(placeholder, replaced)
        if count:
            total_matches += count
            matched_terms.extend([compiled.term] * count)

    dense = total_matches >= density_threshold if density_threshold > 0 else False
    if dense and total_matches > 0:
        return placeholder, total_matches, matched_terms, True
    return replaced, total_matches, matched_terms, False
