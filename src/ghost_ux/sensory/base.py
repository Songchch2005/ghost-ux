from __future__ import annotations

from abc import ABC, abstractmethod

from ghost_ux.models import DOMElement, FilterEffect


class BaseSensoryFilter(ABC):
    name: str = "base"
    last_effect: FilterEffect | None = None

    @abstractmethod
    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        raise NotImplementedError

    def _record_effect(
        self,
        *,
        removed_count: int = 0,
        modified_count: int = 0,
        reasons: dict[str, int] | None = None,
    ) -> FilterEffect:
        self.last_effect = FilterEffect(
            filter_name=self.name,
            removed_count=removed_count,
            modified_count=modified_count,
            reasons=reasons or {},
        )
        return self.last_effect
