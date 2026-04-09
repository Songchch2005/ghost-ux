from __future__ import annotations

from ghost_ux.models import DOMElement, FilterEffect
from ghost_ux.sensory.base import BaseSensoryFilter


class FilterPipeline:
    def __init__(self, filters: list[BaseSensoryFilter] | None = None):
        self.filters = filters or []

    def apply(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement]]:
        filtered_bytes, filtered_elements, _ = self.apply_with_trace(screenshot_bytes, dom_elements_list)
        return filtered_bytes, filtered_elements

    def apply_with_trace(
        self,
        screenshot_bytes: bytes,
        dom_elements_list: list[DOMElement],
    ) -> tuple[bytes, list[DOMElement], list[FilterEffect]]:
        filtered_bytes = screenshot_bytes
        filtered_elements = list(dom_elements_list)
        effects: list[FilterEffect] = []
        for sensory_filter in self.filters:
            filtered_bytes, filtered_elements = sensory_filter.apply(filtered_bytes, filtered_elements)
            effects.append(
                sensory_filter.last_effect
                or FilterEffect(filter_name=sensory_filter.name)
            )
        return filtered_bytes, filtered_elements, effects

    @property
    def active_filter_names(self) -> list[str]:
        return [sensory_filter.name for sensory_filter in self.filters]
