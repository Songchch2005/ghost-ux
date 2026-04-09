from __future__ import annotations

from ghost_ux.actions.base import BaseActionFilter
from ghost_ux.models import ExecutableAction, Observation, UIAction


class ActionFilterPipeline:
    def __init__(self, filters: list[BaseActionFilter] | None = None):
        self.filters = filters or []

    def apply(
        self,
        action: UIAction,
        observation: Observation,
        plan: ExecutableAction,
    ) -> ExecutableAction:
        current = plan.model_copy(deep=True)
        for action_filter in self.filters:
            current = action_filter.apply(action, observation, current)
        return current

    @property
    def active_filter_names(self) -> list[str]:
        return [action_filter.name for action_filter in self.filters]
