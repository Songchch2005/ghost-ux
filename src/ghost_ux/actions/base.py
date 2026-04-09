from __future__ import annotations

from abc import ABC, abstractmethod

from ghost_ux.models import ExecutableAction, Observation, UIAction


class BaseActionFilter(ABC):
    name: str = "base"

    @abstractmethod
    def apply(
        self,
        action: UIAction,
        observation: Observation,
        plan: ExecutableAction,
    ) -> ExecutableAction:
        raise NotImplementedError
