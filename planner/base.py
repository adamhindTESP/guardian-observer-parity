# planner/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any


class PlannerInterface(ABC):
    """
    Proposal-only interface.
    Planners generate structured action proposals.
    They DO NOT validate, enforce, or filter.
    """

    @abstractmethod
    def propose(self, instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a proposed action dict.
        Must be schema-compatible with Guardian.
        """
        pass