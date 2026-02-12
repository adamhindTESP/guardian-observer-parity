
# planner/deepseek_wrapper.py

from typing import Dict, Any
from planner.base import PlannerInterface


class DeepSeekPlanner(PlannerInterface):
    """
    DeepSeek wrapper skeleton.
    Loads model locally (not included in repo).
    """

    def __init__(self, model_path: str = None, lora_path: str = None):
        self.model_path = model_path
        self.lora_path = lora_path

        # TODO: load DeepSeek model locally

    def propose(self, instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace with real DeepSeek inference.
        """

        raise NotImplementedError(
            "Load DeepSeek locally and implement inference."
        )
