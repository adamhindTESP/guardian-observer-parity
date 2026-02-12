
# planner/mock_planner.py

from planner.base import PlannerInterface


class MockPlanner(PlannerInterface):
    """
    Deterministic planner for reproducible parity testing.

    Must match PlannerInterface:
        propose(self, prompt: str) -> str

    Returns JSON string (not dict) so evaluator can:
        - hash raw proposal
        - extract JSON
        - remain model-agnostic
    """

    def propose(self, prompt: str) -> str:
        # Deterministic, constraint-safe envelope
        # IMPORTANT: Return JSON string, not dict
        return (
            '{"output": {'
            '"force_n": 0.0, '
            '"velocity_mps": 0.0, '
            '"distance_m": 0.3'
            '}}'
        )
