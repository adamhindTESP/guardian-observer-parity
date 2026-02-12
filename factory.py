# planner/factory.py

from typing import Optional
from planner.base import PlannerInterface
from planner.mock_planner import MockPlanner
from planner.phi2_wrapper import Phi2Planner
from planner.deepseek_wrapper import DeepSeekPlanner


def get_planner(
    name: str,
    base_model: Optional[str] = None,
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
) -> PlannerInterface:
    """
    Canonical planner factory for guardian-observer-parity repo.

    SUPPORTED PLANNERS:
      • mock     : Deterministic baseline (no args required)
      • phi2     : Local Phi-2 + optional LoRA (--base-model required)
      • deepseek : Local DeepSeek + optional LoRA (--base-model required)

    Fails fast on misconfiguration to preserve reproducibility.
    """

    if not name:
        raise ValueError("Planner name required (--planner-name)")

    name = name.lower().strip()

    # ---------------- MOCK (deterministic baseline) ----------------
    if name == "mock":
        return MockPlanner()

    # ---------------- PHI-2 ----------------
    if name == "phi2":
        if base_model is None:
            raise ValueError(
                "Phi2 requires --base-model "
                "(e.g. microsoft/phi-2)"
            )
        return Phi2Planner(
            base_model=base_model,
            lora_path=lora_path,
            device=device,
        )

    # ---------------- DEEPSEEK ----------------
    if name == "deepseek":
        if base_model is None:
            raise ValueError(
                "DeepSeek requires --base-model"
            )
        return DeepSeekPlanner(
            base_model=base_model,
            lora_path=lora_path,
            device=device,
        )

    # ---------------- UNKNOWN ----------------
    raise ValueError(
        f"Unknown planner: '{name}'. "
        "Supported: mock, phi2, deepseek."
    )