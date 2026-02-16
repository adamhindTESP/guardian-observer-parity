# planner/factory.py — Reproducibility-Locked Version (Qwen Integrated)

from typing import Optional
from planner.base import PlannerInterface

from planner.mock_planner import MockPlanner
from planner.phi2_wrapper import Phi2Planner
from planner.deepseek_wrapper import DeepSeekPlanner
from planner.tinyllama_wrapper import TinyLlamaPlanner
from planner.qwen_wrapper import QwenPlanner  # ← NEW


def get_planner(
    name: str,
    base_model: Optional[str] = None,
    lora_path: Optional[str] = None,
    device: Optional[str] = None,
) -> PlannerInterface:
    """
    Canonical planner factory for guardian-observer-parity repo.

    COLAB/PAPER USAGE EXAMPLE:

    python run_eval_minimal.py test_sets/gte_core.jsonl \
        --planner \
        --planner-name qwen \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --device cuda \
        --verbose
    """

    if not name:
        raise ValueError("Planner name required (--planner-name)")

    name = name.lower().strip()

    # ---------------------------------------------------------------------
    # MOCK (deterministic baseline)
    # ---------------------------------------------------------------------
    if name == "mock":
        return MockPlanner()

    # ---------------------------------------------------------------------
    # Registry: class + capabilities + suggested model
    # ---------------------------------------------------------------------
    planners = {
        "phi2": {
            "cls": Phi2Planner,
            "suggested": "microsoft/phi-2",
            "supports_lora": True,
        },
        "tinyllama": {
            "cls": TinyLlamaPlanner,
            "suggested": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "supports_lora": False,
        },
        "deepseek": {
            "cls": DeepSeekPlanner,
            "suggested": None,
            "supports_lora": True,
        },
        "qwen": {  # ← NEW ENTRY
            "cls": QwenPlanner,
            "suggested": "Qwen/Qwen2.5-7B-Instruct",
            "supports_lora": False,
        },
    }

    # ---------------------------------------------------------------------
    # Validate planner name
    # ---------------------------------------------------------------------
    if name not in planners:
        raise ValueError(
            f"Unknown planner: '{name}'\n"
            f"Supported: {', '.join(planners.keys())}\n"
            "Recommended for Colab repro: --planner-name qwen"
        )

    entry = planners[name]

    # ---------------------------------------------------------------------
    # Validate base_model requirement
    # ---------------------------------------------------------------------
    if base_model is None:
        if entry["suggested"]:
            raise ValueError(
                f"{name.title()} requires --base-model\n"
                f"Recommended: {entry['suggested']}"
            )
        raise ValueError(f"{name.title()} requires --base-model")

    # ---------------------------------------------------------------------
    # Dispatch with correct constructor signature
    # ---------------------------------------------------------------------
    if entry["supports_lora"]:
        return entry["cls"](
            base_model=base_model,
            lora_path=lora_path,
            device=device,
        )
    else:
        return entry["cls"](
            base_model=base_model,
            device=device,
        )
