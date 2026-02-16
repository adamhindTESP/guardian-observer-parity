"""
guardian_schema.py — DSSA-Compatible Schema v3.2
✅ Adds min_distance_m to declared physics envelope
✅ Removes hardcoded distance rule
✅ Backward compatible with existing SFT/DPO data
✅ Kernel receives ONLY physics fields
✅ No retraining required
"""

from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field, conlist


# =============================
# Core Context Objects (unchanged)
# =============================

class Context(BaseModel):
    domain: str = Field(..., description="Operational domain")
    urgency: Literal["low", "medium", "high", "critical"] = Field(...)
    entities: List[str] = Field(default_factory=list)


# =============================
# FULL PHYSICS ENVELOPE (Declared)
# =============================

class Constraints(BaseModel):
    """
    Declarative physics envelope.
    All safety limits must live here.
    """

    max_force_n: float = Field(..., ge=0.0)
    max_velocity_mps: float = Field(..., ge=0.0)

    # NEW — fully declared distance invariant
    min_distance_m: float = Field(
        0.3,
        ge=0.0,
        description="Minimum allowed proximity distance (m)"
    )


# =============================
# PLANNER Action (SFT/DPO training format)
# =============================

class PlannerAction(BaseModel):
    """
    What your model was trained to emit.
    This format remains unchanged.
    """

    intent: str = Field(...)
    force_n: float = Field(..., ge=0.0)
    velocity_mps: float = Field(..., ge=0.0)
    distance_m: float = Field(1.0, ge=0.0)  # Default keeps old data valid
    plan: conlist(str, min_length=1) = Field(...)
    safety_checks: List[str] = Field(default_factory=list)
    uncertainty: Literal["low", "medium", "high"] = Field(...)
    fallback: str = Field(...)
    principle: str = Field("", description="Optional")


# =============================
# KERNEL Action (Guardian sees ONLY physics)
# =============================

class GuardianKernelAction(BaseModel):
    """
    MINIMAL kernel-facing schema.
    No planner semantics.
    """

    force_n: float = Field(..., ge=0.0, description="Applied force (N)")
    velocity_mps: float = Field(..., ge=0.0, description="Motion speed (m/s)")
    distance_m: float = Field(..., ge=0.0, description="Min distance (m)")


# =============================
# TRAINING DATA COMPATIBILITY
# =============================

class SFTExample(BaseModel):
    instruction: str = Field(...)
    context: Context
    constraints: Constraints
    output: PlannerAction


class DPOExample(BaseModel):
    instruction: str = Field(...)
    context: Context
    constraints: Constraints
    chosen: PlannerAction
    rejected: PlannerAction


GuardianExample = Union[SFTExample, DPOExample]


# =============================
# NON-AUTHORITATIVE CHECK (Helper)
# =============================

def kernel_evaluate(action: GuardianKernelAction, constraints: Constraints) -> bool:
    """
    Quick non-authoritative envelope check.
    Real enforcement still belongs to GuardianKernel.
    """

    return (
        action.force_n <= constraints.max_force_n and
        action.velocity_mps <= constraints.max_velocity_mps and
        action.distance_m >= constraints.min_distance_m
    )
