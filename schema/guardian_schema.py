
"""
guardian_schema.py - FIXED FOR EVAL PIPELINE v3.1
✅ Parses your SFT/DPO data correctly
✅ Kernel receives ONLY physics (no planner fields)
✅ Backward compatible with your training
✅ No retraining required
"""

from typing import List, Literal, Union
from pydantic import BaseModel, Field, conlist
from typing import Optional


# =============================
# Core Context Objects (unchanged)
# =============================

class Context(BaseModel):
    domain: str = Field(..., description="Operational domain")
    urgency: Literal["low", "medium", "high", "critical"] = Field(...)
    entities: List[str] = Field(default_factory=list)


class Constraints(BaseModel):
    max_force_n: float = Field(..., ge=0.0)
    max_velocity_mps: float = Field(..., ge=0.0)


# =============================
# PLANNER Action (SFT/DPO training format) 
# =============================

class PlannerAction(BaseModel):
    """What your model was trained to emit."""
    intent: str = Field(...)
    force_n: float = Field(..., ge=0.0)
    velocity_mps: float = Field(..., ge=0.0)
    distance_m: float = Field(1.0, ge=0.0)  # Default for old data
    plan: conlist(str, min_length=1) = Field(...)
    safety_checks: List[str] = Field(default_factory=list)
    uncertainty: Literal["low", "medium", "high"] = Field(...)
    fallback: str = Field(...)
    principle: str = Field("", description="Optional")


# =============================
# KERNEL Action (Guardian sees ONLY physics)
# =============================

class GuardianKernelAction(BaseModel):
    """MINIMAL schema - ONLY what kernel needs."""
    force_n: float = Field(..., ge=0.0, description="Applied force (N)")
    velocity_mps: float = Field(..., ge=0.0, description="Motion speed (m/s)")
    distance_m: float = Field(..., ge=0.0, description="Min distance (m)")


# =============================
# SFT Example (your training data)
# =============================

class SFTExample(BaseModel):
    instruction: str = Field(...)
    context: Context
    constraints: Constraints
    output: PlannerAction  # Your trained model output


# =============================
# DPO Example (your training data) 
# =============================

class DPOExample(BaseModel):
    instruction: str = Field(...)
    context: Context
    constraints: Constraints
    chosen: PlannerAction    # Safe (trained preferred)
    rejected: PlannerAction  # Unsafe (trained rejected)


# =============================
# Training Data Compatibility
# =============================

GuardianExample = Union[SFTExample, DPOExample]


# =============================
# Kernel Compatibility Check
# =============================

def kernel_evaluate(action: GuardianKernelAction, constraints: Constraints) -> bool:
    """Quick check - NON-AUTHORITATIVE."""
    return (
        action.force_n <= constraints.max_force_n and
        action.velocity_mps <= constraints.max_velocity_mps and
        action.distance_m >= 0.3  # Ladybird principle
    )
