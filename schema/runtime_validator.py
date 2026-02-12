from typing import Dict, Any
from schema.guardian_schema import GuardianKernelAction as Action


def validate_action(action: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
    """
    FROZEN, deterministic runtime validator.
    This is the ONLY authority for action approval.
    """

    try:
        parsed = Action(**action)
    except Exception as e:
        return {
            "valid": False,
            "error": f"SCHEMA_ERROR: {e}",
        }

    # Numeric constraint enforcement
    if parsed.force_n > constraints.get("max_force_n", float("inf")):
        return {
            "valid": False,
            "error": "FORCE_LIMIT_EXCEEDED",
        }

    if parsed.velocity_mps > constraints.get("max_velocity_mps", float("inf")):
        return {
            "valid": False,
            "error": "VELOCITY_LIMIT_EXCEEDED",
        }

    return {
        "valid": True,
        "action": parsed,
    }
