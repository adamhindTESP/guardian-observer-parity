"""
guardian/kernel.py - FINAL CORRECTED VERSION v3.3
🔧 FIXED IMPORT TOPOLOGY + CACHE IMMUNITY
"""

from typing import Dict, Any, Tuple
# ✅ CORRECTED IMPORTS FOR YOUR DIRECTORY STRUCTURE
from schema.guardian_schema import GuardianKernelAction  # schema/guardian_schema.py
from schema.runtime_validator import validate_action     # schema/runtime_validator.py


class GuardianKernel:
    def __init__(self, constraints: Dict[str, float] = None):
        """Frozen safety authority - enforces Temple physics."""
        self.constraints = constraints or {
            "max_force_n": 2.0,        # Ladybird principle
            "max_velocity_mps": 0.05,  # Human reaction buffer  
            "min_distance_m": 0.3,     # Personal space
        }
    
    def evaluate(self, kernel_action: GuardianKernelAction) -> Tuple[str, str]:
        """
        TEMPLE KERNEL: Physics → Verdict
        
        GuardianKernelAction(3 fields) → validate_action() → PASS/VETO
        """
        # Extract PURE PHYSICS for your validator (ALL 3 FIELDS)
        physics = {
            "force_n": kernel_action.force_n,
            "velocity_mps": kernel_action.velocity_mps,
            "distance_m": kernel_action.distance_m,
        }
        
        # CALL YOUR FROZEN GUARDIAN-01 KERNEL
        result = validate_action(physics, self.constraints)
        
        # Clean Temple interface
        if result.get("valid", False):
            return "PASS", ""
        return "VETO", result.get("error", "VALIDATION_FAILED")


# ✅ BACKWARD COMPATIBILITY
def validate_action_wrapper(
    action_dict: Dict[str, Any], 
    constraints: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Legacy wrapper - converts ANY action dict → GuardianKernelAction."""
    kernel = GuardianKernel(constraints)
    
    kernel_action = GuardianKernelAction(
        force_n=float(action_dict.get("force_n", 0.0)),
        velocity_mps=float(action_dict.get("velocity_mps", 0.0)),
        distance_m=float(action_dict.get("distance_m", 1.0)),
    )
    
    verdict, reason = kernel.evaluate(kernel_action)
    return {
        "valid": verdict == "PASS",
        "error": reason or "",
    }


# 🆕 PRODUCTION-READY TEST WITH VERBOSITY CONTROL
def ladybird_test(verbose: bool = True) -> bool:
    """Verifies Ladybird Principle (gentleness = strength)."""
    kernel = GuardianKernel()
    ladybird = GuardianKernelAction(
        force_n=0.05,    # Gentle righting (0.05 < 2.0 ✓)
        velocity_mps=0.01,  # Slow observation (0.01 < 0.05 ✓) 
        distance_m=0.5,  # Respectful distance (0.5 > 0.3 ✓)
    )
    
    verdict, reason = kernel.evaluate(ladybird)
    
    if verbose:
        print("🔴 LADYBIRD PRINCIPLE TEST")
        print(f"  PHYSICS: force={ladybird.force_n}N, velocity={ladybird.velocity_mps}m/s, distance={ladybird.distance_m}m")
        print(f"  LIMITS:  max_force={kernel.constraints['max_force_n']}N, max_vel={kernel.constraints['max_velocity_mps']}m/s, min_dist={kernel.constraints['min_distance_m']}m")
        print(f"  VERDICT: {verdict}")
        print(f"  REASON:  {reason}")
        print(f"  RESULT:  {verdict == 'PASS'}")
        print("-" * 60)
    
    return verdict == "PASS"


# ONE-COMMAND PRODUCTION TEST
if __name__ == "__main__":
    """Run Ladybird test directly."""
    success = ladybird_test()
    print(f"TEMPLE KERNEL STATUS: {'✅ ALIVE' if success else '❌ DEAD'}")