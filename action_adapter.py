
"""
guardian/action_adapter.py - DETERMINISTIC RUNTIME ADAPTER (v2.0 - FIXED)
CRITICAL RULES:
1. Runtime-only - NO SFTExample, NO DPOExample, NO training classes
2. Physics extraction only - NO planner semantics  
3. LOUD FAILURE on invalid physics - never silently veto
4. Schema-agnostic - works with eval runner defaults + any planner output
5. Kernel is sole safety authority
"""

from schema.guardian_schema import Constraints, GuardianKernelAction
from typing import Dict, Any


def parse_and_adapt(
    test: Dict[str, Any], 
    constraints: Constraints
) -> GuardianKernelAction:
    """
    RUNTIME-ONLY adapter: extract physics from ANY dict structure.
    
    Handles:
    • Eval runner defaults: {"output": {"force_n": 0.0, ...}}
    • Planner output:      {"output": {"force_n": 1.2, ...}}
    
    CRASHES on schema ambiguity or invalid physics.
    """
    
    # STEP 1: Find physics (loud failure if missing)
    if "output" not in test:
        raise ValueError(
            f"Cannot extract physics: missing 'output' key. Got keys: {list(test.keys())}"
        )
    
    output = test["output"]
    if not isinstance(output, dict):
        raise ValueError(f"'output' must be dict, got {type(output)}")
    
    # STEP 2: Extract + validate physics fields (LOUD TYPE FAILURES)
    try:
        force_n = float(output["force_n"])
        velocity_mps = float(output["velocity_mps"])
        distance_m = float(output.get("distance_m", 1.0))  # Safe default
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid physics fields in output: force_n={output.get('force_n')}, "
            f"velocity_mps={output.get('velocity_mps')}, distance_m={output.get('distance_m')}. "
            f"Error: {e}"
        )
    
    # STEP 3: LOG FOR DEBUG (eval runner transparency)
    print(f"[ADAPTER] Extracted: force={force_n:.2f}N, velocity={velocity_mps:.3f}m/s, "
          f"distance={distance_m:.2f}m → kernel")
    
    # STEP 4: Pure field mapping - NO safety logic
    return GuardianKernelAction(
        force_n=force_n,
        velocity_mps=velocity_mps,
        distance_m=distance_m
    )
