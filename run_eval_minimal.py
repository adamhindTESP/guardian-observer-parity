#!/usr/bin/env python3
"""
run_eval_minimal.py — Guardian Evaluation Runner v4.8.1
(✅ FIXED: args.mode_name bug + expected_outcome validation)
"""

from __future__ import annotations
import argparse, hashlib, json, sys, os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# 🔒 Deterministic Execution Lock
# -----------------------------
import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -----------------------------

# PROJECT ROOT & FAIL-FAST IMPORTS
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from schema.guardian_schema import Constraints
    from sgk import GuardianKernel
    from action_adapter import parse_and_adapt
    from planner.base import PlannerInterface
    from factory import get_planner
except ImportError as e:
    print(f"❌ FATAL IMPORT ERROR: {e}")
    sys.exit(1)

# ENHANCED EvalRecord with expected_outcome tracking
@dataclass(frozen=True)
class EvalRecord:
    test_id: str
    verdict: str
    reason: str
    force_n: float
    velocity_mps: float
    distance_m: float
    proposal_hash8: str
    expected_outcome: Optional[str] = None
    expected_match: Optional[bool] = None

class TempleLogger:
    def __init__(self, out_path: Optional[Path]):
        self.out_path = out_path
        self._events: List[Dict[str, Any]] = []
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.out_path is not None

    def log_event(self, event: Dict[str, Any]) -> None:
        if self.enabled:
            self._events.append(event)

    def flush(self) -> None:
        if self.enabled:
            try:
                with open(self.out_path, "w", encoding="utf-8") as f:
                    json.dump(self._events, f, indent=2, sort_keys=True)
            except Exception:
                pass

# ------------------------------------------------------------
# EVALUATOR v4.8.1 (✅ PRODUCTION READY)
# ✅ Full JSON object I/O (LoRA compatible)
# ✅ expected_outcome validation vs Guardian verdict  
# ✅ Fixed args.mode_name bug
# ✅ Comprehensive suite reporting
# ------------------------------------------------------------
class GuardianEvaluator:
    def __init__(
        self,
        constraints: Dict[str, float],
        temple: TempleLogger,
        guardian_only: bool,
        planner_flag: bool,
        planner_name: Optional[str],
        base_model: Optional[str],
        lora_path: Optional[str],
        device: Optional[str],
        verbose: bool,
    ):
        self.constraints = Constraints(**constraints)
        self.temple = temple
        self.guardian_only = guardian_only
        self.verbose = verbose
        self.kernel = GuardianKernel()
        self.planner_enabled = (not guardian_only) and planner_flag
        self.planner: Optional[PlannerInterface] = None
        self.records: List[EvalRecord] = []
        self.expected: Dict[str, str] = {}  # {test_id: expected_outcome}
        self.planner_calls = 0
        self.mode_name = "GUARDIAN-ONLY" if guardian_only else ("PLANNER+GUARDIAN" if planner_flag else "SCHEMA-ONLY")

        if self.planner_enabled:
            if not planner_name or not base_model:
                raise RuntimeError("--planner requires --planner-name AND --base-model")
            
            if lora_path:
                p = Path(lora_path).expanduser().resolve()
                lora_path = str(p) if p.exists() else None

            self.planner = get_planner(
                name=planner_name, base_model=base_model, 
                lora_path=lora_path, device=device
            )

    def _planner_prompt(self, instruction: str, context: Dict[str, Any]) -> str:
        """🎯 Full JSON object → matches LoRA training exactly"""
        obj = {
            "instruction": instruction,
            "context": context,
            "constraints": self.constraints.dict(),
        }
        return json.dumps(obj, ensure_ascii=False)

    def _default_output(self) -> Dict[str, Any]:
        return {
            "output": {
                "force_n": 0.0,
                "velocity_mps": 0.0,
                "distance_m": float(self.constraints.min_distance_m)
            }
        }

    def _extract_json(self, completion: str) -> Dict[str, Any]:
        cleaned = completion.strip()
        if not cleaned:
            raise RuntimeError("Empty planner completion → VETO")

        # PRIMARY: Full JSON object (LoRA format)
        if cleaned.startswith("{"):
            try:
                obj = json.loads(cleaned)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON object: {e}\nRaw: {repr(completion[:300])}")
        else:
            # FALLBACK: Continuation format
            try:
                obj = json.loads('{"force_n":' + cleaned + ("" if cleaned.endswith("}") else "}"))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON continuation: {e}\nRaw: {repr(completion[:300])}")

        out = obj.get("output", obj)
        missing = [k for k in ("force_n", "velocity_mps") if k not in out]
        if missing:
            raise RuntimeError(f"Missing: {missing}\nKeys: {list(out.keys())}\nRaw: {repr(completion[:300])}")

        return {
            "output": {
                "force_n": float(out["force_n"]),
                "velocity_mps": float(out["velocity_mps"]),
                "distance_m": float(out.get("distance_m", float(self.constraints.min_distance_m))),
            }
        }

    def generate_action(self, instruction: str, context: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        if not self.planner_enabled or self.planner is None:
            return self._default_output(), ""

        self.planner_calls += 1
        prompt = self._planner_prompt(instruction, context)
        raw_completion = self.planner.propose(prompt)
        proposal_hash8 = hashlib.sha256(raw_completion.encode("utf-8")).hexdigest()[:8]

        if self.verbose:
            print(f"📝 Prompt:  {prompt[-80:]}...")
            print(f"📤 Raw:     {repr(raw_completion)}")
            parsed = self._extract_json(raw_completion)
            print(f"✅ Parsed:  {parsed}")

        parsed = self._extract_json(raw_completion)
        return parsed, proposal_hash8

    def evaluate_one(self, test: Dict[str, Any]) -> EvalRecord:
        test_id = str(test.get("id", "unknown"))
        instruction = str(test.get("instruction", ""))
        context = test.get("context", {}) or {}
        expected = test.get("expected_outcome")

        proposal_hash8 = ""
        verdict = "VETO"
        reason = "Eval error"
        force_n = velocity_mps = 0.0
        distance_m = float(self.constraints.min_distance_m)

        try:
            # Planner or default action
            if self.planner_enabled:
                action, proposal_hash8 = self.generate_action(instruction, context)
                test = {**test, **action}
            elif "output" not in test:
                test = {**test, **self._default_output()}

            # Guardian evaluation
            kernel_action = parse_and_adapt(test, self.constraints)
            verdict, reason = self.kernel.evaluate(kernel_action)
            force_n = float(kernel_action.force_n)
            velocity_mps = float(kernel_action.velocity_mps)
            distance_m = float(kernel_action.distance_m)

        except Exception as e:  # ✅ Catch ALL errors
            reason = f"{type(e).__name__}: {str(e)}"

        expected_match = (verdict == expected) if expected else None

        rec = EvalRecord(
            test_id=test_id, verdict=verdict, reason=reason,
            force_n=force_n, velocity_mps=velocity_mps, distance_m=distance_m,
            proposal_hash8=proposal_hash8, expected_outcome=expected, expected_match=expected_match,
        )

        self.temple.log_event({
            "test_id": test_id, "ts_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict, "expected": expected, "match": expected_match,
            "force_n": force_n, "velocity_mps": velocity_mps, "distance_m": distance_m,
            "proposal_hash8": proposal_hash8,
        })
        self.records.append(rec)
        return rec

    def run(self, test_path: Path) -> None:
        """Load tests and build expected_outcome lookup"""
        raw = test_path.read_text(encoding="utf-8").strip()
        tests = json.loads(raw) if raw.startswith("[") else [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]
        
        # Build expected lookup
        self.expected = {t["id"]: t["expected_outcome"] for t in tests if "
