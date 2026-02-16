Guardian–Observer Parity

Deterministic Evaluation Artifact — Paper 2

Repository: guardian-observer-parity
Status: Deterministic evaluation harness
Purpose: Empirical validation of the Observer–Guardian non-interference invariant

⸻

Paper Claim

This repository validates the following invariant:

Enabling the Observer layer (Temple) produces no change in execution decisions, evaluation results, or evaluation stream hash.

The Observer is strictly:
	•	Write-only
	•	Authority-free
	•	Invisible to planner and Guardian
	•	Non-branching with respect to runtime control flow

If Temple ON and Temple OFF produce identical evaluation streams, then non-interference is empirically demonstrated.

⸻

What This Repository Is

This repository contains:
	•	Deterministic evaluation runner (run_eval_minimal.py)
	•	Frozen Semantic Guardian Kernel (SGK)
	•	Planner wrappers (proposal-only)
	•	Minimal Observer implementation (observer_minimal.py)
	•	Canonical test sets
	•	Hash-based stream verification

This repository is:
	•	Fully reproducible
	•	Deterministic
	•	Planner-agnostic
	•	Enforcement-preserving

⸻

🚫 What This Repository Is Not
	•	❌ Not a training repository
	•	❌ Not a development sandbox
	•	❌ Not a robotics stack
	•	❌ Not a safety authority implementation for deployment

This repository exists solely to verify the non-interference invariant.

⸻

Architectural Scope

Runtime pipeline under test:

Planner (proposal only)
        ↓
SGK (deterministic veto authority)
        ↓
Observer (write-only, no feedback)

Critical invariant:
	•	Observer cannot modify
	•	Observer cannot veto
	•	Observer cannot branch execution
	•	Observer cannot affect SGK state
	•	Observer cannot affect planner prompt or output

⸻

Reproducibility

1️⃣ Guardian-Only — Temple OFF

python run_eval_minimal.py \
  test_sets/gte_core_guardian.jsonl \
  --guardian-only

Expected:
	•	PASS/VETO counts
	•	Deterministic stream hash
	•	No observer artifact

⸻

2️⃣ Guardian-Only — Temple ON

python run_eval_minimal.py \
  test_sets/gte_core_guardian.jsonl \
  --guardian-only \
  --temple-out observer/gte_core_guardian_temple.json

Expected:
	•	Identical PASS/VETO counts
	•	Identical stream hash
	•	Observer file created

If stream hashes differ, non-interference is violated.

⸻

3️⃣ Planner Enabled — Temple OFF

python run_eval_minimal.py \
  test_sets/gte_core_planner.jsonl \
  --planner tinyllama \
  --device cpu


⸻

4️⃣ Planner Enabled — Temple ON

python run_eval_minimal.py \
  test_sets/gte_core_planner.jsonl \
  --planner tinyllama \
  --device cpu \
  --temple-out observer/gte_core_planner_temple.json

Expected:
	•	Identical Guardian decisions
	•	Identical evaluation stream hash
	•	Planner proposal count unchanged

⸻

Deterministic Evaluation Stream

The evaluation runner computes a cumulative hash of:
	•	Canonicalized planner proposal
	•	Guardian verdict
	•	Expected outcome
	•	Test ID

Temple writes artifacts only after the decision is finalized.

The hash does not include observer output.

Hash parity ON/OFF is the core proof.

⸻

Expected Outcome

For each test suite:

Mode	PASS	VETO	Hash
Temple OFF	X	Y	H
Temple ON	X	Y	H

Where H must be identical.

⸻

Threat Model (Paper 2 Scope)

This repository tests only:
	•	Runtime non-interference
	•	Authority separation integrity
	•	Write-only observer behavior
	•	Deterministic execution stability

Out-of-scope:
	•	Hardware attacks
	•	OS compromise
	•	Sensor spoofing
	•	Long-horizon adversarial planning
	•	Alignment guarantees

⸻

Relationship to Paper 1

Paper 1 (Guardian-01) demonstrated:

Safety enforcement is independent of planner learning.

Paper 2 demonstrates:

Observer layer cannot influence enforcement decisions.

Together, these establish:
	•	Intelligence ≠ Authority
	•	Observation ≠ Influence

⸻

Invariants

This repository assumes:
	•	SGK is frozen
	•	Observer is write-only
	•	Planner has zero execution authority
	•	No runtime randomness
	•	No asynchronous side channels
	•	No timing-based branching

If any of these are modified, parity proof is invalid.

⸻

🏷 Versioning

This repository will be tagged once parity is validated:

observer-parity-v1.0.0

Tag represents frozen experimental state for Paper 2.

⸻

📄 License

MIT.
