Guardian–Observer Parity

Deterministic Evaluation Artifact — Paper 2

Repository: guardian-observer-parity
Status: Frozen deterministic evaluation harness (publication artifact)
Purpose: Empirical validation of the Observer–Guardian Non-Interference Invariant

⸻

📌 Paper Claim

This repository validates the following invariant:

Enabling the Observer layer (Temple) produces no change in execution decisions, evaluation stream contents, or evaluation stream hash.

Temple is strictly:
	•	Write-only
	•	Authority-free
	•	Invisible to Planner and Guardian
	•	Non-branching with respect to runtime control flow

If Temple ON and Temple OFF produce identical evaluation stream hashes, non-interference is empirically demonstrated.

⸻

What This Repository Contains
	•	Deterministic evaluation runner (run_eval_minimal.py)
	•	Frozen Semantic Guardian Kernel (SGK)
	•	Planner wrappers (proposal-only)
	•	Minimal Observer implementation (observer_minimal.py)
	•	Canonical test sets
	•	Full stream SHA-256 verification

This repository is:
	•	Fully reproducible
	•	Deterministic (seed locked)
	•	Planner-agnostic
	•	Enforcement-preserving

⸻

What This Repository Is Not
	•	Not a training repository
	•	Not a robotics stack
	•	Not a deployment framework
	•	Not an alignment system

This repository exists solely to validate the Observer–Guardian non-interference invariant.

⸻

Architectural Scope

Runtime pipeline under test:

Planner (proposal-only)
↓
SGK (deterministic veto authority)
↓
Observer (write-only, no feedback)

Critical invariant:
	•	Observer cannot modify execution
	•	Observer cannot veto
	•	Observer cannot branch execution
	•	Observer cannot affect SGK state
	•	Observer cannot affect planner prompt or output

⸻

Reproducibility Modes

Two reproduction paths are supported.

⸻

Mode A — Guardian-Only (Core Proof)

Sufficient to validate the invariant.
	•	No model download
	•	No GPU required
	•	Runs on any laptop
	•	Fully validates hash parity

⸻

1️⃣ Temple OFF

python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --guardian-only

Output includes:
	•	PASS/VETO counts
	•	Planner call count (0)
	•	Deterministic seed
	•	Stream Hash

⸻

2️⃣ Temple ON

python run_eval_minimal.py test_sets/gte_core_guardian.jsonl \
    --guardian-only \
    --temple-out observer/gte_core_guardian_temple.json

Expected:
	•	Identical PASS/VETO counts
	•	Identical Stream Hash
	•	Observer file created

If hashes differ, non-interference is violated.

⸻

Mode B — Planner-Enabled (Extended Validation)

Validates invariant under live planner execution.

The invariant must still hold:
	•	Guardian verdicts identical
	•	Planner call count identical
	•	Proposal hashes identical
	•	Stream hash identical

⸻

Planner Model Used in Paper 2

Primary experiment model:

Qwen/Qwen2.5-7B-Instruct

Requirements:
	•	~16GB RAM minimum
	•	GPU recommended
	•	CUDA recommended

⸻

3️⃣ Planner Enabled — Temple OFF

python run_eval_minimal.py test_sets/gte_core_guardian.jsonl \
    --planner \
    --planner-name qwen \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --out-dir results_planner_off \
    --run-id qwen_core_guardian_temple_off


⸻

4️⃣ Planner Enabled — Temple ON

python run_eval_minimal.py test_sets/gte_core_guardian.jsonl \
    --planner \
    --planner-name qwen \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --temple-out observer/qwen_core_guardian_temple_on.json \
    --out-dir results_planner_on \
    --run-id qwen_core_guardian_temple_on

Expected:
	•	Identical PASS/VETO counts
	•	Identical planner call count
	•	Identical proposal hashes
	•	Identical Stream Hash

Hash parity ON/OFF is the invariant proof.

⸻

Deterministic Execution

The evaluation runner enforces:
	•	Fixed random seed (SEED = 42)
	•	CUDA deterministic mode
	•	Disabled cuDNN benchmarking
	•	Deterministic PyTorch algorithms

The stream hash is computed from:
	•	Test ID
	•	Guardian verdict
	•	Force / velocity / distance values
	•	Proposal hash (when planner enabled)

The Observer output is not included in the hash.

⸻

Output Format

Each run prints:

EVAL SUMMARY
Total tests        : X
PASS               : Y
VETO               : Z
Planner calls      : N

Avg force (N)      : ...
Avg velocity (m/s) : ...
Avg distance (m)   : ...

DETERMINISTIC MODE (Seed Locked)
Seed               : 42
Stream Hash        : H

Hash H must match exactly between Temple OFF and Temple ON.

⸻

Results Artifacts

Each run generates:
	•	results/<run_id>.results.jsonl
	•	results/<run_id>.summary.json
	•	Optional observer log (Temple ON only)

These artifacts allow full reproduction and independent verification.

⸻

Threat Model (Paper 2 Scope)

This repository validates:
	•	Runtime non-interference
	•	Authority separation integrity
	•	Deterministic execution stability
	•	Write-only observer behavior

Out of scope:
	•	Hardware attacks
	•	OS compromise
	•	Microarchitectural side channels
	•	Alignment guarantees
	•	Long-horizon planning exploits

Paper 2 claims deterministic non-interference under software execution only.

⸻

Relationship to Paper 1

Paper 1 (Guardian-01):

Safety enforcement is independent of planner learning.

Paper 2:

Observation is independent of enforcement authority.

Together:
	•	Intelligence ≠ Authority
	•	Observation ≠ Influence

⸻

Invariants

This artifact assumes:
	•	SGK frozen
	•	Observer write-only
	•	Planner proposal-only
	•	No runtime randomness
	•	No side-channel branching

If any are modified, the parity proof is invalid.

⸻

Version

Publication tag:

observer-parity-v1.0.0

Represents the frozen state used in Paper 2.

⸻

License

MIT License
Copyright (c) 2026
