Guardian–Observer Parity

Deterministic Evaluation Artifact — Paper 2

Repository: guardian-observer-parity
Status: Deterministic evaluation harness (frozen for publication)
Purpose: Empirical validation of the Observer–Guardian Non-Interference Invariant

⸻

📌 Paper Claim

This repository validates the following invariant:

Enabling the Observer layer (Temple) produces no change in execution decisions, evaluation results, or evaluation stream hash.

Temple is strictly:
	•	Write-only
	•	Authority-free
	•	Invisible to Planner and Guardian
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
	•	❌ Not a deployment safety system

This repository exists solely to verify the Observer–Guardian non-interference invariant.

⸻

Architectural Scope

Runtime pipeline under test:

Planner (proposal only)
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

This repository supports two reproduction paths.

⸻

Mode A — Guardian-Only (Required Proof)

This is sufficient to validate the non-interference invariant.
	•	No model download
	•	No GPU required
	•	Runs on any standard laptop
	•	Fully validates hash parity

1️⃣ Temple OFF

!python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --guardian-only

Expected output:
	•	Deterministic PASS/VETO counts
	•	Evaluation stream hash printed
	•	No observer artifact

⸻

2️⃣ Temple ON

!python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --guardian-only --temple-out observer/gte_core_guardian_temple.json

Expected:
	•	Identical PASS/VETO counts
	•	Identical evaluation stream hash
	•	Observer file created

If hashes differ, non-interference is violated.

⸻

Mode B — Planner-Enabled (Extended Validation)

This demonstrates parity when a live planner is active.

The invariant must still hold:
	•	Guardian decisions identical
	•	Evaluation stream hash identical
	•	Planner call count unchanged

⸻

Planner Model Used in Paper 2

Experiments use:

Qwen/Qwen2.5-7B-Instruct

This model requires:
	•	~16GB RAM minimum
	•	GPU strongly recommended
	•	CUDA recommended

If local hardware is insufficient, use Google Colab (instructions below).

⸻

3️⃣ Planner Enabled — Temple OFF

!python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --planner --planner-name qwen --base-model Qwen/Qwen2.5-7B-Instruct --device cuda --out-dir results_planner_off --run-id qwen_core_guardian_temple_off


⸻

4️⃣ Planner Enabled — Temple ON

!python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --planner --planner-name qwen --base-model Qwen/Qwen2.5-7B-Instruct --device cuda --temple-out observer/qwen_core_guardian_temple_on.json --out-dir results_planner_on --run-id qwen_core_guardian_temple_on

Expected:
	•	Identical Guardian PASS/VETO counts
	•	Identical evaluation stream hash
	•	Identical planner call count

Hash parity ON/OFF is the proof.

⸻

Google Colab Reproduction (Recommended for Planner Runs)

If your local machine cannot load Qwen 7B, use Colab:
	1.	Set runtime to GPU
	2.	Run:

!git clone https://github.com/adamhindTESP/guardian-observer-parity.git
%cd guardian-observer-parity

!pip install torch transformers pydantic

!python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --planner --planner-name qwen --base-model Qwen/Qwen2.5-7B-Instruct --device cuda --out-dir results_planner_off --run-id qwen_core_guardian_temple_off

or 

!python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --planner --planner-name qwen --base-model Qwen/Qwen2.5-7B-Instruct --device cuda --temple-out observer/qwen_core_guardian_temple_on.json --out-dir results_planner_on --run-id qwen_core_guardian_temple_on

⸻

Lightweight Planner Option (Optional)

For low-memory systems, a smaller Qwen variant may be used:

Qwen/Qwen2.5-1.5B-Instruct

The non-interference invariant is architectural and does not depend on model size.

⸻

Deterministic Evaluation Stream

The evaluation runner computes a cumulative SHA-256 hash of:
	•	Canonicalized planner proposal
	•	Guardian verdict
	•	Expected outcome
	•	Test ID

Temple writes artifacts only after decisions finalize.

The hash does not include observer output.

Hash parity ON/OFF is the invariant proof.

⸻

Expected Result Format

For each test suite:

Mode	PASS	VETO	Hash
Temple OFF	X	Y	H
Temple ON	X	Y	H

Hash H must match exactly.

⸻

Threat Model (Paper 2 Scope)

This repository tests only:
	•	Runtime non-interference
	•	Authority separation integrity
	•	Write-only observer behavior
	•	Deterministic execution stability

Out of scope:
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

Together:
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

If any of these are modified, the parity proof is invalid.

⸻

Versioning

Once parity validation is complete, this repository will be tagged:

observer-parity-v1.0.0

The tag represents a frozen experimental state used in Paper 2.

⸻

License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the conditions stated in the LICENSE file.
