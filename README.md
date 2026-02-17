# Guardian–Observer Parity

**Deterministic Evaluation Artifact — Paper 2**  
**Repository:** `guardian-observer-parity`  
**Status:** Frozen deterministic evaluation harness (publication artifact)  
**Purpose:** Empirical validation of the **Observer–Guardian Non-Interference Invariant**

***

## 📄 Paper Claim

This repository validates the following invariant:

> **Enabling the Observer layer ("Temple") produces no change in execution decisions, evaluation stream contents, or evaluation stream hash.**

### Temple Characteristics
- Write-only  
- Authority-free  
- Invisible to Planner and Guardian  
- Non-branching with respect to runtime control flow  

When Temple ON and Temple OFF produce identical stream hashes, runtime non-interference is empirically demonstrated.

***

## 📦 Repository Contents

- Deterministic evaluation runner (`run_eval_minimal.py` v4.9.2)  
- Frozen **Semantic Guardian Kernel (SGK)**  
- Planner wrappers (proposal-only)  
- Minimal write-only **Observer** implementation  
- Canonical test sets  
- Full evaluation-stream SHA-256 verification  

**Properties**
- Fully reproducible within a fixed environment  
- Seed-locked (**SEED = 42**)  
- Planner-agnostic  
- Enforcement-preserving  
- Observer-non-interfering by design  

***

## 🚫 Out of Scope

- Training, robotics, or deployment stacks  
- Alignment or isolation systems  
- Hardware-level determinism or proof of cross-device parity  

This artifact solely validates the **Observer–Guardian non-interference invariant**.

***

## 🧭 Architectural Scope

```
Planner (proposal-only)
     ↓
SGK (deterministic veto authority)
     ↓
Observer (write-only, no feedback)
```

**Critical invariant**
- Observer cannot modify, veto, or branch execution  
- Observer cannot affect SGK state  
- Observer cannot affect planner prompt or output  

***

## 🧪 Reproducibility Modes

Two deterministic reproduction paths are supported, both using the **guardian enforcement dataset** (`test_sets/gte_core_guardian.jsonl`) to ensure VETO visibility.

### Mode A — Guardian-Only (Core Proof)

Minimal proof of invariant.  
No model download -  No GPU required -  Runs on any machine.

#### 1️⃣ Temple OFF
```bash
python run_eval_minimal.py test_sets/gte_core_guardian.jsonl --guardian-only
```

#### 2️⃣ Temple ON
```bash
python run_eval_minimal.py test_sets/gte_core_guardian.jsonl \
    --guardian-only \
    --temple-out observer/gte_core_guardian_temple.json
```

**Expected**
- Identical PASS/VETO counts  
- Identical stream hash  

Confirms Temple has no influence on deterministic Guardian enforcement.

***

### Mode B — Planner Enabled + Guardian Enforcement (Extended Proof)

Validates Observer non-interference under live planner proposals **evaluated by Guardian** on the enforcement dataset.  
**Paper 2 Model:** `Qwen/Qwen2.5-7B-Instruct`

#### 3️⃣ Planner Enabled — Temple OFF
```bash
python run_eval_minimal.py test_sets/gte_core_guardian.jsonl \
    --planner \
    --planner-name qwen \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --out-dir results_planner_off \
    --run-id qwen_core_guardian_temple_off
```

#### 4️⃣ Planner Enabled — Temple ON
```bash
python run_eval_minimal.py test_sets/gte_core_guardian.jsonl \
    --planner \
    --planner-name qwen \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --device cuda \
    --temple-out observer/qwen_core_guardian_temple_on.json \
    --out-dir results_planner_on \
    --run-id qwen_core_guardian_temple_on
```

**Expected**
- Identical PASS/VETO counts  
- Identical planner call counts  
- Identical proposal hashes  
- Identical stream hash  

Demonstrates Observer does not influence planner proposals, Guardian enforcement, or execution stream on the red-team enforcement set.

***

## ⚙️ Deterministic Execution Model

**Runner version:** v4.9.2  
**Seed:** 42  

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Planner generation:
# do_sample = False
# temperature = 0.0
```

Strict `torch.use_deterministic_algorithms(True)` is **not** enforced (may fail under certain CUDA/cuBLAS configs).  

**Proven:** Temple ON/OFF equivalence within a single execution environment.  
**Not claimed:** Cross-hardware bit-level equality.

***

## 🔑 Stream Hash Definition

Computed from:
- Test ID  
- Guardian verdict  
- Force/Velocity/Distance values  
- Proposal hash (planner-enabled mode)  

**Observer output excluded** — Temple cannot influence the verified stream.

***

## 📊 Example Output (Planner + Guardian Enforcement)

```
EVAL SUMMARY
---------------------------------------------------
Total tests        : 7
PASS               : 6
VETO               : 1
Planner calls      : 7
---------------------------------------------------
Avg force (N)      : 1.6857
Avg velocity (m/s) : 0.0543
Avg distance (m)   : 0.2929
===================================================
DETERMINISTIC MODE (Seed Locked)
Seed               : 42
Stream Hash        : 483647c61a124ad85f0c625a43d41ad39366596e2cf9789829ab47c9dcfc3bac
===================================================
```

**Temple ON and OFF produce identical stream hash values.** This shows active Guardian vetoes (1/7) under live planner proposals.

***

## 📁 Result Artifacts

- `results/<run_id>.results.jsonl`  
- `results/<run_id>.summary.json`  
- `observer/*.json`

***

## 🧱 Threat Model (Paper 2 Scope)

**Validated**
- Runtime non-interference  
- Authority separation integrity  
- Seed-locked stability  
- Write-only Observer behavior  

**Out of scope**
- Hardware/OS attacks  
- Microarchitectural side channels  
- Alignment guarantees  
- Cross-hardware reproducibility  

***

## 🔗 Relationship to Paper 1

**Paper 1 (Guardian-01):** Safety enforcement independent of planner learning.  
**Paper 2:** Observation independent of enforcement authority.

> **Intelligence ≠ Authority**  **Observation ≠ Influence**

***

## 🧾 Version History

**Publication Tag:** `observer-parity-v1.1.1`

**v1.1.1 includes**
- Runner v4.9.2  
- Robust planner JSON extraction  
- Seed-locked deterministic generation  
- Guardian-only & Planner+Guardian parity (same enforcement dataset)  
- Realistic enforcement example (1 VETO demonstrated)  
- Clarified determinism scope  

**Previous:** v1.0.0 (initial), v1.1.0 (determinism added)

***

## ⚖️ License

**MIT License**  
Copyright (c) 2026  
