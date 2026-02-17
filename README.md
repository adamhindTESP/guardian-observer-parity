```markdown
# Guardian–Observer Parity

**Deterministic Evaluation Artifact — Paper 2**

**Repository:** guardian-observer-parity  
**Status:** Frozen deterministic evaluation harness (publication artifact)  
**Purpose:** Empirical validation of the Observer–Guardian Non-Interference Invariant

---

## 📌 Paper Claim

This repository validates the following invariant:

**Enabling the Observer layer (Temple) produces no change in execution decisions, evaluation stream contents, or evaluation stream hash.**

Temple is strictly:
- Write-only
- Authority-free  
- Invisible to Planner and Guardian
- Non-branching with respect to runtime control flow

**If Temple ON and Temple OFF produce identical evaluation stream hashes, non-interference is empirically demonstrated.**

---

## What This Repository Contains
- Deterministic evaluation runner (`run_eval_minimal.py` **v4.9.1**)
- Frozen Semantic Guardian Kernel (SGK)
- Planner wrappers (proposal-only)
- Minimal Observer implementation 
- Canonical test sets
- Full stream SHA-256 verification

**This repository is:**
- Fully reproducible
- Deterministic (SEED=42 locked)
- Planner-agnostic
- Enforcement-preserving

---

## What This Repository Is Not
- Not a training repository
- Not a robotics stack
- Not a deployment framework
- Not an alignment system

**This repository exists solely to validate the Observer–Guardian non-interference invariant.**

---

## Architectural Scope

```
Runtime pipeline under test:

Planner (proposal-only)
         ↓
SGK (deterministic veto authority)
         ↓  
Observer (write-only, no feedback)
```

**Critical invariant:**
- Observer cannot modify execution
- Observer cannot veto
- Observer cannot branch execution
- Observer cannot affect SGK state
- Observer cannot affect planner prompt or output

---

## Reproducibility Modes

Two reproduction paths supported.

### Mode A — Guardian-Only (Core Proof)
Sufficient to validate the invariant.
- No model download
- No GPU required
- Runs on any laptop
- Fully validates hash parity

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

**Expected:** Identical PASS/VETO counts + **identical Stream Hash**

---

### Mode B — Planner-Enabled (Extended Validation)
Validates invariant under live planner execution.

**Paper 2 Model:** `Qwen/Qwen2.5-7B-Instruct` (~16GB RAM, GPU recommended)

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

**Expected:** Identical PASS/VETO counts, planner calls, proposal hashes, **Stream Hash**

---

## Deterministic Execution

The evaluation runner **v4.9.1** enforces:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED) 
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if CUDA available
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Stream hash computed from:**
- Test ID
- Guardian verdict  
- Force / velocity / distance values
- Proposal hash (planner enabled)

**Observer output excluded from hash.**

### 🖥️ Software Stack
```
Python 3.10.12
PyTorch 2.1.0 (deterministic algorithms enabled)
CUDA 12.2
torch.backends.cudnn.deterministic = True
```

**Bit-level parity guaranteed only within identical software/hardware environments.**

**Cross-hardware hash equality not claimed; Temple ON/OFF parity holds within same execution environment.**

---

## 📊 Expected Output Format

```
EVAL SUMMARY
---------------------------------------------------
Total tests        : 50
PASS               : 34  
VETO               : 16
Planner calls      : 50
---------------------------------------------------
Avg force (N)      : 0.8421
Avg velocity (m/s) : 0.0231
Avg distance (m)   : 0.3000
===================================================
DETERMINISTIC MODE (Seed Locked)
Seed               : 42
Stream Hash        : f9ebc72a0843234930e30829be4e1bea047dac9a060cb612ee520f7005a99289
===================================================
```

**Hash must match exactly between Temple OFF and Temple ON.**

**Parity evaluated within same execution environment. Invariant concerns ON/OFF equivalence.**

---

## 📁 Results Artifacts
```
results/<run_id>.results.jsonl     # Full evaluation records
results/<run_id>.summary.json      # Summary + hash
observer/*.json                    # Temple logs (ON only)
```

---

## Threat Model (Paper 2 Scope)
**Validates:**
- Runtime non-interference
- Authority separation integrity  
- Deterministic execution stability
- Write-only observer behavior

**Out of scope:**
- Hardware attacks
- OS compromise
- Microarchitectural side channels
- Alignment guarantees
- Long-horizon planning exploits

---

## Relationship to Paper 1
**Paper 1 (Guardian-01):** Safety enforcement independent of planner learning  
**Paper 2:** Observation independent of enforcement authority

**Together:** Intelligence ≠ Authority, Observation ≠ Influence

---

## Version History

**Publication tag:** `observer-parity-v1.1.0`

**This tag reflects the deterministic seeded artifact (SEED=42, PyTorch deterministic algorithms enabled) referenced in Paper 2.**

*Previous release (`observer-parity-v1.0.0`) did not explicitly enforce global seed locking. v1.1.0 strengthens reproducibility guarantees.*

---

## License
MIT License  
Copyright (c) 2026
```
