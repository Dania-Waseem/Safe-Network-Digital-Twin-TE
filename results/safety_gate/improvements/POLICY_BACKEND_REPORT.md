# Policy Backend Report

**Stage:** evaluation infrastructure (no full A–F experiment).  
**Module:** `safety/improvements/policy_backend.py`  
**Graph cache:** `safety/improvements/graph_cache.py`

---

## Post-hardening verification

After negative mean/capacity safeguards were added to `gate_logic.py`, the live auditor was **re-run**:

| Check | Result |
| --- | --- |
| Live checks | **61 / 61 passed** |
| Issues | 0 |
| Original twin SHA256 | `7b74fee9…b73685` (unchanged) |
| Original `safety/gate.py` SHA256 | `bb8399b9…7aca2d` (unchanged) |

Remaining auditor warning: `full_af_not_started` (expected).

---

## Design

`PolicyBackend` wraps **`framework.general_policy.GeneralPolicy`** without modifying it.

| Responsibility | Behaviour |
| --- | --- |
| Checkpoint load | `GeneralPolicy.load(path)` → `payload["model"]` |
| Default policy | `improved_stable` → `models/policy/improved/recommended_stable.pt` |
| Inference | Deterministic `propose` / `propose_from_demand` |
| Action identity | Returns `PolicyAction` with topology, `n_real_paths`, `n_demands`, graph fingerprint |
| Graph reuse | Injects graphs from `TopologyGraphCache` into `policy._graphs[topo.name]` |

Configured alternate checkpoints (read-only): `improved_best`, `general_3epoch`.

---

## Graph cache isolation

`TopologyGraphCache` keys graphs by `topology_fingerprint(name + capacity/hops/n_paths/demand_row digests)`.

Verified:
- B4 and GEANT never share a cached `PolicyGraph` object.
- Re-getting the same topology returns the identical object (deterministic reuse).
- Clearing one topology leaves others intact.
- Proposed action length equals `sum(n_paths)` (real path table identity).

---

## Reproducibility smoke

On B4 val matrix 0:
- Two successive `propose` calls → **bit-identical** actions.
- Topology switch B4 → GEANT updates fingerprint and path counts correctly.

---

## Preservation

- `framework/general_policy.py` not modified.
- RL checkpoints not modified or overwritten.
- No training performed.

## Tests

`tests/test_safety_gate_eval_infra.py`: PolicyBackend + GraphCache + preservation cases — **passed**.
