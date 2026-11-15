# Safety gate

This package implements the safety gate that accepts or rejects a policy proposal using the digital
twin, and the fallback that replaces rejected proposals.

| File | Contents |
|---|---|
| `gate.py`, `gate_config.json` | The original gate: accept iff the twin's predicted load satisfies the capacity bound on every link |
| `evaluate.py` | Evaluation of the original gate with the twin in `models/twin/ensemble.pt`; results in `results/safety/` (`metrics.json`, `SAFETY_REPORT.md`, per-topology decision logs) |
| `improvements/` | The improved gate and its evaluation infrastructure |

The `improvements/` package contains the upper-confidence-bound decision rule μ + kσ ≤ capacity
(`gate_logic.py`, `decision.py`), the twin and policy backends (`twin_backend.py`, `policy_backend.py`,
`graph_cache.py`), the physics oracle (`oracle.py`), the shortest/equal least-violating fallback and the
conditional LP fallback (`fallback.py`, `lp_fallback_*.py`), the paired A–F evaluation runner with
resumable decision logs (`run_eval.py`, `resume.py`, `logging_util.py`), metrics with explicit
denominators (`metrics.py`), and the uncertainty and fallback diagnostics. Its configuration and results
are in `results/safety_gate/improvements/`. The gate never retrains the twin and never reads the ASN
shards.
