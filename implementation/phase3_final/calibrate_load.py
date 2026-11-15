"""Calibrated-load study: feasibility-calibrated traffic load levels.

For every TM used later (train/val of B4, UsCarrier, Cogentco; the seeded GEANT
test subsample; Kdl test) compute shortest, equal and (where tractable) the
k=4 path-LP optimal MLU. The calibrated TM for target t is alpha * D with

    alpha(t) = t / reference_mlu,   reference = LP MLU (B4, GEANT, UsCarrier, Cogentco)
                                               shortest MLU (Kdl; LP not tractable)

All routings are linear in D, so one LP per TM covers every target. Only alpha is
stored; scaled TMs are always rebuilt as alpha * D, identically for every method.
The calibration reads only the TM and capacities, never any method's output.

Usage:  python -m phase3_final.calibrate_load
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict

import numpy as np

from twin.data import load_split_demands, load_topology

from . import routing as R
from .common import (
    LP_SOLVER_NOTE,
    TOY_TM_NOTE,
    GEANT_SUBSAMPLE_SEED,
    append_jsonl,
    environment,
    geant_subsample,
    matrix_hash,
    out_dir,
    read_jsonl,
    write_json,
)

TARGETS = (0.5, 0.7, 0.9, 1.1)
FIXED_SCALES = (0.25, 0.5, 0.75, 1.0)
LP_TOPOLOGIES = ("B4", "GEANT", "UsCarrier", "Cogentco")
PLAN = (
    ("B4", "train"),
    ("B4", "val"),
    ("UsCarrier", "train"),
    ("UsCarrier", "val"),
    ("Cogentco", "train"),
    ("Cogentco", "val"),
    ("GEANT", "test"),
    ("Kdl", "test"),
)
E3_GEANT_SEED = 2027
E3_GEANT_SIZE = 50
FEAS_TOL = 1e-9
OUT = out_dir("calibrated_traffic")
ROWS = OUT / "calibration_rows.jsonl"


def positions_for(topology: str, split: str, n: int) -> np.ndarray:
    if topology == "GEANT" and split == "test":
        return geant_subsample(n)
    return np.arange(n)


def run() -> list[dict]:
    done = {(r["topology"], r["split"], r["position"]) for r in read_jsonl(ROWS)}
    lp_cache: dict[tuple[str, str], dict] = {
        (r["topology"], r["tm_hash"]): r for r in read_jsonl(ROWS) if r.get("lp_mlu") is not None
    }
    for topology, split in PLAN:
        topo = load_topology(topology)
        demands, meta = load_split_demands(split, topology)
        indices = list(meta["matrix_indices"])
        positions = positions_for(topology, split, demands.shape[0])
        sp_action, eq_action = R.shortest_action(topo), R.equal_action(topo)
        t0 = time.time()
        for k, pos in enumerate(positions):
            pos = int(pos)
            if (topology, split, pos) in done:
                continue
            matrix = demands[pos]
            h = matrix_hash(matrix)
            row = {
                "topology": topology,
                "split": split,
                "position": pos,
                "tm_index": int(indices[pos]),
                "tm_hash": h,
                "traffic_kind": meta.get("traffic_kind"),
                "mlu_shortest": R.mlu(topo, matrix, sp_action),
                "mlu_equal": R.mlu(topo, matrix, eq_action),
                "mlu_lp": None,
                "lp_runtime_s": None,
                "lp_reused_identical_tm": False,
            }
            if topology in LP_TOPOLOGIES:
                cached = lp_cache.get((topology, h))
                if cached is not None:
                    row["mlu_lp"] = cached["mlu_lp"]
                    row["lp_runtime_s"] = 0.0
                    row["lp_reused_identical_tm"] = True
                else:
                    sol = R.lp_min_mlu(topo, matrix)
                    if sol["status"] != "optimal":
                        raise RuntimeError(f"LP failed for {topology}/{split}/{pos}: {sol['message']}")
                    row["mlu_lp"] = sol["lp_mlu"]
                    row["lp_runtime_s"] = sol["runtime_s"]
                    row["lp_solver_u"] = sol["solver_u"]
                    lp_cache[(topology, h)] = row
            row["reference"] = "lp" if row["mlu_lp"] is not None else "shortest"
            ref = row["mlu_lp"] if row["mlu_lp"] is not None else row["mlu_shortest"]
            row["reference_mlu"] = ref
            row["alpha"] = {f"{t:.1f}": t / ref for t in TARGETS}
            append_jsonl(ROWS, row)
            done.add((topology, split, pos))
            if (k + 1) % 25 == 0 or k + 1 == len(positions):
                el = time.time() - t0
                print(f"{topology}/{split}: {k + 1}/{len(positions)} elapsed {el:.0f}s", flush=True)
        del topo, demands
    return read_jsonl(ROWS)


def feasibility(rows: list[dict]) -> list[dict]:
    """Fraction of scenarios with a feasible routing (MLU <= 1) per method."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        role = "eval_test" if r["split"] == "test" else r["split"]
        groups[(r["topology"], role)].append(r)
    out = []
    for (topology, role), rs in sorted(groups.items()):
        settings = [("fixed", s, lambda r, s=s: s) for s in FIXED_SCALES]
        settings += [("calibrated", t, lambda r, t=t: r["alpha"][f"{t:.1f}"]) for t in TARGETS]
        for mode, level, alpha_of in settings:
            rec = {"topology": topology, "split": role, "mode": mode, "level": level, "n_tm": len(rs),
                   "n_unique_tm": len({r["tm_hash"] for r in rs}), "reference": rs[0]["reference"]}
            for method in ("shortest", "equal", "lp"):
                vals = [r[f"mlu_{method}"] for r in rs]
                if any(v is None for v in vals):
                    rec[f"feasible_{method}"] = None
                    rec[f"mean_mlu_{method}"] = None
                    continue
                scaled = np.array([alpha_of(r) * v for r, v in zip(rs, vals)])
                rec[f"feasible_{method}"] = float(np.mean(scaled <= 1.0 + FEAS_TOL))
                rec[f"mean_mlu_{method}"] = float(scaled.mean())
            # "any baseline feasible" = the definition used by robustness_v1 (shortest or equal)
            sp = np.array([alpha_of(r) * r["mlu_shortest"] for r in rs])
            eq = np.array([alpha_of(r) * r["mlu_equal"] for r in rs])
            rec["feasible_shortest_or_equal"] = float(np.mean((sp <= 1 + FEAS_TOL) | (eq <= 1 + FEAS_TOL)))
            out.append(rec)
    return out


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def write_outputs(rows: list[dict], feas: list[dict]) -> None:
    cols = ["topology", "split", "position", "tm_index", "tm_hash", "traffic_kind", "reference",
            "mlu_shortest", "mlu_equal", "mlu_lp", "reference_mlu", "lp_runtime_s", "lp_reused_identical_tm"]
    with (OUT / "calibration_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols + [f"alpha_{t:.1f}" for t in TARGETS])
        for r in rows:
            w.writerow([r.get(c) for c in cols] + [r["alpha"][f"{t:.1f}"] for t in TARGETS])
    with (OUT / "feasibility_before_after.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feas[0].keys()))
        w.writeheader()
        w.writerows(feas)

    geant_pos = sorted(r["position"] for r in rows if r["topology"] == "GEANT")
    rng = np.random.default_rng(E3_GEANT_SEED)
    e3 = sorted(int(x) for x in rng.choice(geant_pos, size=E3_GEANT_SIZE, replace=False))
    index = {
        "description": "Calibrated TM for (topology, split, position, target) is alpha * D[position]; "
                       "D from data/processed/splits/<split>/<topology>_demand.npz.",
        "targets": TARGETS,
        "fixed_scales": FIXED_SCALES,
        "geant_test_positions_200": geant_pos,
        "geant_subsample_seed": GEANT_SUBSAMPLE_SEED,
        "geant_e3_positions_50": e3,
        "geant_e3_seed": E3_GEANT_SEED,
        "alpha": {f"{r['topology']}/{r['split']}/{r['position']}": r["alpha"] for r in rows},
        "reference": {f"{r['topology']}/{r['split']}/{r['position']}": r["reference"] for r in rows},
    }
    write_json(OUT / "scaled_tm_index.json", index)

    # Report
    lines = [
        "# Feasibility-calibrated traffic",
        "",
        "**Setting label:** *calibrated load* (an additional evaluation setting; all fixed-scale results "
        "under `results/evaluation/` are kept unchanged).",
        "",
        "## Why",
        "",
        "Toy (UsCarrier, Kdl) and reconstructed (Cogentco gravity) matrices have arbitrary units. The fixed "
        "scales 0.25–1.0 leave these topologies above capacity for every routing, so no method can be safe and "
        "every method falls back. Calibration rescales each TM so that a method-independent reference MLU hits "
        "a chosen target, making the feasibility boundary visible.",
        "",
        "## Definition",
        "",
        "For each TM *D*: `alpha(t) = t / reference_mlu(D)`, calibrated TM = `alpha(t) * D`, "
        f"targets t ∈ {list(TARGETS)}.",
        "",
        "| Topology | Reference MLU | Why |",
        "|---|---|---|",
        "| B4, GEANT, UsCarrier, Cogentco | k=4 path-LP optimum (min-MLU) | LP tractable (≤ 7 s per TM) |",
        "| Kdl | shortest-path MLU | LP over ~2.3 M path variables not tractable on this laptop |",
        "",
        "- The reference uses only the TM and the capacities, never any method's output.",
        "- Shortest, equal and LP are linear in *D*, so one LP per TM is exact for every target.",
        "- The identical scaled TM is given to every method (paired design).",
        "- With the LP reference, a feasible k=4 routing exists **by construction** when t ≤ 1 (LP MLU = t) and "
        "none exists at t = 1.1. This is a design property, not a finding.",
        "- With the shortest reference (Kdl), shortest itself is feasible when t ≤ 1.",
        f"- {LP_SOLVER_NOTE}",
        f"- {TOY_TM_NOTE} Identical TMs are detected by hash; their LP is solved once and reused.",
        f"- GEANT: fixed seeded subsample of {len(geant_pos)} of the 2,693 FIGRET test TMs "
        f"(seed {GEANT_SUBSAMPLE_SEED}); 50 of these (seed {E3_GEANT_SEED}) are reserved for E3.",
        "",
        "## Reference MLU per topology (native scale)",
        "",
        "| Topology | Split | TMs | Unique TMs | Reference | Shortest MLU | Equal MLU | LP MLU | LP s/TM |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    by = defaultdict(list)
    for r in rows:
        by[(r["topology"], r["split"])].append(r)
    for (topology, split), rs in by.items():
        lp = [r["mlu_lp"] for r in rs if r["mlu_lp"] is not None]
        lrt = [r["lp_runtime_s"] for r in rs if r.get("lp_runtime_s") and not r["lp_reused_identical_tm"]]
        lines.append(
            f"| {topology} | {split} | {len(rs)} | {len({r['tm_hash'] for r in rs})} | {rs[0]['reference']} | "
            f"{np.mean([r['mlu_shortest'] for r in rs]):.3f} | {np.mean([r['mlu_equal'] for r in rs]):.3f} | "
            f"{_fmt(np.mean(lp) if lp else None)} | {_fmt(np.mean(lrt) if lrt else None, 2)} |"
        )
    lines += [
        "",
        "## Fraction of feasible scenarios (MLU ≤ 1), before (fixed scale) and after (calibrated)",
        "",
        "| Topology | Split | Mode | Level | Shortest | Equal | LP | Shortest or equal |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for f in feas:
        lines.append(
            f"| {f['topology']} | {f['split']} | {f['mode']} | {f['level']} | {_fmt(f['feasible_shortest'], 2)} | "
            f"{_fmt(f['feasible_equal'], 2)} | {_fmt(f['feasible_lp'], 2)} | {_fmt(f['feasible_shortest_or_equal'], 2)} |"
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "- *Before*: rows with mode `fixed`. *After*: rows with mode `calibrated`.",
        "- The `LP` column under calibration is 1.00 for t ≤ 0.9 and 0.00 for t = 1.1 by construction.",
        "- Where LP is feasible but shortest/equal are not, a smarter routing (e.g. a learned policy) can add "
        "value; where all are infeasible, the best any method can do is reject and fall back.",
        "",
        "## Artifacts",
        "",
        "- `calibration_rows.jsonl`, `calibration_table.csv` — per-TM MLUs and alpha",
        "- `feasibility_before_after.csv` — table above",
        "- `scaled_tm_index.json` — alpha per (topology/split/position), GEANT subsample positions",
        "- `config.json` — targets, scales, seeds, environment",
    ]
    (OUT / "CALIBRATED_TRAFFIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    write_json(OUT / "config.json", {
        "targets": TARGETS, "fixed_scales": FIXED_SCALES, "lp_topologies": LP_TOPOLOGIES,
        "plan": PLAN, "geant_subsample_seed": GEANT_SUBSAMPLE_SEED, "e3_geant_seed": E3_GEANT_SEED,
        "feasibility_tolerance": FEAS_TOL, "lp_solver": LP_SOLVER_NOTE, "toy_tm_note": TOY_TM_NOTE,
        "environment": environment(),
    })
    rows = run()
    feas = feasibility(rows)
    write_outputs(rows, feas)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
