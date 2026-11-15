"""E2 holdout evaluation: paired A-F evaluation on the unseen GEANT and Kdl topologies.

Reuses ``safety.improvements.run_eval.evaluate_paired_case`` and the frozen gate
config of ``af_eval_k1_calibrated_devtopos_v1`` without any change to thresholds,
twins, conformal scale or fallback. ``run_eval`` itself refuses held-out
topologies by design, so this wrapper builds the holdout cases and calls the same
per-case function. Nothing is tuned here.

Cases per topology:
  fixed scales {0.25, 0.5, 0.75, 1.0} and calibrated targets {0.5, 0.7, 0.9, 1.1}
  (alpha from the calibrated-load study), on
  * GEANT: the 200 seeded FIGRET test TMs (calibrated-load subsample)
  * Kdl:   1 test TM (all 8 Kdl test TMs are identical toy copies)

Usage:
  python -m phase3_final.holdout_af --topologies GEANT
  python -m phase3_final.holdout_af --topologies Kdl
  python -m phase3_final.holdout_af --report-only
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safety.improvements.config import load_improvements_config
from safety.improvements.decision import VARIANT_IDS
from safety.improvements.graph_cache import TopologyGraphCache
from safety.improvements.metrics import compute_safety_metrics
from safety.improvements.policy_backend import policy_backend_from_config
from safety.improvements.run_eval import evaluate_paired_case
from safety.improvements.twin_backend import backends_from_config
from twin.data import ROOT, load_split_demands, load_topology

from . import routing as R
from .common import (
    LP_SOLVER_NOTE,
    TOY_TM_NOTE,
    append_jsonl,
    environment,
    fmt_eta,
    out_dir,
    read_jsonl,
    write_json,
)

FROZEN_CONFIG = (
    ROOT / "results" / "safety_gate" / "improvements" / "eval_runs"
    / "af_eval_k1_calibrated_devtopos_v1" / "frozen_gate_improvements_config.json"
)
CALIBRATION = ROOT / "results" / "phase3_final" / "calibrated_traffic" / "scaled_tm_index.json"
FIXED_SCALES = (0.25, 0.5, 0.75, 1.0)
TARGETS = (0.5, 0.7, 0.9, 1.1)
CASE_SEED = 2026  # policies are deterministic; the seed field only labels the case
SHORT = {
    "A_policy_only": "A",
    "B_original_twin_mean": "B",
    "C_original_twin_uncertainty": "C",
    "D_improved_twin_mean": "D",
    "E_improved_twin_calibrated_uncertainty": "E",
    "F_physics_oracle": "F",
}


def build_cases(topologies: list[str]) -> list[dict[str, Any]]:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    cases = []
    for name in topologies:
        demands, meta = load_split_demands("test", name)
        ids = meta["matrix_indices"]
        positions = cal["geant_test_positions_200"] if name == "GEANT" else [0]
        for pos in positions:
            alpha = cal["alpha"][f"{name}/test/{pos}"]
            levels = [("fixed", s, s) for s in FIXED_SCALES]
            levels += [("calibrated", t, alpha[f"{t:.1f}"]) for t in TARGETS]
            for mode, level, scale in levels:
                cases.append({
                    "seed": CASE_SEED, "topology": name, "split": "test",
                    "local_index": int(pos), "matrix_id": int(ids[pos]),
                    "scale": float(scale), "load_mode": mode, "load_level": float(level),
                    "unseen": True,
                })
    return cases


def case_key(case: dict[str, Any], policy_id: str) -> str:
    return (f"{case['topology']}|pos={case['local_index']}|{case['load_mode']}={case['load_level']}"
            f"|policy={policy_id}")


def holdout_config(policy_id: str | None, checkpoint: str | None) -> dict[str, Any]:
    cfg = load_improvements_config(FROZEN_CONFIG)
    cfg = copy.deepcopy(cfg)
    if checkpoint:
        cfg["policy"]["checkpoint_id"] = policy_id
        cfg["policy"]["default_checkpoint"] = str(Path(checkpoint).resolve())
    ev = cfg["evaluation"]
    ev["topology_authorization"] = {
        "included": ["GEANT", "Kdl"],
        "reason": "E2 holdout evaluation: single pass on the held-out test topologies; "
                  "frozen thresholds, twins and conformal scale; no tuning or recalibration.",
    }
    ev["never_tune_on"] = ["GEANT", "Kdl"]
    ev["tune_on"] = ["B4", "UsCarrier", "Cogentco"]
    return cfg


def run(topologies: list[str], out: Path, policy_id: str | None, checkpoint: str | None,
        limit: int | None = None) -> None:
    cfg = holdout_config(policy_id, checkpoint)
    write_json(out / f"config_{cfg['policy']['checkpoint_id']}.json", {
        "frozen_config_source": FROZEN_CONFIG.relative_to(ROOT),
        "gate": cfg["gate"], "variants": list(cfg["variants"]), "twins": cfg["twins"],
        "policy": cfg["policy"], "fallback": cfg["fallback"],
        "fixed_scales": FIXED_SCALES, "calibrated_targets": TARGETS,
        "calibration_index": CALIBRATION.relative_to(ROOT),
        "topology_authorization": cfg["evaluation"]["topology_authorization"],
        "lp_solver": LP_SOLVER_NOTE, "toy_tm_note": TOY_TM_NOTE, "environment": environment(),
    })
    device = torch.device("cpu")
    graph_cache = TopologyGraphCache(device)
    policy = policy_backend_from_config(cfg, device=device, graph_cache=graph_cache)
    policy.load()
    pid = policy.checkpoint_id
    backends = backends_from_config(cfg, device)
    decisions = out / "decisions.jsonl"
    compact = out / "case_rows.jsonl"
    done = {r["case_key"] for r in read_jsonl(compact)}
    cases = [c for c in build_cases(topologies) if case_key(c, pid) not in done]
    if limit:
        cases = cases[:limit]
    print(f"{len(cases)} cases to run ({len(done)} already done)", flush=True)

    topo_cache: dict[str, Any] = {}
    ref_cache: dict[tuple[str, int], dict[str, Any]] = {}
    t0 = time.time()
    for i, case in enumerate(cases):
        name = case["topology"]
        if name not in topo_cache:
            topo_cache.clear()
            topo_cache[name] = (load_topology(name), load_split_demands("test", name)[0])
        topo, demands = topo_cache[name]
        base = demands[case["local_index"]]
        matrix = base * case["scale"]
        rkey = (name, case["local_index"])
        if rkey not in ref_cache:
            ref = {"shortest": R.shortest_action(topo), "equal": R.equal_action(topo), "lp": None}
            if name != "Kdl":
                ref["lp"] = R.lp_min_mlu(topo, base)["action"]
            ref_cache[rkey] = ref
        ref = ref_cache[rkey]

        tp = time.perf_counter()
        proposal = policy.propose(topo, matrix)
        policy_ms = (time.perf_counter() - tp) * 1000.0
        rows = evaluate_paired_case(
            case=case, topo=topo, matrix=matrix, policy_action=proposal.action,
            policy_meta=proposal.to_log_dict(), backends=backends, cfg=cfg, variants=VARIANT_IDS,
        )
        sat = {k: R.satisfaction_p1(topo, matrix, a) for k, a in
               (("policy", proposal.action), ("shortest", ref["shortest"]), ("equal", ref["equal"]))}
        refs = {
            "mlu_shortest": R.mlu(topo, matrix, ref["shortest"]),
            "mlu_equal": R.mlu(topo, matrix, ref["equal"]),
            "mlu_lp": None if ref["lp"] is None else R.mlu(topo, matrix, ref["lp"]),
            "sat_shortest": sat["shortest"], "sat_equal": sat["equal"],
            "sat_lp": None if ref["lp"] is None else R.satisfaction_p1(topo, matrix, ref["lp"]),
        }
        key = case_key(case, pid)
        for row in rows:
            row.update({"case_key": key, "load_mode": case["load_mode"], "load_level": case["load_level"],
                        "decision_status": "ok", "policy_ms": policy_ms})
            row["satisfaction_applied"] = sat.get(row["applied_action_source"])
            row["satisfaction_proposed"] = sat["policy"]
        for row in rows:
            append_jsonl(decisions, row)
        compact_row = {
            "case_key": key, "topology": name, "position": case["local_index"],
            "matrix_id": case["matrix_id"], "load_mode": case["load_mode"],
            "load_level": case["load_level"], "scale": case["scale"], "policy_id": pid,
            "policy_ms": policy_ms, "mlu_proposed": rows[0]["mlu_proposed"],
            "sat_proposed": sat["policy"], "oracle_feasible_proposed": rows[0]["oracle_feasible_proposed"],
            **refs,
            "variants": {SHORT[r["gate_variant"]]: {
                "accept": r["acceptance"], "unsafe_accept": r["unsafe_accept"],
                "false_reject": r["false_reject"], "fallback_used": r["fallback_used"],
                "applied": r["applied_action_source"], "mlu_applied": r["mlu_applied"],
                "sat_applied": r["satisfaction_applied"],
                "overload_applied": r["capacity_violations_applied"] > 0,
                "no_feasible_fallback": r["no_feasible_fallback"],
                "mlu_mean_pred": r["mlu_mean_predicted"], "mlu_ucb_pred": r["mlu_ucb_predicted"],
                "decision_ms": r["decision_runtime_ms"], "twin_ms": r["twin_runtime_ms"],
            } for r in rows},
        }
        append_jsonl(compact, compact_row)
        el = time.time() - t0
        if (i + 1) % 20 == 0 or i + 1 == len(cases) or name == "Kdl":
            eta = el / (i + 1) * (len(cases) - i - 1)
            print(f"[{i + 1}/{len(cases)}] {key} elapsed {fmt_eta(el)} eta {fmt_eta(eta)}", flush=True)


def _rate(vals: list[bool]) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def summarize(out: Path) -> dict[str, Any]:
    rows = read_jsonl(out / "case_rows.jsonl")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["policy_id"], r["topology"], r["load_mode"], r["load_level"])].append(r)
        groups[(r["policy_id"], r["topology"], r["load_mode"], "all")].append(r)
        groups[(r["policy_id"], r["topology"], "all", "all")].append(r)
    table = []
    for (pid, topo, mode, level), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        for v in "ABCDEF":
            vs = [r["variants"][v] for r in rs]
            safe = [not r["oracle_feasible_proposed"] for r in rs]  # proposal unsafe
            n_unsafe_prop = int(sum(safe))
            table.append({
                "policy_id": pid, "topology": topo, "load_mode": mode, "load_level": level,
                "variant": v, "n_cases": len(rs), "n_unsafe_proposals": n_unsafe_prop,
                "accept_rate": _rate([x["accept"] for x in vs]),
                "unsafe_accept_rate": _rate([x["unsafe_accept"] for x in vs]),
                "unsafe_accept_given_unsafe": _rate([x["unsafe_accept"] for x, u in zip(vs, safe) if u]),
                "false_reject_rate": _rate([x["false_reject"] for x in vs]),
                "false_reject_given_safe": _rate([x["false_reject"] for x, u in zip(vs, safe) if not u]),
                "fallback_rate": _rate([x["fallback_used"] for x in vs]),
                "no_feasible_fallback_rate": _rate([bool(x["no_feasible_fallback"]) for x in vs]),
                "applied_overload_rate": _rate([x["overload_applied"] for x in vs]),
                "mean_mlu_applied": float(np.mean([x["mlu_applied"] for x in vs])),
                "mean_sat_applied": float(np.mean([x["sat_applied"] for x in vs])),
                "mean_decision_ms": float(np.mean([x["decision_ms"] for x in vs])),
            })
    refs = []
    for (pid, topo, mode, level), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        def m(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        def feas(key):
            vals = [r[key] <= 1 + 1e-9 for r in rs if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        refs.append({
            "policy_id": pid, "topology": topo, "load_mode": mode, "load_level": level, "n_cases": len(rs),
            "mlu_policy": m("mlu_proposed"), "mlu_shortest": m("mlu_shortest"), "mlu_equal": m("mlu_equal"),
            "mlu_lp": m("mlu_lp"), "sat_policy": m("sat_proposed"), "sat_shortest": m("sat_shortest"),
            "sat_equal": m("sat_equal"), "sat_lp": m("sat_lp"),
            "feasible_policy": feas("mlu_proposed"), "feasible_shortest": feas("mlu_shortest"),
            "feasible_equal": feas("mlu_equal"), "feasible_lp": feas("mlu_lp"),
            "policy_over_shortest_mlu": float(np.mean([r["mlu_proposed"] / r["mlu_shortest"] for r in rs])),
            "mean_policy_ms": m("policy_ms"),
        })
    # D vs E discordance
    disc = []
    for (pid, topo, mode, level), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if level != "all":
            continue
        d_only = sum(r["variants"]["D"]["unsafe_accept"] and not r["variants"]["E"]["unsafe_accept"] for r in rs)
        e_only = sum(r["variants"]["E"]["unsafe_accept"] and not r["variants"]["D"]["unsafe_accept"] for r in rs)
        acc_diff = sum(r["variants"]["D"]["accept"] != r["variants"]["E"]["accept"] for r in rs)
        ef_diff = sum(r["variants"]["E"]["accept"] != r["variants"]["F"]["accept"] for r in rs)
        disc.append({"policy_id": pid, "topology": topo, "load_mode": mode, "n_cases": len(rs),
                     "D_only_unsafe_accepts": d_only, "E_only_unsafe_accepts": e_only,
                     "D_E_accept_discordant": acc_diff, "E_F_accept_discordant": ef_diff})
    summary = {"variant_table": table, "reference_table": refs, "d_vs_e": disc, "n_case_rows": len(rows)}
    write_json(out / "summary.json", summary)
    return summary


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def report(out: Path, summary: dict[str, Any]) -> None:
    vt = summary["variant_table"]
    names = {"A": "A policy only", "B": "B orig. mean", "C": "C orig. unc.", "D": "D imp. mean",
             "E": "E imp. cal. unc. (k=1)", "F": "F physics oracle"}
    lines = [
        "# E2 holdout evaluation — gate variants A–F on unseen GEANT and Kdl",
        "",
        "Frozen improved stack from `af_eval_k1_calibrated_devtopos_v1`: same policy (`recommended_stable`), "
        "same twins, same conformal scale, gate k=1 / utilisation limit 1.0, same shortest/equal "
        "least-violating fallback. **No recalibration, no threshold change, no training.** "
        "GEANT and Kdl were never used for training, tuning or calibration.",
        "",
        "- GEANT (22 nodes / 72 links, FIGRET real TMs): 200 seeded test TMs × 8 load levels.",
        f"- Kdl (754/1790): 1 test TM × 8 load levels. {TOY_TM_NOTE}",
        "- Load levels: fixed scales 0.25–1.0 and calibrated targets 0.5–1.1 (calibrated-load study; GEANT reference = LP "
        "MLU, Kdl reference = shortest MLU).",
        "- *Unsafe accept* = accepted proposal that overloads at least one link (physics ground truth). "
        "Rates are per case; `| unsafe` / `| safe` columns condition on the proposal.",
        "- Satisfaction = P1 definition (eqs. 23–28 adjustment, delivered / total demand).",
        f"- {LP_SOLVER_NOTE} LP is not run on Kdl.",
        "",
    ]
    for pid, topo in sorted({(r["policy_id"], r["topology"]) for r in vt}, key=lambda x: (x[0] != "improved_stable", x)):
        for mode in ("all", "fixed", "calibrated"):
            sub = [r for r in vt if r["policy_id"] == pid and r["topology"] == topo and r["load_mode"] == mode
                   and r["load_level"] == "all"]
            if not sub:
                continue
            n = sub[0]["n_cases"]
            nu = sub[0]["n_unsafe_proposals"]
            lines += [
                f"## [{pid}] {topo} — {'all load levels' if mode == 'all' else mode + ' load'} "
                f"(n = {n} cases, {nu} unsafe proposals)",
                "",
                "| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | "
                "Fallback | No feasible FB | Applied overload | Applied MLU | Applied sat. | ms/decision |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in sub:
                lines.append(
                    f"| {names[r['variant']]} | {_f(r['accept_rate'])} | {_f(r['unsafe_accept_rate'])} | "
                    f"{_f(r['unsafe_accept_given_unsafe'])} | {_f(r['false_reject_given_safe'])} | "
                    f"{_f(r['fallback_rate'])} | {_f(r['no_feasible_fallback_rate'])} | "
                    f"{_f(r['applied_overload_rate'])} | {_f(r['mean_mlu_applied'])} | "
                    f"{_f(r['mean_sat_applied'], 4)} | {_f(r['mean_decision_ms'], 0)} |"
                )
            lines.append("")
    lines += [
        "## Per load level: E (full framework) vs references",
        "",
        "| Policy | Topology | Mode | Level | n | Policy MLU | Shortest | Equal | LP | Policy/shortest | "
        "Policy feasible | E accept | E unsafe acc. | E false rej. | E applied MLU | E applied sat. | LP sat. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    refs = {(r["policy_id"], r["topology"], r["load_mode"], r["load_level"]): r for r in summary["reference_table"]}
    for r in vt:
        if r["variant"] != "E" or r["load_level"] == "all":
            continue
        ref = refs[(r["policy_id"], r["topology"], r["load_mode"], r["load_level"])]
        lines.append(
            f"| {r['policy_id']} | {r['topology']} | {r['load_mode']} | {r['load_level']} | {r['n_cases']} | {_f(ref['mlu_policy'])} | "
            f"{_f(ref['mlu_shortest'])} | {_f(ref['mlu_equal'])} | {_f(ref['mlu_lp'])} | "
            f"{_f(ref['policy_over_shortest_mlu'])} | {_f(ref['feasible_policy'], 2)} | {_f(r['accept_rate'], 2)} | "
            f"{_f(r['unsafe_accept_rate'], 2)} | {_f(r['false_reject_rate'], 2)} | {_f(r['mean_mlu_applied'])} | "
            f"{_f(r['mean_sat_applied'], 4)} | {_f(ref['sat_lp'], 4)} |"
        )
    lines += [
        "",
        "## Mean-only (D) vs uncertainty-aware (E) and E vs oracle (F)",
        "",
        "| Policy | Topology | Mode | n | D-only unsafe accepts | E-only unsafe accepts | D/E accept discordant | "
        "E/F accept discordant |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for d in summary["d_vs_e"]:
        lines.append(f"| {d['policy_id']} | {d['topology']} | {d['load_mode']} | {d['n_cases']} | {d['D_only_unsafe_accepts']} | "
                     f"{d['E_only_unsafe_accepts']} | {d['D_E_accept_discordant']} | {d['E_F_accept_discordant']} |")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `case_rows.jsonl`: one compact row per case with all six variants and the references.",
        "- `decisions.jsonl`: full decision records (same schema as the dev A–F run; large, not committed).",
        "- `summary.json`, `config_<policy_id>.json` (frozen stack: `improved_stable`; policy v2 test pass: "
        "`v2_calload_normscore`).",
    ]
    (out / "AF_HOLDOUT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topologies", nargs="+", default=["GEANT"], choices=["GEANT", "Kdl"])
    ap.add_argument("--out", default="e2_holdout_af")
    ap.add_argument("--policy-id", default=None)
    ap.add_argument("--checkpoint", default=None, help="override policy checkpoint (policy v2 pass)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke: run only the first N cases")
    args = ap.parse_args()
    out = out_dir(args.out)
    if not args.report_only:
        run(args.topologies, out, args.policy_id, args.checkpoint, args.limit)
    report(out, summarize(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
