"""Holdout A-F, link-failure and twin-error experiments on the SNDlib topologies (Abilene, Germany50).

Frozen stack, exactly as in the GEANT experiments; nothing is trained, recalibrated or tuned here:
  policy ``recommended_stable`` (``models/policy/improved/recommended_stable.pt``), the original and the
  improved twin ensembles, conformal std scale 0.955 (stored in the calibrated twin checkpoint), gate
  k = 1 / utilisation limit 1.0, frozen shortest/equal least-violating fallback
  (``af_eval_k1_calibrated_devtopos_v1/frozen_gate_improvements_config.json``). In the twin-error sweep the
  margin-control limits L of Dm are read from the GEANT/dev sweep (``e4_twin_error/summary.json``), where
  they were chosen on dev validation; they are not refitted.

Load: the calibrated targets 0.5 / 0.7 / 0.9 / 1.1 (alpha from
``data/processed/calibrated_load/sndlib_scaled_tm_index.json``; k = 4 min-MLU path LP reference).

Subcommands
  holdout   A-F on 200 TMs x 4 targets per topology, plus shortest, ECMP (equal split), LP (min-MLU),
            frozen P1 actor and the v2 actor on the same scaled matrices  -> results/holdout_sndlib/
  failures  0-3 failed directed links (seeds 100..1000), 50 TMs x 4 targets, paths rebuilt around the
            failed links (``failures.apply_failure``)                        -> results/link_failures_sndlib/
  twin      member-wise twin-error sweep sigma in {0, 0.05, 0.1, 0.2, 0.4}, 3 replicates, gates A, D, Dm,
            E (k = 1, frozen c), F                                           -> results/twin_error_sndlib/
  stats     Wilcoxon (Holm, Cliff's delta), McNemar (Holm), Friedman + Nemenyi -> results/stats_sndlib/
  figures   300-dpi PNG + PDF figures in each result folder
  report    Markdown reports in each result folder

Usage (from implementation/):
  python -m phase3_final.sndlib_eval holdout
  python -m phase3_final.sndlib_eval failures
  python -m phase3_final.sndlib_eval twin
  python -m phase3_final.sndlib_eval stats
  python -m phase3_final.sndlib_eval figures
  python -m phase3_final.sndlib_eval report
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from framework.general_policy import GeneralPolicy
from framework.policy_adapter import FrozenP1Policy
from safety.improvements.decision import VARIANT_IDS, decide_variant
from safety.improvements.fallback import select_physics_fallback
from safety.improvements.graph_cache import TopologyGraphCache
from safety.improvements.oracle import physics_violations
from safety.improvements.policy_backend import policy_backend_from_config
from safety.improvements.run_eval import evaluate_paired_case
from safety.improvements.twin_backend import backends_from_config
from twin.data import ROOT, load_split_demands, load_topology
from twin.improvements.infer import load_ensemble_checkpoint
from twin.physics import od_demands

from . import routing as R
from .common import LP_SOLVER_NOTE, append_jsonl, environment, fmt_eta, read_jsonl, write_json
from .failures import apply_failure, load_cases, routable_matrix
from .holdout_af import FROZEN_CONFIG, SHORT, holdout_config
from .holdout_af import summarize as summarize_af
from .twin_error_sweep import IMPROVED, NOISE_SEED, REPLICATES, SIGMAS, gate_accepts, member_stack, noise, perturbed

TOPOS = ("Abilene", "Germany50")
TARGETS = (0.5, 0.7, 0.9, 1.1)
RESULTS = ROOT / "results"
HOLD = RESULTS / "holdout_sndlib"
LF = RESULTS / "link_failures_sndlib"
TE = RESULTS / "twin_error_sndlib"
ST = RESULTS / "stats_sndlib"
CAL = ROOT / "data" / "processed" / "calibrated_load" / "sndlib_scaled_tm_index.json"
QUALITY = ROOT / "data" / "raw" / "sndlib" / "QUALITY_REPORT.md"
F3 = RESULTS / "phase3_final"
V2 = ROOT / "models" / "policy_v2" / "v2_calload_normscore" / "best.pt"
STABLE = ROOT / "models" / "policy" / "improved" / "recommended_stable.pt"
FROZEN_P1 = ROOT / "models" / "baseline_p1" / "b4_freeze" / "grl_te_b4_freeze_seed100.pt"
E4_SUMMARY = F3 / "e4_twin_error" / "summary.json"
LF_VARIANTS = {"A": "A_policy_only", "D": "D_improved_twin_mean", "E": "E_improved_twin_calibrated_uncertainty",
               "F": "F_physics_oracle"}
TOPO_NOTE = {"Abilene": "unseen topology with related traffic (same measurement campaign as the B4 training traffic)",
             "Germany50": "unseen topology, unrelated traffic (DFN 2005)"}


def cal_index() -> dict:
    return json.loads(CAL.read_text(encoding="utf-8"))


def scaled(cal: dict, name: str, pos: int, target: float) -> float:
    return float(cal["alpha"][f"{name}/test/{pos}"][f"{target:.1f}"])


def sndlib_config() -> dict:
    cfg = holdout_config(None, None)
    ev = cfg["evaluation"]
    ev["topology_authorization"] = {
        "included": list(TOPOS),
        "reason": "SNDlib holdout evaluation: single pass on additional unseen test topologies; frozen thresholds, "
                  "twins and conformal scale; no tuning or recalibration.",
    }
    ev["never_tune_on"] = ["GEANT", "Kdl", *TOPOS]
    return cfg


def load_actor(path: Path) -> GeneralPolicy:
    p = GeneralPolicy(torch.device("cpu"))
    p.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    p.model.eval()
    return p


def _method_metrics(topo, matrix, action) -> dict:
    load = R.loads(topo, matrix, action)
    return {"mlu": float(R.utilisation(load, topo.capacity).max()), "sat": R.satisfaction_p1(topo, matrix, action),
            "overload": bool(np.any(load > topo.capacity + 1e-6))}


# ------------------------------------------------------------------ holdout A-F
def run_holdout(names: list[str]) -> None:
    HOLD.mkdir(parents=True, exist_ok=True)
    cfg = sndlib_config()
    cal = cal_index()
    write_json(HOLD / "config.json", {
        "frozen_config_source": FROZEN_CONFIG.relative_to(ROOT), "gate": cfg["gate"], "variants": list(cfg["variants"]),
        "twins": cfg["twins"], "policy": cfg["policy"], "fallback": cfg["fallback"], "calibrated_targets": TARGETS,
        "calibration_index": CAL.relative_to(ROOT), "topology_authorization": cfg["evaluation"]["topology_authorization"],
        "baselines": {"frozen_p1": FROZEN_P1.relative_to(ROOT), "actor_v2": V2.relative_to(ROOT)},
        "lp_solver": LP_SOLVER_NOTE, "environment": environment()})
    device = torch.device("cpu")
    policy = policy_backend_from_config(cfg, device=device, graph_cache=TopologyGraphCache(device))
    policy.load()
    pid = policy.checkpoint_id
    backends = backends_from_config(cfg, device)
    frozen = FrozenP1Policy(FROZEN_P1)
    v2 = load_actor(V2)
    rows_path = HOLD / "case_rows.jsonl"
    done = {r["case_key"] for r in read_jsonl(rows_path)}
    for name in names:
        topo = load_topology(name)
        demands, meta = load_split_demands("test", name)
        sp, eq = R.shortest_action(topo), R.equal_action(topo)
        cases = [(pos, t) for pos in cal["positions_200"][name] for t in TARGETS
                 if f"{name}|pos={pos}|calibrated={t}|policy={pid}" not in done]
        print(f"{name}: {len(cases)} cases to run", flush=True)
        lp_cache: dict[int, tuple[np.ndarray, float]] = {}
        t0 = time.time()
        for i, (pos, target) in enumerate(cases):
            scale = scaled(cal, name, pos, target)
            base = demands[pos]
            matrix = base * scale
            if pos not in lp_cache:
                tl = time.perf_counter()
                lp_cache[pos] = (R.lp_min_mlu(topo, base)["action"], (time.perf_counter() - tl) * 1000)
            lp_action, lp_ms = lp_cache[pos]
            case = {"seed": 2026, "topology": name, "split": "test", "local_index": int(pos),
                    "matrix_id": int(meta["matrix_indices"][pos]), "scale": scale, "load_mode": "calibrated",
                    "load_level": float(target), "unseen": True}
            tp = time.perf_counter()
            proposal = policy.propose(topo, matrix)
            policy_ms = (time.perf_counter() - tp) * 1000.0
            rows = evaluate_paired_case(case=case, topo=topo, matrix=matrix, policy_action=proposal.action,
                                        policy_meta=proposal.to_log_dict(), backends=backends, cfg=cfg,
                                        variants=VARIANT_IDS)
            tf = time.perf_counter()
            fp1 = frozen.propose(topo, od_demands(matrix))
            fp1_ms = (time.perf_counter() - tf) * 1000.0
            tv = time.perf_counter()
            with torch.no_grad():
                av2 = v2.propose(topo, od_demands(matrix))
            v2_ms = (time.perf_counter() - tv) * 1000.0
            methods = {"shortest": _method_metrics(topo, matrix, sp), "equal": _method_metrics(topo, matrix, eq),
                       "lp": _method_metrics(topo, matrix, lp_action), "frozen_p1": _method_metrics(topo, matrix, fp1),
                       "actor_stable": _method_metrics(topo, matrix, proposal.action),
                       "actor_v2": _method_metrics(topo, matrix, av2)}
            sat = {"policy": methods["actor_stable"]["sat"], "shortest": methods["shortest"]["sat"],
                   "equal": methods["equal"]["sat"]}
            key = f"{name}|pos={pos}|calibrated={target}|policy={pid}"
            append_jsonl(rows_path, {
                "case_key": key, "topology": name, "position": int(pos), "matrix_id": case["matrix_id"],
                "matrix_time": meta["matrix_times"][pos], "load_mode": "calibrated", "load_level": float(target),
                "scale": scale, "policy_id": pid, "policy_ms": policy_ms, "mlu_proposed": rows[0]["mlu_proposed"],
                "sat_proposed": sat["policy"], "oracle_feasible_proposed": rows[0]["oracle_feasible_proposed"],
                "mlu_shortest": methods["shortest"]["mlu"], "mlu_equal": methods["equal"]["mlu"],
                "mlu_lp": methods["lp"]["mlu"], "sat_shortest": methods["shortest"]["sat"],
                "sat_equal": methods["equal"]["sat"], "sat_lp": methods["lp"]["sat"],
                "methods": methods, "ms": {"lp": lp_ms, "frozen_p1": fp1_ms, "actor_v2": v2_ms, "actor_stable": policy_ms},
                "variants": {SHORT[r["gate_variant"]]: {
                    "accept": r["acceptance"], "unsafe_accept": r["unsafe_accept"], "false_reject": r["false_reject"],
                    "fallback_used": r["fallback_used"], "applied": r["applied_action_source"],
                    "mlu_applied": r["mlu_applied"], "sat_applied": sat.get(r["applied_action_source"]),
                    "overload_applied": r["capacity_violations_applied"] > 0,
                    "no_feasible_fallback": r["no_feasible_fallback"], "mlu_mean_pred": r["mlu_mean_predicted"],
                    "mlu_ucb_pred": r["mlu_ucb_predicted"], "decision_ms": r["decision_runtime_ms"],
                    "twin_ms": r["twin_runtime_ms"]} for r in rows},
            })
            if (i + 1) % 50 == 0 or i + 1 == len(cases):
                el = time.time() - t0
                print(f"{name} [{i + 1}/{len(cases)}] elapsed {fmt_eta(el)} eta {fmt_eta(el / (i + 1) * (len(cases) - i - 1))}",
                      flush=True)
        policy.graph_cache.clear(name)
        frozen.release(name)
        v2.release(name)


def summarize_holdout() -> dict:
    summary = summarize_af(HOLD)  # same variant / reference tables as the GEANT holdout (writes summary.json)
    rows = read_jsonl(HOLD / "case_rows.jsonl")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["topology"], r["load_level"])].append(r)
        groups[(r["topology"], "all")].append(r)
    base = []
    for (topo, lvl), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        for m in ("shortest", "equal", "lp", "frozen_p1", "actor_stable", "actor_v2"):
            vs = [r["methods"][m] for r in rs]
            ms = [r["ms"].get(m) for r in rs if r["ms"].get(m) is not None]
            base.append({"topology": topo, "load_level": lvl, "method": m, "n": len(vs),
                         "sat_mean": float(np.mean([v["sat"] for v in vs])),
                         "mlu_mean": float(np.mean([min(v["mlu"], 1e6) for v in vs])),
                         "overload_rate": float(np.mean([v["overload"] for v in vs])),
                         "median_ms": float(np.median(ms)) if ms else (0.0 if m in ("shortest", "equal") else None)})
    summary["baseline_table"] = base
    write_json(HOLD / "summary.json", summary)
    return summary


# ------------------------------------------------------------------ link failures
LF_LEVELS = TARGETS


def run_failures(names: list[str]) -> None:
    LF.mkdir(parents=True, exist_ok=True)
    cfg = sndlib_config()
    backends = backends_from_config(cfg, torch.device("cpu"))
    stable = load_actor(STABLE)
    frozen = FrozenP1Policy(FROZEN_P1)
    cal = cal_index()
    rows_path = LF / "case_rows.jsonl"
    done = {r["key"] for r in read_jsonl(rows_path)}
    write_json(LF / "config.json", {"policy": STABLE.relative_to(ROOT), "frozen_p1": FROZEN_P1.relative_to(ROOT),
                                    "calibrated_targets": LF_LEVELS, "gate": cfg["gate"], "variants": LF_VARIANTS,
                                    "tms": {n: cal["e3_positions_50"][n] for n in names},
                                    "failure_cases": "data/scenarios/link_failures/<topology>.json",
                                    "lp_solver": LP_SOLVER_NOTE, "environment": environment()})
    for name in names:
        base_topo = load_topology(name)
        demands = load_split_demands("test", name)[0]
        positions = cal["e3_positions_50"][name]
        cases = load_cases(name)
        todo = [(c, pos, lvl) for c in cases for pos in positions for lvl in LF_LEVELS
                if f"{c.key}|pos={pos}|calibrated={lvl}" not in done]
        print(f"{name}: {len(todo)} cases to run", flush=True)
        t0, k = time.time(), 0
        for case in cases:
            mine = [x for x in todo if x[0] is case]
            if not mine:
                continue
            ft = apply_failure(base_topo, case)
            topo = ft.topo
            sp, eq = R.shortest_action(topo), R.equal_action(topo)
            lp_cache: dict[int, np.ndarray | None] = {}
            for _c, pos, lvl in mine:
                scale = scaled(cal, name, pos, lvl)
                original = demands[pos] * scale
                routed = routable_matrix(ft, original)
                if pos not in lp_cache:
                    lp_cache[pos] = R.lp_min_mlu(topo, routable_matrix(ft, demands[pos]))["action"]
                row = {"key": f"{case.key}|pos={pos}|calibrated={lvl}", "topology": name, "seed": case.seed,
                       "n_failed": case.n_failed, "failed_edges": case.edges, "position": pos, "load_mode": "calibrated",
                       "load_level": lvl, "scale": scale, "unservable_pairs": int(ft.unservable_pairs.size),
                       "unservable_demand_frac": 1 - float(od_demands(routed).sum()) / float(od_demands(original).sum()),
                       "pairs_rerouted": ft.n_pairs_rerouted, "methods": {}}
                acts = {"shortest": sp, "equal": eq, "lp": lp_cache[pos]}
                tp = time.perf_counter()
                acts["frozen_p1"] = frozen.propose(topo, od_demands(routed))
                row["frozen_p1_ms"] = (time.perf_counter() - tp) * 1000
                tp = time.perf_counter()
                with torch.no_grad():
                    acts["actor_stable"] = stable.propose(topo, od_demands(routed))
                row["actor_stable_ms"] = (time.perf_counter() - tp) * 1000
                for mname, a in acts.items():
                    if a is None:
                        continue
                    load = R.loads(topo, routed, a)
                    row["methods"][mname] = {"mlu": float(R.utilisation(load, topo.capacity).max()),
                                             "sat": sat_eff(topo, routed, original, a),
                                             "overload": bool(np.any(load > topo.capacity + 1e-6))}
                pa = acts["actor_stable"]
                for short, vid in LF_VARIANTS.items():
                    rec = decide_variant(vid, topo=topo, matrix=routed, policy_action=pa, backends=backends, cfg=cfg)
                    applied = {"policy": pa, "shortest": sp, "equal": eq}[rec["applied_action_source"]]
                    row["methods"][f"fw_stable_{short}"] = {
                        "accept": rec["acceptance"], "unsafe_accept": rec["unsafe_accept"],
                        "false_reject": rec["false_reject"], "fallback": rec["fallback_used"],
                        "applied": rec["applied_action_source"], "mlu": rec["mlu_applied"],
                        "sat": sat_eff(topo, routed, original, applied),
                        "overload": rec["capacity_violations_applied"] > 0,
                        "proposal_unsafe": not rec["oracle_feasible_proposed"],
                        "decision_ms": rec["decision_runtime_ms"]}
                append_jsonl(rows_path, row)
                k += 1
                if k % 200 == 0 or k == len(todo):
                    el = time.time() - t0
                    print(f"{name} [{k}/{len(todo)}] elapsed {fmt_eta(el)} eta {fmt_eta(el / k * (len(todo) - k))}", flush=True)
            for p in (stable, frozen):
                p.release(topo.name)


def sat_eff(topo, routed, original, action) -> float:
    total = float(od_demands(original).sum())
    return R.satisfaction_p1(topo, routed, action) * float(od_demands(routed).sum()) / total if total > 0 else 1.0


def summarize_failures() -> dict:
    rows = read_jsonl(LF / "case_rows.jsonl")
    groups = defaultdict(list)
    for r in rows:
        for lvl in (r["load_level"], "all"):
            groups[(r["topology"], lvl, r["n_failed"])].append(r)
            if r["load_level"] in (0.7, 0.9):
                groups[(r["topology"], "0.7+0.9", r["n_failed"])].append(r)
    table = []
    for (topo, lvl, nf), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        for m in sorted({m for r in rs for m in r["methods"]}):
            vs = [r["methods"][m] for r in rs if m in r["methods"]]
            rec = {"topology": topo, "load_level": lvl, "n_failed": nf, "method": m, "n": len(vs),
                   "sat_mean": float(np.mean([v["sat"] for v in vs])),
                   "mlu_mean": float(np.mean([min(v["mlu"], 1e6) for v in vs])),
                   "violation_rate": float(np.mean([v["overload"] for v in vs]))}
            if m.startswith("fw_"):
                unsafe = [v for v in vs if v["proposal_unsafe"]]
                safe = [v for v in vs if not v["proposal_unsafe"]]
                rec.update({"accept_rate": float(np.mean([v["accept"] for v in vs])),
                            "unsafe_accept_rate": float(np.mean([v["unsafe_accept"] for v in vs])),
                            "unsafe_accept_given_unsafe": float(np.mean([v["accept"] for v in unsafe])) if unsafe else None,
                            "false_reject_given_safe": float(np.mean([v["false_reject"] for v in safe])) if safe else None,
                            "fallback_rate": float(np.mean([v["fallback"] for v in vs])),
                            "n_unsafe_proposals": len(unsafe),
                            "median_decision_ms": float(np.median([v["decision_ms"] for v in vs]))})
            table.append(rec)
    unserv = defaultdict(list)
    for r in rows:
        unserv[(r["topology"], r["n_failed"])].append(r["unservable_demand_frac"])
    out = {"table": table, "n_rows": len(rows), "unservable": {f"{t}|{n}": float(np.mean(v)) for (t, n), v in unserv.items()},
           "pairs_rerouted_mean": {f"{t}|{n}": float(np.mean([r["pairs_rerouted"] for r in rows if r["topology"] == t and r["n_failed"] == n]))
                                   for (t, n) in unserv}}
    write_json(LF / "summary.json", out)
    return out


# ------------------------------------------------------------------ twin-error sweep
def build_twin_cache(names: list[str]) -> None:
    cache = TE / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    cfg = sndlib_config()
    device = torch.device("cpu")
    policy = policy_backend_from_config(cfg, device=device, graph_cache=TopologyGraphCache(device))
    policy.load()
    model, payload = load_ensemble_checkpoint(IMPROVED, device)
    cal = cal_index()
    for name in names:
        target = cache / f"{name}.npz"
        if target.exists():
            print(f"{name}: cache exists")
            continue
        topo = load_topology(name)
        demands = load_split_demands("test", name)[0]
        cases = [{"topology": name, "split": "test", "position": int(p), "load_mode": "calibrated", "load_level": t,
                  "scale": scaled(cal, name, p, t)} for p in cal["positions_200"][name] for t in TARGETS]
        n, e = len(cases), topo.n_links
        stack = np.zeros((n, 5, e))
        true_load, fb_load = np.zeros((n, e)), np.zeros((n, e))
        fb_feasible = np.zeros(n, dtype=bool)
        sat_prop, sat_fb, policy_ms, twin_ms = (np.zeros(n) for _ in range(4))
        t0 = time.time()
        for i, c in enumerate(cases):
            m = demands[c["position"]] * c["scale"]
            tp = time.perf_counter()
            action = policy.propose(topo, m).action
            policy_ms[i] = (time.perf_counter() - tp) * 1000
            true_load[i] = R.loads(topo, m, action)
            sat_prop[i] = R.satisfaction_p1(topo, m, action)
            fb = select_physics_fallback(topo, m)
            fb_load[i], fb_feasible[i] = fb.selected_load, fb.feasible_fallback
            sat_fb[i] = R.satisfaction_p1(topo, m, fb.selected_action)
            tt = time.perf_counter()
            stack[i] = member_stack(model, topo, m, action)
            twin_ms[i] = (time.perf_counter() - tt) * 1000
            if (i + 1) % 100 == 0 or i + 1 == n:
                el = time.time() - t0
                print(f"{name} [{i + 1}/{n}] elapsed {fmt_eta(el)} eta {fmt_eta(el / (i + 1) * (n - i - 1))}", flush=True)
        np.savez_compressed(target, capacity=topo.capacity, true_load=true_load, fb_load=fb_load, fb_feasible=fb_feasible,
                            sat_prop=sat_prop, sat_fb=sat_fb, policy_ms=policy_ms, twin_ms=twin_ms,
                            meta=json.dumps(cases), stack_improved=stack,
                            conformal=json.dumps({"improved": float(payload.get("conformal_std_scale", 1.0))}))
        policy.graph_cache.clear(name)


def run_twin_sweep(names: list[str]) -> None:
    e4 = json.loads(E4_SUMMARY.read_text(encoding="utf-8"))
    limits = {float(k.split("|")[1]): v["L"] for k, v in e4["margin_control_L"].items() if k.startswith("improved|")}
    rows = []
    for name in names:
        z = np.load(TE / "cache" / f"{name}.npz", allow_pickle=False)
        cases = json.loads(str(z["meta"]))
        c_frozen = json.loads(str(z["conformal"]))["improved"]
        cap, stack, true_all = z["capacity"], z["stack_improved"], z["true_load"]
        for sigma in SIGMAS:
            for rep in range(REPLICATES if sigma > 0 else 1):
                for i, case in enumerate(cases):
                    st = stack[i]
                    mu, sd = perturbed(st, sigma, noise(i, name, rep, st.shape))
                    true = true_all[i]
                    unsafe = physics_violations(true, cap) > 0
                    fb_over = physics_violations(z["fb_load"][i], cap) > 0
                    pred_mlu = float(np.max(mu / cap))
                    acc = {"A": True, "D": gate_accepts(mu, sd, cap, 0.0), "Dm": pred_mlu <= limits[sigma],
                           "E": gate_accepts(mu, c_frozen * sd, cap, 1.0), "F": not unsafe}
                    mlu_p, mlu_fb = float(np.max(true / cap)), float(np.max(z["fb_load"][i] / cap))
                    for v, a in acc.items():
                        rows.append({"topology": name, "sigma": sigma, "rep": rep, "case": i, "load_level": case["load_level"],
                                     "variant": v, "unsafe_proposal": bool(unsafe), "accept": bool(a),
                                     "unsafe_accept": bool(a and unsafe), "false_reject": bool((not a) and (not unsafe)),
                                     "applied_mlu": mlu_p if a else mlu_fb, "applied_overload": bool(unsafe if a else fb_over),
                                     "applied_sat": float(z["sat_prop"][i] if a else z["sat_fb"][i]),
                                     "pred_mlu_mean": pred_mlu, "true_mlu": mlu_p})
    with gzip.open(TE / "paired_rows.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        for scope in (r["topology"], "sndlib_pooled"):
            groups[(scope, r["sigma"], r["variant"])].append(r)
    table = []
    for (scope, sigma, v), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        unsafe = [r for r in rs if r["unsafe_proposal"]]
        safe = [r for r in rs if not r["unsafe_proposal"]]
        mean = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else None  # noqa: E731
        table.append({"scope": scope, "sigma": sigma, "variant": v, "n": len(rs), "n_unsafe_prop": len(unsafe),
                      "accept_rate": mean(rs, "accept"), "unsafe_accept_rate": mean(rs, "unsafe_accept"),
                      "unsafe_accept_given_unsafe": mean(unsafe, "accept"),
                      "false_reject_rate": mean(rs, "false_reject"), "false_reject_given_safe": mean(safe, "false_reject"),
                      "applied_mlu": mean(rs, "applied_mlu"), "applied_overload_rate": mean(rs, "applied_overload"),
                      "applied_sat": mean(rs, "applied_sat")})
    with (TE / "sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    margin = {}
    for name in names:
        z = np.load(TE / "cache" / f"{name}.npz", allow_pickle=False)
        mlu = np.max(z["true_load"] / z["capacity"], axis=1)
        margin[name] = {"n_cases": int(mlu.size), "share_unsafe": float(np.mean([physics_violations(t, z["capacity"]) > 0 for t in z["true_load"]])),
                        "share_true_mlu_in_[0.9,1.1]": float(np.mean((mlu >= 0.9) & (mlu <= 1.1))),
                        "median_policy_ms": float(np.median(z["policy_ms"])), "median_twin_ms": float(np.median(z["twin_ms"])),
                        "frozen_c": json.loads(str(z["conformal"]))["improved"]}
    write_json(TE / "summary.json", {"table": table, "margin_control_L_frozen": limits, "proposal_margin": margin})
    write_json(TE / "config.json", {"sigmas": SIGMAS, "replicates": REPLICATES, "noise_seed": NOISE_SEED,
                                    "twin": IMPROVED.relative_to(ROOT), "variants": ["A", "D", "Dm", "E", "F"],
                                    "dm_limits_source": E4_SUMMARY.relative_to(ROOT), "dm_limits": limits,
                                    "calibrated_targets": TARGETS, "environment": environment()})


# ------------------------------------------------------------------ statistics
def statistics() -> dict:
    from scipy import stats as sps

    from .stats_final import cliffs_delta, friedman_nemenyi, holm, mcnemar_row, wilcoxon_row

    ST.mkdir(parents=True, exist_ok=True)
    new = read_jsonl(HOLD / "case_rows.jsonl")
    e2 = [r for r in read_jsonl(F3 / "e2_holdout_af" / "case_rows.jsonl") if r["policy_id"] == "improved_stable"]
    calib = lambda rs: [r for r in rs if r["load_mode"] == "calibrated"]  # noqa: E731
    pools = {"Abilene": [r for r in new if r["topology"] == "Abilene"],
             "Germany50": [r for r in new if r["topology"] == "Germany50"],
             "GEANT (calibrated)": [r for r in calib(e2) if r["topology"] == "GEANT"],
             "Kdl (calibrated)": [r for r in calib(e2) if r["topology"] == "Kdl"]}
    pools["SNDlib pooled (Abilene + Germany50)"] = pools["Abilene"] + pools["Germany50"]
    pools["all unseen pooled (GEANT + Kdl + Abilene + Germany50, calibrated)"] = (
        pools["GEANT (calibrated)"] + pools["Kdl (calibrated)"] + pools["Abilene"] + pools["Germany50"])
    wil = []
    for scope, rs in pools.items():
        if scope.startswith("Kdl"):
            continue  # 4 cases; reported only inside the pooled test
        for metric, hib in (("sat", True), ("mlu", False)):
            key = "sat_applied" if metric == "sat" else "mlu_applied"
            x = [min(r["variants"]["E"][key], 1e6) for r in rs]
            y = [min(r["variants"]["A"][key], 1e6) for r in rs]
            wil.append(wilcoxon_row(f"framework (E) vs policy only (A) · {scope} · applied {metric}", x, y, hib))
        x = [r["variants"]["E"]["overload_applied"] for r in rs]
        y = [r["variants"]["A"]["overload_applied"] for r in rs]
        mc = mcnemar_row(f"applied overload framework (E) vs policy only (A) · {scope}", x, y)
        mc["family"] = "overload"
        wil.append({**mc, "n": mc["n_pairs"], "mean_diff": float(np.mean(x) - np.mean(y)), "cliffs_delta": None})
    adj = holm([r["p"] for r in wil])
    for r, a in zip(wil, adj):
        r["p_holm"], r["significant_holm_0.05"] = a, a < 0.05

    # McNemar D vs E on the twin-error sweep
    mcn = []
    by = defaultdict(lambda: defaultdict(dict))
    with gzip.open(TE / "paired_rows.csv.gz", "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by[(r["topology"], float(r["sigma"]))][(r["rep"], r["case"])][r["variant"]] = r
    pooled = defaultdict(list)
    for (topo, sigma), cases in sorted(by.items()):
        vals = list(cases.values())
        pooled[sigma].extend(vals)
        if sigma == 0:
            continue
        for ref in ("D", "Dm"):
            for what in ("unsafe_accept", "false_reject"):
                mcn.append(mcnemar_row(f"{what.replace('_', ' ')} {ref} vs E · {topo} · sigma={sigma}",
                                       [v[ref][what] == "True" for v in vals], [v["E"][what] == "True" for v in vals]))
    for sigma, vals in sorted(pooled.items()):
        if sigma == 0:
            continue
        for what in ("unsafe_accept", "false_reject"):
            mcn.append(mcnemar_row(f"{what.replace('_', ' ')} D vs E · SNDlib pooled · sigma={sigma}",
                                   [v["D"][what] == "True" for v in vals], [v["E"][what] == "True" for v in vals]))
    adj = holm([r["p"] for r in mcn])
    for r, a in zip(mcn, adj):
        r["p_holm"], r["significant_holm_0.05"] = a, a < 0.05

    # Friedman + Nemenyi with the new topologies added to the existing grid
    grid = read_jsonl(F3 / "stats" / "main_grid_rows.jsonl")
    for r in new:
        grid.append({"key": r["case_key"], "topology": r["topology"], "load_mode": r["load_mode"],
                     "load_level": r["load_level"], "position": r["position"],
                     "methods": {**{m: r["methods"][m] for m in ("shortest", "equal", "lp", "frozen_p1", "actor_stable", "actor_v2")},
                                 "framework": {"mlu": r["variants"]["E"]["mlu_applied"], "sat": r["variants"]["E"]["sat_applied"],
                                               "overload": r["variants"]["E"]["overload_applied"]}}})
    fr_all = friedman_nemenyi(grid)
    fr_cal = friedman_nemenyi([r for r in grid if r["load_mode"] == "calibrated"])
    fr_unseen_cal = friedman_nemenyi([r for r in grid if r["load_mode"] == "calibrated"
                                      and r["topology"] in ("GEANT", "Kdl", "Abilene", "Germany50")])
    fr_prev = json.loads((F3 / "stats" / "friedman_nemenyi.json").read_text(encoding="utf-8"))
    out = {"wilcoxon_holm": wil, "mcnemar_holm": mcn,
           "friedman": {"all_blocks_with_sndlib": fr_all, "calibrated_blocks_with_sndlib": fr_cal,
                        "unseen_calibrated_blocks": fr_unseen_cal, "previous_without_sndlib": fr_prev},
           "notes": ["GEANT and Kdl rows are the frozen-stack rows of results/phase3_final/e2_holdout_af/case_rows.jsonl, "
                     "restricted to the calibrated load levels so that every topology contributes the same four levels.",
                     "Kdl (1 TM x 4 levels) is not tested on its own; it enters the pooled test.",
                     "Holm correction is applied within each family (Wilcoxon + overload McNemar; twin-error McNemar)."]}
    write_json(ST / "statistics.json", out)
    return out


# ------------------------------------------------------------------ figures
def _plt():
    from .figures import _plt as base_plt
    return base_plt()


def _save(fig, folder: Path, name: str) -> None:
    (folder / "figures").mkdir(parents=True, exist_ok=True)
    if not fig.legends:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / "figures" / f"{name}.{ext}", dpi=300, facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


def figures() -> None:
    from .figures import COL, MRK, MUTED, line

    plt = _plt()
    # 1. Holdout: satisfaction and unsafe accept, A vs E, per topology (GEANT calibrated as reference)
    new = read_jsonl(HOLD / "case_rows.jsonl")
    e2 = [r for r in read_jsonl(F3 / "e2_holdout_af" / "case_rows.jsonl")
          if r["policy_id"] == "improved_stable" and r["load_mode"] == "calibrated" and r["topology"] == "GEANT"]
    panels = (("GEANT", e2), ("Abilene", [r for r in new if r["topology"] == "Abilene"]),
              ("Germany50", [r for r in new if r["topology"] == "Germany50"]))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
    for col, (topo, rs) in enumerate(panels):
        for row, (metric, ylab) in enumerate((("sat_applied", "applied demand satisfaction"),
                                               ("unsafe_accept", "unsafe accepts (share of cases)"))):
            ax = axes[row, col]
            for v, key, lab in (("A", "policy", "A policy only"), ("E", "framework", "E framework (gate k=1)")):
                pts = defaultdict(list)
                for r in rs:
                    pts[r["load_level"]].append(float(r["variants"][v][metric]))
                xs = sorted(pts)
                line(ax, xs, [np.mean(pts[x]) for x in xs], key, lab)
            if row == 0:
                pts = defaultdict(list)
                for r in rs:
                    if r.get("sat_lp") is not None:
                        pts[r["load_level"]].append(r["sat_lp"])
                xs = sorted(pts)
                line(ax, xs, [np.mean(pts[x]) for x in xs], "lp", "LP (min-MLU)", ls="--", lw=1.3)
                ax.set_title(f"{topo}{' (reference, calibrated levels)' if topo == 'GEANT' else ''}", fontsize=10, loc="left")
            else:
                ax.set_ylim(-0.03, 1.05)
                ax.set_xlabel("calibrated load target")
            if col == 0:
                ax.set_ylabel(ylab)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, HOLD, "holdout_sat_unsafe_A_vs_E")

    # 2. Link failures
    t = json.loads((LF / "summary.json").read_text(encoding="utf-8"))["table"]
    nf = [0, 1, 2, 3]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for col, topo in enumerate(TOPOS):
        ax = axes[0, col]
        for m, key, lab, kw in (("shortest", "shortest", "shortest", {"ls": "--", "lw": 1.4}),
                                ("equal", "equal", "ECMP (equal split)", {"lw": 1.4}),
                                ("lp", "lp", "LP (min-MLU)", {}), ("frozen_p1", "frozen_p1", "frozen P1 actor", {"lw": 1.4}),
                                ("actor_stable", "policy", "policy only (A)", {}),
                                ("fw_stable_E", "framework", "framework (gate E)", {"lw": 2.6, "zorder": 4})):
            y = [next(r for r in t if r["topology"] == topo and r["load_level"] == "all" and r["n_failed"] == n
                      and r["method"] == m)["sat_mean"] * 100 for n in nf]
            line(ax, nf, y, key, lab, **kw)
        ax.set_title(f"{topo}: satisfaction", fontsize=10, loc="left")
        ax.set_ylabel("demand satisfaction (%)")
        ax = axes[1, col]
        for m, key, lab in (("fw_stable_A", "A", "A policy only"), ("fw_stable_D", "D", "D mean-only gate"),
                            ("fw_stable_E", "E1", "E gate k=1")):
            y = [next(r for r in t if r["topology"] == topo and r["load_level"] == "all" and r["n_failed"] == n
                      and r["method"] == m)["unsafe_accept_rate"] for n in nf]
            line(ax, nf, y, key, lab)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{topo}: unsafe accepts", fontsize=10, loc="left")
        ax.set_ylabel("unsafe accepts (share of cases)")
        ax.set_xlabel("failed directed links")
        ax.set_xticks(nf)
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, LF, "link_failure_curves")

    # 3. Twin-error curves
    rows = list(csv.DictReader(open(TE / "sweep.csv", encoding="utf-8")))
    labels = {"D": "D mean-only", "Dm": "Dm mean-only + margin (dev-tuned L)", "E": "E k=1 (frozen c)", "F": "F oracle"}
    keys = {"D": "D", "Dm": "Dm", "E": "E1", "F": "F"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
    for col, topo in enumerate(TOPOS):
        for row, (metric, ylab) in enumerate((("unsafe_accept_given_unsafe", "unsafe accepts / unsafe proposals"),
                                               ("false_reject_given_safe", "false rejects / safe proposals"))):
            ax = axes[row, col]
            for v in ("D", "Dm", "E", "F"):
                pts = sorted((float(r["sigma"]), float(r[metric])) for r in rows if r["scope"] == topo and r["variant"] == v
                             and r[metric] not in ("", "None"))
                if pts:
                    line(ax, [p[0] for p in pts], [p[1] for p in pts], keys[v], labels[v], lw=1.6)
            ax.set_ylabel(ylab)
            ax.set_title(f"{topo}: {'unsafe accepts' if row == 0 else 'false rejects'}", fontsize=10, loc="left")
            if row == 1:
                ax.set_xlabel("member-wise twin error σ")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, TE, "twin_error_curves")
    _ = (COL, MRK, MUTED)


# ------------------------------------------------------------------ reports
def _f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def _p(x):
    if x is None:
        return "–"
    return f"{x:.2g}" if x < 1e-3 else f"{x:.3f}"


def report_holdout() -> None:
    s = json.loads((HOLD / "summary.json").read_text(encoding="utf-8"))
    vt, bt = s["variant_table"], s["baseline_table"]
    names = {"A": "A policy only", "B": "B orig. twin, mean", "C": "C orig. twin, μ+σ", "D": "D improved twin, mean",
             "E": "E improved twin, calibrated μ+σ (k=1)", "F": "F physics oracle"}
    L = ["# Holdout A–F evaluation on SNDlib Abilene and Germany50", "",
         "Frozen stack of the GEANT holdout evaluation (`results/phase3_final/e2_holdout_af/`): policy "
         "`recommended_stable`, original and improved twin ensembles, conformal std scale 0.955, gate k = 1 with "
         "utilisation limit 1.0, shortest/ECMP least-violating fallback. Nothing was trained, recalibrated or tuned on "
         "these topologies.", "",
         "- Abilene (12 nodes / 30 directed links): " + TOPO_NOTE["Abilene"] + ". Germany50 (50 / 176): " + TOPO_NOTE["Germany50"] + ".",
         "- Cases: 200 seeded real traffic matrices per topology × calibrated targets 0.5 / 0.7 / 0.9 / 1.1 "
         "(alpha from the k = 4 min-MLU LP) = 800 cases per topology.",
         "- *Unsafe accept* = accepted proposal that overloads at least one link (physics ground truth). "
         "*False reject* = rejected proposal that was safe. Satisfaction = P1 definition (eqs. 23–28 adjustment).",
         f"- {LP_SOLVER_NOTE}", "",
         "Data: `data/raw/sndlib/QUALITY_REPORT.md`; calibration: `data/processed/calibrated_load/sndlib_scaled_tm_index.json`.", ""]
    for topo in TOPOS:
        sub = [r for r in vt if r["topology"] == topo and r["load_mode"] == "all" and r["load_level"] == "all"]
        if not sub:
            continue
        L += [f"## {topo} — all calibrated levels (n = {sub[0]['n_cases']} cases, {sub[0]['n_unsafe_proposals']} unsafe proposals)", "",
              "| Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False reject given safe | Fallback | "
              "No feasible fallback | Applied overload | Applied MLU | Applied sat. | ms/decision |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sub:
            L.append(f"| {names[r['variant']]} | {_f(r['accept_rate'])} | {_f(r['unsafe_accept_rate'])} | "
                     f"{_f(r['unsafe_accept_given_unsafe'])} | {_f(r['false_reject_given_safe'])} | {_f(r['fallback_rate'])} | "
                     f"{_f(r['no_feasible_fallback_rate'])} | {_f(r['applied_overload_rate'])} | {_f(r['mean_mlu_applied'])} | "
                     f"{_f(r['mean_sat_applied'], 4)} | {_f(r['mean_decision_ms'], 1)} |")
        L += ["", f"Baselines on the same {sub[0]['n_cases']} scaled matrices:", "",
              "| Method | Satisfaction | MLU | Overload rate | median ms |", "|---|---:|---:|---:|---:|"]
        for r in [b for b in bt if b["topology"] == topo and b["load_level"] == "all"]:
            L.append(f"| {r['method']} | {_f(r['sat_mean'], 4)} | {_f(r['mlu_mean'])} | {_f(r['overload_rate'])} | {_f(r['median_ms'], 1)} |")
        L += ["", "Per load level:", "",
              "| Target | A sat. | E sat. | LP sat. | Shortest sat. | ECMP sat. | Frozen P1 sat. | A unsafe acc. | E unsafe acc. | "
              "E false rej. | E fallback | E overload | LP overload | Policy MLU | LP MLU |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for t in TARGETS:
            a = next(r for r in vt if r["topology"] == topo and r["load_level"] == t and r["variant"] == "A")
            e = next(r for r in vt if r["topology"] == topo and r["load_level"] == t and r["variant"] == "E")
            b = {r["method"]: r for r in bt if r["topology"] == topo and r["load_level"] == t}
            L.append(f"| {t} | {_f(a['mean_sat_applied'], 4)} | {_f(e['mean_sat_applied'], 4)} | {_f(b['lp']['sat_mean'], 4)} | "
                     f"{_f(b['shortest']['sat_mean'], 4)} | {_f(b['equal']['sat_mean'], 4)} | {_f(b['frozen_p1']['sat_mean'], 4)} | "
                     f"{_f(a['unsafe_accept_rate'])} | {_f(e['unsafe_accept_rate'])} | {_f(e['false_reject_rate'])} | "
                     f"{_f(e['fallback_rate'])} | {_f(e['applied_overload_rate'])} | {_f(b['lp']['overload_rate'])} | "
                     f"{_f(b['actor_stable']['mlu_mean'])} | {_f(b['lp']['mlu_mean'])} |")
        L.append("")
    L += ["## D vs E and E vs oracle F", "", "| Topology | n | D-only unsafe accepts | E-only unsafe accepts | D/E accept discordant | E/F accept discordant |",
          "|---|---:|---:|---:|---:|---:|"]
    for d in s["d_vs_e"]:
        if d["load_mode"] == "all":
            L.append(f"| {d['topology']} | {d['n_cases']} | {d['D_only_unsafe_accepts']} | {d['E_only_unsafe_accepts']} | "
                     f"{d['D_E_accept_discordant']} | {d['E_F_accept_discordant']} |")
    L += ["", "## Files", "", "- `case_rows.jsonl`: one row per case (all six variants, all baselines, timings)",
          "- `summary.json`: variant table (as for GEANT), D/E discordance, baseline table; `config.json`: frozen stack",
          "- `figures/holdout_sat_unsafe_A_vs_E.{png,pdf}`"]
    (HOLD / "HOLDOUT_SNDLIB_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def report_failures() -> None:
    s = json.loads((LF / "summary.json").read_text(encoding="utf-8"))
    t = s["table"]

    def get(topo, lvl, nf, m):
        return next((r for r in t if r["topology"] == topo and r["load_level"] == lvl and r["n_failed"] == nf
                     and r["method"] == m), None)
    L = ["# Link failures on SNDlib Abilene and Germany50", "",
         "Same protocol as the GEANT link-failure experiment (`results/phase3_final/e3_link_failures/`): failure cases "
         "from `data/scenarios/link_failures/<topology>.json` (P1 §5.5 sampler, seeds 100–1000, 0–3 failed directed "
         "links); candidate paths through a failed link are dropped and Yen k = 4 is recomputed for pairs left without a "
         "path; demand of pairs disconnected by the failure is unservable and counted as undelivered. 50 seeded TMs per "
         "topology × calibrated targets 0.5 / 0.7 / 0.9 / 1.1 (alpha from the intact topology) × 40 failure cases = "
         "8,000 cases per topology. Frozen stack (policy `recommended_stable`, gate k = 1, frozen fallback).", "",
         "GEANT was run at fixed scales 0.5 / 1.0 and calibrated 0.7 / 0.9; the `0.7+0.9` rows below are the levels "
         "shared with GEANT.", ""]
    order = [("lp", "LP (min-MLU)"), ("shortest", "shortest"), ("equal", "ECMP (equal split)"), ("frozen_p1", "frozen P1 actor"),
             ("actor_stable", "policy only (actor)"), ("fw_stable_A", "framework A (no gate)"), ("fw_stable_D", "framework D (mean-only gate)"),
             ("fw_stable_E", "framework E (gate k=1)"), ("fw_stable_F", "framework F (oracle)")]
    for topo in TOPOS:
        for lvl in ("all", "0.7+0.9"):
            unserv = ", ".join(f"{n}: {s['unservable'].get(f'{topo}|{n}', 0) * 100:.2f}%" for n in range(4))
            L += [f"## {topo} — load levels: {lvl} (unservable demand by failed links — {unserv})", "",
                  "| Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | MLU 0 | MLU 3 | Overload 0 | Overload 3 | Unsafe acc. 0/1/2/3 | "
                  "False rej. given safe 0/1/2/3 | Fallback 0/1/2/3 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
            for m, lab in order:
                rs = [get(topo, lvl, n, m) for n in range(4)]
                if rs[0] is None:
                    continue
                ua = "/".join(_f(r.get("unsafe_accept_rate"), 2) if "unsafe_accept_rate" in r else "–" for r in rs)
                fr = "/".join(_f(r.get("false_reject_given_safe"), 2) if "accept_rate" in r else "–" for r in rs)
                fb = "/".join(_f(r.get("fallback_rate"), 2) if "fallback_rate" in r else "–" for r in rs)
                L.append(f"| {lab} | " + " | ".join(_f(r["sat_mean"] * 100, 2) for r in rs) +
                         f" | {_f(rs[0]['mlu_mean'])} | {_f(rs[3]['mlu_mean'])} | {_f(rs[0]['violation_rate'], 2)} | "
                         f"{_f(rs[3]['violation_rate'], 2)} | {ua} | {fr} | {fb} |")
            L.append("")
    L += ["Satisfaction in %, P1 definition, counting unservable demand as undelivered. MLU capped at 1e6 for averaging. "
          "Overload = any link above capacity after the method's routing (for framework rows: after gate and fallback).", "",
          "## Files", "", "- `case_rows.jsonl` (one row per failure case × TM × load), `summary.json`, `config.json`",
          "- `figures/link_failure_curves.{png,pdf}`"]
    (LF / "LINK_FAILURES_SNDLIB_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def report_twin() -> None:
    s = json.loads((TE / "summary.json").read_text(encoding="utf-8"))
    st = json.loads((ST / "statistics.json").read_text(encoding="utf-8")) if (ST / "statistics.json").exists() else None
    t = s["table"]
    L = ["# Twin-error sweep on SNDlib Abilene and Germany50", "",
         "Same error model as the GEANT sweep (`results/phase3_final/e4_twin_error/`): member-wise perturbation of the "
         "improved twin, `member_j(e) ← max(0, member_j(e)·(1 + σ z_j(e)))`, z ~ N(0, 1) with seed 4040 per "
         "(topology, case, replicate), the same z for every σ and every gate, 3 replicates for σ > 0. μ and std are "
         "recomputed from the perturbed members.", "",
         "Gates on identical inputs: A no gate · D mean-only (k = 0) · Dm mean-only with the tighter utilisation limit L "
         "chosen on dev validation in the GEANT sweep (L = " +
         ", ".join(f"{v} at σ={k}" for k, v in sorted(s["margin_control_L_frozen"].items(), key=lambda kv: float(kv[0]))) +
         "; not refitted) · E μ + 1·c·std with the frozen conformal scale c = "
         f"{next(iter(s['proposal_margin'].values()))['frozen_c']:.4f} · F physics oracle. Rejections use the frozen fallback.", "",
         "Cases: 200 TMs × 4 calibrated targets = 800 per topology (× 3 replicates for σ > 0).", ""]
    for scope in (*TOPOS, "sndlib_pooled"):
        L += [f"## {scope}", "", "| σ | Variant | Accept | Unsafe acc. given unsafe | False rej. given safe | Applied MLU | "
              "Applied overload | Applied sat. |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for sigma in SIGMAS:
            for v in ("A", "D", "Dm", "E", "F"):
                r = next((x for x in t if x["scope"] == scope and x["sigma"] == sigma and x["variant"] == v), None)
                if r:
                    L.append(f"| {sigma} | {v} | {_f(r['accept_rate'])} | {_f(r['unsafe_accept_given_unsafe'])} | "
                             f"{_f(r['false_reject_given_safe'])} | {_f(r['applied_mlu'])} | {_f(r['applied_overload_rate'])} | "
                             f"{_f(r['applied_sat'], 4)} |")
        L.append("")
    if st:
        L += ["## Exact McNemar tests (Holm-corrected within this family)", "",
              "| Comparison (left vs right) | pairs | left-only | right-only | p | p (Holm) |", "|---|---:|---:|---:|---:|---:|"]
        for r in st["mcnemar_holm"]:
            L.append(f"| {r['comparison']} | {r['n_pairs']} | {r['left_only']} | {r['right_only']} | {_p(r['p'])} | "
                     f"{_p(r['p_holm'])}{' *' if r['significant_holm_0.05'] else ''} |")
        L.append("")
    L += ["## Proposal margin", "", "| Topology | Cases | Unsafe share | True MLU in [0.9, 1.1] | median policy ms | median twin ms |",
          "|---|---:|---:|---:|---:|---:|"]
    for n, m in s["proposal_margin"].items():
        L.append(f"| {n} | {m['n_cases']} | {_f(m['share_unsafe'], 2)} | {_f(m['share_true_mlu_in_[0.9,1.1]'], 2)} | "
                 f"{_f(m['median_policy_ms'], 1)} | {_f(m['median_twin_ms'], 1)} |")
    L += ["", "## Files", "", "- `sweep.csv`, `paired_rows.csv.gz`, `summary.json`, `config.json`",
          "- `figures/twin_error_curves.{png,pdf}`; `cache/` (member stacks, not committed)"]
    (TE / "TWIN_ERROR_SNDLIB_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def report_stats() -> None:
    st = json.loads((ST / "statistics.json").read_text(encoding="utf-8"))
    L = ["# Statistics with the SNDlib topologies", "", *[f"- {n}" for n in st["notes"]], "",
         "## Paired Wilcoxon signed-rank, framework (E) vs policy only (A), with Cliff's δ; McNemar on applied overload", "",
         "| Comparison (left vs right) | n | mean diff | left better | right better | p | p (Holm) | Cliff's δ |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in st["wilcoxon_holm"]:
        if r.get("family") == "overload":
            L.append(f"| {r['comparison']} (McNemar; left-only {r['left_only']}, right-only {r['right_only']}) | {r['n']} | "
                     f"{r['mean_diff']:+.4f} | – | – | {_p(r['p'])} | {_p(r['p_holm'])}{' *' if r['significant_holm_0.05'] else ''} | – |")
        else:
            L.append(f"| {r['comparison']} | {r['n']} | {r['mean_diff']:+.4f} | {r['share_left_better']:.2f} | "
                     f"{r['share_right_better']:.2f} | {_p(r['p'])} | {_p(r['p_holm'])}{' *' if r['significant_holm_0.05'] else ''} | "
                     f"{r['cliffs_delta']:+.3f} |")
    L += ["", "Satisfaction: higher is better; MLU: lower is better. \\* significant at 0.05 after Holm correction.", "",
          "## Friedman test + Nemenyi post-hoc", "",
          "Blocks are (topology, load mode, level) cells; each block value is the mean over the cell's TMs. "
          "`framework` = recommended_stable + improved twin + gate E + fallback.", "",
          "| Block set | Test | Blocks | χ² | p | Mean ranks (1 = best) |", "|---|---|---:|---:|---:|---|"]
    for setname, d in st["friedman"].items():
        for k, v in d.items():
            ranks = ", ".join(f"{m} {rk:.2f}" for m, rk in sorted(v["mean_rank (1 = best)"].items(), key=lambda kv: kv[1]))
            L.append(f"| {setname} | {k} | {v['n_blocks']} | {v['friedman_chi2']:.1f} | {_p(v['friedman_p'])} | {ranks} |")
    L += ["", "`previous_without_sndlib` repeats `results/phase3_final/stats/friedman_nemenyi.json` for comparison.", ""]
    (ST / "STATISTICS_SNDLIB_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["holdout", "failures", "twin", "stats", "figures", "report"])
    ap.add_argument("--topologies", nargs="+", default=list(TOPOS), choices=list(TOPOS))
    a = ap.parse_args()
    if a.cmd == "holdout":
        run_holdout(a.topologies)
        summarize_holdout()
    elif a.cmd == "failures":
        run_failures(a.topologies)
        summarize_failures()
    elif a.cmd == "twin":
        TE.mkdir(parents=True, exist_ok=True)
        build_twin_cache(a.topologies)
        run_twin_sweep(list(TOPOS))
    elif a.cmd == "stats":
        statistics()
    elif a.cmd == "figures":
        figures()
    else:
        summarize_holdout()
        summarize_failures()
        report_holdout()
        report_failures()
        report_twin()
        report_stats()


if __name__ == "__main__":
    main()
