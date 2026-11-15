"""E4 twin-error sweep, mean-only gate D vs uncertainty-aware gate E.

``data/scenarios/twin_errors/spec.json`` perturbs only a single predicted load
(``load_hat = max(0, load_true * (1 + sigma z))``). That changes the mean but not
the ensemble spread, so it cannot test an uncertainty-aware gate fairly. Two
documented error models are used instead, applied identically to D and E:

1. Member-wise perturbation of the improved (physics-anchored) twin:
       member_j(e) <- max(0, member_j(e) * (1 + sigma * z_j(e)))
   z ~ N(0, 1) per (case, replicate, member, link), generator seed 4040, the same
   z for every sigma (common random numbers) and for every gate. mu and std are
   recomputed from the perturbed members, so the spread grows with twin error.
   sigma in {0, 0.05, 0.10, 0.20, 0.40} (from spec.json); 3 replicates per case.
2. Degraded twins (``train_degraded_twin.py``): weaker checkpoints evaluated as-is.

Gate variants (utilisation_limit = 1.0, max_relative_std off, as frozen):
  A    no gate (always apply)
  D    mean-only, k = 0
  E1   mu + 1 * c * std   (c = conformal std scale fitted on clean validation; frozen)
  E2   mu + 2 * c * std
  E1r  mu + 1 * c_sigma * std, c_sigma refitted for each error level on the dev
       *validation* cases with the unchanged rule (smallest c with 2-std coverage >= 0.9).
       In-sample for dev topologies, out-of-sample for GEANT/Kdl.
  F    physics oracle
Rejected proposals use the frozen shortest/equal least-violating fallback.

Cases (load: fixed 0.25-1.0 and calibrated 0.5-1.1 from the calibrated-load study):
  dev:     B4 val (8 TMs), UsCarrier val (1 TM; identical toy copies), Cogentco val (8 TMs)
  holdout: GEANT (200 seeded test TMs), Kdl (1 TM; identical toy copies)

Usage:
  python -m phase3_final.twin_error_sweep cache --topologies B4 UsCarrier Cogentco GEANT
  python -m phase3_final.twin_error_sweep cache --topologies Kdl
  python -m phase3_final.twin_error_sweep sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from safety.improvements.fallback import select_physics_fallback
from safety.improvements.gate_logic import ImprovedGateConfig, inspect_utilisation_ucb
from safety.improvements.graph_cache import TopologyGraphCache
from safety.improvements.oracle import physics_violations
from safety.improvements.policy_backend import policy_backend_from_config
from twin.data import ROOT, load_split_demands, load_topology
from twin.improvements.infer import load_ensemble_checkpoint
from twin.train import _sample, _tensors

from . import routing as R
from .common import LP_SOLVER_NOTE, TOY_TM_NOTE, environment, fmt_eta, out_dir, write_json
from .holdout_af import CALIBRATION, FIXED_SCALES, TARGETS, holdout_config

SIGMAS = (0.0, 0.05, 0.10, 0.20, 0.40)
REPLICATES = 3
NOISE_SEED = 4040
OUT = out_dir("e4_twin_error")
CACHE = OUT / "cache"
IMPROVED = ROOT / "models" / "twin_improved" / "ensemble_physics_anchored_conformal.pt"
DEGRADED_DIR = ROOT / "models" / "twin_degraded"
TWINS = ("improved", "degraded_f10", "degraded_f25", "degraded_f50")
DEV = ("B4", "UsCarrier", "Cogentco")
HOLDOUT = ("GEANT", "Kdl")
C_GRID = tuple(float(x) for x in np.round(np.concatenate([np.linspace(0.25, 8.0, 32), [10, 12, 16, 24, 32]]), 4))
VARIANTS = ("A", "D", "E1", "E2", "E1r", "F")
REPORT_VARIANTS = ("A", "D", "Dm", "E1", "E2", "E1r", "F")


def twin_paths() -> dict[str, Path]:
    paths = {"improved": IMPROVED}
    for name in TWINS[1:]:
        p = DEGRADED_DIR / f"ensemble_{name}.pt"
        if p.exists():
            paths[name] = p
    return paths


def cases_for(name: str) -> list[dict[str, Any]]:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if name in DEV:
        split = "val"
        positions = [0] if name == "UsCarrier" else list(range(8))
    else:
        split = "test"
        positions = cal["geant_test_positions_200"] if name == "GEANT" else [0]
    out = []
    for pos in positions:
        alpha = cal["alpha"][f"{name}/{split}/{pos}"]
        for mode, level, scale in ([("fixed", s, s) for s in FIXED_SCALES]
                                   + [("calibrated", t, alpha[f"{t:.1f}"]) for t in TARGETS]):
            out.append({"topology": name, "split": split, "position": int(pos), "load_mode": mode,
                        "load_level": float(level), "scale": float(scale)})
    return out


@torch.no_grad()
def member_stack(model, topo, matrix, action) -> np.ndarray:
    batch = _tensors(_sample(topo, matrix, action), topo, torch.device("cpu"))
    _, _, stack = model.predict(batch["features"], batch["flow"], batch["incidence_path"],
                                batch["incidence_link"], topo.n_links)
    return stack.numpy().astype(np.float64) * topo.denominator  # [M, E] absolute load


def build_cache(names: list[str]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = holdout_config(None, None)
    device = torch.device("cpu")
    policy = policy_backend_from_config(cfg, device=device, graph_cache=TopologyGraphCache(device))
    policy.load()
    paths = twin_paths()
    models = {k: load_ensemble_checkpoint(p, device) for k, p in paths.items()}
    for name in names:
        target = CACHE / f"{name}.npz"
        if target.exists():
            print(f"{name}: cache exists, skipping")
            continue
        topo = load_topology(name)
        cases = cases_for(name)
        demands = load_split_demands(cases[0]["split"], name)[0]
        n, e = len(cases), topo.n_links
        stacks = {k: np.zeros((n, 5, e)) for k in models}
        true_load = np.zeros((n, e))
        fb_load = np.zeros((n, e))
        fb_feasible = np.zeros(n, dtype=bool)
        sat_prop = np.zeros(n)
        sat_fb = np.zeros(n)
        policy_ms = np.zeros(n)
        twin_ms = np.zeros(n)
        t0 = time.time()
        for i, c in enumerate(cases):
            matrix = demands[c["position"]] * c["scale"]
            tp = time.perf_counter()
            action = policy.propose(topo, matrix).action
            policy_ms[i] = (time.perf_counter() - tp) * 1000
            true_load[i] = R.loads(topo, matrix, action)
            sat_prop[i] = R.satisfaction_p1(topo, matrix, action)
            fb = select_physics_fallback(topo, matrix)
            fb_load[i] = fb.selected_load
            fb_feasible[i] = fb.feasible_fallback
            sat_fb[i] = R.satisfaction_p1(topo, matrix, fb.selected_action)
            for k, (model, _payload) in models.items():
                tt = time.perf_counter()
                stacks[k][i] = member_stack(model, topo, matrix, action)
                if k == "improved":
                    twin_ms[i] = (time.perf_counter() - tt) * 1000
            if (i + 1) % 50 == 0 or i + 1 == n or name == "Kdl":
                el = time.time() - t0
                print(f"{name} [{i + 1}/{n}] elapsed {fmt_eta(el)} eta {fmt_eta(el / (i + 1) * (n - i - 1))}",
                      flush=True)
        np.savez_compressed(
            target, capacity=topo.capacity, true_load=true_load, fb_load=fb_load, fb_feasible=fb_feasible,
            sat_prop=sat_prop, sat_fb=sat_fb, policy_ms=policy_ms, twin_ms=twin_ms,
            meta=json.dumps(cases), twins=json.dumps(list(models)),
            conformal=json.dumps({k: float(p.get("conformal_std_scale", 1.0)) for k, (_m, p) in models.items()}),
            **{f"stack_{k}": v for k, v in stacks.items()},
        )
        print(f"{name}: cached {n} cases -> {target.name}", flush=True)
        del topo, demands


def load_cache(name: str) -> dict[str, Any]:
    z = np.load(CACHE / f"{name}.npz", allow_pickle=False)
    d = {k: z[k] for k in z.files if not k.startswith("stack_") and k not in ("meta", "twins", "conformal")}
    d["cases"] = json.loads(str(z["meta"]))
    d["twins"] = json.loads(str(z["twins"]))
    d["conformal"] = json.loads(str(z["conformal"]))
    d["stacks"] = {k: z[f"stack_{k}"] for k in d["twins"]}
    return d


def noise(case_index: int, topo_name: str, rep: int, shape: tuple[int, int]) -> np.ndarray:
    tid = sum(ord(ch) for ch in topo_name)
    return np.random.default_rng([NOISE_SEED, tid, case_index, rep]).standard_normal(shape)


def perturbed(stack: np.ndarray, sigma: float, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    members = np.maximum(0.0, stack * (1.0 + sigma * z)) if sigma > 0 else stack
    return members.mean(axis=0), members.std(axis=0)  # population std, as EnsembleTwin.predict


def gate_accepts(mu, std, cap, k) -> bool:
    return bool(inspect_utilisation_ucb(mu, std, cap, ImprovedGateConfig(k=k, utilisation_limit=1.0,
                                                                          max_relative_std=None)).accepted)


def coverage(mu, std, true, c, mult) -> float:
    return float(np.mean(np.abs(mu - true) <= mult * c * std + 1e-12 * np.maximum(true, 1.0)))


def iter_conditions(data: dict[str, Any]):
    """Yield (twin, sigma, rep) conditions: member-wise noise on the improved twin, degraded twins clean."""
    for sigma in SIGMAS:
        for rep in range(REPLICATES if sigma > 0 else 1):
            yield "improved", sigma, rep
    for t in data["twins"]:
        if t != "improved":
            yield t, 0.0, 0


def fit_recalibrated_c(devs: dict[str, dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    """Per (twin, sigma): smallest c on C_GRID with mean dev-val 2-std coverage >= 0.9 (unchanged rule)."""
    out = {}
    conds = sorted({(t, s) for d in devs.values() for t, s, _r in iter_conditions(d)})
    for twin, sigma in conds:
        errs, stds = [], []
        for name, d in devs.items():
            if twin not in d["stacks"]:
                continue
            for i in range(len(d["cases"])):
                for rep in range(REPLICATES if sigma > 0 else 1):
                    st = d["stacks"][twin][i]
                    mu, sd = perturbed(st, sigma, noise(i, name, rep, st.shape))
                    errs.append(np.abs(mu - d["true_load"][i]))
                    stds.append(sd)
        err, sd = np.concatenate(errs), np.concatenate(stds)
        grid = [{"c": c, "coverage_2std": float(np.mean(err <= 2 * c * sd + 1e-12))} for c in C_GRID]
        ok = [g for g in grid if g["coverage_2std"] >= 0.9]
        chosen = ok[0] if ok else grid[-1]
        out[(twin, sigma)] = {"c": chosen["c"], "coverage_2std": chosen["coverage_2std"], "target_met": bool(ok)}
    return out


def sweep() -> None:
    names = [n for n in DEV + HOLDOUT if (CACHE / f"{n}.npz").exists()]
    data = {n: load_cache(n) for n in names}
    recal = fit_recalibrated_c({n: data[n] for n in DEV if n in data})
    rows = []
    for name in names:
        d = data[name]
        cap = d["capacity"]
        for twin, sigma, rep in iter_conditions(d):
            c_frozen = d["conformal"][twin]
            c_r = recal[(twin, sigma)]["c"]
            for i, case in enumerate(d["cases"]):
                st = d["stacks"][twin][i]
                mu, sd = perturbed(st, sigma, noise(i, name, rep, st.shape))
                true = d["true_load"][i]
                unsafe_prop = physics_violations(true, cap) > 0  # same oracle as A-F (atol 1e-6)
                fb_over = physics_violations(d["fb_load"][i], cap) > 0
                acc = {"A": True, "D": gate_accepts(mu, sd, cap, 0.0),
                       "E1": gate_accepts(mu, c_frozen * sd, cap, 1.0),
                       "E2": gate_accepts(mu, c_frozen * sd, cap, 2.0),
                       "E1r": gate_accepts(mu, c_r * sd, cap, 1.0), "F": not unsafe_prop}
                mlu_prop = float(np.max(true / cap))
                mlu_fb = float(np.max(d["fb_load"][i] / cap))
                base = {"topology": name, "group": "dev" if name in DEV else "holdout", "twin": twin,
                        "sigma": sigma, "rep": rep, "case": i, "load_mode": case["load_mode"],
                        "load_level": case["load_level"], "unsafe_proposal": unsafe_prop,
                        "cov1_frozen": coverage(mu, sd, true, c_frozen, 1),
                        "cov2_frozen": coverage(mu, sd, true, c_frozen, 2),
                        "cov2_recal": coverage(mu, sd, true, c_r, 2),
                        "pred_mlu_mean": float(np.max(mu / cap)), "true_mlu": mlu_prop,
                        "mlu_fb": mlu_fb, "fb_over": fb_over, "sat_prop": float(d["sat_prop"][i]),
                        "sat_fb": float(d["sat_fb"][i])}
                for v in VARIANTS:
                    a = acc[v]
                    rows.append({**base, "variant": v, "accept": a, "unsafe_accept": a and unsafe_prop,
                                 "false_reject": (not a) and (not unsafe_prop),
                                 "applied_mlu": mlu_prop if a else mlu_fb,
                                 "applied_overload": unsafe_prop if a else fb_over,
                                 "applied_sat": float(d["sat_prop"][i] if a else d["sat_fb"][i])})
    margins = add_margin_control(rows)
    write_rows(rows, recal, data, margins)


L_GRID = tuple(round(1.0 - 0.005 * i, 3) for i in range(0, 101))  # 1.000 ... 0.500


def add_margin_control(rows: list[dict]) -> dict:
    """Dm: mean-only gate with a tighter utilisation limit L (no uncertainty).

    For each (twin, sigma), L is the largest value on L_GRID whose dev false-reject rate
    (given a safe proposal) is >= E1's dev false-reject rate, i.e. Dm is at least as
    conservative as E1 on dev validation. L is chosen on dev only and applied unchanged
    to holdout. Dm accepts iff max predicted mean utilisation <= L (same as the gate with k=0).
    """
    by_cond: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[(r["twin"], r["sigma"])].append(r)
    chosen = {}
    new_rows = []
    for (twin, sigma), rs in by_cond.items():
        dev_e1 = [r for r in rs if r["group"] == "dev" and r["variant"] == "E1" and not r["unsafe_proposal"]]
        target = float(np.mean([r["false_reject"] for r in dev_e1])) if dev_e1 else 0.0
        dev_d = [r for r in rs if r["group"] == "dev" and r["variant"] == "D" and not r["unsafe_proposal"]]
        pick = L_GRID[-1]
        for L in L_GRID:
            fr = float(np.mean([r["pred_mlu_mean"] > L for r in dev_d])) if dev_d else 0.0
            if fr >= target - 1e-12:
                pick = L
                break
        chosen[(twin, sigma)] = {"L": pick, "dev_target_false_reject_E1": target}
        for r in rs:
            if r["variant"] != "D":
                continue
            a = r["pred_mlu_mean"] <= pick
            new_rows.append({**r, "variant": "Dm", "accept": a, "unsafe_accept": a and r["unsafe_proposal"],
                             "false_reject": (not a) and (not r["unsafe_proposal"]),
                             "applied_mlu": r["true_mlu"] if a else r["mlu_fb"],
                             "applied_overload": r["unsafe_proposal"] if a else r["fb_over"],
                             "applied_sat": r["sat_prop"] if a else r["sat_fb"]})
    rows.extend(new_rows)
    return chosen


def _agg(rs: list[dict]) -> dict[str, float]:
    unsafe = [r for r in rs if r["unsafe_proposal"]]
    safe = [r for r in rs if not r["unsafe_proposal"]]
    m = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else float("nan")  # noqa: E731
    return {"n": len(rs), "n_unsafe_prop": len(unsafe), "accept_rate": m(rs, "accept"),
            "unsafe_accept_rate": m(rs, "unsafe_accept"), "unsafe_accept_given_unsafe": m(unsafe, "accept"),
            "false_reject_rate": m(rs, "false_reject"), "false_reject_given_safe": m(safe, "false_reject"),
            "applied_mlu": m(rs, "applied_mlu"), "applied_overload_rate": m(rs, "applied_overload"),
            "applied_sat": m(rs, "applied_sat"), "cov1_frozen": m(rs, "cov1_frozen"),
            "cov2_frozen": m(rs, "cov2_frozen"), "cov2_recal": m(rs, "cov2_recal")}


def write_rows(rows: list[dict], recal: dict, data: dict, margins: dict) -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        for topo_key in (r["topology"], r["group"]):
            groups[(topo_key, r["twin"], r["sigma"], r["variant"], "all")].append(r)
            groups[(topo_key, r["twin"], r["sigma"], r["variant"], r["load_mode"])].append(r)
    table = []
    for (topo, twin, sigma, variant, mode), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        table.append({"scope": topo, "twin": twin, "sigma": sigma, "variant": variant, "load_mode": mode,
                      **_agg(rs)})
    with (OUT / "sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # Paired D vs E1 discordance per (scope, twin, sigma), over all cases x replicates.
    paired: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["topology"], r["twin"], r["sigma"], r["rep"], r["case"])
        for scope in (r["topology"], r["group"]):
            paired[(scope, r["twin"], r["sigma"])][key].append(r)
    disc = []
    for (scope, twin, sigma), cases in sorted(paired.items(), key=lambda kv: tuple(map(str, kv[0]))):
        by = [{x["variant"]: x for x in v} for v in cases.values()]
        for ev, ref in (("E1", "D"), ("E2", "D"), ("E1r", "D"), ("E1", "Dm")):
            d_only = sum(b[ref]["unsafe_accept"] and not b[ev]["unsafe_accept"] for b in by)
            e_only = sum(b[ev]["unsafe_accept"] and not b[ref]["unsafe_accept"] for b in by)
            fr_d = sum(b[ref]["false_reject"] and not b[ev]["false_reject"] for b in by)
            fr_e = sum(b[ev]["false_reject"] and not b[ref]["false_reject"] for b in by)
            disc.append({"scope": scope, "twin": twin, "sigma": sigma, "E": ev, "ref": ref, "n_pairs": len(by),
                         "D_only_unsafe_accepts": d_only, "E_only_unsafe_accepts": e_only,
                         "D_only_false_rejects": fr_d, "E_only_false_rejects": fr_e})
    with (OUT / "d_vs_e_discordance.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(disc[0].keys()))
        w.writeheader()
        w.writerows(disc)
    # Compact per-row file for the statistical analysis (paired by case x replicate).
    import gzip

    with gzip.open(OUT / "paired_rows.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        keys = ["topology", "group", "twin", "sigma", "rep", "case", "load_mode", "load_level", "variant",
                "unsafe_proposal", "accept", "unsafe_accept", "false_reject", "applied_mlu", "applied_overload",
                "applied_sat", "cov2_frozen", "pred_mlu_mean", "true_mlu"]
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([r[k] for k in keys])
    margin = {}
    for name, d in data.items():
        mlu = np.max(d["true_load"] / d["capacity"], axis=1)
        margin[name] = {"n_cases": int(mlu.size), "share_true_mlu_in_[0.9,1.1]": float(np.mean((mlu >= 0.9) & (mlu <= 1.1))),
                        "share_true_mlu_in_[0.8,1.2]": float(np.mean((mlu >= 0.8) & (mlu <= 1.2))),
                        "share_unsafe": float(np.mean([physics_violations(t, d["capacity"]) > 0 for t in d["true_load"]])),
                        "median_policy_ms": float(np.median(d["policy_ms"])),
                        "median_twin_ms": float(np.median(d["twin_ms"]))}
    summary = {"margin_control_L": {f"{t}|{s}": v for (t, s), v in margins.items()}, "recalibrated_c": {f"{t}|{s}": v for (t, s), v in recal.items()},
               "frozen_c": {n: d["conformal"] for n, d in data.items()}, "proposal_margin": margin,
               "table": table, "discordance": disc}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "config.json", {"sigmas": SIGMAS, "replicates": REPLICATES, "noise_seed": NOISE_SEED,
                                     "twins": list(TWINS), "variants": VARIANTS, "c_grid": C_GRID,
                                     "fixed_scales": FIXED_SCALES, "calibrated_targets": TARGETS,
                                     "toy_tm_note": TOY_TM_NOTE, "lp_solver": LP_SOLVER_NOTE,
                                     "environment": environment()})
    plot(table)
    report(table, disc, recal, margin, margins)


def plot(table: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (OUT / "plots").mkdir(exist_ok=True)
    colors = {"D": "#1f77b4", "E1": "#d62728", "E2": "#ff7f0e", "E1r": "#9467bd", "F": "#2ca02c"}
    labels = {"D": "D mean-only", "E1": "E k=1 (frozen c)", "E2": "E k=2 (frozen c)",
              "E1r": "E k=1 (recal. c)", "F": "F oracle"}
    scopes = [s for s in ("dev", "holdout", "B4", "Cogentco", "GEANT", "Kdl") if any(r["scope"] == s for r in table)]

    def series(scope, v, metric):
        return sorted((r["sigma"], r[metric]) for r in table if r["scope"] == scope and r["twin"] == "improved"
                      and r["variant"] == v and r["load_mode"] == "all")

    for metric, ylabel, fname in (
            ("unsafe_accept_given_unsafe", "Unsafe accepts / unsafe proposals", "unsafe_accept_vs_sigma"),
            ("false_reject_given_safe", "False rejects / safe proposals", "false_reject_vs_sigma")):
        fig, axes = plt.subplots(1, len(scopes), figsize=(2.9 * len(scopes), 3.1))
        for ax, scope in zip(np.atleast_1d(axes), scopes):
            for v in colors:
                pts = series(scope, v, metric)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3.5, lw=1.6,
                            color=colors[v], label=labels[v])
            a = series(scope, "A", "unsafe_accept_given_unsafe")
            title = scope if metric != "unsafe_accept_given_unsafe" else f"{scope} (A no gate = {a[0][1]:.2f})"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("member-wise twin error σ")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel(ylabel)
        np.atleast_1d(axes)[-1].legend(fontsize=7, loc="best")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / "plots" / f"{fname}.{ext}", dpi=300)
        plt.close(fig)

    # Trade-off: false rejects vs unsafe accepts, one point per sigma, for dev and holdout.
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, scope in zip(axes, ("dev", "holdout")):
        for v in ("D", "E1", "E2", "E1r"):
            xs = series(scope, v, "false_reject_given_safe")
            ys = series(scope, v, "unsafe_accept_given_unsafe")
            ax.plot([x[1] for x in xs], [y[1] for y in ys], marker="o", ms=4, lw=1.4, color=colors[v], label=labels[v])
            for (sg, x), (_s, y) in zip(xs, ys):
                if sg in (0.1, 0.4):
                    ax.annotate(f"σ={sg}", (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_title(scope, fontsize=10)
        ax.set_xlabel("False rejects / safe proposals")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Unsafe accepts / unsafe proposals")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "plots" / f"tradeoff.{ext}", dpi=300)
    plt.close(fig)


def _f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def report(table, disc, recal, margin, margins) -> None:
    def get(scope, twin, sigma, v, mode="all"):
        for r in table:
            if (r["scope"], r["twin"], r["sigma"], r["variant"], r["load_mode"]) == (scope, twin, sigma, v, mode):
                return r
        return None

    lines = [
        "# E4 twin-error sweep: mean-only gate (D) vs uncertainty-aware gate (E)",
        "",
        "## Error models",
        "",
        "1. **Member-wise perturbation** of the improved physics-anchored twin: "
        "`member_j(e) ← max(0, member_j(e)·(1 + σ z_j(e)))`, z ~ N(0,1), seed 4040, the same z for every σ "
        f"and every gate, {REPLICATES} replicates per case. μ and std are recomputed from the perturbed members, "
        "so ensemble spread grows with twin error. σ ∈ {0, 0.05, 0.10, 0.20, 0.40} (spec.json levels).",
        "2. **Degraded twins**: physics-anchored ensembles trained with 10/25/50% of the train TMs and 1/2/4 "
        "epochs (`degraded_twins/`), evaluated as-is at σ = 0, each with its own validation-fitted conformal scale.",
        "",
        "`spec.json`'s single-load perturbation is not used: it moves μ without changing the spread, which cannot "
        "test an uncertainty-aware gate fairly.",
        "",
        "## Gates (identical inputs; utilisation limit 1.0)",
        "",
        "A no gate · D mean-only (k=0) · Dm mean-only with a tighter limit L (control; L chosen on dev validation to be at least as conservative as E1) · E1 μ+1·c·std · E2 μ+2·c·std (c = frozen validation conformal scale) · "
        "E1r μ+1·c_σ·std with c_σ refitted per error level on the **dev validation** cases by the unchanged rule "
        "(smallest c with 2-std coverage ≥ 0.9; in-sample for dev, out-of-sample for GEANT/Kdl) · F physics oracle. "
        "Rejections use the frozen shortest/equal least-violating fallback.",
        "",
        f"Cases: dev = B4 val (8 TMs), UsCarrier val (1 TM), Cogentco val (8 TMs); holdout = GEANT (200 seeded "
        f"test TMs), Kdl (1 TM). Each at fixed scales 0.25–1.0 and calibrated targets 0.5–1.1. {TOY_TM_NOTE}",
        "",
    ]
    for scope in ("dev", "holdout", "B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"):
        if get(scope, "improved", 0.0, "A") is None:
            continue
        r0 = get(scope, "improved", 0.0, "A")
        lines += [f"## {scope}: member-wise error (n = {r0['n']} cases per σ=0; ×{REPLICATES} replicates for σ>0; "
                  f"{r0['n_unsafe_prop']} unsafe proposals at σ=0)", "",
                  "| σ | Variant | Accept | Unsafe accept | Unsafe acc. given unsafe | False rej. given safe | "
                  "Applied MLU | Applied overload | Applied sat. | Coverage@1σ (frozen c) | Coverage@2σ (frozen c) |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in SIGMAS:
            for v in REPORT_VARIANTS:
                r = get(scope, "improved", s, v)
                if r is None:
                    continue
                lines.append(f"| {s} | {v} | {_f(r['accept_rate'])} | {_f(r['unsafe_accept_rate'])} | "
                             f"{_f(r['unsafe_accept_given_unsafe'])} | {_f(r['false_reject_given_safe'])} | "
                             f"{_f(r['applied_mlu'])} | {_f(r['applied_overload_rate'])} | {_f(r['applied_sat'], 4)} | "
                             f"{_f(r['cov1_frozen'])} | {_f(r['cov2_frozen'])} |")
        lines.append("")
    lines += ["## Degraded twins (σ = 0)", "",
              "| Scope | Twin | Variant | Accept | Unsafe accept | False rej. given safe | Applied MLU | Coverage@2σ |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for scope in ("dev", "holdout"):
        for twin in TWINS:
            for v in ("D", "E1", "E2", "F"):
                r = get(scope, twin, 0.0, v)
                if r is not None:
                    lines.append(f"| {scope} | {twin} | {v} | {_f(r['accept_rate'])} | {_f(r['unsafe_accept_rate'])} | "
                                 f"{_f(r['false_reject_given_safe'])} | {_f(r['applied_mlu'])} | {_f(r['cov2_frozen'])} |")
    lines += ["", "## Paired discordance: D vs E (counts over cases × replicates)", "",
              "| Scope | Twin | σ | E variant | vs | Pairs | ref-only unsafe accepts | E-only unsafe accepts | "
              "ref-only false rejects | E-only false rejects |", "|---|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for d in disc:
        if d["scope"] in ("dev", "holdout"):
            lines.append(f"| {d['scope']} | {d['twin']} | {d['sigma']} | {d['E']} | {d['ref']} | {d['n_pairs']} | "
                         f"{d['D_only_unsafe_accepts']} | {d['E_only_unsafe_accepts']} | "
                         f"{d['D_only_false_rejects']} | {d['E_only_false_rejects']} |")
    lines += ["", "## Margin control Dm: mean-only gate with tighter limit L (chosen on dev validation)", "",
              "L is the largest limit whose dev false-reject rate is ≥ E1's dev false-reject rate at the same σ. "
              "If E1 still has fewer unsafe accepts than Dm on holdout, uncertainty adds value beyond a constant "
              "safety margin.", "", "| Twin | σ | L | E1 dev false-reject target |", "|---|---:|---:|---:|"]
    for (t, s), v in sorted(margins.items()):
        lines.append(f"| {t} | {s} | {v['L']} | {_f(v['dev_target_false_reject_E1'])} |")
    lines += ["", "## Recalibrated conformal scale c_σ (dev validation only)", "",
              "| Twin | σ | c_σ | dev coverage@2σ | target met |", "|---|---:|---:|---:|---|"]
    for (t, s), v in sorted(recal.items()):
        lines.append(f"| {t} | {s} | {v['c']} | {_f(v['coverage_2std'])} | {v['target_met']} |")
    lines += ["", "## How close are proposals to the capacity boundary?", "",
              "| Topology | Cases | Unsafe share | True MLU in [0.9, 1.1] | in [0.8, 1.2] | median policy ms | "
              "median twin ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for n, m in margin.items():
        lines.append(f"| {n} | {m['n_cases']} | {_f(m['share_unsafe'], 2)} | {_f(m['share_true_mlu_in_[0.9,1.1]'], 2)} | "
                     f"{_f(m['share_true_mlu_in_[0.8,1.2]'], 2)} | {_f(m['median_policy_ms'], 1)} | "
                     f"{_f(m['median_twin_ms'], 1)} |")
    lines += ["", "## Artifacts", "",
              "- `sweep.csv` (all scopes/twins/σ/variants/load modes), `d_vs_e_discordance.csv`, `paired_rows.csv.gz`",
              "- `summary.json`, `config.json`, `plots/unsafe_accept_vs_sigma.{png,pdf}`, "
              "`plots/false_reject_vs_sigma.{png,pdf}`",
              "- `cache/*.npz` (member stacks; not committed), `degraded_twins/*_meta.json` "
              "(checkpoints `.pt` not committed)"]
    (OUT / "E4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--topologies", nargs="+", required=True, choices=list(DEV + HOLDOUT))
    sub.add_parser("sweep")
    args = ap.parse_args()
    if args.cmd == "cache":
        build_cache(args.topologies)
    else:
        sweep()


if __name__ == "__main__":
    main()
