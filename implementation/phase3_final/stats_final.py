"""Statistical analysis: statistics, overhead and failure analysis.

Subcommands
-----------
grid     Paired main-comparison grid on every topology x load level (fixed 0.25-1.0,
         calibrated 0.5-1.1): shortest, equal, LP (min-MLU, HiGHS; not Kdl), frozen P1
         actor, stable actor, v2 actor and the full framework (stable + improved twin +
         gate E + fallback, from the ablation ``full`` rows). Times every component.
analyze  Statistics on all result files:
  * Friedman test + Nemenyi post-hoc (scikit-posthocs) across (topology, load level) blocks
    for the main methods (satisfaction and MLU; with and without LP, LP excludes Kdl);
  * paired Wilcoxon signed-rank with Holm correction and Cliff's delta (continuous outcomes):
    framework vs policy-only, v2 vs frozen P1, v2 vs shortest, v2 vs stable;
  * exact McNemar tests with Holm correction (paired binary outcomes): D vs E1 unsafe accepts and
    false rejects per twin-error level, E1 vs the margin control Dm;
  * unified overhead table (median ms per component and topology);
  * failure taxonomy for every unsafe or overloaded applied framework decision, plus an
    oracle-twin counterfactual (same proposals with a perfect twin).

Usage: python -m phase3_final.stats_final grid --topologies B4 UsCarrier Cogentco GEANT
       python -m phase3_final.stats_final grid --topologies Kdl
       python -m phase3_final.stats_final analyze
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from itertools import combinations

import numpy as np
import torch
from scipy import stats

from framework.general_policy import GeneralPolicy
from framework.policy_adapter import FrozenP1Policy
from safety.improvements.fallback import select_physics_fallback
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import od_demands

from . import routing as R
from .common import LP_SOLVER_NOTE, TOY_TM_NOTE, append_jsonl, environment, out_dir, read_jsonl, write_json
from .holdout_af import CALIBRATION, FIXED_SCALES, TARGETS

OUT = out_dir("stats")
GRID = OUT / "main_grid_rows.jsonl"
F = ROOT / "results" / "phase3_final"
POLICIES = {"actor_stable": ROOT / "models" / "policy" / "improved" / "recommended_stable.pt",
            "actor_v2": ROOT / "models" / "policy_v2" / "v2_calload_normscore" / "best.pt"}
FROZEN_P1 = ROOT / "models" / "baseline_p1" / "b4_freeze" / "grl_te_b4_freeze_seed100.pt"
DEV = ("B4", "UsCarrier", "Cogentco")


def _positions(name):
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if name in ("B4", "Cogentco"):
        return "val", list(range(8)), cal
    if name == "UsCarrier":
        return "val", [0], cal
    if name == "GEANT":
        return "test", cal["geant_test_positions_200"], cal
    return "test", [0], cal


# ------------------------------------------------------------------ grid
def grid(names):
    pols = {}
    for k, p in POLICIES.items():
        g = GeneralPolicy(torch.device("cpu"))
        g.model.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"])
        g.model.eval()
        pols[k] = g
    frozen = FrozenP1Policy(FROZEN_P1)
    full = {r["key"]: r["configs"]["full"] for r in read_jsonl(F / "ablations_v2" / "sequence_rows.jsonl")}
    done = {r["key"] for r in read_jsonl(GRID)}
    for name in names:
        topo = load_topology(name)
        split, positions, cal = _positions(name)
        demands = load_split_demands(split, name)[0]
        sp, eq = R.shortest_action(topo), R.equal_action(topo)
        for pos in positions:
            lp_action, lp_ms = None, None
            if name != "Kdl":
                t = time.perf_counter()
                lp_action = R.lp_min_mlu(topo, demands[pos])["action"]
                lp_ms = (time.perf_counter() - t) * 1000
            for mode, lvl in [("fixed", s) for s in FIXED_SCALES] + [("calibrated", t) for t in TARGETS]:
                key = f"{name}|{mode}={lvl}|pos={pos}"
                if key in done:
                    continue
                scale = lvl if mode == "fixed" else cal["alpha"][f"{name}/{split}/{pos}"][f"{lvl:.1f}"]
                m = demands[pos] * scale
                row = {"key": key, "topology": name, "load_mode": mode, "load_level": lvl, "position": pos,
                       "methods": {}, "ms": {"lp": lp_ms}}
                acts = {"shortest": sp, "equal": eq, "lp": lp_action}
                t = time.perf_counter()
                acts["frozen_p1"] = frozen.propose(topo, od_demands(m))
                row["ms"]["frozen_p1"] = (time.perf_counter() - t) * 1000
                for k, g in pols.items():
                    t = time.perf_counter()
                    with torch.no_grad():
                        acts[k] = g.propose(topo, od_demands(m))
                    row["ms"][k] = (time.perf_counter() - t) * 1000
                t = time.perf_counter()
                select_physics_fallback(topo, m)
                row["ms"]["fallback_select"] = (time.perf_counter() - t) * 1000
                for k, a in acts.items():
                    if a is None:
                        continue
                    load = R.loads(topo, m, a)
                    row["methods"][k] = {"mlu": float(R.utilisation(load, topo.capacity).max()),
                                         "sat": R.satisfaction_p1(topo, m, a),
                                         "overload": bool(np.any(load > topo.capacity + 1e-6))}
                if key in full:
                    fr = full[key]
                    row["methods"]["framework"] = {"mlu": fr["mlu"], "sat": fr["sat"], "overload": fr["overload"]}
                append_jsonl(GRID, row)
        print(f"{name}: grid done", flush=True)
        for g in list(pols.values()) + [frozen]:
            g.release(topo.name)


# ------------------------------------------------------------------ helpers
def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    return float((gt - lt) / (len(x) * len(y)))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def wilcoxon_row(name, x, y, higher_is_better):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)
    return {"comparison": name, "n": int(len(d)), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
            "share_left_better": float(np.mean(d > 1e-12) if higher_is_better else np.mean(d < -1e-12)),
            "share_right_better": float(np.mean(d < -1e-12) if higher_is_better else np.mean(d > 1e-12)),
            "p": p, "cliffs_delta": cliffs_delta(x, y)}


def mcnemar_row(name, a, b):
    """Exact McNemar on paired booleans: a-only vs b-only counts."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 > 0 else 1.0
    return {"comparison": name, "n_pairs": int(len(a)), "left_only": n10, "right_only": n01,
            "rate_left": float(a.mean()), "rate_right": float(b.mean()), "p": p}


# ------------------------------------------------------------------ analyses
def friedman_nemenyi(grid_rows):
    import scikit_posthocs as sp

    out = {}
    cells = defaultdict(lambda: defaultdict(list))
    for r in grid_rows:
        for m, v in r["methods"].items():
            cells[(r["topology"], r["load_mode"], r["load_level"])][m].append(v)
    for label, methods, exclude in (
            ("without_LP_all_topologies", ["shortest", "equal", "frozen_p1", "actor_stable", "actor_v2", "framework"], ()),
            ("with_LP_excluding_Kdl", ["lp", "shortest", "equal", "frozen_p1", "actor_stable", "actor_v2", "framework"],
             ("Kdl",))):
        for metric, better_high in (("sat", True), ("mlu", False)):
            blocks, keys = [], []
            for k, d in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
                if k[0] in exclude or not all(m in d for m in methods):
                    continue
                blocks.append([float(np.mean([min(v[metric], 1e6) for v in d[m]])) for m in methods])
                keys.append("|".join(map(str, k)))
            data = np.array(blocks)
            if data.shape[0] < 3:
                continue
            fr = stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
            ranks = stats.rankdata(-data if better_high else data, axis=1).mean(axis=0)
            nem = sp.posthoc_nemenyi_friedman(data)
            nem.index = nem.columns = methods
            out[f"{label}|{metric}"] = {"methods": methods, "n_blocks": int(data.shape[0]), "blocks": keys,
                                        "friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
                                        "mean_rank (1 = best)": dict(zip(methods, map(float, ranks))),
                                        "nemenyi_p": nem.round(6).to_dict()}
    return out


def paired_tests(grid_rows, abl_rows):
    rows = []
    for scope_name, scope in (("dev", set(DEV)), ("holdout", {"GEANT", "Kdl"})):
        sub = [r for r in abl_rows if r["topology"] in scope]
        if sub:
            for metric, hib in (("sat", True), ("mlu", False)):
                rows.append(wilcoxon_row(f"framework vs policy-only · {scope_name} · applied {metric}",
                                         [r["configs"]["full"][metric] if metric == "sat" else min(r["configs"]["full"]["mlu"], 1e6) for r in sub],
                                         [r["configs"]["no_gate"][metric] if metric == "sat" else min(r["configs"]["no_gate"]["mlu"], 1e6) for r in sub], hib))
        g = [r for r in grid_rows if r["topology"] in scope and all(m in r["methods"] for m in ("actor_v2", "frozen_p1", "shortest", "actor_stable"))]
        if g:
            for other in ("frozen_p1", "shortest", "actor_stable"):
                for metric, hib in (("sat", True), ("mlu", False)):
                    rows.append(wilcoxon_row(f"v2 vs {other} · {scope_name} · actor {metric}",
                                             [min(r["methods"]["actor_v2"][metric], 1e6) for r in g],
                                             [min(r["methods"][other][metric], 1e6) for r in g], hib))
            for metric, hib in (("sat", True), ("mlu", False)):
                rows.append(wilcoxon_row(f"stable vs frozen_p1 · {scope_name} · actor {metric}",
                                         [min(r["methods"]["actor_stable"][metric], 1e6) for r in g],
                                         [min(r["methods"]["frozen_p1"][metric], 1e6) for r in g], hib))
    adj = holm([r["p"] for r in rows])
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["significant_holm_0.05"] = a < 0.05
    return rows


def mcnemar_tests():
    rows = defaultdict(lambda: defaultdict(list))
    with gzip.open(F / "e4_twin_error" / "paired_rows.csv.gz", "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["twin"] != "improved":
                continue
            key = (r["group"], float(r["sigma"]), r["topology"], r["rep"], r["case"])
            rows[key][r["variant"]] = r
    by = defaultdict(list)
    for (group, sigma, *_), v in rows.items():
        by[(group, sigma)].append(v)
    out = []
    for (group, sigma), vs in sorted(by.items()):
        if sigma == 0:
            continue
        tb = lambda v, var, k: v[var][k] == "True"  # noqa: E731
        for ev in ("E1", "E2", "E1r"):
            out.append(mcnemar_row(f"unsafe accept D vs {ev} · {group} · sigma={sigma}",
                                   [tb(v, "D", "unsafe_accept") for v in vs], [tb(v, ev, "unsafe_accept") for v in vs]))
            out.append(mcnemar_row(f"false reject D vs {ev} · {group} · sigma={sigma}",
                                   [tb(v, "D", "false_reject") for v in vs], [tb(v, ev, "false_reject") for v in vs]))
    # Dm rows are not in paired_rows (derived); recompute from summary-level discordance file
    disc = list(csv.DictReader(open(F / "e4_twin_error" / "d_vs_e_discordance.csv", encoding="utf-8")))
    for d in disc:
        if d["ref"] == "Dm" and d["twin"] == "improved" and d["scope"] in ("dev", "holdout") and float(d["sigma"]) > 0:
            for what, a, b in (("unsafe accept", "D_only_unsafe_accepts", "E_only_unsafe_accepts"),
                               ("false reject", "D_only_false_rejects", "E_only_false_rejects")):
                n10, n01 = int(d[a]), int(d[b])
                p = float(stats.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
                out.append({"comparison": f"{what} Dm vs E1 · {d['scope']} · sigma={float(d['sigma'])}",
                            "n_pairs": int(d["n_pairs"]), "left_only": n10, "right_only": n01,
                            "rate_left": None, "rate_right": None, "p": p})
    adj = holm([r["p"] for r in out])
    for r, a in zip(out, adj):
        r["p_holm"] = a
        r["significant_holm_0.05"] = a < 0.05
    return out


def overhead(grid_rows):
    e2 = [r for r in read_jsonl(F / "e2_holdout_af" / "case_rows.jsonl") if r["policy_id"] == "improved_stable"]
    calib = read_jsonl(F / "calibrated_traffic" / "calibration_rows.jsonl")
    import numpy as _np
    e4 = {}
    for name in ("B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"):
        p = F / "e4_twin_error" / "cache" / f"{name}.npz"
        if p.exists():
            z = _np.load(p)
            e4[name] = {"policy_ms": float(_np.median(z["policy_ms"])), "twin_ms": float(_np.median(z["twin_ms"]))}
    table = []
    for name in ("B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"):
        g = [r for r in grid_rows if r["topology"] == name]
        e = [r for r in e2 if r["topology"] == name]
        med = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
        twin = med([r["variants"]["E"]["twin_ms"] for r in e]) if e else e4.get(name, {}).get("twin_ms")
        gate = med([r["variants"]["E"]["decision_ms"] - r["variants"]["E"]["twin_ms"] for r in e]) if e else None
        lp = med([r["lp_runtime_s"] * 1000 for r in calib if r["topology"] == name and r.get("lp_runtime_s")
                  and not r.get("lp_reused_identical_tm")])
        table.append({"topology": name,
                      "policy_ms (stable GNN)": med([r["ms"]["actor_stable"] for r in g]) or e4.get(name, {}).get("policy_ms"),
                      "twin_ms (5-member ensemble)": twin,
                      "gate+fallback_ms (decision minus twin)": gate,
                      "fallback_select_ms": med([r["ms"]["fallback_select"] for r in g]),
                      "lp_ms (min-MLU, HiGHS)": lp,
                      "frozen_p1_actor_ms": med([r["ms"]["frozen_p1"] for r in g]),
                      "source": "grid rows + E2 case rows (GEANT/Kdl) + E4 cache (dev twin) + calibrated-load LP timings"})
    return table


def taxonomy():
    """Classify every unsafe or overloaded applied decision of the framework (gate E1/E, frozen c)."""
    res = {}
    # sigma = 0: E2 holdout (frozen stack) + ablations dev 'full' rows; under twin error: E4 paired rows.
    calib = {(r["topology"], r["position"]): r for r in read_jsonl(F / "calibrated_traffic" / "calibration_rows.jsonl")}
    abl = read_jsonl(F / "ablations_v2" / "sequence_rows.jsonl")
    cats = defaultdict(lambda: defaultdict(int))
    for r in abl:
        fr = r["configs"]["full"]
        scope = "dev" if r["topology"] in DEV else "holdout"
        if not fr["overload"]:
            continue
        if fr["accept"]:
            cats[scope]["unsafe accept (twin under-prediction)"] += 1
            continue
        c = calib.get((r["topology"], r["position"]))
        ref = None if c is None else (c["mlu_lp"] if c["mlu_lp"] is not None else None)
        scale = r["load_level"] if r["load_mode"] == "fixed" else (r["load_level"] / c["reference_mlu"] if c else None)
        lp_feasible = None if (ref is None or scale is None) else ref * scale <= 1 + 1e-9
        if lp_feasible is None:
            cats[scope]["fallback infeasible (LP not available: Kdl)"] += 1
        elif lp_feasible:
            cats[scope]["fallback infeasible, but a feasible k=4 routing exists (fallback limitation)"] += 1
        else:
            cats[scope]["no feasible k=4 routing exists (capacity/catalog infeasible)"] += 1
        if r["configs"]["full"]["proposal_unsafe"]:
            cats[scope]["  of which proposal unsafe (policy out of distribution / overloaded)"] += 1
    res["sigma0_framework"] = {k: dict(v) for k, v in cats.items()}
    # oracle counterfactual at sigma = 0 (perfect twin = physics twin, ablation (e))
    cf = defaultdict(lambda: {"framework_overloaded": 0, "oracle_overloaded": 0, "n": 0})
    for r in abl:
        scope = "dev" if r["topology"] in DEV else "holdout"
        cf[scope]["n"] += 1
        cf[scope]["framework_overloaded"] += int(r["configs"]["full"]["overload"])
        cf[scope]["oracle_overloaded"] += int(r["configs"]["physics_twin"]["overload"])
    res["oracle_counterfactual_sigma0"] = dict(cf)
    # under twin error: E1 unsafe accepts, split by coverage
    tw = defaultdict(lambda: defaultdict(int))
    with gzip.open(F / "e4_twin_error" / "paired_rows.csv.gz", "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["twin"] != "improved" or r["variant"] not in ("E1", "F"):
                continue
            key = f"{r['group']}|sigma={r['sigma']}"
            if r["variant"] == "F":
                tw[key]["oracle (F) overloaded applied"] += int(r["applied_overload"] == "True")
                continue
            tw[key]["n"] += 1
            if r["applied_overload"] != "True":
                continue
            if r["accept"] == "True":
                if float(r["cov2_frozen"]) < 0.9:
                    tw[key]["unsafe accept: gate miscalibration (case coverage@2σ < 0.9)"] += 1
                else:
                    tw[key]["unsafe accept: twin under-prediction (coverage OK)"] += 1
            elif r["unsafe_proposal"] == "False":
                tw[key]["false reject -> overloaded fallback"] += 1
            else:
                tw[key]["rejected unsafe proposal -> fallback infeasible"] += 1
            tw[key]["E1 overloaded applied"] += 1
    res["twin_error_E1"] = {k: dict(v) for k, v in sorted(tw.items())}
    return res


def _fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{nd}f}"
    return str(x)


def analyze():
    grid_rows = read_jsonl(GRID)
    abl = read_jsonl(F / "ablations_v2" / "sequence_rows.jsonl")
    fn = friedman_nemenyi(grid_rows)
    pt = paired_tests(grid_rows, abl)
    mc = mcnemar_tests()
    oh = overhead(grid_rows)
    tx = taxonomy()
    write_json(OUT / "friedman_nemenyi.json", fn)
    write_json(OUT / "paired_wilcoxon_holm.json", pt)
    write_json(OUT / "mcnemar_holm.json", mc)
    write_json(OUT / "overhead.json", oh)
    write_json(OUT / "failure_taxonomy.json", tx)
    write_json(OUT / "config.json", {"grid_rows": len(grid_rows), "ablation_rows": len(abl), "environment": environment(),
                                     "notes": [LP_SOLVER_NOTE, TOY_TM_NOTE]})
    L = ["# Statistics, overhead and failure analysis", "",
         f"Sources: `main_grid_rows.jsonl` ({len(grid_rows)} paired cases: 5 topologies × TMs × 8 load levels), "
         "`../ablations_v2/sequence_rows.jsonl`, `../e4_twin_error/paired_rows.csv.gz`, `../e2_holdout_af/case_rows.jsonl`, "
         "`../calibrated_traffic/calibration_rows.jsonl`. "
         f"{TOY_TM_NOTE} {LP_SOLVER_NOTE}", "",
         "## 1. Friedman + Nemenyi across (topology, load level) blocks", "",
         "Each block is the mean over the TMs of one (topology, load mode, level) cell. Mean rank 1 = best "
         "(highest satisfaction / lowest MLU). `framework` = stable policy + improved twin + gate E + fallback.", ""]
    for k, v in fn.items():
        L += [f"### {k}  (blocks = {v['n_blocks']}, Friedman χ² = {v['friedman_chi2']:.2f}, p = {v['friedman_p']:.3g})", "",
              "| Method | Mean rank |", "|---|---:|"]
        for m, rk in sorted(v["mean_rank (1 = best)"].items(), key=lambda kv: kv[1]):
            L.append(f"| {m} | {rk:.2f} |")
        L += ["", "Nemenyi p-values (pairs with p < 0.05):", ""]
        sig = [(a, b, v["nemenyi_p"][a][b]) for a, b in combinations(v["methods"], 2) if v["nemenyi_p"][a][b] < 0.05]
        L += [", ".join(f"{a}–{b} ({p:.3g})" for a, b, p in sig) or "none", ""]
    L += ["## 2. Paired Wilcoxon signed-rank (Holm-corrected) with Cliff's δ", "",
          "| Comparison (left vs right) | n | mean diff | left better | right better | p | p (Holm) | Cliff's δ |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pt:
        L.append(f"| {r['comparison']} | {r['n']} | {r['mean_diff']:+.4f} | {r['share_left_better']:.2f} | "
                 f"{r['share_right_better']:.2f} | {_fmt(r['p'])} | {_fmt(r['p_holm'])}{' *' if r['significant_holm_0.05'] else ''} | "
                 f"{r['cliffs_delta']:+.3f} |")
    L += ["", "\\* significant at 0.05 after Holm correction. Satisfaction: higher is better; MLU: lower is better.", "",
          "## 3. Exact McNemar tests on paired gate outcomes (Holm-corrected)", "",
          "| Comparison (left vs right) | pairs | left-only | right-only | p | p (Holm) |", "|---|---:|---:|---:|---:|---:|"]
    for r in mc:
        L.append(f"| {r['comparison']} | {r['n_pairs']} | {r['left_only']} | {r['right_only']} | {_fmt(r['p'])} | "
                 f"{_fmt(r['p_holm'])}{' *' if r['significant_holm_0.05'] else ''} |")
    L += ["", "## 4. Unified overhead (median milliseconds per decision)", "",
          "| Topology | Policy (GNN) | Twin (5-member) | Gate + fallback | Fallback selection | LP (min-MLU) | Frozen P1 actor |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in oh:
        L.append(f"| {r['topology']} | {_fmt(r['policy_ms (stable GNN)'], 1)} | {_fmt(r['twin_ms (5-member ensemble)'], 1)} | "
                 f"{_fmt(r['gate+fallback_ms (decision minus twin)'], 1)} | {_fmt(r['fallback_select_ms'], 1)} | "
                 f"{_fmt(r['lp_ms (min-MLU, HiGHS)'], 1)} | {_fmt(r['frozen_p1_actor_ms'], 1)} |")
    L += ["", "Gate + fallback time is only measured where the A–F runner logged it (GEANT, Kdl). LP time is per TM "
          "(calibrated-load study); LP is not tractable on Kdl. Timings are single-process CPU wall clock on the laptop and include "
          "Python overhead.", "", "## 5. Failure taxonomy", "",
          "### No twin error (σ = 0): every overloaded applied decision of the full framework", "",
          "| Scope | Cause | Count |", "|---|---|---:|"]
    for scope, d in tx["sigma0_framework"].items():
        for k, v in d.items():
            L.append(f"| {scope} | {k} | {v} |")
    L += ["", "### Oracle-twin counterfactual (σ = 0): same proposals, perfect twin", "",
          "| Scope | Cases | Framework overloaded | Oracle-twin overloaded |", "|---|---:|---:|---:|"]
    for scope, d in tx["oracle_counterfactual_sigma0"].items():
        L.append(f"| {scope} | {d['n']} | {d['framework_overloaded']} | {d['oracle_overloaded']} |")
    L += ["", "### Under member-wise twin error: overloaded applied decisions of gate E1", "",
          "| Scope / σ | Cause | Count |", "|---|---|---:|"]
    for key, d in tx["twin_error_E1"].items():
        for k, v in d.items():
            L.append(f"| {key} | {k} | {v} |")
    L += ["", "## Artifacts", "", "- `friedman_nemenyi.json`, `paired_wilcoxon_holm.json`, `mcnemar_holm.json`, "
          "`overhead.json`, `failure_taxonomy.json`, `main_grid_rows.jsonl`, `config.json`"]
    (OUT / "STATISTICS_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'STATISTICS_REPORT.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--topologies", nargs="+", required=True)
    sub.add_parser("analyze")
    a = ap.parse_args()
    grid(a.topologies) if a.cmd == "grid" else analyze()


if __name__ == "__main__":
    main()
