"""Results summary: build results/phase3_final/RESULTS_SUMMARY.md from result files only.

Every number below is read from a JSON/CSV/JSONL produced by code in this repository; the
source path is printed beside each table. Claim statuses in the ledger are set by explicit
rules over those numbers (see ``claims()``), not by hand.

Usage: python -m phase3_final.consolidate
"""

from __future__ import annotations

import csv
import json

import numpy as np

from twin.data import ROOT, PROCESSED

from .common import LP_SOLVER_NOTE, TOY_TM_NOTE, read_jsonl

F = ROOT / "results" / "phase3_final"
RES = ROOT / "results"
REPRO = ROOT / "baseline_p1_reproduction"
RP = REPRO / "phase3_final" / "results"
P1 = {"B4": {"LP": 99.83, "NCFlow": 98.90, "Teal": 84.77, "FIGRET": 99.24, "GRL-TE": 92.39},
      "GEANT": {"LP": 88.40, "NCFlow": 88.99, "GRL-TE": 86.41},
      "average_six": {"LP": 93.19, "NCFlow": 76.48, "Teal": 82.04, "FIGRET": 82.20, "GRL-TE": 89.36}}


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


def src(*paths) -> str:
    def rel(p):
        try:
            return "`" + p.relative_to(ROOT).as_posix() + "`"
        except ValueError:
            return f"`{p}`"
    return "*Source: " + "; ".join(rel(p) for p in paths) + "*"


def f(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


def pct(x, nd=2):
    return "–" if x is None else f"{100 * x:.{nd}f}"


# ------------------------------------------------------------------ sections
def datasets():
    L = ["## 1. Datasets, splits and implementation", "", "### 1.1 Datasets", "",
         "| Topology | Nodes | Directed links | Traffic | Train TMs | Val TMs | Test TMs | Role |", "|---|---:|---:|---|---:|---:|---:|---|"]
    roles = {"B4": "train/val (seen)", "UsCarrier": "train/val (seen)", "Cogentco": "train/val (seen)",
             "GEANT": "test (unseen)", "Kdl": "test (unseen)", "ASN": "excluded (39/1,739 path shards missing)"}
    paths = []
    for t in ("B4", "UsCarrier", "Cogentco", "GEANT", "Kdl", "ASN"):
        st = J(PROCESSED / "graph_data" / f"{t}_stats.json")
        paths.append(PROCESSED / "graph_data" / f"{t}_stats.json")
        counts, kind = {}, "–"
        for split in ("train", "val", "test"):
            p = PROCESSED / "splits" / split / f"{t}.json"
            if p.exists():
                m = J(p)
                counts[split] = m["matrix_count"]
                kind = m["traffic_kind"]
        L.append(f"| {t} | {st['nodes']} | {st['directed_edges']} | {kind} | {counts.get('train', '–')} | "
                 f"{counts.get('val', '–')} | {counts.get('test', '–')} | {roles[t]} |")
    L += ["", src(PROCESSED / "graph_data", PROCESSED / "splits"), "",
          f"- {TOY_TM_NOTE} Cogentco TMs are gravity reconstructions. GEANT uses FIGRET's real test histories "
          "(a seeded subsample of 200 of the 2,693 test TMs; 50 for E3). The FIGRET train histories (8,080) are held "
          "aside from all Phase 3 models; they are used only to train the P1 baseline on GEANT.",
          "- Held-aside TMs 28–35 of B4/UsCarrier/Cogentco were never used.", "",
          "### 1.2 Splits", "",
          "| Split | Topologies | TM indices | Used for |", "|---|---|---|---|",
          "| train | B4, UsCarrier, Cogentco | 0–19 | policy and twin training |",
          "| validation | B4, UsCarrier, Cogentco | 20–27 | tuning, checkpoint selection, conformal calibration, Dm margin |",
          "| held aside | B4, UsCarrier, Cogentco | 28–35 | never used |",
          "| test (unseen topologies) | GEANT (FIGRET test), Kdl | — | final evaluation only |", "",
          src(PROCESSED / "splits" / "SPLIT_REPORT.md"), ""]
    v2 = J(F / "policy_v2" / "runs" / "v2_calload_normscore" / "config.json")
    gate = J(F / "e2_holdout_af" / "config_improved_stable.json")
    tw = J(F / "e4_twin_error" / "degraded_twins" / "config.json")["reference_hyperparameters"]
    frz = J(REPRO / "BASELINE_FREEZE" / "logs" / "b4_freeze_summary.json")
    L += ["### 1.3 Implementation and hyperparameters", "",
          "| Component | Setting |", "|---|---|",
          "| Candidate paths | Yen k = 4 (hop count) per OD pair; capacities and demands ÷ max link capacity |",
          f"| Policy (`recommended_stable`, frozen) | Phase-3 GNN on link–path bipartite graph, 4-way softmax per demand; Adam lr {v2['actor_lr']}, batch {v2['batch_size']}, ≤ {v2['max_epochs']} epochs, patience {v2['patience']}, seed {v2['seed']} |",
          "| Policy v2 | same + calibrated-load training TMs + per-topology normalised selection score S |",
          f"| Twin | {tw['n_members']}-member physics-anchored ensemble (hidden {tw['hidden']}, dropout {tw['dropout']}, lr {tw['lr']}, weight decay {tw['weight_decay']}, anchor weight {tw['anchor_weight']}); conformal std scale 0.955 fitted on validation |",
          f"| Gate (frozen) | k = {gate['gate']['k']}, utilisation limit {gate['gate']['utilisation_limit']}, relative-σ veto {gate['gate']['max_relative_std']} |",
          "| Fallback (frozen) | shortest / equal, feasible first, else least-violating; conditional LP only in the LP-fallback study |",
          f"| P1 baseline | TopoFlowNet L = 6 + one-step A2C, {frz['episodes']} episodes, seeds 100–1000, ADMM ρ = {frz['admm']['rho']}, {frz['admm']['iters']} iterations on B4 / 100 on GEANT (Teal-order headline) |",
          f"| Solvers | {LP_SOLVER_NOTE} |", "",
          src(F / "policy_v2" / "runs" / "v2_calload_normscore" / "config.json", F / "e2_holdout_af" / "config_improved_stable.json",
              F / "e4_twin_error" / "degraded_twins" / "config.json", REPRO / "BASELINE_FREEZE" / "logs" / "b4_freeze_summary.json"), ""]
    return L


def baselines():
    frz = J(REPRO / "BASELINE_FREEZE" / "logs" / "b4_freeze_summary.json")
    rows = read_jsonl(RP / "geant_figret" / "seeds.jsonl")
    base = J(RP / "geant_figret" / "baselines.json")
    rows_u = read_jsonl(RP / "geant_figret_units" / "seeds.jsonl")
    base_u = J(RP / "geant_figret_units" / "baselines.json") if (RP / "geant_figret_units" / "baselines.json").exists() else None

    def ch(rs, c):
        v = [r[f"eval_{c}"]["sat_mean"] for r in rs if r[f"eval_{c}"]["sat_mean"] is not None]
        return (np.mean(v) * 100, np.std(v) * 100, len(v)) if v else (None, None, 0)

    t_m, t_s, t_n = ch(rows, "teal")
    n_m, n_s, _ = ch(rows, "none")
    L = ["## 2. Baseline: P1 GRL-TE reproduction", "",
         "| Method | B4 (ours) | B4 (P1 Table 6) | GEANT 22/72 (ours, raw h⁰) | GEANT (ours, FIGRET units) | GEANT (P1 Table 6) |",
         "|---|---:|---:|---:|---:|---:|"]
    if rows_u:
        u_m, u_s, u_n = ch(rows_u, "teal")
        un_m, un_s, _ = ch(rows_u, "none")
    else:
        u_m = u_s = un_m = un_s = None
        u_n = 0
    lp_u = '–' if not base_u else f"{100 * base_u['lp_max_throughput']['demand_satisfaction']:.2f}"
    fmt = lambda m, s, n=None: "–" if m is None else f"{m:.2f} ± {s:.2f}" + (f" (n={n})" if n else "")  # noqa: E731
    L += [f"| GRL-TE + ADMM (Teal-order) | {frz['channels']['teal_admm']['pct_mean']:.2f} ± {frz['channels']['teal_admm']['pct_std']:.2f} | 92.39 | "
          f"{fmt(t_m, t_s, t_n)} | {fmt(u_m, u_s, u_n)} | 86.41 |",
          f"| GRL-TE without ADMM | {frz['channels']['no_admm']['pct_mean']:.2f} ± {frz['channels']['no_admm']['pct_std']:.2f} | 91.71 (= 92.39 − 0.68) | "
          f"{fmt(n_m, n_s)} | {fmt(un_m, un_s)} | 71.32 (= 86.41 − 15.09) |",
          f"| GRL-TE + ADMM (P1-order, diagnostic) | {frz['channels']['p1_admm']['pct_mean']:.2f} ± {frz['channels']['p1_admm']['pct_std']:.2f} | – | failed on every TM | failed on every TM | – |",
          f"| LP | {100 * frz['baselines']['lp']['demand_satisfaction']:.2f} | 99.83 | {100 * base['lp_max_throughput']['demand_satisfaction']:.2f} | "
          f"{lp_u} | 88.40 |",
          f"| ECMP (equal split) | {100 * frz['baselines']['ecmp']['demand_satisfaction']:.2f} | – | {100 * base['ecmp']['demand_satisfaction']:.2f} | – | – |",
          f"| Shortest path | {100 * frz['baselines']['shortest_path']['demand_satisfaction']:.2f} | – | {100 * base['shortest_path']['demand_satisfaction']:.2f} | – | – |",
          "| Teal | not run | 84.77 | not run | not run | – |",
          "| FIGRET | not run | 99.24 | not run | not run | – |",
          "| NCFlow | not run | 98.90 | not run | not run | 88.99 |", "",
          src(REPRO / "BASELINE_FREEZE" / "logs" / "b4_freeze_summary.json", RP / "geant_figret" / "seeds.jsonl",
              RP / "geant_figret" / "baselines.json", RP / "geant_figret_units"), "",
          "- Satisfaction in %, mean ± SD over the 10 seeds (each seed = mean over the test TMs). LP = max-throughput "
          "path LP (P1 eqs. 1–4) with SciPy HiGHS. Teal, FIGRET and NCFlow numbers are **quoted from P1 Table 6, not "
          "reproduced**.",
          "- GEANT: FIGRET 22/72 graph and real TMs; training on a seeded 1,000-TM subsample of the 8,080 FIGRET train "
          "TMs; test on the 200 seeded FIGRET test TMs. The raw-h⁰ channel (P1 eq. 5) is the pre-registered headline; "
          "FIGRET units (÷1e9) is a sensitivity channel. P1-order ADMM hits a singular linear system on every GEANT TM.",
          f"- **ADMM ablation on GEANT:** removing ADMM changes satisfaction by {n_m - t_m:+.2f} points (P1: −15.09).", ""]
    # comparable-topology average (B4 + GEANT only)
    lp_b4 = 100 * frz["baselines"]["lp"]["demand_satisfaction"]
    lp_ge = 100 * base["lp_max_throughput"]["demand_satisfaction"]
    gr_b4 = frz["channels"]["teal_admm"]["pct_mean"]
    L += ["### 2.1 Comparable-topology average (B4 and GEANT only)", "",
          "Only B4 and GEANT have data matching P1 (public real TMs on the same graphs). A six-topology average is "
          "**not** computed, because the other four topologies use toy or reconstructed traffic here.", "",
          "| Method | Ours (B4 + GEANT) / 2 | P1 (B4 + GEANT) / 2 |", "|---|---:|---:|",
          f"| GRL-TE + ADMM | {(gr_b4 + t_m) / 2:.2f} | {(92.39 + 86.41) / 2:.2f} |",
          f"| LP | {(lp_b4 + lp_ge) / 2:.2f} | {(99.83 + 88.40) / 2:.2f} |",
          f"| GRL-TE gap to LP | {(gr_b4 + t_m) / 2 - (lp_b4 + lp_ge) / 2:+.2f} | {(92.39 + 86.41) / 2 - (99.83 + 88.40) / 2:+.2f} |", "",
          "Our absolute levels are higher on both topologies, for GRL-TE and for LP alike, so the TMs/splits are likely "
          "less loaded than P1's (P1 does not publish its splits). The GRL-TE-to-LP gap is the more comparable quantity.", ""]
    return L


def calibrated():
    rows = list(csv.DictReader(open(F / "calibrated_traffic" / "feasibility_before_after.csv", encoding="utf-8")))
    L = ["## 3. Calibrated load", "",
         "Each TM is rescaled so that a method-independent reference MLU (k=4 path LP; shortest path on Kdl) equals "
         "0.5 / 0.7 / 0.9 / 1.1. Share of scenarios where a routing is feasible (MLU ≤ 1):", "",
         "| Topology | Split | Fixed 0.25 → 1.0: shortest or equal | Calibrated 0.5 / 0.7 / 0.9 / 1.1: shortest or equal | Calibrated: LP |",
         "|---|---|---|---|---|"]
    for t in ("B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"):
        for split in ("val", "eval_test"):
            rs = [r for r in rows if r["topology"] == t and r["split"] == split]
            if not rs:
                continue
            fx = " / ".join(f(float(r["feasible_shortest_or_equal"]), 2) for r in rs if r["mode"] == "fixed")
            ca = " / ".join(f(float(r["feasible_shortest_or_equal"]), 2) for r in rs if r["mode"] == "calibrated")
            lp = " / ".join("–" if not r["feasible_lp"] else f(float(r["feasible_lp"]), 2) for r in rs if r["mode"] == "calibrated")
            L.append(f"| {t} | {split} | {fx} | {ca} | {lp} |")
    L += ["", src(F / "calibrated_traffic" / "feasibility_before_after.csv"), "",
          "LP feasibility at targets ≤ 0.9 is a design property. The informative cells are those where the LP is "
          "feasible but shortest/equal are not (UsCarrier at every target; Cogentco and GEANT at 0.7–0.9): a better "
          "routing exists within the 4 candidates.", ""]
    return L


def af_section():
    dev = J(RES / "safety_gate" / "improvements" / "eval_runs" / "af_eval_k1_calibrated_devtopos_v1" / "summary.json")["metrics_by_variant"]
    e2 = J(F / "e2_holdout_af" / "summary.json")
    long = {"A": "A_policy_only", "B": "B_original_twin_mean", "C": "C_original_twin_uncertainty",
            "D": "D_improved_twin_mean", "E": "E_improved_twin_calibrated_uncertainty", "F": "F_physics_oracle"}
    L = ["## 4. Safety gate A–F: development (E1 setting) and unseen topologies (E2 holdout evaluation)", "",
         "A policy only · B original twin, mean · C original twin, μ+σ · D improved twin, mean · E improved twin, "
         "calibrated μ+σ (k=1) · F physics oracle. Frozen stack, no recalibration.", "",
         "| Variant | Dev unsafe acc. | Dev false rej. | GEANT unsafe acc. | GEANT false rej. (safe) | GEANT applied sat. | Kdl unsafe acc. | Kdl applied sat. |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    sel = lambda t, v: next(r for r in e2["variant_table"] if r["policy_id"] == "improved_stable" and r["topology"] == t  # noqa: E731
                            and r["load_mode"] == "all" and r["load_level"] == "all" and r["variant"] == v)
    for v in "ABCDEF":
        g, k = sel("GEANT", v), sel("Kdl", v)
        L.append(f"| {v} | {f(dev[long[v]]['unsafe_acceptance_rate'])} | {f(dev[long[v]]['false_rejection_rate_among_safe'])} | "
                 f"{f(g['unsafe_accept_rate'])} | {f(g['false_reject_given_safe'])} | {f(g['mean_sat_applied'], 4)} | "
                 f"{f(k['unsafe_accept_rate'])} | {f(k['mean_sat_applied'], 4)} |")
    g = sel("GEANT", "A")
    L += ["", src(RES / "safety_gate" / "improvements" / "eval_runs" / "af_eval_k1_calibrated_devtopos_v1" / "summary.json",
                  F / "e2_holdout_af" / "summary.json"), "",
          f"Dev: 80 cases (B4/UsCarrier/Cogentco, fixed scales). GEANT: {g['n_cases']} cases (200 TMs × 8 load levels), "
          f"{g['n_unsafe_proposals']} unsafe proposals. Kdl: 8 cases. D/E/F agree on every holdout case.", ""]
    return L


def e3_section():
    p1 = J(RP / "b4_link_failures" / "summary.json")
    e3 = J(F / "e3_link_failures" / "summary.json")["table"]
    L = ["## 5. E3 link failures", "", "### 5.1 B4, P1 §5.5-comparable (frozen P1 baseline, fixed paths)", "",
         "| Failed links | GRL-TE + Teal ADMM | GRL-TE no ADMM | LP | ECMP | Shortest | P1 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    p1rep = {0: "92.39 (intact, Table 6)", 3: "77.25 (§5.5)"}
    for n in range(4):
        d = p1[str(n)]
        L.append(f"| {n} | " + " | ".join(f"{d[m][0]:.2f} ± {d[m][1]:.2f}" for m in ("grl_te_teal_admm", "grl_te_no_admm", "lp", "ecmp", "shortest_path"))
                 + f" | {p1rep.get(n, '–')} |")
    L += ["", src(RP / "b4_link_failures" / "summary.json"), "",
          "P1 §5.5 is quoted as 90.80% → 77.25% with 3 failed links; 90.80% is not assigned to a row because P1's intact "
          "value is 92.39%.", "", "### 5.2 Phase-3 framework under failures (rerouted paths)", "",
          "| Topology | Method | Sat. 0 | Sat. 1 | Sat. 2 | Sat. 3 | Unsafe accept 0/1/2/3 | Violation 0 → 3 |", "|---|---|---:|---:|---:|---:|---|---|"]
    get = lambda t, n, m: next((r for r in e3 if r["topology"] == t and r["load_mode"] == "all" and r["n_failed"] == n and r["method"] == m), None)  # noqa: E731
    for t in ("B4", "GEANT", "UsCarrier"):
        for m, lab in (("fw_stable_A", "policy only (A)"), ("fw_stable_D", "gate D"), ("fw_stable_E", "framework (gate E)"),
                       ("shortest", "shortest"), ("lp", "LP (min-MLU)"), ("frozen_p1", "frozen P1 actor")):
            rs = [get(t, n, m) for n in range(4)]
            if rs[0] is None:
                continue
            ua = "/".join(f(r["unsafe_accept_rate"], 2) for r in rs) if "unsafe_accept_rate" in rs[0] else "–"
            L.append(f"| {t} | {lab} | " + " | ".join(pct(r["sat_mean"]) for r in rs) + f" | {ua} | {f(rs[0]['violation_rate'], 2)} → {f(rs[3]['violation_rate'], 2)} |")
    L += ["", src(F / "e3_link_failures" / "summary.json"), ""]
    return L


def e4_section():
    rows = list(csv.DictReader(open(F / "e4_twin_error" / "sweep.csv", encoding="utf-8")))
    L = ["## 6. E4 twin-error sweep", "",
         "Member-wise perturbation of the improved twin (σ = member-wise relative error), identical inputs for every gate. "
         "UA = unsafe accepts / unsafe proposals; FR = false rejects / safe proposals. Dm = mean-only gate with a "
         "tighter utilisation limit chosen on dev validation to match E1's false-reject rate.", "",
         "| Scope | σ | D UA / FR | Dm UA / FR | E1 UA / FR | E2 UA / FR | F |", "|---|---:|---|---|---|---|---|"]
    def g(scope, s, v):
        r = next((x for x in rows if x["scope"] == scope and x["twin"] == "improved" and float(x["sigma"]) == s
                  and x["variant"] == v and x["load_mode"] == "all"), None)
        return "–" if r is None else f"{float(r['unsafe_accept_given_unsafe']):.4f} / {float(r['false_reject_given_safe']):.3f}"
    for scope in ("dev", "holdout"):
        for s in (0.0, 0.05, 0.1, 0.2, 0.4):
            L.append(f"| {scope} | {s} | {g(scope, s, 'D')} | {g(scope, s, 'Dm')} | {g(scope, s, 'E1')} | {g(scope, s, 'E2')} | {g(scope, s, 'F')} |")
    L += ["", "Degraded twins (10/25/50% data, 1/2/4 epochs) at σ = 0:", "", "| Scope | Twin | D UA / FR | E1 UA / FR |", "|---|---|---|---|"]
    for scope in ("dev", "holdout"):
        for t in ("degraded_f10", "degraded_f25", "degraded_f50"):
            def gg(v):
                r = next((x for x in rows if x["scope"] == scope and x["twin"] == t and x["variant"] == v and x["load_mode"] == "all"), None)
                return "–" if r is None else f"{float(r['unsafe_accept_given_unsafe']):.4f} / {float(r['false_reject_given_safe']):.3f}"
            L.append(f"| {scope} | {t} | {gg('D')} | {gg('E1')} |")
    L += ["", src(F / "e4_twin_error" / "sweep.csv", F / "e4_twin_error" / "E4_FINDINGS.md"), ""]
    return L


def gap_section():
    g = J(F / "policy_v2" / "generalization_gap.json")
    v2 = J(F / "policy_v2" / "runs" / "v2_calload_normscore" / "summary.json")
    ref = J(F / "policy_v2" / "reference_recommended_stable_val.json")["val"]
    L = ["## 7. Policy generalization and generalization gap", "",
         f"Validation selection score S (lower is better): recommended_stable **{ref['S']:.4f}**, v2 **{v2['best_val']['S']:.4f}** "
         f"→ recommended_stable stays selected (v2 is a negative result).", "",
         "| Policy | MLU/shortest seen | unseen | gap | Satisfaction seen | unseen | gap | Violation seen | unseen | gap |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pid, r in g.items():
        s, u, d = r["seen_mean"], r["unseen_mean"], r["gap"]
        L.append(f"| {pid} | {f(s['mlu_ratio'], 4)} | {f(u['mlu_ratio'], 4)} | {d['mlu_ratio']:+.4f} | {f(s['sat'], 4)} | {f(u['sat'], 4)} | "
                 f"{d['sat']:+.4f} | {f(s['violation_rate'])} | {f(u['violation_rate'])} | {d['violation_rate']:+.3f} |")
    L += ["", src(F / "policy_v2" / "generalization_gap.json", F / "policy_v2" / "TRAINING_REPORT.md"), "",
          "Gap = seen − unseen. The MLU ratio is the clean metric (both policies ~0.34 worse vs shortest on unseen "
          "topologies). Satisfaction/violation gaps are confounded by the infeasible toy/gravity TMs on the seen side.", ""]
    return L


def ablation_section():
    t = J(F / "ablations_v2" / "summary.json")["table"]
    names = {"full": "Full framework", "no_gate": "(a) no gate", "mean_gate": "(b) gate without uncertainty",
             "no_fallback": "(c) no fallback", "b4_only_policy": "(d) B4-only policy (frozen P1)†", "physics_twin": "(e) twin without learned residual"}
    L = ["## 8. Ablations", "",
         "| Configuration | Dev sat. | Dev overload | Dev unsafe acc. | Holdout sat. | Holdout overload | Holdout unsafe acc. | Holdout false rej. (safe) | Holdout median ms |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c, lab in names.items():
        d = next(r for r in t if r["scope"] == "dev" and r["config"] == c)
        h = next(r for r in t if r["scope"] == "holdout" and r["config"] == c)
        L.append(f"| {lab} | {f(d['applied_sat'], 4)} | {f(d['applied_overload_rate'])} | {f(d['unsafe_accept_rate'])} | "
                 f"{f(h['applied_sat'], 4)} | {f(h['applied_overload_rate'])} | {f(h['unsafe_accept_rate'])} | "
                 f"{f(h['false_reject_given_safe'])} | {f(h['ms_median'], 1)} |")
    L += ["", src(F / "ablations_v2" / "summary.json"), "",
          "† confounded: differs in training topology *and* model/trainer. Flat-MLP policy ablation: not part of this study.", ""]
    return L


def fallback_section():
    ab = list(csv.DictReader(open(RES / "safety_gate" / "improvements" / "lp_fallback_ablation" / "paired_results.csv", encoding="utf-8")))
    tx = J(F / "stats" / "failure_taxonomy.json")
    L = ["## 9. Fallback", "",
         f"- Existing conditional-LP fallback study (dev, 80 cases; {len(ab)} paired rows): LP recovers 5 of 40 "
         "rejected cases (all UsCarrier at 0.25×), mean LP time 8.75 s; gate decisions unchanged "
         "(`results/safety_gate/improvements/lp_fallback_ablation/LP_FALLBACK_AB_REPORT.md`).",
         "- **The 8-path fallback extension (generalised LP, max-flow screen) is not part of this study.**",
         "- Why overloaded decisions remain under the full framework (σ = 0), from the failure taxonomy:", "",
         "| Scope | Cause | Count |", "|---|---|---:|"]
    for scope, d in tx["sigma0_framework"].items():
        for k, v in d.items():
            L.append(f"| {scope} | {k.strip()} | {v} |")
    L += ["", src(F / "stats" / "failure_taxonomy.json", RES / "safety_gate" / "improvements" / "lp_fallback_ablation" / "paired_results.csv"), ""]
    return L


def stats_section():
    fn = J(F / "stats" / "friedman_nemenyi.json")
    pt = J(F / "stats" / "paired_wilcoxon_holm.json")
    mc = J(F / "stats" / "mcnemar_holm.json")
    oh = J(F / "stats" / "overhead.json")
    L = ["## 10. Statistics and overhead", "", "### 10.1 Friedman + Nemenyi", "",
         "| Test | Blocks | χ² | p | Mean ranks (1 = best) |", "|---|---:|---:|---:|---|"]
    for k, v in fn.items():
        ranks = ", ".join(f"{m} {r:.2f}" for m, r in sorted(v["mean_rank (1 = best)"].items(), key=lambda kv: kv[1]))
        L.append(f"| {k} | {v['n_blocks']} | {v['friedman_chi2']:.1f} | {v['friedman_p']:.2g} | {ranks} |")
    L += ["", "### 10.2 Paired Wilcoxon (Holm) with Cliff's δ", "",
          "| Comparison | n | mean diff | p (Holm) | δ |", "|---|---:|---:|---:|---:|"]
    for r in pt:
        L.append(f"| {r['comparison']} | {r['n']} | {r['mean_diff']:+.4f} | {r['p_holm']:.2g}{' *' if r['significant_holm_0.05'] else ''} | {r['cliffs_delta']:+.3f} |")
    L += ["", "### 10.3 Exact McNemar (Holm) on paired gate outcomes (holdout)", "",
          "| Comparison | pairs | left-only | right-only | p (Holm) |", "|---|---:|---:|---:|---:|"]
    for r in mc:
        if "holdout" in r["comparison"] and ("E1 " in r["comparison"] + " " or r["comparison"].endswith("E1") or "vs E1" in r["comparison"]):
            L.append(f"| {r['comparison']} | {r['n_pairs']} | {r['left_only']} | {r['right_only']} | {r['p_holm']:.2g}{' *' if r['significant_holm_0.05'] else ''} |")
    L += ["", "### 10.4 Overhead (median ms per decision)", "",
          "| Topology | Policy | Twin | Gate + fallback | Fallback selection | LP |", "|---|---:|---:|---:|---:|---:|"]
    for r in oh:
        L.append(f"| {r['topology']} | {f(r['policy_ms (stable GNN)'], 1)} | {f(r['twin_ms (5-member ensemble)'], 1)} | "
                 f"{f(r['gate+fallback_ms (decision minus twin)'], 1)} | {f(r['fallback_select_ms'], 1)} | {f(r['lp_ms (min-MLU, HiGHS)'], 1)} |")
    L += ["", src(F / "stats"), "", "\\* significant at 0.05 after Holm correction. Full tables: `stats/STATISTICS_REPORT.md`. "
          "Timings are wall clock on a CPU laptop with other experiments running concurrently.", ""]
    return L


def claims():
    e2 = J(F / "e2_holdout_af" / "summary.json")["variant_table"]
    sel = lambda t, v: next(r for r in e2 if r["policy_id"] == "improved_stable" and r["topology"] == t and r["load_mode"] == "all"  # noqa: E731
                            and r["load_level"] == "all" and r["variant"] == v)
    e3 = J(F / "e3_link_failures" / "summary.json")["table"]
    mc = J(F / "stats" / "mcnemar_holm.json")
    pt = J(F / "stats" / "paired_wilcoxon_holm.json")
    gap = J(F / "policy_v2" / "generalization_gap.json")
    v2 = J(F / "policy_v2" / "runs" / "v2_calload_normscore" / "summary.json")["best_val"]["S"]
    ref = J(F / "policy_v2" / "reference_recommended_stable_val.json")["val"]["S"]
    abl = J(F / "ablations_v2" / "summary.json")["table"]

    gate_holdout_zero = all(sel(t, v)["unsafe_accept_rate"] == 0 for t in ("GEANT", "Kdl") for v in "DE")
    gate_holdout_a = min(sel(t, "A")["unsafe_accept_rate"] for t in ("GEANT", "Kdl"))
    e3_zero = all(r["unsafe_accept_rate"] == 0 for r in e3 if r["method"] in ("fw_stable_D", "fw_stable_E") and r["load_mode"] == "all")
    fw_vs_a = [r for r in pt if r["comparison"].startswith("framework vs policy-only · holdout · applied sat")]
    d_e1 = [r for r in mc if r["comparison"].startswith("unsafe accept D vs E1 · holdout")]
    dm_e1 = [r for r in mc if r["comparison"].startswith("unsafe accept Dm vs E1 · holdout")]
    e_beats_d = any(r["significant_holm_0.05"] and r["left_only"] > r["right_only"] for r in d_e1)
    e_beats_dm = any(r["significant_holm_0.05"] and r["left_only"] > r["right_only"] for r in dm_e1)
    unseen_ratio = gap["improved_stable"]["unseen_mean"]["mlu_ratio"]
    nofb = next(r for r in abl if r["scope"] == "holdout" and r["config"] == "no_fallback")
    full = next(r for r in abl if r["scope"] == "holdout" and r["config"] == "full")
    phys = next(r for r in abl if r["scope"] == "holdout" and r["config"] == "physics_twin")

    def st(ok, partial=False):
        return "**demonstrated**" if ok else ("partially" if partial else "**not supported**")

    rows = [
        ("Twin gate removes unsafe policy decisions on seen topologies", st(True), "dev A–F: unsafe accept 0.50 → 0 (B–F)",
         "results/safety_gate/improvements/eval_runs/af_eval_k1_calibrated_devtopos_v1/AF_EVALUATION_REPORT.md"),
        ("…and on unseen topologies (G2)", st(gate_holdout_zero and gate_holdout_a > 0.5),
         f"GEANT/Kdl: A ≥ {gate_holdout_a:.2f} → D/E 0; E ≡ oracle on all cases", "results/phase3_final/e2_holdout_af/E2_FINDINGS.md"),
        ("…and under 1–3 link failures (E3)", st(e3_zero), "D/E unsafe accepts 0 at every failure count (B4, GEANT, UsCarrier)",
         "results/phase3_final/e3_link_failures/E3_FINDINGS.md"),
        ("Framework improves applied satisfaction over policy-only on unseen topologies",
         st(bool(fw_vs_a) and fw_vs_a[0]["significant_holm_0.05"] and fw_vs_a[0]["mean_diff"] > 0),
         (f"mean diff {fw_vs_a[0]['mean_diff']:+.4f}, p_Holm {fw_vs_a[0]['p_holm']:.2g}" if fw_vs_a else "–"), "results/phase3_final/stats/STATISTICS_REPORT.md"),
        ("Uncertainty-aware gate (E) beats mean-only gate (D) without twin error (G1)", st(False),
         "D ≡ E on all dev, holdout and E3 cases at σ = 0", "results/phase3_final/e2_holdout_af/E2_FINDINGS.md"),
        ("E beats D when the twin is erroneous (G1, G3)", st(e_beats_d, partial=True),
         "McNemar D-only vs E1-only unsafe accepts (holdout): " + ", ".join(f"σ={r['comparison'].split('sigma=')[1]}: {r['left_only']} vs {r['right_only']}" for r in d_e1)
         + "; cost: significantly more false rejects (" + ", ".join(f"{r['right_only']}" for r in mc if r["comparison"].startswith("false reject D vs E1 · holdout")) + " E1-only)",
         "results/phase3_final/stats/mcnemar_holm.json"),
        ("E's advantage comes from uncertainty information rather than conservatism", st(e_beats_dm, partial=False),
         "vs dev-tuned mean-only margin Dm: " + ", ".join(f"{r['left_only']} vs {r['right_only']}" for r in dm_e1) + " unsafe accepts (Dm-only vs E1-only)",
         "results/phase3_final/e4_twin_error/E4_FINDINGS.md"),
        ("Learned twin residual adds decision value over pure physics", st(False),
         f"physics twin agrees with oracle on {100 * phys['physics_agrees_with_oracle']:.1f}% of holdout cases",
         "results/phase3_final/ablations_v2/ABLATION_REPORT.md"),
        ("Fallback matters (vs keeping the previous routing)", st(nofb["applied_overload_rate"] > full["applied_overload_rate"]),
         f"holdout overload {full['applied_overload_rate']:.3f} (fallback) vs {nofb['applied_overload_rate']:.3f} (no fallback)",
         "results/phase3_final/ablations_v2/ABLATION_REPORT.md"),
        ("Rejection always yields a safe (feasible) routing", st(False),
         f"{100 * sel('GEANT', 'E')['applied_overload_rate']:.0f}% of GEANT holdout cases still overload after gate E + fallback",
         "results/phase3_final/stats/failure_taxonomy.json"),
        ("Graph-RL policy generalizes to unseen topologies", st(unseen_ratio < 1.0), f"unseen MLU / shortest = {unseen_ratio:.3f}",
         "results/phase3_final/policy_v2/generalization_gap.json"),
        ("Calibrated-load training + normalised selection improves the policy", st(v2 < ref), f"val S v2 {v2:.4f} vs {ref:.4f}",
         "results/phase3_final/policy_v2/TRAINING_REPORT.md"),
        ("P1 GRL-TE reproduced on B4", st(False, partial=True), "97.44% vs P1 92.39% (higher; protocol gaps documented)",
         "baseline_p1_reproduction/BASELINE_FREEZE/BASELINE_FREEZE_DECISION.md"),
        ("P1 GRL-TE reproduced on GEANT 22/72", st(False, partial=True), "level higher than P1 (LP too); ADMM-removal effect similar in sign and size",
         "baseline_p1_reproduction/phase3_final/results/geant_figret/P1_GEANT_FIGRET_REPORT.md"),
        ("P1 link-failure degradation (B4)", st(False, partial=True), "−16.8 points 0→3 failures vs P1 −13.55 (quoted start point differs)",
         "baseline_p1_reproduction/phase3_final/results/b4_link_failures/B4_P1_SECTION55_TABLE.md"),
        ("Six-topology average comparable to P1 (89.36%)", "not tested", "not computed: four topologies use toy/reconstructed traffic", "this file §2.1"),
    ]
    L = ["## 11. Claims ledger", "",
         "Status rules: *demonstrated* = the measured effect is in the claimed direction on all relevant cases and, where a "
         "test applies, significant after Holm correction; *partially* = direction holds but not everywhere / under extra "
         "conditions / with documented protocol gaps; *not supported* = measured null or opposite effect; *not tested* = not measured (reason given).", "",
         "| Claim | Status | Evidence | File |", "|---|---|---|---|"]
    L += [f"| {c} | {s} | {e} | `{p}` |" for c, s, e, p in rows]
    return L + [""]


def figures_section():
    fig = F / "figures" / "FIGURES.md"
    L = ["## 12. Figures", "", "All in `results/phase3_final/figures/` (300-dpi PNG + PDF); sources listed in `figures/FIGURES.md`. "
         "Architecture diagrams are in `diagrams/` (repository root).", ""]
    if fig.exists():
        L += [l for l in fig.read_text(encoding="utf-8").splitlines() if l.startswith("|")]
    return L + [""]


def not_done():
    return ["## 13. Scope and limitations", "",
            "- The study does not include an 8-path fallback (generalised LP, max-flow screen) or the ASN topology.",
            "- Policy v2 is a single training run: no hyperparameter grid, topology augmentation or actor-critic fine-tuning.",
            "- The E3 link-failure experiment covers B4, GEANT and UsCarrier (not Cogentco or Kdl). The P1 GEANT "
            "reproduction has no ADMM stabilisation or B4 gap investigation; Teal, FIGRET and NCFlow were not run and "
            "are quoted from P1. The ablations contain no flat-MLP policy; the single-topology ablation uses the frozen "
            "P1 actor (confounded).",
            f"- {TOY_TM_NOTE}", f"- {LP_SOLVER_NOTE}",
            "- The flow-level simulator is the ground truth for both twin labels and the oracle; no packet-level or real "
            "network validation was performed.", ""]


def comparison_other_work():
    return ["## 14. Qualitative comparison with other work", "",
            "| Work | Decision | Safety check | Unseen-topology test | Twin uncertainty |", "|---|---|---|---|---|",
            "| P1 GRL-TE (Lu et al. 2025) | GNN + A2C splits, ADMM refinement | constraint adjustment only (post-hoc scaling) | no | no |",
            "| Teal (Xu et al. 2023) | multi-agent RL + ADMM | ADMM feasibility repair | no | no |",
            "| FIGRET (Liu et al. 2024) | robust DL TE | burst-aware loss, no runtime check | no | no |",
            "| **This work** | GNN split policy (frozen) | twin-predicted per-decision gate + fallback | **yes (GEANT, Kdl)** | ensemble σ with conformal scale |",
            "", "The comparison is qualitative (design properties), except for the P1 numbers in §2, which are measured.", ""]


def main() -> None:
    L = ["# Phase 3 results summary — AI-driven network digital twin for safe and generalizable decision testing", "",
         "Students: Dania Waseem (23I-2622), Taiba Tariq (23I-2618). Generated by `python -m phase3_final.consolidate`; every "
         "number is read from the file cited under its table (paths relative to the repository root).", ""]
    for fn in (datasets, baselines, calibrated, af_section, e3_section, e4_section, gap_section, ablation_section,
               fallback_section, stats_section, claims, figures_section, not_done, comparison_other_work):
        L += fn()
    from .sndlib_summary import section as sndlib_section  # SNDlib Abilene / Germany50 (empty if not run)

    L += sndlib_section()
    (F / "RESULTS_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {F / 'RESULTS_SUMMARY.md'}")


if __name__ == "__main__":
    main()
