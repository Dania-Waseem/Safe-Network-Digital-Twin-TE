"""Final figures (300-dpi PNG + PDF) under results/phase3_final/figures/.

Every figure reads only result files produced by this repository; the source of each
figure is written to figures/FIGURES.md. Colours follow the method (entity), never its
rank, and are identical in every figure (validated categorical palette; the three
low-contrast hues are always paired with a legend and a marker shape).

Usage: python -m phase3_final.figures
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

import numpy as np

from twin.data import ROOT

from .common import out_dir, read_jsonl

F = ROOT / "results" / "phase3_final"
OUT = out_dir("figures")
REPRO = ROOT / "baseline_p1_reproduction" / "phase3_final" / "results"

# Fixed method -> colour / marker (validated palette, light mode)
COL = {"framework": "#2a78d6", "policy": "#eb6834", "shortest": "#1baf7a", "lp": "#eda100", "equal": "#e87ba4",
       "frozen_p1": "#008300", "v2": "#6250d6", "D": "#e34948", "E1": "#2a78d6", "E2": "#6250d6", "E1r": "#1baf7a",
       "Dm": "#eda100", "F": "#008300", "A": "#eb6834"}
MRK = {"framework": "o", "policy": "s", "shortest": "^", "lp": "D", "equal": "v", "frozen_p1": "P", "v2": "X",
       "D": "s", "E1": "o", "E2": "^", "E1r": "v", "Dm": "D", "F": "P", "A": "X"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SOURCES: dict[str, str] = {}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
                         "lines.markersize": 6, "figure.dpi": 100})
    return plt


def save(fig, name: str, source: str) -> None:
    if not getattr(fig, "_rect_layout", False) and not fig.legends:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    SOURCES[name] = source
    import matplotlib.pyplot as plt
    plt.close(fig)


def line(ax, x, y, key, label, **kw):
    ax.plot(x, y, color=COL[key], marker=MRK[key], label=label, markeredgecolor="white", markeredgewidth=0.8, **kw)


# ---------------------------------------------------------------- figures
def fig_training():
    plt = _plt()
    curve = read_jsonl(F / "policy_v2" / "runs" / "v2_calload_normscore" / "training_curve.jsonl")
    ref = json.loads((F / "policy_v2" / "reference_recommended_stable_val.json").read_text(encoding="utf-8"))["val"]
    ep = [c["epoch"] + 1 for c in curve]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    line(axes[0], ep, [c["val"]["S"] for c in curve], "v2", "v2 (calibrated-load training)")
    axes[0].axhline(ref["S"], color=COL["framework"], ls="--", lw=1.5, label=f"recommended_stable ({ref['S']:.4f})")
    axes[0].set_xlabel("epoch (0 = untrained)")
    axes[0].set_ylabel("validation score S (policy MLU / shortest MLU)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Selection score (lower is better)", fontsize=10, loc="left")
    for topo, key in (("B4", "framework"), ("UsCarrier", "policy"), ("Cogentco", "shortest")):
        line(axes[1], ep, [c["val"]["ratio_by_topology"][topo] for c in curve], key, topo)
    axes[1].axhline(1.0, color=MUTED, lw=1, ls=":")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MLU ratio to shortest path")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Per-topology validation ratio (v2)", fontsize=10, loc="left")
    save(fig, "fig01_training_curves", "policy_v2/runs/v2_calload_normscore/training_curve.jsonl; "
         "policy_v2/reference_recommended_stable_val.json")


def fig_mlu_vs_load():
    plt = _plt()
    rows = read_jsonl(F / "stats" / "main_grid_rows.jsonl")
    if not rows:
        return
    topos = ["B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, t in zip(axes, topos):
        rs = [r for r in rows if r["topology"] == t and r["load_mode"] == "calibrated"]
        for key, m, lab in (("lp", "lp", "LP (min-MLU)"), ("shortest", "shortest", "shortest"), ("equal", "equal", "equal"),
                            ("policy", "actor_stable", "policy only"), ("framework", "framework", "framework (gate E)")):
            pts = defaultdict(list)
            for r in rs:
                if m in r["methods"]:
                    pts[r["load_level"]].append(min(r["methods"][m]["mlu"], 1e6))
            if pts:
                xs = sorted(pts)
                line(ax, xs, [np.mean(pts[x]) for x in xs], key, lab)
        ax.axhline(1.0, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{t}{' (unseen)' if t in ('GEANT', 'Kdl') else ''}", fontsize=10, loc="left")
        ax.set_xlabel("calibrated load target")
    axes[0].set_ylabel("mean MLU (lower is better)")
    h, lab = axes[3].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig02_mlu_vs_load", "stats/main_grid_rows.jsonl (calibrated load levels; Kdl reference = shortest, no LP)")


def fig_safety_bars():
    plt = _plt()
    dev = json.loads((ROOT / "results" / "safety_gate" / "improvements" / "eval_runs" / "af_eval_k1_calibrated_devtopos_v1"
                      / "summary.json").read_text(encoding="utf-8"))["metrics_by_variant"]
    e2 = json.loads((F / "e2_holdout_af" / "summary.json").read_text(encoding="utf-8"))["variant_table"]
    variants = ["A", "B", "C", "D", "E", "F"]
    long = {"A": "A_policy_only", "B": "B_original_twin_mean", "C": "C_original_twin_uncertainty",
            "D": "D_improved_twin_mean", "E": "E_improved_twin_calibrated_uncertainty", "F": "F_physics_oracle"}
    scopes = {"dev (B4/UsCarrier/Cogentco, n=80)": {v: (dev[long[v]]["unsafe_acceptance_rate"], dev[long[v]]["false_rejection_rate_among_safe"]) for v in variants}}
    for t in ("GEANT", "Kdl"):
        sel = {r["variant"]: r for r in e2 if r["policy_id"] == "improved_stable" and r["topology"] == t
               and r["load_mode"] == "all" and r["load_level"] == "all"}
        n = sel["A"]["n_cases"]
        scopes[f"{t} (unseen, n={n})"] = {v: (sel[v]["unsafe_accept_rate"], sel[v]["false_reject_given_safe"] or 0.0) for v in variants}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    x = np.arange(len(variants))
    for ax, (title, d) in zip(axes, scopes.items()):
        ua = [d[v][0] for v in variants]
        fr = [d[v][1] for v in variants]
        ax.bar(x - 0.2, ua, 0.38, color=COL["policy"], label="unsafe accept (share of cases)", edgecolor="white", lw=2)
        ax.bar(x + 0.2, fr, 0.38, color=COL["framework"], label="false reject (share of safe proposals)", edgecolor="white", lw=2)
        for xi, v in zip(x - 0.2, ua):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x, ["A\nno gate", "B\norig μ", "C\norig μ+σ", "D\nimp μ", "E\nimp μ+σ", "F\noracle"], fontsize=8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, 1.08)
    axes[0].set_ylabel("rate")
    axes[0].legend(fontsize=8, loc="upper right")
    save(fig, "fig03_safety_af", "safety_gate/improvements/eval_runs/af_eval_k1_calibrated_devtopos_v1/summary.json; "
         "phase3_final/e2_holdout_af/summary.json")


def fig_twin_error():
    plt = _plt()
    rows = list(csv.DictReader(open(F / "e4_twin_error" / "sweep.csv", encoding="utf-8")))
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    labels = {"D": "D mean-only", "Dm": "Dm mean-only + margin (dev-tuned)", "E1": "E k=1", "E2": "E k=2",
              "E1r": "E k=1, recalibrated c", "F": "F oracle"}
    for col, scope in enumerate(("dev", "holdout")):
        for row_i, (metric, ylab) in enumerate((("unsafe_accept_given_unsafe", "unsafe accepts / unsafe proposals"),
                                                ("false_reject_given_safe", "false rejects / safe proposals"))):
            ax = axes[col * 2 + row_i]
            for v in ("D", "Dm", "E1", "E2", "E1r", "F"):
                pts = sorted((float(r["sigma"]), float(r[metric])) for r in rows if r["scope"] == scope
                             and r["twin"] == "improved" and r["variant"] == v and r["load_mode"] == "all")
                if pts:
                    line(ax, [p[0] for p in pts], [p[1] for p in pts], v, labels[v], lw=1.6)
            ax.set_xlabel("member-wise twin error σ")
            ax.set_ylabel(ylab)
            ax.set_title(f"{scope}: {'unsafe accepts' if row_i == 0 else 'false rejects'}", fontsize=10, loc="left")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "fig04_unsafe_accept_vs_twin_error", "e4_twin_error/sweep.csv (improved twin, member-wise error)")


def fig_link_failures():
    plt = _plt()
    p1 = json.loads((REPRO / "b4_link_failures" / "summary.json").read_text(encoding="utf-8"))
    e3 = json.loads((F / "e3_link_failures" / "summary.json").read_text(encoding="utf-8"))["table"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    nf = [0, 1, 2, 3]
    ax = axes[0]
    for m, key, lab in (("grl_te_teal_admm", "frozen_p1", "frozen P1 GRL-TE (+ADMM)"), ("lp", "lp", "LP (max throughput)"),
                        ("shortest_path", "shortest", "shortest"), ("ecmp", "equal", "ECMP")):
        y = [p1[str(n)][m][0] for n in nf]
        e = [p1[str(n)][m][1] for n in nf]
        ax.errorbar(nf, y, yerr=e, color=COL[key], marker=MRK[key], label=lab, capsize=3, lw=2, markeredgecolor="white")
    ax.scatter([0, 3], [92.39, 77.25], s=80, facecolor="white", edgecolor=INK, zorder=5, label="P1 reported (92.39 intact; 77.25 at 3)")
    ax.set_title("B4, P1 §5.5 protocol (test TMs, fixed paths)", fontsize=10, loc="left")
    ax.set_ylabel("demand satisfaction (%)")
    ax.legend(fontsize=7, loc="lower left")
    for ax, topo in zip(axes[1:], ("GEANT", "UsCarrier")):
        for m, key, lab, kw in (("shortest", "shortest", "shortest (dashed)", {"ls": "--", "lw": 1.4, "zorder": 2}),
                                ("lp", "lp", "LP (min-MLU)", {"zorder": 2}),
                                ("actor_stable", "policy", "policy only", {"zorder": 2}),
                                ("fw_stable_E", "framework", "framework (gate E)", {"lw": 2.6, "zorder": 4})):
            y = [next(r for r in e3 if r["topology"] == topo and r["load_mode"] == "all" and r["n_failed"] == n
                      and r["method"] == m)["sat_mean"] * 100 for n in nf]
            line(ax, nf, y, key, lab, **kw)
        ax.set_title(f"{topo} (Phase-3 framework, rerouted paths)", fontsize=10, loc="left")
    for ax in axes:
        ax.set_xlabel("failed directed links")
        ax.set_xticks(nf)
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.66, -0.02),
               title="GEANT / UsCarrier panels (framework coincides with shortest on UsCarrier)", title_fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig05_link_failure_satisfaction", "baseline_p1_reproduction/phase3_final/results/b4_link_failures/summary.json; "
         "phase3_final/e3_link_failures/summary.json")


def fig_generalization_gap():
    plt = _plt()
    g = json.loads((F / "policy_v2" / "generalization_gap.json").read_text(encoding="utf-8"))
    topos = ["B4", "UsCarrier", "Cogentco", "GEANT", "Kdl"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(topos))
    for off, (pid, key, lab) in zip((-0.2, 0.2), (("improved_stable", "framework", "recommended_stable"),
                                                   ("v2_calload_normscore", "v2", "v2 (calibrated-load)"))):
        vals = [(g[pid]["seen"].get(t) or g[pid]["unseen"].get(t))["mlu_ratio"] for t in topos]
        ax.bar(x + off, vals, 0.38, color=COL[key], label=lab, edgecolor="white", lw=2)
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.axvline(2.5, color=MUTED, lw=1)
    ax.text(1, ax.get_ylim()[1] * 0.97, "seen (validation)", ha="center", color=MUTED, fontsize=8.5)
    ax.text(3.5, ax.get_ylim()[1] * 0.97, "unseen (test)", ha="center", color=MUTED, fontsize=8.5)
    ax.set_xticks(x, topos)
    ax.set_ylabel("policy MLU / shortest MLU (lower is better)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig06_generalization_gap", "policy_v2/generalization_gap.json")


def fig_overhead():
    plt = _plt()
    oh = json.loads((F / "stats" / "overhead.json").read_text(encoding="utf-8"))
    comps = (("policy_ms (stable GNN)", "policy", "policy (GNN)"), ("twin_ms (5-member ensemble)", "framework", "twin"),
             ("fallback_select_ms", "shortest", "fallback selection"), ("lp_ms (min-MLU, HiGHS)", "lp", "LP (min-MLU)"))
    topos = [r["topology"] for r in oh]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = np.arange(len(topos))
    for i, (k, key, lab) in enumerate(comps):
        vals = [r.get(k) or np.nan for r in oh]
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.19, color=COL[key], label=lab, edgecolor="white", lw=1.5)
    ax.set_yscale("log")
    ax.set_xticks(x, topos)
    ax.set_ylabel("median time per decision (ms, log)")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    save(fig, "fig07_overhead", "stats/overhead.json (LP not run on Kdl)")


def fig_ablation():
    plt = _plt()
    t = json.loads((F / "ablations_v2" / "summary.json").read_text(encoding="utf-8"))["table"]
    cfgs = ["full", "no_gate", "mean_gate", "no_fallback", "b4_only_policy", "physics_twin"]
    names = ["full", "(a) no gate", "(b) no σ", "(c) no fallback", "(d) B4-only†", "(e) physics twin"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    x = np.arange(len(cfgs))
    for ax, (metric, ylab) in zip(axes, (("applied_sat", "applied satisfaction"), ("applied_overload_rate", "applied overload rate"))):
        for off, (scope, key) in zip((-0.2, 0.2), (("dev", "policy"), ("holdout", "framework"))):
            vals = [next(r for r in t if r["scope"] == scope and r["config"] == c)[metric] for c in cfgs]
            ax.bar(x + off, vals, 0.38, color=COL[key], label=scope, edgecolor="white", lw=2)
        ax.set_xticks(x, names, rotation=20, fontsize=8)
        ax.set_ylabel(ylab)
        if metric == "applied_sat":
            ax.set_ylim(0.6, 1.0)
    axes[0].legend(fontsize=8)
    save(fig, "fig08_ablations", "ablations_v2/summary.json († confounded: different model and trainer)")


def fig_taxonomy():
    plt = _plt()
    tx = json.loads((F / "stats" / "failure_taxonomy.json").read_text(encoding="utf-8"))["sigma0_framework"]
    cats = sorted({k for d in tx.values() for k in d if not k.startswith("  ")})
    fig, ax = plt.subplots(figsize=(9, 3.2))
    y = np.arange(len(cats))
    for off, (scope, key) in zip((-0.2, 0.2), (("dev", "policy"), ("holdout", "framework"))):
        vals = [tx.get(scope, {}).get(c, 0) for c in cats]
        ax.barh(y + off, vals, 0.38, color=COL[key], label=scope, edgecolor="white", lw=2)
        for yi, v in zip(y + off, vals):
            ax.text(v + 2, yi, str(v), va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [c.replace(" (", "\n(") for c in cats], fontsize=8)
    ax.set_xlabel("overloaded applied decisions (full framework, no twin error)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig09_failure_taxonomy", "stats/failure_taxonomy.json")


def main() -> None:
    for fn in (fig_training, fig_mlu_vs_load, fig_safety_bars, fig_twin_error, fig_link_failures,
               fig_generalization_gap, fig_overhead, fig_ablation, fig_taxonomy):
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except FileNotFoundError as exc:
            print(f"skip {fn.__name__}: {exc}")
    lines = ["# Final figures", "", "All figures are 300-dpi PNG + PDF, generated by `python -m phase3_final.figures`.",
             "Sources are relative to `results/` unless they start with `baseline_p1_reproduction/`.", "",
             "| Figure | Source |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in SOURCES.items()]
    (OUT / "FIGURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
