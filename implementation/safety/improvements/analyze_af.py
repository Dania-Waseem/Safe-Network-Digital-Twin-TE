"""Post-process a completed paired A–F evaluation run."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .decision import VARIANT_IDS
from .metrics import compute_safety_metrics
from .resume import read_decision_rows


def _bootstrap_rate_ci(
    flags: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    arr = np.asarray(flags, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(np.mean(arr)), "ci95_low": float(lo), "ci95_high": float(hi), "n": int(arr.size)}


def _case_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("case_id")
        or (
            row.get("seed"),
            row.get("topology"),
            row.get("matrix_id"),
            row.get("scale"),
            row.get("policy_checkpoint_id"),
        )
    )


def paired_binary_comparison(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    """Paired case-level comparison of a boolean field (e.g. unsafe_accept)."""
    map_a = {_case_key(r): r for r in rows_a if r.get("decision_status", "ok") == "ok"}
    map_b = {_case_key(r): r for r in rows_b if r.get("decision_status", "ok") == "ok"}
    keys = sorted(set(map_a) & set(map_b), key=lambda x: str(x))
    a_flags = []
    b_flags = []
    for key in keys:
        a_flags.append(bool(map_a[key].get(field)))
        b_flags.append(bool(map_b[key].get(field)))
    a_arr = np.asarray(a_flags, dtype=bool)
    b_arr = np.asarray(b_flags, dtype=bool)
    # Discordant pairs for McNemar-style summary.
    a_only = int(np.sum(a_arr & ~b_arr))
    b_only = int(np.sum(~a_arr & b_arr))
    both = int(np.sum(a_arr & b_arr))
    neither = int(np.sum(~a_arr & ~b_arr))
    diff = a_arr.astype(np.float64) - b_arr.astype(np.float64)
    # Bootstrap mean difference.
    if diff.size:
        rng = np.random.default_rng(7)
        boots = [float(np.mean(rng.choice(diff, size=diff.size, replace=True))) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        diff_summary = {
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        }
    else:
        diff_summary = {
            "mean_diff_a_minus_b": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    return {
        "n_paired_cases": len(keys),
        "rate_a": float(np.mean(a_arr)) if a_arr.size else float("nan"),
        "rate_b": float(np.mean(b_arr)) if b_arr.size else float("nan"),
        "discordant": {"a_only": a_only, "b_only": b_only, "both": both, "neither": neither},
        "difference": diff_summary,
        "field": field,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    decisions_path = run_dir / "decisions.jsonl"
    rows = read_decision_rows(decisions_path)
    ok = [r for r in rows if r.get("decision_status", "ok") == "ok" and "acceptance" in r]
    failed = [r for r in rows if r.get("decision_status") == "failed"]

    by_variant: dict[str, list] = defaultdict(list)
    for r in ok:
        by_variant[str(r.get("gate_variant"))].append(r)

    metrics_by_variant = {vid: compute_safety_metrics(by_variant.get(vid, [])) for vid in VARIANT_IDS}

    # Applied-action overload rate (among completed).
    applied = {}
    for vid, subset in by_variant.items():
        flags = [int(r.get("capacity_violations_applied") or 0) > 0 for r in subset]
        applied[vid] = {
            "applied_overload_rate": _bootstrap_rate_ci(flags, seed=11),
            "mean_mlu_applied": float(np.nanmean([r.get("mlu_applied", float("nan")) for r in subset])),
            "mean_mlu_proposed": float(np.nanmean([r.get("mlu_proposed", float("nan")) for r in subset])),
            "mean_decision_runtime_ms": float(
                np.nanmean([r.get("decision_runtime_ms", float("nan")) for r in subset])
            ),
        }

    # Breakdowns by topology / scale for each variant.
    breakdowns: dict[str, Any] = {"by_topology": {}, "by_scale": {}}
    for topo in sorted({r.get("topology") for r in ok}):
        breakdowns["by_topology"][str(topo)] = {
            vid: compute_safety_metrics([r for r in by_variant.get(vid, []) if r.get("topology") == topo])
            for vid in VARIANT_IDS
        }
    for scale in sorted({float(r.get("scale")) for r in ok}):
        breakdowns["by_scale"][str(scale)] = {
            vid: compute_safety_metrics(
                [r for r in by_variant.get(vid, []) if float(r.get("scale")) == float(scale)]
            )
            for vid in VARIANT_IDS
        }

    comparisons = {
        "D_vs_E_unsafe_accept": paired_binary_comparison(
            by_variant.get("D_improved_twin_mean", []),
            by_variant.get("E_improved_twin_calibrated_uncertainty", []),
            field="unsafe_accept",
        ),
        "D_vs_E_false_reject": paired_binary_comparison(
            by_variant.get("D_improved_twin_mean", []),
            by_variant.get("E_improved_twin_calibrated_uncertainty", []),
            field="false_reject",
        ),
        "D_vs_E_acceptance": paired_binary_comparison(
            by_variant.get("D_improved_twin_mean", []),
            by_variant.get("E_improved_twin_calibrated_uncertainty", []),
            field="acceptance",
        ),
        "B_vs_D_unsafe_accept": paired_binary_comparison(
            by_variant.get("B_original_twin_mean", []),
            by_variant.get("D_improved_twin_mean", []),
            field="unsafe_accept",
        ),
        "C_vs_E_unsafe_accept": paired_binary_comparison(
            by_variant.get("C_original_twin_uncertainty", []),
            by_variant.get("E_improved_twin_calibrated_uncertainty", []),
            field="unsafe_accept",
        ),
        "A_vs_E_unsafe_accept": paired_binary_comparison(
            by_variant.get("A_policy_only", []),
            by_variant.get("E_improved_twin_calibrated_uncertainty", []),
            field="unsafe_accept",
        ),
        "E_vs_F_acceptance": paired_binary_comparison(
            by_variant.get("E_improved_twin_calibrated_uncertainty", []),
            by_variant.get("F_physics_oracle", []),
            field="acceptance",
        ),
    }

    # Completeness / resumability checks.
    case_ids = {r.get("case_id") for r in ok if r.get("case_id")}
    variant_counts = {
        cid: len([r for r in ok if r.get("case_id") == cid]) for cid in case_ids
    }
    incomplete = {cid: n for cid, n in variant_counts.items() if n != len(VARIANT_IDS)}
    fingerprints = defaultdict(set)
    for r in ok:
        fingerprints[r.get("case_id")].add(r.get("policy_action_fingerprint"))
    fp_bad = {cid: sorted(fps) for cid, fps in fingerprints.items() if len(fps) != 1}

    d_eq_e_accept = comparisons["D_vs_E_acceptance"]
    uncertainty_adds_value = None
    # Claim rule: E must show lower unsafe_accept than D with CI excluding 0 favoring E,
    # without a large rise in false rejects — otherwise do not claim value.
    d_vs_e_unsafe = comparisons["D_vs_E_unsafe_accept"]
    d_vs_e_fr = comparisons["D_vs_E_false_reject"]
    unsafe_diff = d_vs_e_unsafe["difference"]["mean_diff_a_minus_b"]  # D - E
    fr_diff = d_vs_e_fr["difference"]["mean_diff_a_minus_b"]
    # Positive unsafe_diff means D has more unsafe accepts than E (E better).
    if (
        d_vs_e_unsafe["n_paired_cases"] > 0
        and d_vs_e_unsafe["difference"]["ci95_low"] > 0
        and fr_diff <= 0.05
    ):
        uncertainty_adds_value = True
    elif d_vs_e_unsafe["rate_a"] == d_vs_e_unsafe["rate_b"] and d_vs_e_fr["rate_a"] == d_vs_e_fr["rate_b"]:
        uncertainty_adds_value = False
    else:
        uncertainty_adds_value = False

    analysis = {
        "n_logged_lines": len(rows),
        "n_completed_rows": len(ok),
        "n_failed_rows": len(failed),
        "n_unique_cases": len(case_ids),
        "expected_variants": len(VARIANT_IDS),
        "incomplete_cases": incomplete,
        "fingerprint_violations": fp_bad,
        "metrics_by_variant": metrics_by_variant,
        "applied_action_stats": applied,
        "breakdowns": breakdowns,
        "paired_comparisons": comparisons,
        "uncertainty_adds_value_claim": {
            "value": uncertainty_adds_value,
            "rule": (
                "Claim True only if paired (D-E) unsafe_accept mean difference CI lower bound > 0 "
                "(D worse than E) and false_reject increase (E-D) is not large; "
                "identical rates => False (no incremental value demonstrated)."
            ),
            "d_unsafe_rate": d_vs_e_unsafe["rate_a"],
            "e_unsafe_rate": d_vs_e_unsafe["rate_b"],
            "d_false_reject_rate": d_vs_e_fr["rate_a"],
            "e_false_reject_rate": d_vs_e_fr["rate_b"],
        },
    }
    return analysis


def make_plots(analysis: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    variants = list(VARIANT_IDS)
    metrics = analysis["metrics_by_variant"]

    def _series(key: str) -> list[float]:
        return [float(metrics[v].get(key, float("nan"))) for v in variants]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(variants))
    ax.bar(x - 0.2, _series("unsafe_acceptance_rate"), width=0.4, label="unsafe accept")
    ax.bar(x + 0.2, _series("false_rejection_rate"), width=0.4, label="false reject")
    ax.set_xticks(x)
    ax.set_xticklabels([v.split("_")[0] for v in variants])
    ax.set_ylabel("Rate")
    ax.set_title("Unsafe acceptance vs false rejection by variant")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "unsafe_vs_false_reject.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, _series("acceptance_rate"))
    ax.set_xticks(x)
    ax.set_xticklabels([v.split("_")[0] for v in variants])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Acceptance rate by variant")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "acceptance_rate.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, _series("fallback_frequency"), width=0.4, label="fallback")
    ax.bar(
        x + 0.2,
        _series("no_feasible_fallback_frequency"),
        width=0.4,
        label="no feasible fallback",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([v.split("_")[0] for v in variants])
    ax.set_title("Fallback analysis")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "fallback.png", dpi=140)
    plt.close(fig)

    # Topology unsafe accept for D and E
    fig, ax = plt.subplots(figsize=(7, 4))
    topos = sorted(analysis["breakdowns"]["by_topology"])
    d_rates = [
        analysis["breakdowns"]["by_topology"][t]["D_improved_twin_mean"].get(
            "unsafe_acceptance_rate", float("nan")
        )
        for t in topos
    ]
    e_rates = [
        analysis["breakdowns"]["by_topology"][t]["E_improved_twin_calibrated_uncertainty"].get(
            "unsafe_acceptance_rate", float("nan")
        )
        for t in topos
    ]
    xx = np.arange(len(topos))
    ax.bar(xx - 0.15, d_rates, width=0.3, label="D mean-only")
    ax.bar(xx + 0.15, e_rates, width=0.3, label="E uncertainty")
    ax.set_xticks(xx)
    ax.set_xticklabels(topos)
    ax.set_ylabel("Unsafe acceptance rate")
    ax.set_title("D vs E unsafe acceptance by topology")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "d_vs_e_by_topology.png", dpi=140)
    plt.close(fig)


def write_report(run_dir: Path, analysis: dict[str, Any], meta: dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    m = analysis["metrics_by_variant"]
    claim = analysis["uncertainty_adds_value_claim"]
    lines = [
        "# Paired A–F Safety Gate Evaluation Report",
        "",
        f"**Run directory:** `{run_dir}`",
        "",
        "## Configuration (frozen)",
        "",
        "- Policy: `improved_stable`",
        "- Twin D: physics-anchored (mean-only)",
        "- Twin E: physics-anchored + conformal calibration, **k=1.0**, utilisation_limit=**1.0**",
        "- Catalog: robustness_v1",
        "- Topologies: **B4, UsCarrier, Cogentco only**",
        "- **GEANT/Kdl:** excluded (authorization ambiguity; left completely untouched)",
        "",
        "## Completeness",
        "",
        f"- Logged lines: {analysis['n_logged_lines']}",
        f"- Completed decision rows: {analysis['n_completed_rows']}",
        f"- Failed case rows: {analysis['n_failed_rows']}",
        f"- Unique cases: {analysis['n_unique_cases']}",
        f"- Incomplete A–F blocks: {len(analysis['incomplete_cases'])}",
        f"- Fingerprint violations: {len(analysis['fingerprint_violations'])}",
        f"- Resume skips (meta): {meta.get('n_cases_skipped_resume')}",
        "",
        "## Metrics by variant (raw-count denominators)",
        "",
        "| Variant | Accept | Unsafe accept | False reject | Fallback | Applied overload |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for vid in VARIANT_IDS:
        mm = m[vid]
        app = analysis["applied_action_stats"].get(vid, {}).get("applied_overload_rate", {})
        lines.append(
            f"| {vid} | {mm.get('acceptance_rate', float('nan')):.4f} | "
            f"{mm.get('unsafe_acceptance_rate', float('nan')):.4f} | "
            f"{mm.get('false_rejection_rate', float('nan')):.4f} | "
            f"{mm.get('fallback_frequency', float('nan')):.4f} | "
            f"{app.get('mean', float('nan')):.4f} |"
        )

    d_e = analysis["paired_comparisons"]["D_vs_E_unsafe_accept"]
    d_e_fr = analysis["paired_comparisons"]["D_vs_E_false_reject"]
    lines.extend(
        [
            "",
            "## Primary comparison: D vs E (incremental uncertainty value)",
            "",
            f"- Paired cases: **{d_e['n_paired_cases']}**",
            f"- Unsafe accept D: **{d_e['rate_a']:.4f}**; E: **{d_e['rate_b']:.4f}**",
            f"- Mean difference (D−E): {d_e['difference']['mean_diff_a_minus_b']:.4f} "
            f"[{d_e['difference']['ci95_low']:.4f}, {d_e['difference']['ci95_high']:.4f}]",
            f"- Discordant unsafe accepts: D-only={d_e['discordant']['a_only']}, "
            f"E-only={d_e['discordant']['b_only']}",
            f"- False reject D: **{d_e_fr['rate_a']:.4f}**; E: **{d_e_fr['rate_b']:.4f}**",
            f"- Mean FR difference (D−E): {d_e_fr['difference']['mean_diff_a_minus_b']:.4f} "
            f"[{d_e_fr['difference']['ci95_low']:.4f}, {d_e_fr['difference']['ci95_high']:.4f}]",
            "",
            "### Claim: does calibrated uncertainty add measurable value?",
            "",
            f"**Measured answer: `{claim['value']}`**",
            "",
            f"Rule: {claim['rule']}",
            "",
            "## Other paired comparisons (unsafe accept unless noted)",
            "",
        ]
    )
    for key in (
        "B_vs_D_unsafe_accept",
        "C_vs_E_unsafe_accept",
        "A_vs_E_unsafe_accept",
        "E_vs_F_acceptance",
    ):
        block = analysis["paired_comparisons"][key]
        lines.append(
            f"- **{key}**: A={block['rate_a']:.4f}, B={block['rate_b']:.4f}, "
            f"diff(A−B)={block['difference']['mean_diff_a_minus_b']:.4f} "
            f"[{block['difference']['ci95_low']:.4f}, {block['difference']['ci95_high']:.4f}]"
        )

    lines.extend(
        [
            "",
            "## Measured findings vs hypotheses",
            "",
            "- **Measured:** variant rates, paired D−E differences with bootstrap CIs, topology/scale breakdowns, fallback and applied-overload rates.",
            "- **Hypothesis (not claimed unless D≻E on unsafe accept with CI support):** uncertainty-aware gating improves safety over mean-only on this twin.",
            "",
            "## Preservation",
            "",
            "Original gate/Twin/RL/splits/baselines untouched. GEANT/Kdl unused. No retraining or threshold retuning.",
            "",
            "## Artifacts",
            "",
            "- `decisions.jsonl`, `run_meta.json`, `summary.json`",
            "- `tables/analysis.json`",
            "- `plots/*.png`",
            "- `AF_EVALUATION_REPORT.md`",
            "",
        ]
    )
    (run_dir / "AF_EVALUATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    analysis = analyze_run(run_dir)
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    make_plots(analysis, run_dir)
    write_report(run_dir, analysis, meta)
    return analysis
