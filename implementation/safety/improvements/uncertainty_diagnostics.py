"""Digital Twin uncertainty diagnostics on development topologies only.

Uses the robustness_v1 catalog for B4 / UsCarrier / Cogentco (val split).
Never reads GEANT/Kdl for calibration, threshold selection, or model selection.
Does not modify Twin / gate / RL checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twin.data import ROOT, load_split_demands, load_topology

from .config import load_improvements_config
from .diagnostics import CAPACITY_BINS, capacity_proximity_bin
from .graph_cache import TopologyGraphCache
from .oracle import physics_violations
from .policy_backend import policy_backend_from_config
from .resume import append_decision_rows
from .twin_backend import backends_from_config

OUT_DIR = ROOT / "results" / "safety_gate" / "improvements" / "uncertainty_diagnostics"
CATALOG_PATH = ROOT / "results" / "evaluation" / "robustness_v1" / "catalog.json"
DEV_TOPOS = ("B4", "UsCarrier", "Cogentco")
SCALES = (0.25, 0.5, 0.75, 1.0)
NOMINAL_K = (1.0, 1.5, 2.0, 2.5, 3.0)
# Seeds used only for threshold selection (still val matrices; not GEANT/Kdl).
THRESHOLD_SEEDS = (404, 505)
DIAGNOSTIC_SEEDS = (101, 202, 303, 404, 505)


def _bootstrap_mean_ci(values: list[float], *, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(np.mean(sample)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(np.mean(arr)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n": int(arr.size),
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def load_dev_catalog(path: Path | None = None) -> dict[str, Any]:
    payload = json.loads((path or CATALOG_PATH).read_text(encoding="utf-8"))
    topologies = {k: v for k, v in payload["topologies"].items() if k in DEV_TOPOS}
    for name, spec in topologies.items():
        if spec.get("unseen"):
            raise RuntimeError(f"{name} marked unseen in catalog; refused")
        if name in ("GEANT", "Kdl"):
            raise RuntimeError("GEANT/Kdl must not enter uncertainty diagnostics")
    return {"seeds": list(payload["seeds"]), "topologies": topologies, "source": str(path or CATALOG_PATH)}


def _case_coverage(abs_err: np.ndarray, std: np.ndarray, k: float) -> float:
    return float(np.mean(abs_err <= float(k) * np.maximum(std, 0.0)))


def _case_width(std: np.ndarray, k: float) -> tuple[float, float]:
    half = float(k) * np.maximum(std, 0.0)
    width = 2.0 * half
    return float(np.mean(width)), float(np.median(width))


def evaluate_case_uncertainty(
    *,
    true_load: np.ndarray,
    mean_load: np.ndarray,
    raw_std: np.ndarray,
    cal_std: np.ndarray,
    capacity: np.ndarray,
) -> dict[str, Any]:
    cap = np.maximum(capacity, 1e-12)
    abs_err = np.abs(mean_load - true_load)
    util_true = true_load / cap
    util_mean = mean_load / cap
    util_err = np.abs(util_mean - util_true)
    raw_std_u = raw_std / cap
    cal_std_u = cal_std / cap

    out: dict[str, Any] = {
        "n_links": int(true_load.size),
        "mae_abs": float(np.mean(abs_err)),
        "mae_util": float(np.mean(util_err)),
        "mean_raw_std_util": float(np.mean(raw_std_u)),
        "mean_cal_std_util": float(np.mean(cal_std_u)),
        "spearman_raw_std_vs_abserr": _spearman(raw_std, abs_err),
        "spearman_cal_std_vs_abserr": _spearman(cal_std, abs_err),
        "spearman_raw_std_util_vs_utilerr": _spearman(raw_std_u, util_err),
        "spearman_cal_std_util_vs_utilerr": _spearman(cal_std_u, util_err),
        "actual_mlu": float(np.max(util_true)),
        "capacity_proximity": capacity_proximity_bin(float(np.max(util_true))),
    }
    for k in NOMINAL_K:
        out[f"coverage_raw_k{k:g}"] = _case_coverage(abs_err, raw_std, k)
        out[f"coverage_cal_k{k:g}"] = _case_coverage(abs_err, cal_std, k)
        out[f"coverage_raw_util_k{k:g}"] = _case_coverage(util_err, raw_std_u, k)
        out[f"coverage_cal_util_k{k:g}"] = _case_coverage(util_err, cal_std_u, k)
        mw_r, md_r = _case_width(raw_std_u, k)
        mw_c, md_c = _case_width(cal_std_u, k)
        out[f"mean_width_raw_util_k{k:g}"] = mw_r
        out[f"median_width_raw_util_k{k:g}"] = md_r
        out[f"mean_width_cal_util_k{k:g}"] = mw_c
        out[f"median_width_cal_util_k{k:g}"] = md_c
        # Nominal coverage for Gaussian-like two-sided interval approx via erf is not assumed;
        # report nominal as the k-label itself for the symmetric |err|<=k*std rule.
        out[f"nominal_k"] = None  # placeholder cleared below
    out.pop("nominal_k", None)
    return out


def link_records_for_case(
    *,
    true_load: np.ndarray,
    mean_load: np.ndarray,
    raw_std: np.ndarray,
    cal_std: np.ndarray,
    capacity: np.ndarray,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    cap = np.maximum(capacity, 1e-12)
    rows = []
    for i in range(true_load.size):
        util_t = float(true_load[i] / cap[i])
        abs_err = float(abs(mean_load[i] - true_load[i]))
        util_err = float(abs(mean_load[i] / cap[i] - util_t))
        rows.append(
            {
                **meta,
                "link": int(i),
                "true_load": float(true_load[i]),
                "mean_load": float(mean_load[i]),
                "raw_std": float(raw_std[i]),
                "cal_std": float(cal_std[i]),
                "capacity": float(capacity[i]),
                "util_true": util_t,
                "util_mean": float(mean_load[i] / cap[i]),
                "abs_err": abs_err,
                "util_err": util_err,
                "raw_std_util": float(raw_std[i] / cap[i]),
                "cal_std_util": float(cal_std[i] / cap[i]),
                "capacity_proximity": capacity_proximity_bin(util_t),
            }
        )
    return rows


def alarm_metrics(
    util_true: np.ndarray,
    util_mean: np.ndarray,
    std_util: np.ndarray,
    *,
    k: float,
    utilisation_limit: float,
) -> dict[str, float]:
    """Gate-proxy alarms: UCB util > limit vs actual util > 1 (per link, then case-reduced)."""
    ucb = util_mean + float(k) * np.maximum(std_util, 0.0)
    alarm = ucb > float(utilisation_limit)
    unsafe = util_true > 1.0 + 1e-9
    # Case-level: any-link reduction matching gate semantics.
    case_alarm = bool(np.any(alarm))
    case_unsafe = bool(np.any(unsafe))
    return {
        "case_alarm": float(case_alarm),
        "case_unsafe": float(case_unsafe),
        "link_alarm_rate": float(np.mean(alarm)),
        "link_unsafe_rate": float(np.mean(unsafe)),
        "error_detection_case": float(case_alarm) if case_unsafe else float("nan"),
        "false_alarm_case": float(case_alarm) if (not case_unsafe) else float("nan"),
    }


def uncertainty_error_detection(
    util_err: np.ndarray,
    std_util: np.ndarray,
    *,
    err_quantile: float = 0.9,
    unc_quantile: float = 0.9,
) -> dict[str, float]:
    """Does high uncertainty flag large util errors? (link-level within a case pool)."""
    if util_err.size < 10:
        return {
            "error_detection_rate": float("nan"),
            "false_alarm_rate": float("nan"),
            "err_threshold": float("nan"),
            "unc_threshold": float("nan"),
        }
    err_thr = float(np.quantile(util_err, err_quantile))
    unc_thr = float(np.quantile(std_util, unc_quantile))
    large_err = util_err >= err_thr
    high_unc = std_util >= unc_thr
    u = int(np.sum(large_err))
    s = int(np.sum(~large_err))
    detect = int(np.sum(high_unc & large_err))
    false_al = int(np.sum(high_unc & ~large_err))
    return {
        "error_detection_rate": float(detect / u) if u else float("nan"),
        "false_alarm_rate": float(false_al / s) if s else float("nan"),
        "err_threshold": err_thr,
        "unc_threshold": unc_thr,
        "n_large_err": u,
        "n_high_unc": int(np.sum(high_unc)),
    }


def run_diagnostics(
    *,
    out_dir: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir or OUT_DIR)
    raw_dir = out_dir / "raw"
    tables = out_dir / "tables"
    plots = out_dir / "plots"
    for d in (raw_dir, tables, plots):
        d.mkdir(parents=True, exist_ok=True)

    cfg = load_improvements_config()
    device = device or torch.device(str(cfg.get("evaluation", {}).get("device", "cpu")))
    catalog = load_dev_catalog()
    graph_cache = TopologyGraphCache(device)
    policy = policy_backend_from_config(cfg, device=device, graph_cache=graph_cache)
    policy.load()
    backends = backends_from_config(cfg, device)
    improved = backends["improved"]
    calibrated = backends["improved_calibrated"]
    scale = float(calibrated.resolved_conformal_scale())

    case_path = raw_dir / "case_records.jsonl"
    link_path = raw_dir / "link_records.jsonl"
    if case_path.exists():
        case_path.unlink()
    if link_path.exists():
        link_path.unlink()

    case_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for topo_name, spec in catalog["topologies"].items():
        topo = load_topology(topo_name)
        demands, meta = load_split_demands(spec["split"], topo_name)
        matrix_ids = meta.get("matrix_indices", list(range(demands.shape[0])))
        for seed in catalog["seeds"]:
            entry = spec["by_seed"][str(seed)]
            for local_i, matrix_id in zip(entry["local_indices"], entry["matrix_ids"]):
                base = demands[int(local_i)]
                for scale_traffic in SCALES:
                    matrix = base * float(scale_traffic)
                    proposal = policy.propose(topo, matrix)
                    action = proposal.action
                    # Raw (no conformal) and calibrated predictions — same weights.
                    pred_raw = improved.predict(topo, matrix, action)
                    pred_cal = calibrated.predict(topo, matrix, action)
                    # Prefer raw std from improved backend; cal from calibrated backend.
                    raw_std = pred_raw.std_load
                    cal_std = pred_cal.std_load
                    mean_load = pred_cal.mean_load  # identical to raw mean when weights match
                    true_load = pred_raw.true_load
                    np.testing.assert_allclose(pred_raw.mean_load, pred_cal.mean_load, rtol=1e-5, atol=1e-8)

                    stats = evaluate_case_uncertainty(
                        true_load=true_load,
                        mean_load=mean_load,
                        raw_std=raw_std,
                        cal_std=cal_std,
                        capacity=topo.capacity,
                    )
                    cap = np.maximum(topo.capacity, 1e-12)
                    util_true = true_load / cap
                    util_mean = mean_load / cap
                    for label, std_u in (
                        ("raw", raw_std / cap),
                        ("cal", cal_std / cap),
                    ):
                        for k in (1.0, 2.0):
                            am = alarm_metrics(
                                util_true,
                                util_mean,
                                std_u,
                                k=k,
                                utilisation_limit=1.0,
                            )
                            for key, val in am.items():
                                stats[f"alarm_{label}_k{k:g}_{key}"] = val

                    row = {
                        "seed": int(seed),
                        "topology": topo_name,
                        "split": spec["split"],
                        "local_index": int(local_i),
                        "matrix_id": int(matrix_id),
                        "scale": float(scale_traffic),
                        "threshold_selection_seed": int(seed) in THRESHOLD_SEEDS,
                        "policy_checkpoint_id": policy.checkpoint_id,
                        "conformal_std_scale": scale,
                        "n_violations_proposed": int(physics_violations(true_load, topo.capacity)),
                        "oracle_feasible": bool(physics_violations(true_load, topo.capacity) == 0),
                        "action_fingerprint": proposal.graph_fingerprint,
                        **stats,
                    }
                    case_rows.append(row)
                    append_decision_rows(case_path, [row])

                    meta_link = {
                        "seed": int(seed),
                        "topology": topo_name,
                        "matrix_id": int(matrix_id),
                        "scale": float(scale_traffic),
                        "threshold_selection_seed": int(seed) in THRESHOLD_SEEDS,
                    }
                    links = link_records_for_case(
                        true_load=true_load,
                        mean_load=mean_load,
                        raw_std=raw_std,
                        cal_std=cal_std,
                        capacity=topo.capacity,
                        meta=meta_link,
                    )
                    link_rows.extend(links)
                    append_decision_rows(link_path, links)

    elapsed = time.perf_counter() - started
    summary = build_summaries(case_rows, link_rows, conformal_scale=scale)
    summary["elapsed_s"] = elapsed
    summary["n_cases"] = len(case_rows)
    summary["n_link_rows"] = len(link_rows)
    summary["catalog_source"] = catalog["source"]
    summary["held_out_from_this_study"] = ["GEANT", "Kdl"]
    summary["protocol"] = {
        "diagnostic_seeds": list(DIAGNOSTIC_SEEDS),
        "threshold_selection_seeds": list(THRESHOLD_SEEDS),
        "development_topologies": list(DEV_TOPOS),
        "split": "val (robustness_v1 catalog)",
        "measured_vs_hypothesis": "Coverage/width/error are measured; usefulness for gating is hypothesized until A–F.",
    }

    threshold_report = select_thresholds(case_rows, link_rows)
    (tables / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (tables / "threshold_selection.json").write_text(
        json.dumps(threshold_report, indent=2), encoding="utf-8"
    )
    config = {
        "dev_topologies": list(DEV_TOPOS),
        "scales": list(SCALES),
        "nominal_k": list(NOMINAL_K),
        "threshold_seeds": list(THRESHOLD_SEEDS),
        "conformal_std_scale": scale,
        "catalog": catalog["source"],
        "policy": policy.checkpoint_id,
        "twins": {
            "improved": str(improved.path),
            "improved_calibrated": str(calibrated.path),
        },
        "never_used": ["GEANT", "Kdl"],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    make_plots(case_rows, link_rows, plots)
    write_threshold_markdown(out_dir / "THRESHOLD_SELECTION.md", threshold_report)
    write_main_report(out_dir / "UNCERTAINTY_DIAGNOSTIC_REPORT.md", summary, threshold_report, config)
    return {"summary": summary, "threshold_report": threshold_report, "out_dir": str(out_dir)}


def build_summaries(
    case_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
    *,
    conformal_scale: float,
) -> dict[str, Any]:
    def case_metric(key: str, subset: list[dict[str, Any]] | None = None) -> dict[str, float]:
        rows = subset if subset is not None else case_rows
        return _bootstrap_mean_ci([float(r[key]) for r in rows if key in r])

    overall = {
        "mae_util": case_metric("mae_util"),
        "mae_abs": case_metric("mae_abs"),
        "spearman_cal_std_util_vs_utilerr": case_metric("spearman_cal_std_util_vs_utilerr"),
        "spearman_raw_std_util_vs_utilerr": case_metric("spearman_raw_std_util_vs_utilerr"),
        "conformal_std_scale": conformal_scale,
    }
    for k in NOMINAL_K:
        overall[f"coverage_raw_util_k{k:g}"] = case_metric(f"coverage_raw_util_k{k:g}")
        overall[f"coverage_cal_util_k{k:g}"] = case_metric(f"coverage_cal_util_k{k:g}")
        overall[f"mean_width_raw_util_k{k:g}"] = case_metric(f"mean_width_raw_util_k{k:g}")
        overall[f"mean_width_cal_util_k{k:g}"] = case_metric(f"mean_width_cal_util_k{k:g}")
        overall[f"median_width_cal_util_k{k:g}"] = case_metric(f"median_width_cal_util_k{k:g}")

    by_topo: dict[str, Any] = {}
    for topo in DEV_TOPOS:
        subset = [r for r in case_rows if r["topology"] == topo]
        by_topo[topo] = {
            "n": len(subset),
            "mae_util": case_metric("mae_util", subset),
            "coverage_cal_util_k2": case_metric("coverage_cal_util_k2", subset),
            "coverage_raw_util_k2": case_metric("coverage_raw_util_k2", subset),
            "mean_width_cal_util_k2": case_metric("mean_width_cal_util_k2", subset),
            "spearman_cal": case_metric("spearman_cal_std_util_vs_utilerr", subset),
        }

    by_scale: dict[str, Any] = {}
    for scale in SCALES:
        subset = [r for r in case_rows if float(r["scale"]) == float(scale)]
        by_scale[str(scale)] = {
            "n": len(subset),
            "mae_util": case_metric("mae_util", subset),
            "coverage_cal_util_k2": case_metric("coverage_cal_util_k2", subset),
            "mean_width_cal_util_k2": case_metric("mean_width_cal_util_k2", subset),
            "actual_mlu": case_metric("actual_mlu", subset),
        }

    # Capacity proximity: aggregate link-level (report n_links; CI still bootstrap by case means).
    by_prox: dict[str, Any] = {}
    for name, _lo, _hi in CAPACITY_BINS:
        links = [r for r in link_rows if r["capacity_proximity"] == name]
        if not links:
            by_prox[name] = {"n_links": 0}
            continue
        util_err = np.array([r["util_err"] for r in links], dtype=np.float64)
        raw_u = np.array([r["raw_std_util"] for r in links], dtype=np.float64)
        cal_u = np.array([r["cal_std_util"] for r in links], dtype=np.float64)
        # Case-pooled detection within bin.
        det_raw = uncertainty_error_detection(util_err, raw_u)
        det_cal = uncertainty_error_detection(util_err, cal_u)
        by_prox[name] = {
            "n_links": len(links),
            "mae_util": float(np.mean(util_err)),
            "mean_raw_std_util": float(np.mean(raw_u)),
            "mean_cal_std_util": float(np.mean(cal_u)),
            "spearman_cal_std_vs_utilerr": _spearman(cal_u, util_err),
            "spearman_raw_std_vs_utilerr": _spearman(raw_u, util_err),
            "large_error_detection_raw": det_raw,
            "large_error_detection_cal": det_cal,
            "note": "Link rows within a bin are not independent seeds; Spearman is descriptive.",
        }

    # Reliability: bin predicted cal std_util and measure empirical MAE util.
    reliability = _reliability_curve(link_rows, std_key="cal_std_util", err_key="util_err", n_bins=8)
    reliability_raw = _reliability_curve(link_rows, std_key="raw_std_util", err_key="util_err", n_bins=8)

    return {
        "overall": overall,
        "by_topology": by_topo,
        "by_scale": by_scale,
        "by_capacity_proximity": by_prox,
        "reliability_calibrated": reliability,
        "reliability_raw": reliability_raw,
    }


def _reliability_curve(
    link_rows: list[dict[str, Any]],
    *,
    std_key: str,
    err_key: str,
    n_bins: int,
) -> list[dict[str, float]]:
    if not link_rows:
        return []
    std = np.array([r[std_key] for r in link_rows], dtype=np.float64)
    err = np.array([r[err_key] for r in link_rows], dtype=np.float64)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(std, qs))
    if edges.size < 3:
        return []
    out = []
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (std >= lo) & (std <= hi if i == edges.size - 2 else std < hi)
        if not np.any(mask):
            continue
        out.append(
            {
                "bin": i,
                "std_lo": float(lo),
                "std_hi": float(hi),
                "mean_predicted_std": float(np.mean(std[mask])),
                "empirical_mae": float(np.mean(err[mask])),
                "n": int(np.sum(mask)),
            }
        )
    return out


def select_thresholds(
    case_rows: list[dict[str, Any]],
    link_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select (k, utilisation_limit) on threshold seeds only (val development topos)."""
    select_cases = [r for r in case_rows if r.get("threshold_selection_seed")]
    holdout_cases = [r for r in case_rows if not r.get("threshold_selection_seed")]
    k_grid = [0.5, 1.0, 1.5, 2.0, 2.5]
    limit_grid = [0.95, 1.0, 1.05]

    def score_subset(subset: list[dict[str, Any]], k: float, limit: float, unc: str) -> dict[str, float]:
        # Reconstruct from stored alarm fields when available; else approximate via case fields.
        # We recompute from link rows filtered to selection seeds for accuracy.
        return {}  # filled below

    # Use link rows for selection seeds to evaluate UCB alarms vs actual util>1.
    def eval_config(seeds_flag: bool, k: float, limit: float, unc_key: str) -> dict[str, float]:
        links = [r for r in link_rows if bool(r.get("threshold_selection_seed")) == seeds_flag]
        # Aggregate to cases.
        groups: dict[tuple, list] = defaultdict(list)
        for r in links:
            groups[(r["seed"], r["topology"], r["matrix_id"], r["scale"])].append(r)
        detections = []
        false_alarms = []
        accepts_safe = []
        unsafe_accepts = []
        for _key, rows in groups.items():
            util_true = np.array([x["util_true"] for x in rows])
            util_mean = np.array([x["util_mean"] for x in rows])
            std_u = np.array([x[unc_key] for x in rows])
            ucb = util_mean + k * std_u
            reject = bool(np.any(ucb > limit))
            unsafe = bool(np.any(util_true > 1.0 + 1e-9))
            accept = not reject
            if unsafe:
                detections.append(1.0 if reject else 0.0)
                if accept:
                    unsafe_accepts.append(1.0)
                else:
                    unsafe_accepts.append(0.0)
            else:
                false_alarms.append(1.0 if reject else 0.0)
                accepts_safe.append(1.0 if accept else 0.0)
        n = len(groups)
        return {
            "n_cases": float(n),
            "n_unsafe": float(len(detections)),
            "n_safe": float(len(false_alarms)),
            "error_detection_rate": float(np.mean(detections)) if detections else float("nan"),
            "false_alarm_rate": float(np.mean(false_alarms)) if false_alarms else float("nan"),
            "unsafe_acceptance_rate": float(np.mean(unsafe_accepts)) if unsafe_accepts else float("nan"),
            "safe_acceptance_rate": float(np.mean(accepts_safe)) if accepts_safe else float("nan"),
        }

    candidates = []
    for unc_key, unc_name in (("cal_std_util", "calibrated"), ("raw_std_util", "raw")):
        for k in k_grid:
            for limit in limit_grid:
                sel = eval_config(True, k, limit, unc_key)
                hol = eval_config(False, k, limit, unc_key)
                # Objective: maximize detection, then minimize false alarms, prefer narrower k.
                det = sel["error_detection_rate"]
                far = sel["false_alarm_rate"]
                if not np.isfinite(det):
                    det = 0.0
                if not np.isfinite(far):
                    far = 1.0
                score = det - 0.5 * far - 0.02 * k
                candidates.append(
                    {
                        "uncertainty": unc_name,
                        "unc_key": unc_key,
                        "k": k,
                        "utilisation_limit": limit,
                        "selection": sel,
                        "holdout_diagnostic_seeds": hol,
                        "score": score,
                    }
                )

    # Prefer calibrated configs.
    cal_cands = [c for c in candidates if c["uncertainty"] == "calibrated"]
    ranked = sorted(cal_cands, key=lambda c: (-c["score"], c["k"], c["utilisation_limit"]))
    best = ranked[0] if ranked else None
    # Also best raw for comparison.
    raw_ranked = sorted(
        [c for c in candidates if c["uncertainty"] == "raw"],
        key=lambda c: (-c["score"], c["k"], c["utilisation_limit"]),
    )
    return {
        "selection_seeds": list(THRESHOLD_SEEDS),
        "holdout_seeds_for_sensitivity": [s for s in DIAGNOSTIC_SEEDS if s not in THRESHOLD_SEEDS],
        "grid": {"k": k_grid, "utilisation_limit": limit_grid},
        "objective": "score = error_detection_rate - 0.5*false_alarm_rate - 0.02*k on selection seeds",
        "selected": best,
        "best_raw_comparator": raw_ranked[0] if raw_ranked else None,
        "all_candidates": candidates,
        "rationale": (
            "Thresholds chosen on val matrices for B4/UsCarrier/Cogentco using seeds "
            f"{THRESHOLD_SEEDS} only. GEANT/Kdl unused. Diagnostic seeds "
            f"{[s for s in DIAGNOSTIC_SEEDS if s not in THRESHOLD_SEEDS]} used for sensitivity, not selection."
        ),
    }


def make_plots(case_rows: list[dict[str, Any]], link_rows: list[dict[str, Any]], plots: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Coverage vs k
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(NOMINAL_K)
    raw_c = [np.mean([r[f"coverage_raw_util_k{k:g}"] for r in case_rows]) for k in ks]
    cal_c = [np.mean([r[f"coverage_cal_util_k{k:g}"] for r in case_rows]) for k in ks]
    ax.plot(ks, raw_c, marker="o", label="raw")
    ax.plot(ks, cal_c, marker="s", label="calibrated")
    ax.axhline(0.9, color="gray", ls="--", lw=1, label="0.90 reference")
    ax.set_xlabel("k (nominal |err| ≤ k·std)")
    ax.set_ylabel("Empirical coverage (util)")
    ax.set_title("Prediction interval coverage (case-mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "coverage_vs_k.png", dpi=140)
    plt.close(fig)

    # Width vs k
    fig, ax = plt.subplots(figsize=(7, 4))
    raw_w = [np.mean([r[f"mean_width_raw_util_k{k:g}"] for r in case_rows]) for k in ks]
    cal_w = [np.mean([r[f"mean_width_cal_util_k{k:g}"] for r in case_rows]) for k in ks]
    ax.plot(ks, raw_w, marker="o", label="raw")
    ax.plot(ks, cal_w, marker="s", label="calibrated")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean interval width (util)")
    ax.set_title("Interval width vs k")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "width_vs_k.png", dpi=140)
    plt.close(fig)

    # Reliability
    rel = _reliability_curve(link_rows, std_key="cal_std_util", err_key="util_err", n_bins=8)
    if rel:
        fig, ax = plt.subplots(figsize=(5, 5))
        xs = [r["mean_predicted_std"] for r in rel]
        ys = [r["empirical_mae"] for r in rel]
        ax.plot(xs, ys, marker="o")
        lim = max(max(xs), max(ys)) * 1.05 if xs else 1.0
        ax.plot([0, lim], [0, lim], ls="--", color="gray", label="y=x")
        ax.set_xlabel("Mean predicted std (util)")
        ax.set_ylabel("Empirical MAE (util)")
        ax.set_title("Reliability curve (calibrated)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / "reliability_calibrated.png", dpi=140)
        plt.close(fig)

    # Uncertainty vs error scatter (subsample)
    fig, ax = plt.subplots(figsize=(6, 5))
    idx = np.linspace(0, len(link_rows) - 1, num=min(4000, len(link_rows)), dtype=int)
    xs = [link_rows[i]["cal_std_util"] for i in idx]
    ys = [link_rows[i]["util_err"] for i in idx]
    ax.scatter(xs, ys, s=4, alpha=0.25)
    ax.set_xlabel("Calibrated std (util)")
    ax.set_ylabel("|util error|")
    ax.set_title("Uncertainty vs absolute util error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "uncertainty_vs_error.png", dpi=140)
    plt.close(fig)

    # Capacity proximity bars
    fig, ax = plt.subplots(figsize=(7, 4))
    names = [b[0] for b in CAPACITY_BINS]
    mae = []
    stdm = []
    for name in names:
        links = [r for r in link_rows if r["capacity_proximity"] == name]
        mae.append(float(np.mean([r["util_err"] for r in links])) if links else 0.0)
        stdm.append(float(np.mean([r["cal_std_util"] for r in links])) if links else 0.0)
    x = np.arange(len(names))
    ax.bar(x - 0.15, mae, width=0.3, label="MAE util")
    ax.bar(x + 0.15, stdm, width=0.3, label="mean cal std util")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.set_title("Capacity-proximity analysis")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "capacity_proximity.png", dpi=140)
    plt.close(fig)

    # Coverage by scale
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, key in (("raw", "coverage_raw_util_k2"), ("cal", "coverage_cal_util_k2")):
        ys = []
        for scale in SCALES:
            subset = [r for r in case_rows if float(r["scale"]) == float(scale)]
            ys.append(float(np.mean([r[key] for r in subset])) if subset else float("nan"))
        ax.plot(list(SCALES), ys, marker="o", label=label)
    ax.set_xlabel("Traffic scale")
    ax.set_ylabel("Coverage @ k=2 (util)")
    ax.set_title("Coverage across traffic scales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots / "coverage_by_scale.png", dpi=140)
    plt.close(fig)


def write_threshold_markdown(path: Path, report: dict[str, Any]) -> None:
    sel = report.get("selected") or {}
    lines = [
        "# Threshold selection report",
        "",
        "Selection used **validation** matrices on B4/UsCarrier/Cogentco only.",
        f"Selection seeds: `{report.get('selection_seeds')}`.",
        f"Sensitivity seeds (not used for selection): `{report.get('holdout_seeds_for_sensitivity')}`.",
        "",
        "**GEANT and Kdl were not used.**",
        "",
        f"Objective: `{report.get('objective')}`.",
        "",
        "## Selected configuration",
        "",
    ]
    if sel:
        lines.extend(
            [
                f"- Uncertainty: **{sel.get('uncertainty')}**",
                f"- k: **{sel.get('k')}**",
                f"- utilisation_limit: **{sel.get('utilisation_limit')}**",
                f"- Selection score: {sel.get('score'):.4f}",
                f"- Selection error-detection: {sel['selection'].get('error_detection_rate')}",
                f"- Selection false-alarm: {sel['selection'].get('false_alarm_rate')}",
                f"- Holdout error-detection: {sel['holdout_diagnostic_seeds'].get('error_detection_rate')}",
                f"- Holdout false-alarm: {sel['holdout_diagnostic_seeds'].get('false_alarm_rate')}",
                "",
                f"Rationale: {report.get('rationale')}",
                "",
            ]
        )
    else:
        lines.append("No calibrated candidate selected.")
    raw = report.get("best_raw_comparator")
    if raw:
        lines.extend(
            [
                "## Best raw-uncertainty comparator",
                "",
                f"- k={raw.get('k')}, limit={raw.get('utilisation_limit')}, score={raw.get('score'):.4f}",
                f"- Selection detection={raw['selection'].get('error_detection_rate')}, FAR={raw['selection'].get('false_alarm_rate')}",
                "",
            ]
        )
    lines.extend(["## Candidate grid", "", "See `tables/threshold_selection.json` for all candidates.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_report(
    path: Path,
    summary: dict[str, Any],
    threshold_report: dict[str, Any],
    config: dict[str, Any],
) -> None:
    ov = summary["overall"]
    sel = threshold_report.get("selected") or {}
    lines = [
        "# Uncertainty Diagnostic Report",
        "",
        "**Type:** Digital Twin uncertainty diagnostic (not the final A–F safety evaluation).",
        "",
        "## Scope",
        "",
        f"- Topologies: {config['dev_topologies']} (robustness_v1 **val** catalog)",
        f"- Scales: {config['scales']}",
        f"- Policy: `{config['policy']}`",
        f"- Conformal scale (frozen): **{config['conformal_std_scale']:.6f}**",
        f"- Cases: **{summary['n_cases']}**; link rows: **{summary['n_link_rows']}**",
        f"- Held out of this study: {summary['held_out_from_this_study']}",
        "",
        "### Measured vs hypothesis",
        "",
        "- **Measured:** coverage, interval width, MAE, rank correlation of std vs error, capacity-bin stats, threshold grid scores on selection seeds.",
        "- **Hypothesis (not claimed):** that the selected (k, limit) will improve Phase-3 A–F safety metrics — deferred to the full evaluation.",
        "",
        "## Overall uncertainty quality (case-level means, 95% bootstrap CI)",
        "",
        f"- MAE util: {ov['mae_util']['mean']:.6g} [{ov['mae_util']['ci95_low']:.6g}, {ov['mae_util']['ci95_high']:.6g}] (n={ov['mae_util']['n']})",
        f"- Spearman(cal std util, util err): {ov['spearman_cal_std_util_vs_utilerr']['mean']:.4f}",
        f"- Spearman(raw std util, util err): {ov['spearman_raw_std_util_vs_utilerr']['mean']:.4f}",
        "",
        "### Coverage @ utilisation space",
        "",
    ]
    for k in NOMINAL_K:
        rk = f"coverage_raw_util_k{k:g}"
        ck = f"coverage_cal_util_k{k:g}"
        lines.append(
            f"- k={k:g}: raw={ov[rk]['mean']:.4f} [{ov[rk]['ci95_low']:.4f}, {ov[rk]['ci95_high']:.4f}]; "
            f"cal={ov[ck]['mean']:.4f} [{ov[ck]['ci95_low']:.4f}, {ov[ck]['ci95_high']:.4f}]"
        )
    lines.extend(
        [
            "",
            "### Interval width @ k=2 (util)",
            "",
            f"- Raw mean width: {ov['mean_width_raw_util_k2']['mean']:.6g}",
            f"- Cal mean width: {ov['mean_width_cal_util_k2']['mean']:.6g}",
            f"- Cal median width: {ov['median_width_cal_util_k2']['mean']:.6g}",
            "",
            "## Raw vs calibrated",
            "",
            f"Calibration multiplies ensemble std by {config['conformal_std_scale']:.6f} (slightly **narrower** than raw).",
            "If cal coverage remains near raw while width shrinks, calibration is beneficial; see coverage/width tables.",
            "",
            "## By topology",
            "",
        ]
    )
    for topo, block in summary["by_topology"].items():
        lines.append(
            f"- **{topo}** (n={block['n']}): MAE util={block['mae_util']['mean']:.6g}; "
            f"cov raw@2={block['coverage_raw_util_k2']['mean']:.4f}; "
            f"cov cal@2={block['coverage_cal_util_k2']['mean']:.4f}; "
            f"Spearman cal={block['spearman_cal']['mean']:.4f}"
        )
    lines.extend(["", "## By traffic scale", ""])
    for scale, block in summary["by_scale"].items():
        lines.append(
            f"- scale={scale}: MAE util={block['mae_util']['mean']:.6g}; "
            f"cov cal@2={block['coverage_cal_util_k2']['mean']:.4f}; "
            f"mean MLU={block['actual_mlu']['mean']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Capacity-proximity analysis",
            "",
            "Bins use **actual** link utilisation. Link rows are descriptive; they are not independent experimental seeds.",
            "",
        ]
    )
    for name, block in summary["by_capacity_proximity"].items():
        if block.get("n_links", 0) == 0:
            lines.append(f"- **{name}**: no links")
            continue
        det = block["large_error_detection_cal"]
        lines.append(
            f"- **{name}** (n_links={block['n_links']}): MAE util={block['mae_util']:.6g}; "
            f"mean cal std={block['mean_cal_std_util']:.6g}; "
            f"Spearman(cal,err)={block['spearman_cal_std_vs_utilerr']:.4f}; "
            f"large-error detection(cal)={det.get('error_detection_rate')}, FAR={det.get('false_alarm_rate')}"
        )
    lines.extend(
        [
            "",
            "## Threshold selection (post-diagnostic)",
            "",
            f"See `THRESHOLD_SELECTION.md`. Selected: k={sel.get('k')}, "
            f"utilisation_limit={sel.get('utilisation_limit')}, uncertainty={sel.get('uncertainty')}.",
            "",
            "## Artifacts",
            "",
            "- `raw/case_records.jsonl`, `raw/link_records.jsonl`",
            "- `tables/summary.json`, `tables/threshold_selection.json`",
            "- `plots/coverage_vs_k.png`, `width_vs_k.png`, `reliability_calibrated.png`,",
            "  `uncertainty_vs_error.png`, `capacity_proximity.png`, `coverage_by_scale.png`",
            "",
            "## Preservation",
            "",
            "No Twin/RL/gate checkpoints or prior evaluation trees were modified. GEANT/Kdl unused. Full A–F not launched.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Twin uncertainty diagnostics (dev topos only)")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    result = run_diagnostics(out_dir=args.output_dir)
    print(
        json.dumps(
            {
                "out_dir": result["out_dir"],
                "n_cases": result["summary"]["n_cases"],
                "selected_k": (result["threshold_report"].get("selected") or {}).get("k"),
                "selected_limit": (result["threshold_report"].get("selected") or {}).get(
                    "utilisation_limit"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
