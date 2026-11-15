"""P1 GEANT reproduction: P1 GRL-TE on the correct GEANT (22 nodes / 72 directed links, FIGRET).

Earlier reproduction attempts used a 40-node Topology Zoo GEANT with gravity TMs
(incomparable). This run uses the FIGRET GEANT graph and FIGRET's real TMs:

* topology : ``data/processed/topologies/GEANT.json`` (FIGRET 22/72),
             registered here under the name ``GEANT_FIGRET`` so the old Zoo path cache
             (``data/candidate_paths/GEANT-4-paths_min-hop.pkl``) can never be picked up.
             Paths are computed in memory (``cache=False``); nothing in ``data/`` is written.
* train TMs: FIGRET train histories (8,080; ``splits/held_aside/GEANT_figret_train``).
             Phase 3's own models never use these; here GEANT is a *seen* topology for P1,
             matching P1's per-topology training setting.
* test TMs : the 200 seeded FIGRET test TMs used throughout Phase 3 (calibrated-load subsample).
* model    : unchanged reproduction (TopoFlowNet L=6 + one-step A2C, 3 episodes, P1 learning
             rates, seeds 100..1000), Yen k=4 min-hop paths, constraint adjustment eqs. (23)-(28).
* ADMM     : rho = 1, 100 iterations (P1 setting for GEANT, ``config.ADMM_ITERS``). Headline channel
             is Teal-order (as in the frozen B4 baseline); P1-order is a diagnostic channel; no-ADMM
             gives the "ADMM removal" ablation. Channels are never averaged together.
* h0       : raw capacities and demands (P1 eq. 5), pre-registered rule: switch to FIGRET units
             (capacities and demands / 1e9, FIGRET's own preprocessing) only if training produces
             non-finite values. Whichever is used is recorded in the results.
* LP / ECMP / shortest: ``experiments/baselines.py`` (LP = max-throughput path LP, P1 eqs. 1-4,
             SciPy HiGHS; no Gurobi licence was available).

Usage (from baseline_p1_reproduction/):
  python -m phase3_final.geant_figret --seeds 100            # first seed / probe
  python -m phase3_final.geant_figret                        # all 10 seeds (resumes)
  python -m phase3_final.geant_figret --report-only
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from config import ADMM_ITERS, EPISODES, P1_PUBLISHED, SEEDS  # noqa: E402
from experiments.baselines import (  # noqa: E402
    ecmp_allocate,
    evaluate_baseline,
    lp_allocate,
    shortest_path_allocate,
)
from models.a2c import GRLTEModel  # noqa: E402
from optimization.admm import admm_refine_p1, admm_refine_teal  # noqa: E402
from optimization.constraint_adjustment import adjust_allocation, evaluate_allocation  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from training.train import train_model  # noqa: E402

IMPL = REPO.parent  # repository root (data/, results/, models/)
TOPO_SRC = IMPL / "data" / "processed" / "topologies" / "GEANT.json"
TRAIN_NPZ = IMPL / "data" / "processed" / "splits" / "held_aside" / "GEANT_figret_train_demand.npz"
TEST_NPZ = IMPL / "data" / "processed" / "splits" / "test" / "GEANT_demand.npz"
CALIB = IMPL / "results" / "phase3_final" / "calibrated_traffic" / "scaled_tm_index.json"
OUT = HERE / "results" / "geant_figret"
TOPO_DIR = HERE / "data" / "topologies"
MODEL_DIR = REPO.parent / "models" / "baseline_p1" / "geant_figret"  # checkpoints (Git LFS)
NAME = "GEANT_FIGRET"
ADMM_N = ADMM_ITERS["GEANT"]
FIGRET_UNIT = 1e9
TRAIN_SUBSAMPLE = 1000  # runtime deviation: seeded subsample of the 8,080 FIGRET train TMs
TRAIN_SUBSAMPLE_SEED = 2026


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup(unit: float):
    TOPO_DIR.mkdir(parents=True, exist_ok=True)
    dst = TOPO_DIR / f"{NAME}.json"
    if not dst.exists():
        shutil.copyfile(TOPO_SRC, dst)
    topology = load_topology(NAME, topology_dir=TOPO_DIR)
    assert (topology.num_nodes, topology.num_edges) == (22, 72), (topology.num_nodes, topology.num_edges)
    topology.capacities = topology.capacities / unit
    paths = generate_candidate_paths(topology, k=4, metric="min-hop", cache=False)
    problem = build_bipartite_graph(topology, paths, k=4, device=torch.device("cpu"))
    train_all = np.load(TRAIN_NPZ)["demand"]
    pick = np.sort(np.random.default_rng(TRAIN_SUBSAMPLE_SEED).choice(train_all.shape[0], TRAIN_SUBSAMPLE, replace=False))
    train = [train_all[i] / unit for i in pick]
    positions = json.loads(CALIB.read_text(encoding="utf-8"))["geant_test_positions_200"]
    test_all = np.load(TEST_NPZ)["demand"]
    test = [test_all[p] / unit for p in positions]
    return topology, problem, train, test, positions


@torch.no_grad()
def evaluate_channel(model, problem, tms, admm: str | None) -> dict:
    model.eval()
    sats, mlus, viols = [], [], []
    failures = []
    for i, tm in enumerate(tms):
        demands = problem.demand_vector(tm)
        h0 = problem.initial_features(demands, scale=False)
        out = model.act(h0, problem.bipartite_index, problem.num_edges, deterministic=True)
        flow = (out["fractions"] * demands.unsqueeze(-1)).reshape(-1)
        try:
            if admm == "teal":
                flow = admm_refine_teal(out["fractions"], demands, problem, num_iters=ADMM_N)
            elif admm == "p1":
                flow = admm_refine_p1(out["fractions"], demands, problem, num_iters=ADMM_N)
        except Exception as exc:  # noqa: BLE001 - P1-order ADMM can hit a singular system
            failures.append({"tm": i, "error": f"{type(exc).__name__}: {str(exc)[:120]}"})
            continue
        if isinstance(flow, tuple):
            flow = flow[0]
        flow = adjust_allocation(flow, demands, problem)
        m = evaluate_allocation(flow, demands, problem)
        sats.append(m["demand_satisfaction"])
        mlus.append(m["max_link_utilization"])
        viols.append(m["capacity_violations"])
    return {"channel": admm or "none", "n": len(tms), "n_ok": len(sats),
            "sat_mean": float(np.mean(sats)) if sats else None,
            "sat_std_over_tms": float(np.std(sats)) if sats else None, "mlu_mean": float(np.mean(mlus)) if mlus else None,
            "violations_mean": float(np.mean(viols)) if viols else None,
            "finite": bool(np.all(np.isfinite(sats))), "n_failed": len(failures), "failures": failures[:10]}


def run_seed(seed: int, problem, train, test, unit: float) -> dict:
    set_seed(seed)
    model = GRLTEModel(num_layers=6, num_paths=4)
    t0 = time.perf_counter()
    hist = train_model(model, problem, train, episodes=EPISODES, log_every=10**9)
    train_s = time.perf_counter() - t0
    finite = all(np.isfinite(p.detach().numpy()).all() for p in model.parameters())
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": seed, "unit": unit}, MODEL_DIR / f"grl_te_geant_figret_seed{seed}.pt")
    row = {"seed": seed, "unit": unit, "h0": "raw (P1 eq. 5)" if unit == 1.0 else "FIGRET units (/1e9)",
           "train_seconds": train_s, "params_finite": bool(finite), "episodes": EPISODES,
           "n_train_tms": len(train), "train_history_tail": hist.get("history", [])[-3:] if isinstance(hist, dict) else None}
    t1 = time.perf_counter()
    for ch in (None, "teal", "p1"):
        row[f"eval_{ch or 'none'}"] = evaluate_channel(model, problem, test, ch)
    row["eval_seconds"] = time.perf_counter() - t1
    return row


def baselines(problem, test) -> dict:
    out = {}
    for name, fn in (("lp_max_throughput", lp_allocate), ("ecmp", ecmp_allocate), ("shortest_path", shortest_path_allocate)):
        res = evaluate_baseline(fn, problem, test)
        out[name] = {k: v for k, v in res.items() if not isinstance(v, (list, torch.Tensor))}
    return out


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def report() -> None:
    rows = read_rows(OUT / "seeds.jsonl")
    base = json.loads((OUT / "baselines.json").read_text(encoding="utf-8"))
    p1 = P1_PUBLISHED["GEANT"]

    def agg(ch):
        v = np.array([r[f"eval_{ch}"]["sat_mean"] for r in rows if r[f"eval_{ch}"]["sat_mean"] is not None]) * 100
        return (v.mean() if v.size else float("nan")), (v.std(ddof=0) if v.size else float("nan")), len(v)

    lines = ["# P1 GRL-TE on the correct GEANT (FIGRET 22/72, real TMs)", "",
             "Reproduction code unchanged; new runner `phase3_final/geant_figret.py`. "
             f"Train: {rows[0]['n_train_tms'] if rows else '–'} FIGRET train TMs; test: 200 seeded FIGRET test TMs "
             "(same subsample as Phase 3). Yen k=4 min-hop paths, TopoFlowNet L=6 + one-step A2C, 3 episodes, "
             f"seeds {', '.join(str(r['seed']) for r in rows)}; ADMM rho=1, {ADMM_N} iterations. "
             f"h0: {rows[0]['h0'] if rows else '–'}.", "",
             f"Training uses a seeded subsample of {TRAIN_SUBSAMPLE} of the 8,080 FIGRET train TMs (seed "
             f"{TRAIN_SUBSAMPLE_SEED}; runtime deviation, fixed before any result).", "",
             "LP = max-throughput path LP (P1 eqs. 1–4) solved with SciPy HiGHS (no Gurobi licence available).", "",
             "## Demand satisfaction (%), mean ± SD over seeds (TM-mean per seed)", "",
             "| Method | This reproduction | P1 Table 6 (GEANT) |", "|---|---:|---:|"]
    for label, ch in (("GRL-TE + ADMM (Teal-order, headline)", "teal"), ("GRL-TE + ADMM (P1-order, diagnostic)", "p1"),
                      ("GRL-TE without ADMM", "none")):
        if rows:
            m, s, n = agg(ch)
            pub = f"{p1['GRL-TE']:.2f}" if ch == "teal" else ("–" if ch == "p1" else f"{p1['GRL-TE'] - 15.09:.2f} (= 86.41 − 15.09)")
            cell = (f"{m:.2f} ± {s:.2f} (n={n})" if n else
                    "failed on every TM (singular linear system in the F-update)")
            lines.append(f"| {label} | {cell} | {pub} |")
    lines += [f"| LP (max throughput) | {base['lp_max_throughput']['demand_satisfaction'] * 100:.2f} | {p1['LP']:.2f} |",
              f"| ECMP (equal split over 4 paths) | {base['ecmp']['demand_satisfaction'] * 100:.2f} | – |",
              f"| Shortest path | {base['shortest_path']['demand_satisfaction'] * 100:.2f} | – |",
              f"| NCFlow | not run | {p1['NCFlow']:.2f} (reported in P1, not reproduced here) |", ""]
    if rows:
        mt, _, _ = agg("teal")
        mn, _, _ = agg("none")
        lines += ["## ADMM ablation", "",
                  f"Removing ADMM changes satisfaction by **{mn - mt:+.2f} points** (no-ADMM − Teal-ADMM); "
                  "P1 reports −15.09 points on GEANT.", "",
                  "## Per seed", "", "| Seed | No ADMM | Teal ADMM | P1 ADMM | MLU (Teal) | Train s | Params finite |",
                  "|---:|---:|---:|---:|---:|---:|---|"]
        def pct(x):
            return "–" if x is None else f"{x * 100:.2f}"
        for r in rows:
            p1e = r["eval_p1"]
            lines.append(f"| {r['seed']} | {pct(r['eval_none']['sat_mean'])} | {pct(r['eval_teal']['sat_mean'])} | "
                         f"{pct(p1e['sat_mean'])} ({p1e.get('n_failed', 0)} of {p1e['n']} TMs failed) | "
                         f"{r['eval_teal']['mlu_mean']:.3f} | {r['train_seconds']:.0f} | {r['params_finite']} |")
        nf = sum(r["eval_p1"].get("n_failed", 0) for r in rows)
        lines += ["", f"P1-order ADMM failed (singular linear system in the F-update) on {nf} of "
                  f"{sum(r['eval_p1']['n'] for r in rows)} seed×TM evaluations; its mean is over successful TMs only."]
    lines += ["", "## Caveats", "",
              "- P1 does not publish its GEANT TM split; we use FIGRET's published train/test histories.",
              "- Teal-order ADMM is the frozen headline (stable); P1's printed order is reported separately and is "
              "known to be unstable at small iteration counts (BASELINE_FREEZE/ADMM_VARIANT_COMPARISON.md).",
              "- Teal, FIGRET and NCFlow were not run (not part of this study); their numbers are quoted from P1 only."]
    (OUT / "P1_GEANT_FIGRET_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--figret-units", action="store_true",
                    help="sensitivity channel: h0 in FIGRET units (/1e9); writes to results/geant_figret_units/")
    args = ap.parse_args()
    global OUT, MODEL_DIR
    if args.figret_units:
        OUT = HERE / "results" / "geant_figret_units"
        MODEL_DIR = REPO.parent / "models" / "baseline_p1" / "geant_figret_units"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report()
        return
    rows_path = OUT / "seeds.jsonl"
    done = {r["seed"]: r for r in read_rows(rows_path)}
    unit = next(iter(done.values()))["unit"] if done else (FIGRET_UNIT if args.figret_units else 1.0)
    topology, problem, train, test, positions = setup(unit)
    if not (OUT / "baselines.json").exists():
        (OUT / "baselines.json").write_text(json.dumps({"unit": unit, **baselines(problem, test)}, indent=2), encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps({
        "topology": "FIGRET GEANT 22/72", "name": NAME, "test_positions": positions, "n_train": len(train),
        "train_subsample": {"n": TRAIN_SUBSAMPLE, "of": 8080, "seed": TRAIN_SUBSAMPLE_SEED,
                            "reason": "runtime: all 8,080 TMs x 3 episodes ~76 min/seed on this CPU"},
        "episodes": EPISODES, "admm_iters": ADMM_N, "rho": 1.0, "seeds": args.seeds,
        "h0_rule": ("sensitivity channel: FIGRET units /1e9 (not the headline)" if args.figret_units else
                    "headline: raw unless non-finite, then FIGRET units /1e9"), "lp_solver": "SciPy HiGHS"}, indent=2), encoding="utf-8")
    for seed in args.seeds:
        if seed in done:
            continue
        print(f"seed {seed}: training on {len(train)} TMs x {EPISODES} episodes (unit={unit})", flush=True)
        row = run_seed(seed, problem, train, test, unit)
        if unit == 1.0 and not args.figret_units and not (row["params_finite"] and all(row[f"eval_{c}"]["finite"] for c in ("none", "teal"))):
            print("non-finite values with raw h0 -> switching to FIGRET units (/1e9) per pre-registered rule", flush=True)
            unit = FIGRET_UNIT
            (OUT / "h0_switch.json").write_text(json.dumps({"failed_raw_seed": seed, "raw_row": row}, indent=2, default=str),
                                                encoding="utf-8")
            topology, problem, train, test, positions = setup(unit)
            (OUT / "baselines.json").write_text(json.dumps({"unit": unit, **baselines(problem, test)}, indent=2), encoding="utf-8")
            row = run_seed(seed, problem, train, test, unit)
        with rows_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        fmt = lambda v: "failed" if v is None else f"{v:.4f}"  # noqa: E731
        print(f"seed {seed}: none={fmt(row['eval_none']['sat_mean'])} teal={fmt(row['eval_teal']['sat_mean'])} "
              f"p1={fmt(row['eval_p1']['sat_mean'])} train {row['train_seconds']:.0f}s eval {row['eval_seconds']:.0f}s", flush=True)
    report()


if __name__ == "__main__":
    main()
