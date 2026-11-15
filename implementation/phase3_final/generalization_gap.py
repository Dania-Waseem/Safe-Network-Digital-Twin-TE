"""Policy generalization: generalization gap of the learned policies (actor only, no gate).

gap = metric(seen validation) - metric(unseen test), for
  * MLU ratio to shortest path (lower is better),
  * P1 demand satisfaction (higher is better),
  * violation rate = share of proposals that overload at least one link.

Seen validation: B4 val (8 TMs), UsCarrier val (1 TM; toy copies), Cogentco val (8 TMs).
Unseen test: GEANT (200 seeded test TMs) and Kdl (1 TM), read from the E2 holdout
``case_rows.jsonl`` (same proposals as the A-F evaluation). Both at the same 8 load levels
(fixed 0.25-1.0, calibrated 0.5-1.1). Each side is the unweighted mean over its topologies,
so Cogentco or GEANT cannot dominate by case count.

Usage: python -m phase3_final.generalization_gap
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch

from framework.general_policy import GeneralPolicy
from twin.data import ROOT, load_split_demands, load_topology
from twin.physics import od_demands

from . import routing as R
from .common import TOY_TM_NOTE, out_dir, read_jsonl, write_json
from .holdout_af import CALIBRATION, FIXED_SCALES, TARGETS

OUT = out_dir("policy_v2")
POLICIES = {
    "improved_stable": ROOT / "models" / "policy" / "improved" / "recommended_stable.pt",
    "v2_calload_normscore": ROOT / "models" / "policy_v2" / "v2_calload_normscore" / "best.pt",
}
E2 = ROOT / "results" / "phase3_final" / "e2_holdout_af" / "case_rows.jsonl"


def seen_metrics(pid: str, path) -> dict:
    cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    pol = GeneralPolicy(torch.device("cpu"))
    pol.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
    pol.model.eval()
    out = {}
    for name, positions in (("B4", range(8)), ("UsCarrier", [0]), ("Cogentco", range(8))):
        topo = load_topology(name)
        demands = load_split_demands("val", name)[0]
        sp = R.shortest_action(topo)
        ratios, sats, viols = [], [], []
        for pos in positions:
            alpha = cal["alpha"][f"{name}/val/{pos}"]
            for scale in list(FIXED_SCALES) + [alpha[f"{t:.1f}"] for t in TARGETS]:
                m = demands[pos] * scale
                with torch.no_grad():
                    a = pol.propose(topo, od_demands(m))
                load = R.loads(topo, m, a)
                ratios.append(float((load / topo.capacity).max()) / R.mlu(topo, m, sp))
                sats.append(R.satisfaction_p1(topo, m, a))
                viols.append(bool(np.any(load > topo.capacity + 1e-6)))
        out[name] = {"n": len(ratios), "mlu_ratio": float(np.mean(ratios)), "sat": float(np.mean(sats)),
                     "violation_rate": float(np.mean(viols))}
        pol.release(name)
    return out


def unseen_metrics(pid: str) -> dict:
    by = defaultdict(list)
    for r in read_jsonl(E2):
        if r["policy_id"] == pid:
            by[r["topology"]].append(r)
    return {name: {"n": len(rs), "mlu_ratio": float(np.mean([r["mlu_proposed"] / r["mlu_shortest"] for r in rs])),
                   "sat": float(np.mean([r["sat_proposed"] for r in rs])),
                   "violation_rate": float(np.mean([not r["oracle_feasible_proposed"] for r in rs]))}
            for name, rs in by.items()}


def main() -> None:
    result = {}
    for pid, path in POLICIES.items():
        seen, unseen = seen_metrics(pid, path), unseen_metrics(pid)
        if not unseen:
            print(f"{pid}: no holdout rows yet; skipped")
            continue
        agg = lambda d, k: float(np.mean([v[k] for v in d.values()]))  # noqa: E731
        result[pid] = {"seen": seen, "unseen": unseen,
                       "gap": {k: agg(seen, k) - agg(unseen, k) for k in ("mlu_ratio", "sat", "violation_rate")},
                       "seen_mean": {k: agg(seen, k) for k in ("mlu_ratio", "sat", "violation_rate")},
                       "unseen_mean": {k: agg(unseen, k) for k in ("mlu_ratio", "sat", "violation_rate")}}
    write_json(OUT / "generalization_gap.json", result)
    sel = json.loads((OUT / "reference_recommended_stable_val.json").read_text(encoding="utf-8"))["val"]["S"]
    v2 = json.loads((OUT / "runs" / "v2_calload_normscore" / "summary.json").read_text(encoding="utf-8"))
    curve = read_jsonl(OUT / "runs" / "v2_calload_normscore" / "training_curve.jsonl")
    lines = [
        "# Policy v2: calibrated-load training + per-topology normalised selection",
        "",
        "One training run (`phase3_final/train_policy_v2.py`). Identical to the run behind "
        "`recommended_stable.pt` (GNN, loss, Adam lr 5e-5, batch 1, ≤24 epochs, patience 6, seed 2026, "
        "4 large-graph train TMs) except for two changes: (1) training TMs at calibrated load (2 of the targets "
        "{0.5, 0.7, 0.9, 1.1} per TM per epoch, calibrated-load alpha), (2) selection on "
        "S = mean over B4/UsCarrier/Cogentco of (policy MLU / shortest MLU) on calibrated validation TMs. "
        "GEANT/Kdl and TM indices 28–35 were never read during training or selection.",
        "",
        f"{TOY_TM_NOTE}",
        "",
        "## Validation selection (the only basis for choosing a checkpoint)",
        "",
        "| Checkpoint | S (val) | B4 ratio | UsCarrier ratio | Cogentco ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    ref = json.loads((OUT / "reference_recommended_stable_val.json").read_text(encoding="utf-8"))["val"]
    b = v2["best_val"]
    init = next(c for c in curve if c["epoch"] == -1)["val"]
    for label, v in (("untrained initialisation", init), ("`recommended_stable.pt` (reference)", ref),
                     (f"v2 best (epoch {v2['best_epoch'] + 1} of {v2['epochs_ran']})", b)):
        r = v["ratio_by_topology"]
        lines.append(f"| {label} | {v['S']:.4f} | {r['B4']:.4f} | {r['UsCarrier']:.4f} | {r['Cogentco']:.4f} |")
    winner = "v2" if b["S"] < sel else "`recommended_stable.pt`"
    lines += [
        "",
        f"**Selected on validation: {winner}.** v2 reaches S = {b['S']:.4f} versus {sel:.4f} for the reference "
        f"({'better' if b['S'] < sel else 'worse'} by {abs(b['S'] - sel):.4f}). "
        + ("v2 therefore becomes the 'best new policy'." if b["S"] < sel else
           "The two changes did **not** produce a better policy on validation; this is a negative result. "
           "`recommended_stable.pt` remains the policy of the frozen framework."),
        "",
        "The MLU ratio is almost constant across load targets for both checkpoints "
        f"(v2: {', '.join(f'{k}: {x:.4f}' for k, x in b['ratio_by_target'].items())}), i.e. the actor's split "
        "barely depends on load magnitude even when trained across calibrated loads.",
        "",
        "## Generalization gap (actor only; evaluated once on GEANT/Kdl test)",
        "",
        "| Policy | Side | Topology | n | MLU / shortest | Satisfaction | Violation rate |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for pid, res in result.items():
        for side in ("seen", "unseen"):
            for name, m in res[side].items():
                lines.append(f"| {pid} | {side} | {name} | {m['n']} | {m['mlu_ratio']:.4f} | {m['sat']:.4f} | "
                             f"{m['violation_rate']:.3f} |")
    lines += ["", "| Policy | MLU ratio seen | unseen | gap | Sat. seen | unseen | gap | Viol. seen | unseen | gap |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for pid, res in result.items():
        s, u, g = res["seen_mean"], res["unseen_mean"], res["gap"]
        lines.append(f"| {pid} | {s['mlu_ratio']:.4f} | {u['mlu_ratio']:.4f} | {g['mlu_ratio']:+.4f} | {s['sat']:.4f} | "
                     f"{u['sat']:.4f} | {g['sat']:+.4f} | {s['violation_rate']:.3f} | {u['violation_rate']:.3f} | "
                     f"{g['violation_rate']:+.3f} |")
    lines += ["", "Gap = seen − unseen (unweighted means over topologies). For the MLU ratio a negative gap means "
              "the policy is *worse* on unseen topologies; for satisfaction a positive gap means worse on unseen.", "",
              "**Reading the gap.** The MLU ratio to shortest path is the cleanest generalization metric because it "
              "normalises away load difficulty: both policies are about 0.34 worse relative to shortest path on "
              "unseen GEANT/Kdl than on the seen topologies. Satisfaction and violation rate are confounded by load "
              "difficulty: the seen side includes UsCarrier and Cogentco, whose toy/gravity TMs are infeasible at "
              "most load levels (low satisfaction for every routing), so their 'gaps' have the opposite sign and "
              "should not be read as better generalization. v2 and `recommended_stable` are practically identical on "
              "the test topologies (MLU ratio 1.5139 vs 1.5140): calibrated-load training did not change "
              "generalization.",
              "", "## Artifacts", "", "- `runs/v2_calload_normscore/{config.json,training_curve.jsonl,summary.json}` "
              "(`best.pt`, `last.pt` not committed)", "- `reference_recommended_stable_val.json`, `generalization_gap.json`",
              "- Holdout rows for both policies: `../e2_holdout_af/case_rows.jsonl` (policy_id field)"]
    (OUT / "TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v["gap"] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
