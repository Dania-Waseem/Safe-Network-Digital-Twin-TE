"""Evaluate the new policy on the same matrices and scales as the extended run.

Frozen-actor numbers are read from that earlier log. They are not recomputed,
and the frozen checkpoint is not loaded.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from safety.evaluate import load_config, load_trained_ensemble
from twin.data import ROOT, load_split_demands, load_topology

from .general_controller import GeneralController
from .general_policy import CHECKPOINT, GeneralPolicy
from .scale_eval import JOBS, SCALES

RESULT_DIR = ROOT / "results" / "policy"
PRIOR = ROOT / "results" / "framework" / "extended" / "decisions.jsonl"


def _prior() -> dict[tuple, dict]:
    rows = {}
    if not PRIOR.exists():
        return rows
    for line in PRIOR.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["topology"], row["matrix_index"], row["scale"])] = row
    return rows


def _summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["topology"], row["scale"]), []).append(row)
    out = []
    for (topology, scale), items in sorted(groups.items()):
        def avg(key: str, bundle=items) -> float:
            values = [item[key] for item in bundle if item.get(key) is not None]
            return float(np.mean(values)) if values else float("nan")

        def rate(key: str, bundle=items) -> float:
            return float(np.mean([bool(item[key]) for item in bundle]))

        out.append(
            {
                "topology": topology,
                "scale": scale,
                "cases": len(items),
                "acceptance_rate": rate("proposed_accepted"),
                "unsafe_accept_rate": rate("unsafe_accept"),
                "false_reject_rate": rate("false_reject"),
                "fallback_rate": rate("fallback_used"),
                "unavoidable_overload_rate": rate("unavoidable_overload"),
                "mean_mlu_new": avg("mlu_applied"),
                "mean_mlu_new_actor": avg("mlu_actor"),
                "mean_mlu_shortest": avg("mlu_shortest"),
                "mean_mlu_frozen_actor": avg("frozen_mlu_actor"),
                "mean_violations_new": avg("violations_applied"),
                "mean_violations_new_actor": avg("violations_actor"),
                "mean_prediction_mae": avg("prediction_mae"),
                "mean_policy_s": avg("policy_s"),
            }
        )
    return out


def evaluate() -> dict:
    device = torch.device("cpu")
    policy = GeneralPolicy(device)
    payload_ckpt = policy.load(CHECKPOINT)
    controller = GeneralController(policy, load_trained_ensemble(device), load_config(), device)
    prior = _prior()
    rows = []
    for name, split, stride, limit in JOBS:
        print(f"general-eval {split}/{name}", flush=True)
        topo = load_topology(name)
        demands, meta = load_split_demands(split, name)
        indices = list(range(0, demands.shape[0], stride))[:limit]
        controller.reset(name)
        for index in indices:
            matrix_id = int(meta["matrix_indices"][index])
            for scale in SCALES:
                decision = controller.decide(topo, demands[index] * float(scale))
                frozen = prior.get((name, matrix_id, float(scale)), {})
                decision.update(
                    {
                        "topology": name,
                        "split": split,
                        "matrix_index": matrix_id,
                        "scale": float(scale),
                        "frozen_mlu_actor": frozen.get("mlu_actor"),
                        "frozen_mlu_gate": frozen.get("mlu_framework"),
                    }
                )
                rows.append(decision)
                print(
                    f"  {name} tm {matrix_id} x{scale:.2f} accept={decision['proposed_accepted']} "
                    f"mlu_new={decision['mlu_actor']:.3f} mlu_applied={decision['mlu_applied']:.3f} "
                    f"mlu_sp={decision['mlu_shortest']:.3f} overload={decision['unavoidable_overload']}",
                    flush=True,
                )
        policy.release(name)
        del topo
    summary = _summary(rows)
    result = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_meta": {key: value for key, value in payload_ckpt.items() if key != "model"},
        "summary": summary,
        "cases": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "eval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULT_DIR / "eval_decisions.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    (RESULT_DIR / "GENERAL_POLICY_REPORT.md").write_text(_report(result), encoding="utf-8")
    print("Wrote", RESULT_DIR / "GENERAL_POLICY_REPORT.md", flush=True)
    return result


def _report(result: dict) -> str:
    lines = [
        "# Generalized policy and improved fallback",
        "",
        "The new actor was trained on the B4, UsCarrier, and Cogentco train splits.",
        "Validation matrices selected the checkpoint. GEANT and Kdl were not used until this evaluation.",
        "The frozen B4 checkpoint was not loaded or modified.",
        "",
        "| Topology | Scale | Accept | Unsafe accept | False reject | Fallback | Unavoidable | MLU new actor | MLU applied | MLU shortest | MLU frozen actor | Policy s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        frozen = row["mean_mlu_frozen_actor"]
        frozen_text = f"{frozen:.3g}" if frozen is not None else "n/a"
        lines.append(
            "| {topology} | {scale:.2f} | {acc:.2f} | {ua:.2f} | {fr:.2f} | {fb:.2f} | {uo:.2f} | {na:.3g} | {ap:.3g} | {sp:.3g} | {frz} | {ps:.2f} |".format(
                topology=row["topology"],
                scale=row["scale"],
                acc=row["acceptance_rate"],
                ua=row["unsafe_accept_rate"],
                fr=row["false_reject_rate"],
                fb=row["fallback_rate"],
                uo=row["unavoidable_overload_rate"],
                na=row["mean_mlu_new_actor"],
                ap=row["mean_mlu_new"],
                sp=row["mean_mlu_shortest"],
                frz=frozen_text,
                ps=row["mean_policy_s"],
            )
        )
    lines.extend(
        [
            "",
            "Applied MLU is the new actor when the gate accepts it. Otherwise it is the",
            "feasible fallback, or the least-violating candidate when every fallback overloads.",
            "Frozen-actor MLU is the earlier identical-condition run, not a new frozen-file execution.",
            "",
            "## Training",
            "",
            f"- Checkpoint: `{result['checkpoint']}`",
            f"- Selection metric: {result['checkpoint_meta'].get('val_mean_mlu')}",
            "",
            "## Remaining limits",
            "",
            "- Three epochs and two train matrices on UsCarrier and Cogentco are a short run, not a full P1 A2C schedule.",
            "- The reward is differentiable overload and MLU, not P1's ADMM-adjusted satisfaction.",
            "- ASN was not trained or evaluated.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    evaluate()
