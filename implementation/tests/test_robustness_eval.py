"""Unit checks for multi-seed robustness evaluation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from framework.robustness_eval import (
    IMPROVED_BEST,
    IMPROVED_STABLE,
    SEEDS,
    TOPOLOGY_SPEC,
    aggregate,
    build_catalog,
    sample_matrix_indices,
    write_report,
)
from framework.general_policy import CHECKPOINT


class SamplingTest(unittest.TestCase):
    def test_sample_is_deterministic_and_unique(self) -> None:
        a = sample_matrix_indices(8, 2, 101, "B4")
        b = sample_matrix_indices(8, 2, 101, "B4")
        c = sample_matrix_indices(8, 2, 202, "B4")
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(set(a)))
        self.assertNotEqual(a, c)

    def test_catalog_holds_out_geant_kdl_as_test(self) -> None:
        catalog = build_catalog(seeds=(101, 202))
        self.assertEqual(catalog["topologies"]["GEANT"]["split"], "test")
        self.assertEqual(catalog["topologies"]["Kdl"]["split"], "test")
        self.assertTrue(catalog["topologies"]["GEANT"]["unseen"])
        self.assertTrue(catalog["topologies"]["Kdl"]["unseen"])
        self.assertEqual(catalog["topologies"]["B4"]["split"], "val")
        for name, _split, count in TOPOLOGY_SPEC:
            for seed in ("101", "202"):
                local = catalog["topologies"][name]["by_seed"][seed]["local_indices"]
                self.assertEqual(len(local), count)


class AggregateTest(unittest.TestCase):
    def test_mean_std_across_seeds_and_scenarios(self) -> None:
        rows = []
        for seed, mlu in ((101, 1.0), (202, 3.0)):
            for method in ("shortest", "improved_stable", "improved_best", "frozen_p1", "general_3epoch", "equal"):
                rows.append(
                    {
                        "seed": seed,
                        "topology": "B4",
                        "scale": 0.25,
                        "scenario": "feasible",
                        "feasible_any_baseline": True,
                        "method": method,
                        "mlu_actor": mlu if method != "equal" else mlu * 0.5,
                        "mlu_applied": mlu,
                        "violations_actor": 0,
                        "violations_applied": 0,
                        "acceptance": 1.0 if method.startswith("improved") else 0.0,
                        "unsafe_accept": 0.0,
                        "false_reject": 0.0,
                        "fallback_used": 0.0,
                        "unavoidable_overload": 0.0,
                        "runtime_s": 0.1,
                        "policy_s": 0.01,
                        "twin_ms": 1.0,
                    }
                )
            # infeasible cogentco-like rows
            for method in ("shortest", "improved_stable", "improved_best", "frozen_p1", "general_3epoch", "equal"):
                rows.append(
                    {
                        "seed": seed,
                        "topology": "Cogentco",
                        "scale": 1.0,
                        "scenario": "infeasible",
                        "feasible_any_baseline": False,
                        "method": method,
                        "mlu_actor": 50.0,
                        "mlu_applied": 50.0,
                        "violations_actor": 10,
                        "violations_applied": 10,
                        "acceptance": 0.0,
                        "unsafe_accept": 0.0,
                        "false_reject": 0.0,
                        "fallback_used": 1.0,
                        "unavoidable_overload": 1.0,
                        "runtime_s": 0.2,
                        "policy_s": 0.02,
                        "twin_ms": 2.0,
                    }
                )
        summary = aggregate(rows)
        stable = summary["by_topology_scale"][0]["methods"]["improved_stable"]
        self.assertAlmostEqual(stable["mlu_actor_mean"], 2.0)
        self.assertGreater(stable["mlu_actor_std"], 0.0)
        self.assertIn("without_cogentco", summary["distortion"])
        self.assertLess(
            summary["distortion"]["without_cogentco"]["improved_stable"]["mlu_actor_mean"],
            summary["distortion"]["all_topologies"]["improved_stable"]["mlu_actor_mean"],
        )

    def test_report_mentions_both_improved_checkpoints(self) -> None:
        rows = []
        for seed in (101, 202):
            for topo, scenario, feas in (("B4", "feasible", True), ("Cogentco", "infeasible", False)):
                for method in ("shortest", "equal", "frozen_p1", "general_3epoch", "improved_best", "improved_stable"):
                    rows.append(
                        {
                            "seed": seed,
                            "topology": topo,
                            "scale": 0.5,
                            "scenario": scenario,
                            "feasible_any_baseline": feas,
                            "method": method,
                            "mlu_actor": 2.0,
                            "mlu_applied": 2.0,
                            "violations_actor": 0 if feas else 3,
                            "violations_applied": 0 if feas else 3,
                            "acceptance": 1.0 if feas and method != "shortest" else 0.0,
                            "unsafe_accept": 0.0,
                            "false_reject": 0.0,
                            "fallback_used": 0.0 if feas else 1.0,
                            "unavoidable_overload": 0.0 if feas else 1.0,
                            "runtime_s": 0.1,
                            "policy_s": 0.01,
                            "twin_ms": 1.0,
                        }
                    )
        summary = aggregate(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            text = write_report(
                summary,
                {
                    "seeds": list(SEEDS),
                    "scales": [0.5],
                    "checkpoints": {
                        "improved_best": str(IMPROVED_BEST),
                        "improved_stable": str(IMPROVED_STABLE),
                        "general_3epoch": str(CHECKPOINT),
                    },
                },
                path,
            )
            self.assertTrue(path.exists())
        self.assertIn("best_checkpoint.pt", text)
        self.assertIn("recommended_stable.pt", text)
        self.assertIn("Cogentco", text)


class CheckpointPresenceTest(unittest.TestCase):
    def test_improved_artifacts_exist(self) -> None:
        self.assertTrue(IMPROVED_BEST.exists(), msg=str(IMPROVED_BEST))
        self.assertTrue(IMPROVED_STABLE.exists(), msg=str(IMPROVED_STABLE))
        self.assertTrue(CHECKPOINT.exists(), msg=str(CHECKPOINT))
        self.assertNotEqual(IMPROVED_BEST.resolve(), IMPROVED_STABLE.resolve())


if __name__ == "__main__":
    unittest.main()
