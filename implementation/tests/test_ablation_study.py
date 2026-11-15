"""Unit checks for audit helpers and offline ablation decision rules."""

from __future__ import annotations

import unittest

import numpy as np

from framework.ablation_study import (
    ABLATION_IDS,
    apply_ablation,
    audit_robustness,
    load_robustness_rows,
    paired_test,
)
from safety.gate import GateConfig
from twin.data import load_topology


class AuditTest(unittest.TestCase):
    def test_robustness_audit_has_no_errors(self) -> None:
        rows = load_robustness_rows()
        audit = audit_robustness(rows)
        self.assertEqual(audit["n_rows"], 840)
        self.assertEqual(audit["expected_rows"], 840)
        self.assertEqual(audit["n_errors"], 0)
        codes = {item["code"] for item in audit["findings"] if item["level"] == "ok"}
        self.assertIn("row_count", codes)
        self.assertIn("held_out", codes)
        self.assertIn("paired_matrices", codes)


class AblationRuleTest(unittest.TestCase):
    def test_policy_only_always_accepts_even_if_overload(self) -> None:
        topo = load_topology("B4")
        # Artificial overloaded actor load
        actor_load = topo.capacity * 2.0
        twin_mean = actor_load.copy()
        twin_std = np.zeros_like(actor_load)
        from twin.physics import equal_split, shortest_split

        short_a = shortest_split(topo.n_paths, topo.hops)
        equal_a = equal_split(topo.n_paths)
        matrix = np.ones((12, 12), dtype=np.float64)
        decision = apply_ablation(
            "A_policy_only",
            topo=topo,
            matrix=matrix,
            actor_load=actor_load,
            twin_mean=twin_mean,
            twin_std=twin_std,
            short_action=short_a,
            equal_action=equal_a,
            full_config=GateConfig(),
            propose_s=0.01,
            twin_ms=1.0,
        )
        self.assertTrue(decision["acceptance"])
        self.assertTrue(decision["unsafe_accept"])
        self.assertEqual(decision["mlu_applied"], decision["mlu_actor"])

    def test_full_gate_rejects_clear_overload_prediction(self) -> None:
        topo = load_topology("B4")
        actor_load = topo.capacity * 2.0
        twin_mean = actor_load.copy()
        twin_std = np.zeros_like(actor_load)
        from twin.physics import equal_split, shortest_split

        short_a = shortest_split(topo.n_paths, topo.hops)
        equal_a = equal_split(topo.n_paths)
        matrix = np.ones((12, 12), dtype=np.float64) * 0.01
        decision = apply_ablation(
            "E_full_uncertainty_gate",
            topo=topo,
            matrix=matrix,
            actor_load=actor_load,
            twin_mean=twin_mean,
            twin_std=twin_std,
            short_action=short_a,
            equal_action=equal_a,
            full_config=GateConfig(k=2.0, utilisation_limit=1.0, max_relative_std=0.5),
            propose_s=0.01,
            twin_ms=1.0,
        )
        self.assertFalse(decision["acceptance"])
        self.assertFalse(decision["unsafe_accept"])
        self.assertTrue(decision["fallback_used"])

    def test_ablation_ids_complete(self) -> None:
        self.assertEqual(len(ABLATION_IDS), 5)
        self.assertTrue(any(x.startswith("E_") for x in ABLATION_IDS))


class StatsTest(unittest.TestCase):
    def test_paired_test_detects_shift(self) -> None:
        rng = np.random.default_rng(0)
        a = list(rng.normal(0.0, 0.1, size=40))
        b = [x + 0.5 for x in a]
        row = paired_test(b, a, "b_minus_a")
        self.assertEqual(row["n"], 40)
        self.assertGreater(row["mean_diff"], 0.4)
        self.assertTrue(row["significant_0_05_wilcoxon"])


if __name__ == "__main__":
    unittest.main()
