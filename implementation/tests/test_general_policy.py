"""Checks for the new actor and least-violating fallback controller."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from framework.fallback_modes import classify_candidates
from framework.general_controller import GeneralController
from framework.general_policy import CHECKPOINT, GeneralPolicy, normalized_inputs
from framework.policy_adapter import DEFAULT_CHECKPOINT, collapse_fractions, build_policy_graph
from safety.evaluate import load_config, load_trained_ensemble
from twin.data import load_split_demands, load_topology
from twin.physics import od_demands


class NormalizationTest(unittest.TestCase):
    def test_inputs_are_scale_invariant(self) -> None:
        capacity = torch.tensor([100.0, 200.0, 50.0])
        demand = torch.tensor([10.0, 20.0, 5.0])
        a, b = normalized_inputs(capacity, demand)
        c, d = normalized_inputs(capacity * 1000.0, demand * 1000.0)
        self.assertTrue(torch.allclose(a, c))
        self.assertTrue(torch.allclose(b, d))
        self.assertLessEqual(float(a.max()), 1.0 + 1e-6)


class PaddingCollapseTest(unittest.TestCase):
    def test_uscarrier_padding_sums_to_one_per_demand(self) -> None:
        topo = load_topology("UsCarrier")
        graph = build_policy_graph(topo)
        self.assertGreater(int(np.sum(graph.counts < 4)), 0)
        fractions = np.full((graph.n_demands, 4), 0.25, dtype=np.float64)
        action = collapse_fractions(fractions, graph)
        starts = np.cumsum(graph.counts) - graph.counts
        for demand, count, start in zip(range(graph.n_demands), graph.counts, starts):
            if count <= 0:
                continue
            self.assertAlmostEqual(float(action[start : start + count].sum()), 1.0, places=5)


class FallbackControllerTest(unittest.TestCase):
    def test_least_violating_beats_shortest_when_equal_is_better(self) -> None:
        capacity = np.array([10.0, 10.0])
        choice = classify_candidates(
            [
                ("shortest", np.array([12.0, 12.0])),
                ("equal", np.array([11.0, 5.0])),
            ],
            capacity,
        )
        self.assertEqual(choice["selected"], "equal")
        self.assertTrue(choice["unavoidable_overload"])

    def test_controller_accepts_quarter_scale_b4(self) -> None:
        device = torch.device("cpu")
        policy = GeneralPolicy(device)
        # Untrained weights are fine for a smoke path; checkpoint may be absent.
        if CHECKPOINT.exists():
            policy.load(CHECKPOINT)
        controller = GeneralController(policy, load_trained_ensemble(device), load_config(), device)
        topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        row = controller.decide(topo, demands[0] * 0.25)
        self.assertEqual(row["n_links"], 38)
        self.assertTrue(np.isfinite(row["mlu_actor"]))
        self.assertIn(row["proposed_accepted"], (True, False))
        self.assertLess(row["mlu_shortest"], 1.0)
        policy.release("B4")

    def test_refuses_to_write_into_frozen_repo(self) -> None:
        policy = GeneralPolicy()
        frozen = Path(DEFAULT_CHECKPOINT)
        with self.assertRaises(RuntimeError):
            policy.save(frozen, {"bad": True})


if __name__ == "__main__":
    unittest.main()
