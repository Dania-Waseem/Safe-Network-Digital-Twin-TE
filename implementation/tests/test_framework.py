"""Unit and integration checks for the Step 7 pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from framework.pipeline import Framework, _demand_sums
from framework.policy_adapter import FrozenP1Policy, build_policy_graph, collapse_fractions
from safety.evaluate import load_config, load_trained_ensemble
from twin.data import load_split_demands, load_topology
from twin.physics import od_demands


class CollapseTest(unittest.TestCase):
    def test_four_paths_are_copied_in_order(self) -> None:
        topo = load_topology("B4")
        graph = build_policy_graph(topo)
        fractions = np.zeros((graph.n_demands, 4), dtype=np.float64)
        fractions[:, 0] = 1.0
        action = collapse_fractions(fractions, graph)
        self.assertEqual(action.shape[0], int(topo.n_paths.sum()))
        self.assertTrue(np.allclose(_demand_sums(action, topo.n_paths), 1.0))
        # First path of every B4 demand receives the whole split.
        self.assertTrue(np.all(action[::4] == 1.0))


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cpu")
        cls.policy = FrozenP1Policy(device=cls.device)
        cls.twin = load_trained_ensemble(cls.device)
        cls.framework = Framework(cls.policy, cls.twin, load_config(), cls.device)

    def test_b4_seen_end_to_end(self) -> None:
        topo = load_topology("B4")
        demands, meta = load_split_demands("val", "B4")
        row = self.framework.decide(topo, demands[0])
        self.assertEqual(row["topology"], "B4")
        self.assertEqual(row["n_links"], 38)
        self.assertTrue(row["action_sums_to_demands"])
        self.assertGreater(row["load_checksum_proposed"], 0.0)
        self.assertIn(row["applied_source"], {"proposed", "shortest", "shortest_last_resort", "last_accepted"})
        self.assertTrue(np.isfinite(row["mlu_true_applied"]))
        self.assertEqual(meta["topology"], "B4")
        self.policy.release("B4")

    def test_geant_unseen_end_to_end(self) -> None:
        topo = load_topology("GEANT")
        demands, _meta = load_split_demands("test", "GEANT")
        self.framework.reset_session("GEANT")
        row = self.framework.decide(topo, demands[0])
        self.assertEqual(row["n_links"], 72)
        self.assertTrue(row["action_sums_to_demands"])
        self.assertTrue(np.isfinite(row["prediction_mae_applied"]))
        self.assertGreaterEqual(row["mlu_true_shortest"], 0.0)
        od = od_demands(demands[0])
        self.assertEqual(od.shape[0], topo.n_paths.shape[0])
        self.policy.release("GEANT")


if __name__ == "__main__":
    unittest.main()
