"""Twin physics and a one-step ensemble check on B4. No ASN data."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from twin.data import load_split_demands, load_topology
from twin.model import EnsembleTwin
from twin.physics import (
    demand_per_path,
    equal_split,
    link_load,
    od_demands,
    path_flow_checksum,
    shortest_split,
)
from twin.train import _predict_numpy, _sample


class TwinPhysicsTest(unittest.TestCase):
    def test_b4_equal_split_matches_hop_checksum(self) -> None:
        topo = load_topology("B4")
        demands, _meta = load_split_demands("train", "B4")
        action = equal_split(topo.n_paths)
        self.assertAlmostEqual(float(action.sum()), float(np.sum(topo.n_paths > 0)), places=5)
        od = od_demands(demands[0])
        demand_path = demand_per_path(od, topo.demand_row)
        load = link_load(demand_path, action, topo.incidence_path, topo.incidence_link, topo.n_links)
        self.assertAlmostEqual(float(load.sum()), path_flow_checksum(demand_path, action, topo.hops), places=4)
        self.assertEqual(load.shape, (topo.n_links,))
        self.assertTrue(np.all(load >= -1e-8))

    def test_shortest_split_is_a_distribution(self) -> None:
        topo = load_topology("B4")
        action = shortest_split(topo.n_paths, topo.hops)
        self.assertAlmostEqual(float(action.sum()), float(topo.n_paths.shape[0]), places=5)
        self.assertTrue(np.all((action == 0.0) | (np.isclose(action, 1.0))))

    def test_untrained_ensemble_matches_simulator_scale(self) -> None:
        topo = load_topology("B4")
        demands, _meta = load_split_demands("val", "B4")
        action = equal_split(topo.n_paths)
        sample = _sample(topo, demands[0], action)
        model = EnsembleTwin(n_members=3, hidden=16, dropout=0.0)
        # Zero the hidden weights so every member stays at the physics multiplier.
        with torch.no_grad():
            for member in model.members:
                member.backbone[0].weight.zero_()
                member.backbone[0].bias.zero_()
                member.backbone[3].weight.zero_()
                member.backbone[3].bias.zero_()
        mean_load, std_load = _predict_numpy(model, sample, topo, torch.device("cpu"))
        rel = np.max(np.abs(mean_load - sample["load"])) / max(float(np.max(sample["load"])), 1.0)
        self.assertLess(rel, 1e-4)
        self.assertTrue(np.all(std_load >= 0.0))
        self.assertEqual(mean_load.shape, (topo.n_links,))


if __name__ == "__main__":
    unittest.main()
