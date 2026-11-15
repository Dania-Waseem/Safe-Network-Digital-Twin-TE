"""Scale, fallback classification, and actor-diagnostic checks."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from framework.fallback_modes import classify_candidates
from framework.pipeline import Framework
from framework.policy_adapter import FrozenP1Policy
from safety.evaluate import load_config, load_trained_ensemble
from twin.data import load_split_demands, load_topology
from twin.physics import od_demands


class FallbackClassTest(unittest.TestCase):
    def test_feasible_candidate_is_preferred(self) -> None:
        capacity = np.array([10.0, 10.0])
        choice = classify_candidates(
            [
                ("p1_actor", np.array([12.0, 1.0])),
                ("shortest", np.array([4.0, 4.0])),
                ("equal", np.array([9.0, 11.0])),
            ],
            capacity,
        )
        self.assertEqual(choice["outcome"], "feasible_fallback")
        self.assertFalse(choice["unavoidable_overload"])
        self.assertEqual(choice["selected"], "shortest")
        self.assertEqual(choice["selected_violations"], 0)

    def test_all_infeasible_is_least_violating_and_unavoidable(self) -> None:
        capacity = np.ones(3)
        choice = classify_candidates(
            [
                ("p1_actor", np.array([2.0, 2.0, 2.0])),
                ("shortest", np.array([0.5, 2.0, 0.5])),
                ("equal", np.array([2.0, 2.0, 0.2])),
            ],
            capacity,
        )
        self.assertEqual(choice["outcome"], "least_violating_fallback")
        self.assertTrue(choice["unavoidable_overload"])
        self.assertEqual(choice["selected"], "shortest")
        self.assertEqual(choice["selected_violations"], 1)


class ScaledB4Test(unittest.TestCase):
    def test_quarter_scale_is_scored_and_native_diagnostics_differ_from_geant(self) -> None:
        device = torch.device("cpu")
        policy = FrozenP1Policy(device=device)
        framework = Framework(policy, load_trained_ensemble(device), load_config(), device)
        b4 = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        b4_info = policy.inspect(b4, od_demands(demands[0]))
        geant = load_topology("GEANT")
        geant_demands, _ = load_split_demands("test", "GEANT")
        geant_info = policy.inspect(geant, od_demands(geant_demands[0]))
        self.assertGreater(geant_info["h0_max"], b4_info["h0_max"] * 10.0)
        self.assertEqual(b4_info["demands_with_fewer_than_4_paths"], 0)
        row = framework.decide(b4, demands[0] * 0.25)
        self.assertTrue(row["mlu_true_shortest"] < 1.0)
        self.assertTrue(np.isfinite(row["mlu_true_proposed"]))
        self.assertIn(row["proposed_accepted"], (True, False))
        policy.release()


if __name__ == "__main__":
    unittest.main()
