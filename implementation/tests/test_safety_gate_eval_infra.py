"""Tests for Safety Gate evaluation infrastructure (no full A–F run)."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from safety.improvements.config import load_improvements_config
from safety.improvements.decision import VARIANT_IDS
from safety.improvements.graph_cache import TopologyGraphCache, topology_fingerprint
from safety.improvements.logging_util import validate_decision_record
from safety.improvements.policy_backend import policy_backend_from_config
from safety.improvements.run_eval import run_evaluation
from twin.data import IMPL, ROOT, load_split_demands, load_topology


ORIGINAL_TWIN = ROOT / "models" / "twin" / "ensemble.pt"
ORIGINAL_GATE = IMPL / "safety" / "gate.py"
EXPECTED_TWIN_SHA = "7b74fee96cdf3aa71c78aaa185adbc24c99f262f342689a4ec78d97b7db73685"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GraphCacheTest(unittest.TestCase):
    def test_topology_isolation_and_deterministic_reuse(self) -> None:
        cache = TopologyGraphCache(torch.device("cpu"))
        b4 = load_topology("B4")
        geant = load_topology("GEANT")
        g1 = cache.get(b4)
        g1b = cache.get(b4)
        g2 = cache.get(geant)
        self.assertIs(g1, g1b)
        self.assertIsNot(g1, g2)
        self.assertEqual(g1.n_links, b4.n_links)
        self.assertEqual(g2.n_links, geant.n_links)
        self.assertNotEqual(topology_fingerprint(b4), topology_fingerprint(geant))
        # Clearing one topology must not drop the other.
        cache.clear("B4")
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.metadata()[0]["topology_name"], "GEANT")

    def test_path_identity_matches_real_path_count(self) -> None:
        cache = TopologyGraphCache(torch.device("cpu"))
        topo = load_topology("B4")
        graph = cache.get(topo)
        self.assertEqual(int(graph.counts.sum()), int(topo.n_paths.sum()))
        self.assertEqual(graph.n_demands, int(topo.n_paths.shape[0]))


class PolicyBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_improvements_config()
        cls.device = torch.device("cpu")
        cls.backend = policy_backend_from_config(cls.cfg, device=cls.device)
        cls.topo = load_topology("B4")
        demands, _ = load_split_demands("val", "B4")
        cls.matrix = demands[0]

    def test_loads_actual_rl_checkpoint(self) -> None:
        payload = self.backend.load()
        self.assertIn("model", payload)
        self.assertTrue(Path(self.backend.checkpoint_path).exists())
        self.assertEqual(self.backend.checkpoint_id, "improved_stable")

    def test_action_reproducibility(self) -> None:
        a = self.backend.propose(self.topo, self.matrix)
        b = self.backend.propose(self.topo, self.matrix)
        np.testing.assert_allclose(a.action, b.action, rtol=0, atol=0)
        self.assertEqual(a.n_real_paths, int(self.topo.n_paths.sum()))
        self.assertEqual(a.topology, "B4")
        self.assertTrue(a.deterministic)

    def test_path_identity_preserved(self) -> None:
        proposal = self.backend.propose(self.topo, self.matrix * 0.5)
        self.assertEqual(proposal.action.shape[0], proposal.n_real_paths)
        self.assertEqual(proposal.n_demands, int(self.topo.n_paths.shape[0]))
        self.assertGreaterEqual(float(proposal.action.min()), -1e-6)

    def test_topology_switching_uses_isolated_graphs(self) -> None:
        geant = load_topology("GEANT")
        demands, _ = load_split_demands("test", "GEANT")
        p_b4 = self.backend.propose(self.topo, self.matrix)
        p_g = self.backend.propose(geant, demands[0] * 0.25)
        self.assertNotEqual(p_b4.graph_fingerprint, p_g.graph_fingerprint)
        self.assertEqual(p_b4.n_real_paths, int(self.topo.n_paths.sum()))
        self.assertEqual(p_g.n_real_paths, int(geant.n_paths.sum()))


class SmokeRunnerTest(unittest.TestCase):
    def test_paired_smoke_run_and_logging(self) -> None:
        out = ROOT / "results" / "safety_gate" / "improvements" / "smoke_test_tmp"
        if out.exists():
            for child in out.glob("*"):
                child.unlink()
        result = run_evaluation(smoke=True, allow_full=False, output_dir=out)
        rows = result["rows"]
        self.assertGreaterEqual(len(rows), len(VARIANT_IDS))
        self.assertEqual(len(rows) % len(VARIANT_IDS), 0)

        # Paired fairness within each case block.
        by_case: dict[tuple, list] = {}
        for row in rows:
            key = (row["seed"], row["topology"], row["matrix_id"], row["scale"])
            by_case.setdefault(key, []).append(row)
        for key, block in by_case.items():
            fps = {r["policy_action_fingerprint"] for r in block}
            self.assertEqual(len(fps), 1, msg=str(key))
            variants = [r["gate_variant"] for r in block]
            self.assertEqual(sorted(variants), sorted(VARIANT_IDS))
            for row in block:
                missing = validate_decision_record(row)
                self.assertEqual(missing, [], msg=f"{row['gate_variant']}: {missing}")
                self.assertIn("actual_mlu", row)
                self.assertIn("capacity_proximity", row)
                self.assertIn("proposed_action_source", row)
                self.assertIn("applied_action_source", row)
                if row["gate_variant"] in (
                    "B_original_twin_mean",
                    "C_original_twin_uncertainty",
                    "D_improved_twin_mean",
                    "E_improved_twin_calibrated_uncertainty",
                ):
                    self.assertIn("raw_uncertainty_utilisation", row)
                    self.assertIn("calibrated_uncertainty_utilisation", row)

        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("groups", summary)
        self.assertIn("topology", summary["groups"])
        self.assertIn("scale", summary["groups"])
        self.assertIn("capacity_proximity", summary["groups"])
        self.assertTrue(summary["smoke"])

    def test_full_blocked_without_flag(self) -> None:
        with self.assertRaises(RuntimeError):
            run_evaluation(smoke=False, allow_full=False)


class PreservationTest(unittest.TestCase):
    def test_original_artifacts_untouched(self) -> None:
        self.assertEqual(_sha(ORIGINAL_TWIN), EXPECTED_TWIN_SHA)
        self.assertTrue(ORIGINAL_GATE.exists())
        # GeneralPolicy module still importable / unmodified entrypoint.
        from framework.general_policy import GeneralPolicy

        self.assertTrue(callable(GeneralPolicy.load))


if __name__ == "__main__":
    unittest.main()
