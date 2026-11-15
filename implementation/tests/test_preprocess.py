"""Checks for Step 3. The small graphs are fixtures, not research datasets."""

from __future__ import annotations

import json
import pickle
import unittest
from pathlib import Path

import numpy as np

from preprocessing.bipartite import incidence, tensors_for_matrix
from preprocessing.normalize import apply_demand, parameters
from preprocessing.paths import as_dict, generate, yen_unit
from preprocessing.topology import connectivity_report, from_node_link, from_zoo_gml
from preprocessing.traffic import validate_matrix

from repo_paths import REPO as ROOT
P1_B4_PATHS = (
    ROOT
    / "baseline_p1_reproduction"
    / "data"
    / "candidate_paths"
    / "B4-4-paths_min-hop.pkl"
)
B4_JSON = ROOT / "data" / "raw" / "teal" / "topologies" / "B4.json"


def _link(data, u, v, cap=1.0, **extra):
    item = {"source": u, "target": v, "capacity": cap}
    item.update(extra)
    data["links"].append(item)


def _graph(nodes, links):
    return {"nodes": [{"id": i} for i in nodes], "links": links}


class TopologyTests(unittest.TestCase):
    def test_duplicate_and_self_loop_are_recorded(self):
        data = _graph([0, 1], [])
        _link(data, 0, 1, 5)
        _link(data, 0, 1, 9)
        _link(data, 1, 1, 3)
        topo = from_node_link("toy", data, "fixture", "fixture")
        self.assertEqual(topo.edges, [(0, 1)])
        self.assertEqual(topo.capacities.tolist(), [5.0])
        reasons = {item["reason"] for item in topo.excluded}
        self.assertEqual(reasons, {"duplicate link", "self-loop"})
        self.assertTrue(topo.warnings)

    def test_disconnected_graph_keeps_every_node(self):
        data = _graph([0, 1, 2], [])
        _link(data, 0, 1, 4)
        topo = from_node_link("toy", data, "fixture", "fixture")
        report = connectivity_report(topo)
        self.assertEqual(topo.node_ids, [0, 1, 2])
        self.assertFalse(report["undirected_connected"])
        self.assertEqual(sorted(report["undirected_component_sizes"]), [1, 2])

    def test_missing_capacity_is_not_invented(self):
        data = _graph([0, 1], [{"source": 0, "target": 1}])
        topo = from_node_link("toy", data, "fixture", "fixture")
        self.assertEqual(topo.edges, [])
        self.assertEqual(topo.excluded[0]["reason"], "missing capacity")

    def test_zoo_placeholder_is_labelled(self):
        text = """
        graph [
          node [ id 0 ]
          node [ id 1 ]
          edge [ source 0 target 1 ]
        ]
        """
        topo = from_zoo_gml("Cogentco", text, "fixture", 10000.0)
        self.assertEqual(sorted(topo.edges), [(0, 1), (1, 0)])
        self.assertTrue(np.all(topo.capacities == 10000.0))
        self.assertIn("not a measured capacity", " ".join(topo.warnings))


class TrafficAndScaleTests(unittest.TestCase):
    def test_negative_demand_is_invalid_and_not_clipped(self):
        matrix = np.array([[0.0, -3.0], [4.0, 0.0]])
        check = validate_matrix(matrix, 2)
        self.assertFalse(check["valid"])
        self.assertEqual(check["negative_entries"], 1)
        scaled = apply_demand(matrix, 10.0)
        self.assertTrue(np.isnan(scaled[0, 1]))
        self.assertEqual(scaled[1, 0], 0.4)
        self.assertEqual(matrix[0, 1], -3.0)

    def test_denominator_is_max_capacity_and_ignores_demands(self):
        caps = np.array([10.0, 40.0, 20.0])
        params = parameters(caps)
        self.assertEqual(params["denominator"], 40.0)
        self.assertFalse(params["uses_traffic_matrices"])
        again = parameters(caps)
        self.assertEqual(again["denominator"], params["denominator"])


class PathTests(unittest.TestCase):
    def test_fewer_than_four_paths_are_not_repeated(self):
        data = _graph([0, 1, 2], [])
        _link(data, 0, 1, 1)
        _link(data, 1, 2, 1)
        topo = from_node_link("line", data, "fixture", "fixture")
        arrays = generate(topo, jobs=1)
        paths = as_dict(arrays)
        self.assertEqual(paths[(0, 2)], [[0, 1, 2]])
        self.assertEqual(int(arrays["n_paths"].max()), 1)
        self.assertFalse(arrays["meta"]["padded"])

    def test_unit_weight_yen_matches_networkx_hop_counts_on_b4(self):
        if not B4_JSON.exists():
            self.skipTest("B4 topology is not in the raw folder")
        topo = from_node_link("B4", json.loads(B4_JSON.read_text(encoding="utf-8")), "raw", "teal")
        official = as_dict(generate(topo, jobs=1))
        succ = [[] for _ in range(topo.num_nodes)]
        for u, v in topo.edges:
            succ[u].append(v)
        for (src, dst), paths in official.items():
            yen = yen_unit(succ, src, dst, 4)
            self.assertEqual([len(p) - 1 for p in yen], [len(p) - 1 for p in paths])

    def test_b4_networkx_paths_match_frozen_p1_pickle(self):
        if not B4_JSON.exists() or not P1_B4_PATHS.exists():
            self.skipTest("B4 topology or frozen P1 path pickle is missing")
        topo = from_node_link("B4", json.loads(B4_JSON.read_text(encoding="utf-8")), "raw", "teal")
        got = as_dict(generate(topo, jobs=1))
        with P1_B4_PATHS.open("rb") as handle:
            expected = pickle.load(handle)
        self.assertEqual(got, expected)


class BipartiteTests(unittest.TestCase):
    def test_path_connects_only_to_its_links(self):
        data = _graph([0, 1, 2], [])
        _link(data, 0, 2, 7)
        _link(data, 0, 1, 10)
        _link(data, 1, 2, 5)
        topo = from_node_link("diamond", data, "fixture", "fixture")
        arrays = generate(topo, jobs=1)
        links = incidence(topo, arrays)
        demand = np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        norm = apply_demand(demand, parameters(topo.capacities)["denominator"])
        od_original = _off(demand)
        graph = tensors_for_matrix(topo, arrays, links, od_original, _off(norm))
        self.assertEqual(int(graph["num_link_nodes"]), 3)
        # 0->1, both routes of 0->2, and 1->2. No reverse paths exist.
        self.assertEqual(int(graph["num_path_nodes"]), 4)
        edge = graph["edge_index"].numpy()
        n_links = 3
        forward = {(int(p), int(e)) for p, e in edge.T if p >= n_links}
        # Link order is the order edges were added: 0->2, 0->1, 1->2.
        # Path order: 0->1, then 0->2 direct, then 0->2 via 1, then 1->2.
        self.assertEqual(
            forward,
            {(3, 1), (4, 0), (5, 1), (5, 2), (6, 2)},
        )
        self.assertEqual(od_original.tolist(), [0.0, 4.0, 0.0, 0.0, 0.0, 0.0])


def _off(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.asarray(matrix[mask], dtype=np.float64)


if __name__ == "__main__":
    unittest.main()
