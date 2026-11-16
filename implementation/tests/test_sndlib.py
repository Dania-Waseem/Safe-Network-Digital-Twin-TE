"""Checks for the SNDlib parser and sampler. The network text is a fixture, not a research dataset."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocessing.sndlib import parse_network, read_demand_archive
from preprocessing.sndlib_prepare import stratified_sample

NETWORK = """?SNDlib native format; type: network; version: 1.0
# network tiny

META (
  granularity  = 5min
  time  = 20040301-0005
  unit  = MBITPERSEC
)

NODES (
  A ( 1.0 2.0 )
  B ( 3.0 4.0 )
  C ( 5.0 6.0 )
)

LINKS (
  A_B ( A B ) 9920.00 0.00 0.00 0.00 ( 40000.00 133.00 )
  B_C ( B C ) 0.00 0.00 0.00 0.00 ( 40.00 3290.00 )
)

DEMANDS (
  A_C ( A C ) 1 2.500000 UNLIMITED
  C_A ( C A ) 1 1.000000 UNLIMITED
)
"""


class SndlibParserTests(unittest.TestCase):
    def test_network_sections(self) -> None:
        net = parse_network(NETWORK)
        self.assertEqual(net.name, "tiny")
        self.assertEqual(net.nodes, ["A", "B", "C"])
        self.assertEqual(net.meta["unit"], "MBITPERSEC")
        self.assertEqual([(l.source, l.target, l.pre_installed_capacity) for l in net.links],
                         [("A", "B", 9920.0), ("B", "C", 0.0)])
        self.assertEqual(net.links[1].modules, [(40.0, 3290.0)])
        self.assertEqual(net.demands, [("A", "C", 2.5), ("C", "A", 1.0)])
        self.assertEqual(net.unparsed_lines, [])

    def test_archive_matrix_order_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.tgz"
            with tarfile.open(path, "w:gz") as tf:
                for t in ("20040301-0005", "20040301-0000"):
                    data = NETWORK.replace("20040301-0005", t).encode("utf-8")
                    info = tarfile.TarInfo(f"dir/demandMatrix-tiny-{t}.txt")
                    info.size = len(data)
                    tf.addfile(info, io.BytesIO(data))
            series = read_demand_archive(path, ["A", "B", "C"])
        self.assertEqual(series.times, ["20040301-0000", "20040301-0005"])
        self.assertEqual(series.matrices.shape, (2, 3, 3))
        self.assertEqual(series.matrices[0, 0, 2], 2.5)
        self.assertEqual(series.matrices[0, 2, 0], 1.0)
        self.assertEqual(float(series.matrices[0].sum()), 3.5)
        self.assertTrue(all(series.node_sets_match))
        self.assertEqual(sum(series.unknown_endpoints), 0)


class StratifiedSampleTests(unittest.TestCase):
    def test_one_draw_per_stratum_and_seeded(self) -> None:
        idx = np.arange(1000)
        a = stratified_sample(idx, 10, 7)
        self.assertEqual(a.size, 10)
        for k, v in enumerate(a):
            self.assertTrue(100 * k <= v < 100 * (k + 1))
        np.testing.assert_array_equal(a, stratified_sample(idx, 10, 7))

    def test_small_pool_returns_everything(self) -> None:
        np.testing.assert_array_equal(stratified_sample(np.array([5, 2, 9]), 10, 1), [2, 5, 9])


if __name__ == "__main__":
    unittest.main()
