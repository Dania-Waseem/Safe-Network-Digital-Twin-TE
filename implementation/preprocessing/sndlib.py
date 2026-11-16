"""Parse SNDlib native-format network files and dynamic demand matrices.

SNDlib native format (``?SNDlib native format; type: network; version: 1.0``) has
``META``, ``NODES``, ``LINKS``, ``DEMANDS`` (and optional further) sections, each written as
``NAME ( ... )`` with one record per line. A dynamic demand matrix is a network file
without links; its ``META`` section carries ``granularity``, ``time`` and ``unit``.

Only plain text is read; nothing in the archives is executed.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_SECTION = re.compile(r"^([A-Z_]+)\s*\($")
_LINK = re.compile(
    r"^(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*\(\s*(.*?)\s*\)\s*$"
)
_DEMAND = re.compile(r"^(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
_NODE = re.compile(r"^(\S+)\s*(?:\(\s*(\S+)\s+(\S+)\s*\))?\s*$")


@dataclass
class SndlibLink:
    link_id: str
    source: str
    target: str
    pre_installed_capacity: float
    pre_installed_capacity_cost: float
    routing_cost: float
    setup_cost: float
    modules: list[tuple[float, float]]


@dataclass
class SndlibNetwork:
    name: str
    meta: dict[str, str]
    nodes: list[str]
    coords: dict[str, tuple[float, float]]
    links: list[SndlibLink] = field(default_factory=list)
    demands: list[tuple[str, str, float]] = field(default_factory=list)
    unparsed_lines: list[str] = field(default_factory=list)


def _sections(text: str) -> tuple[str, dict[str, list[str]]]:
    name = ""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("# network "):
            name = line[len("# network "):].strip()
        if not line or line.startswith("#") or line.startswith("?"):
            continue
        if current is None:
            m = _SECTION.match(line)
            if m:
                current = m.group(1)
                sections.setdefault(current, [])
            continue
        if line == ")":
            current = None
            continue
        sections[current].append(line)
    return name, sections


def parse_network(text: str) -> SndlibNetwork:
    name, sec = _sections(text)
    meta: dict[str, str] = {}
    for line in sec.get("META", []):
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    nodes: list[str] = []
    coords: dict[str, tuple[float, float]] = {}
    unparsed: list[str] = []
    for line in sec.get("NODES", []):
        m = _NODE.match(line)
        if not m:
            unparsed.append(f"NODES: {line}")
            continue
        nodes.append(m.group(1))
        if m.group(2) is not None:
            coords[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    links: list[SndlibLink] = []
    for line in sec.get("LINKS", []):
        m = _LINK.match(line)
        if not m:
            unparsed.append(f"LINKS: {line}")
            continue
        mods = [float(x) for x in m.group(8).split()] if m.group(8) else []
        links.append(SndlibLink(
            link_id=m.group(1), source=m.group(2), target=m.group(3),
            pre_installed_capacity=float(m.group(4)), pre_installed_capacity_cost=float(m.group(5)),
            routing_cost=float(m.group(6)), setup_cost=float(m.group(7)),
            modules=[(mods[i], mods[i + 1]) for i in range(0, len(mods) - 1, 2)],
        ))
    demands: list[tuple[str, str, float]] = []
    for line in sec.get("DEMANDS", []):
        m = _DEMAND.match(line)
        if not m:
            unparsed.append(f"DEMANDS: {line}")
            continue
        try:
            value = float(m.group(5))
        except ValueError:
            value = float("nan")
        demands.append((m.group(2), m.group(3), value))
    return SndlibNetwork(name=name, meta=meta, nodes=nodes, coords=coords, links=links, demands=demands,
                         unparsed_lines=unparsed)


@dataclass
class DemandSeries:
    """All matrices of one dynamic archive, in time order."""

    archive: str
    files: list[str]
    times: list[str]                 # META time field, e.g. 20040301-0000
    granularity: list[str]
    units: list[str]
    node_sets_match: list[bool]      # node section identical to the reference node list
    unknown_endpoints: list[int]     # demands whose endpoint is not a reference node
    self_demands: list[int]          # demands with source == target
    duplicate_pairs: list[int]       # repeated (s, t) records within one matrix
    unparsed: list[int]              # lines that did not parse
    matrices: np.ndarray             # [T, n, n] in the file unit; absent pair = 0


def read_demand_archive(archive: Path, node_order: list[str]) -> DemandSeries:
    """Read every ``demandMatrix-*.txt`` member of a native-format ``.tgz`` without extracting it."""
    index = {name: i for i, name in enumerate(node_order)}
    n = len(node_order)
    recs = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            net = parse_network(fh.read().decode("utf-8", errors="replace"))
            mat = np.zeros((n, n), dtype=np.float64)
            unknown = selfd = dup = 0
            seen: set[tuple[int, int]] = set()
            for s, t, v in net.demands:
                if s not in index or t not in index:
                    unknown += 1
                    continue
                i, j = index[s], index[t]
                if i == j:
                    selfd += 1
                    continue
                if (i, j) in seen:
                    dup += 1
                seen.add((i, j))
                mat[i, j] += v
            recs.append((net.meta.get("time", ""), Path(member.name).name, net.meta.get("granularity", ""),
                         net.meta.get("unit", ""), net.nodes == node_order, unknown, selfd, dup,
                         len(net.unparsed_lines), mat))
    recs.sort(key=lambda r: (r[0], r[1]))
    return DemandSeries(
        archive=archive.name,
        files=[r[1] for r in recs], times=[r[0] for r in recs], granularity=[r[2] for r in recs],
        units=[r[3] for r in recs], node_sets_match=[r[4] for r in recs], unknown_endpoints=[r[5] for r in recs],
        self_demands=[r[6] for r in recs], duplicate_pairs=[r[7] for r in recs], unparsed=[r[8] for r in recs],
        matrices=np.stack([r[9] for r in recs]) if recs else np.zeros((0, n, n)),
    )
