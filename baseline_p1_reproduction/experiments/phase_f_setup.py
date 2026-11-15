"""Phase F: copy public topologies/TMs, fetch Zoo graphs, generate Cogentco gravity."""

from __future__ import annotations

import json
import pickle
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PATH_DIR, TM_DIR, TOPOLOGY_DIR  # noqa: E402
from preprocessing.gravity import gravity_matrices  # noqa: E402
from preprocessing.load_graphml import graphml_to_topology, save_teal_json  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402

REF = ROOT.parent / "reference" / "teal"
LOG = ROOT / "results" / "logs" / "phase_f"
GRAPH_DIR = ROOT / "results" / "graphs" / "phase_f"

ZOO_URLS = {
    "Cogentco": [
        "https://raw.githubusercontent.com/afourmy/3D-internet-zoo/master/dataset/Cogentco.gml",
        "https://raw.githubusercontent.com/sk2/topologyzoo/master/dataset/Cogentco.graphml",
    ],
    "GEANT": [
        "https://raw.githubusercontent.com/afourmy/3D-internet-zoo/master/dataset/Geant2012.gml",
        "https://raw.githubusercontent.com/prateekshyap/CPP-in-SDN/main/Geant2012.graphml",
    ],
}

P1_TABLE5 = {
    "B4": (12, 38),
    "GEANT": (22, 72),
    "UsCarrier": (158, 378),
    "Cogentco": (197, 486),
    "Kdl": (754, 1790),
    "ASN": (1739, 8558),
}


def _copy_topology(src_name: str, dest_name: str) -> dict:
    src = REF / "topologies" / f"{src_name}.json"
    dest = TOPOLOGY_DIR / f"{dest_name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    topo = load_topology(dest_name)
    return {
        "name": dest_name,
        "src": str(src),
        "nodes": topo.num_nodes,
        "edges": topo.num_edges,
        "p1": P1_TABLE5.get(dest_name),
        "match_p1_size": (topo.num_nodes, topo.num_edges) == P1_TABLE5.get(dest_name),
    }


def _copy_tms(src_glob_dir: Path, dest_name: str, pattern: str) -> dict:
    dest = TM_DIR / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    files = sorted(src_glob_dir.glob(pattern))
    totals = []
    for i, f in enumerate(files):
        with open(f, "rb") as handle:
            tm = pickle.load(handle)
        tm = np.asarray(tm, dtype=np.float64)
        np.fill_diagonal(tm, 0.0)
        tm = np.maximum(tm, 0.0)
        totals.append(float(tm.sum()))
        out = dest / f"{dest_name}_{i:03d}_traffic-matrix.pkl"
        with open(out, "wb") as handle:
            pickle.dump(tm, handle)
    return {
        "name": dest_name,
        "n": len(files),
        "shape": list(np.asarray(pickle.load(open(files[0], "rb"))).shape) if files else None,
        "mean_total": float(np.mean(totals)) if totals else None,
        "source": str(src_glob_dir),
        "note": "TEAL toy TMs are unit-demand float16 originally; converted to float64",
    }


def _download(urls: list[str], dest: Path) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print("GET", url)
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 200:
                return dest
        except Exception as exc:
            print("  fail", exc)
    return None


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"topologies": [], "traffic": [], "downloads": {}, "limitations": []}

    # TEAL topologies
    for src, dest in [("B4", "B4"), ("UsCarrier", "UsCarrier"), ("Kdl", "Kdl"), ("ASN2k", "ASN")]:
        rec = _copy_topology(src, dest)
        report["topologies"].append(rec)
        print("copied", rec)

    # TMs
    report["traffic"].append(
        {
            "name": "B4",
            "source": "TEAL real Yates-mapped (already in data/traffic_matrices/B4)",
            "type": "public measured-mapped, NOT Microsoft WAN",
        }
    )
    report["traffic"].append(
        _copy_tms(REF / "traffic-matrices" / "toy", "UsCarrier", "UsCarrier.json_toy_*_traffic-matrix.pkl")
        | {"p1_claim": "P1 says Teal TMs; public Teal only ships toy unit-demand"}
    )
    report["traffic"].append(
        _copy_tms(REF / "traffic-matrices" / "toy", "Kdl", "Kdl.json_toy_*_traffic-matrix.pkl")
        | {"p1_claim": "P1 says Teal TMs; public Teal only ships toy unit-demand"}
    )
    report["traffic"].append(
        _copy_tms(REF / "traffic-matrices" / "toy", "ASN", "ASN2k.json_toy_*_traffic-matrix.pkl")
        | {"p1_claim": "P1 says Teal TMs; public Teal only ships toy unit-demand"}
    )

    # Zoo downloads
    raw_dir = ROOT / "data" / "raw_zoo"
    for name, urls in ZOO_URLS.items():
        ext = ".gml" if "gml" in urls[0] else ".graphml"
        dest = raw_dir / f"{name}{ext}"
        got = _download(urls, dest)
        report["downloads"][name] = {"path": str(dest) if got else None, "ok": got is not None}
        if not got:
            report["limitations"].append(f"Could not download {name} Topology Zoo file")
            continue
        topo = graphml_to_topology(got, name)
        save_teal_json(topo)
        rec = {
            "name": name,
            "src": str(got),
            "nodes": topo.num_nodes,
            "edges": topo.num_edges,
            "p1": P1_TABLE5.get(name),
            "match_p1_size": (topo.num_nodes, topo.num_edges) == P1_TABLE5.get(name),
        }
        report["topologies"].append(rec)
        print("zoo", rec)
        if name == "Cogentco":
            tms = gravity_matrices(topo.graph, n_tms=36, seed=2025)
            dest_tm = TM_DIR / "Cogentco"
            dest_tm.mkdir(parents=True, exist_ok=True)
            totals = []
            for i, tm in enumerate(tms):
                totals.append(float(tm.sum()))
                with open(dest_tm / f"Cogentco_{i:03d}_traffic-matrix.pkl", "wb") as handle:
                    pickle.dump(tm, handle)
            report["traffic"].append(
                {
                    "name": "Cogentco",
                    "n": 36,
                    "mean_total": float(np.mean(totals)),
                    "type": "synthetic gravity (degree attractiveness) — reconstructed, not P1 dump",
                }
            )
        if name == "GEANT" and not rec["match_p1_size"]:
            report["limitations"].append(
                f"Downloaded GEANT graph is {rec['nodes']}/{rec['edges']}; P1 Table 5 is 22/72. "
                "FIGRET/Uhlig 22-node TM pack was not present locally and Zoo Geant2012 differs."
            )

    report["limitations"].extend(
        [
            "P1 unpublished: TM split, Yen vs edge-disjoint, ADMM closed forms, Cogentco gravity constants, FIGRET GÉANT files.",
            "Public TEAL UsCarrier/Kdl/ASN TMs are toy (all entries 1.0) — not Microsoft WAN traces used in Teal's paper.",
            "Do not treat toy-TM results as Table 6 reproductions.",
        ]
    )
    (LOG / "setup.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
