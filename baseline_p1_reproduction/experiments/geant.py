"""GÉANT experiment placeholder.

P1 uses FIGRET traffic for GÉANT (22 nodes, 72 edges, 100 ADMM iterations).
Add the FIGRET dump under data/topologies and data/traffic_matrices/GEANT,
then this script can reuse the B4 training/eval pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    raise SystemExit(
        "GÉANT data is not in this folder yet. Download the FIGRET GÉANT "
        "topology and traffic matrices (P1 Section 5.1), then run the same "
        "pipeline as experiments/b4.py with ADMM_ITERS['GEANT']=100."
    )


if __name__ == "__main__":
    main()
