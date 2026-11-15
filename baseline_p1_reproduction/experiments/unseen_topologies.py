"""Held-out topology tests (Phase 3 / our contribution, not part of P1).

P1 trains one model per topology and never tests on an unseen graph. This
script is the hook for that experiment after the GRL-TE baseline is verified.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "Unseen-topology evaluation is part of our proposed method, not P1. "
        "Verify the B4 GRL-TE baseline first (experiments/b4.py)."
    )


if __name__ == "__main__":
    main()
