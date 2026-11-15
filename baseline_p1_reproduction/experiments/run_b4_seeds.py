"""Run B4 GRL-TE for all P1 seeds 100..1000 and aggregate results."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SEEDS, TABLE_DIR, LOG_DIR, P1_PUBLISHED  # noqa: E402


def main() -> None:
    rows = []
    for seed in SEEDS:
        print(f"\n======== SEED {seed} ========")
        cmd = [sys.executable, str(ROOT / "experiments" / "b4.py"), "--seed", str(seed)]
        subprocess.check_call(cmd, cwd=str(ROOT))
        summary = json.loads((LOG_DIR / f"b4_seed{seed}.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "grl_te": summary["ours"]["demand_satisfaction"],
                "grl_te_no_admm": summary["ours_no_admm"]["demand_satisfaction"],
                "lp": summary["lp"]["demand_satisfaction"],
                "ecmp": summary["ecmp"]["demand_satisfaction"],
                "runtime_s": summary["ours"]["runtime_s"],
                "train_time_s": summary["train_time_s"],
            }
        )

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "b4_all_seeds.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(key):
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(vals.mean()), float(vals.std())

    agg = {
        "n_seeds": len(rows),
        "grl_te_mean_std": stats("grl_te"),
        "grl_te_no_admm_mean_std": stats("grl_te_no_admm"),
        "lp_mean_std": stats("lp"),
        "ecmp_mean_std": stats("ecmp"),
        "p1_published_b4": P1_PUBLISHED["B4"],
        "per_seed": rows,
    }
    (LOG_DIR / "b4_all_seeds_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print("\n=== Aggregate over", len(rows), "seeds ===")
    for key in ("grl_te", "grl_te_no_admm", "lp", "ecmp"):
        m, s = stats(key)
        print(f"{key:18s} {100*m:.2f}% ± {100*s:.2f}")
    print("P1 GRL-TE B4:", P1_PUBLISHED["B4"]["GRL-TE"], "%")
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
