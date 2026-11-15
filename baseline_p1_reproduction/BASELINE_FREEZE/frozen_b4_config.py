"""Frozen B4 research-baseline configuration (explicit; no ambiguous defaults).

This file is the single source of truth for the BASELINE FREEZE protocol.
It does NOT modify baseline_p1_reproduction/config.py.

Frozen ADMM for the *reference* research baseline: Teal-order (s→λ→z→x).
P1-order (F→z→ξ→λ) is evaluated in a separate result channel only.
"""

from __future__ import annotations

from pathlib import Path

FREEZE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FREEZE_DIR.parent
ARCHIVE_MODEL_DIR = REPO_ROOT.parent / "models" / "baseline_p1" / "b4"
ARCHIVE_LOG_DIR = REPO_ROOT / "results" / "logs"
FREEZE_MODEL_DIR = REPO_ROOT.parent / "models" / "baseline_p1" / "b4_freeze"
FREEZE_LOG_DIR = FREEZE_DIR / "logs"
FREEZE_RESULT_DIR = FREEZE_DIR / "results"

# --- Identity ---
PROTOCOL_ID = "B4_GRLTE_FREEZE_v1"
PROTOCOL_DATE = "2026-11-07"
PAPER = (
    "Lu et al., Graph-based reinforcement learning for software-defined "
    "networking traffic engineering, J. King Saud Univ. Comput. Inf. Sci. 37:119 (2025)"
)

# --- Data ---
TOPOLOGY = "B4"
TRAFFIC_SOURCE = "TEAL public real TMs (data/traffic_matrices/B4/*_real_*)"
PATH_GENERATOR = "yen_k_shortest"
PATH_K = 4
PATH_METRIC = "min-hop"
EDGE_DISJOINT = False
TM_SPLIT = {"train": (0, 20), "val": (20, 28), "test": (28, 36)}
TM_SPLIT_SOURCE = (
    "TEAL-style consecutive split of 36 public B4 matrices; "
    "P1 does not publish the split (UNKNOWN vs authors)"
)

# --- Model / RL (must match trained checkpoints) ---
NUM_LAYERS = 6
OUTPUT_DIM = 7
NUM_PATHS = 4
ACTOR_IN_DIM = 28
CRITIC_HIDDEN = 256
EPISODES = 3
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
OPTIMIZER = "adam"
SEEDS = list(range(100, 1100, 100))
H0_SCALING = False  # raw capacities and demands (P1 eq. 5)
REWARD = "F_total"  # sum of post-adjust path flows (P1 eqs. 29-30)
RAW_ACTION_CLAMP = 10.0
LOG_STD_MIN = -10.0
LOG_STD_MAX = 10.0

# --- ADMM ---
# Reference baseline channel (frozen for Phase G):
FROZEN_ADMM_VARIANT = "teal"
FROZEN_ADMM_ORDER = "s -> lambda -> z -> x"
# Parallel diagnostic channel (never mixed into frozen mean):
DIAGNOSTIC_ADMM_VARIANT = "p1"
DIAGNOSTIC_ADMM_ORDER = "F -> z -> xi -> lambda"
RHO = 1.0
ADMM_ITERS_B4 = 2

# --- Baselines ---
BASELINES = ("lp", "ecmp", "shortest_path", "grl_te_no_admm")

# --- Claims ---
CLAIMS_FORBIDDEN = (
    "Do NOT claim reproduction of P1 Table 6 B4 GRL-TE 92.39%.",
    "Do NOT claim six-topology / 89.36% average reproduction.",
    "Do NOT average teal_admm and p1_admm results.",
    "Do NOT describe Teal-order ADMM as P1 eqs. (49)-(52).",
)

DISCLAIMER = (
    "This is a qualified B4 research baseline under an explicit protocol. "
    "Overall P1 reproduction status remains PARTIALLY VERIFIED. "
    "Frozen ADMM is Teal-order (stable deploy), labeled as a known deviation "
    "from P1's printed F→z→ξ→λ iteration order."
)
