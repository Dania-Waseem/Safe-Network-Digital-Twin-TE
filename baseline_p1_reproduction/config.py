"""GRL-TE hyperparameters taken from P1 (Lu et al., 2025), Section 5.3.

Paper: Graph-based reinforcement learning for software-defined networking
traffic engineering, J. King Saud Univ. Comput. Inf. Sci. 37:119.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TOPOLOGY_DIR = DATA_DIR / "topologies"
TM_DIR = DATA_DIR / "traffic_matrices"
PATH_DIR = DATA_DIR / "candidate_paths"
RESULT_DIR = ROOT / "results"
TABLE_DIR = RESULT_DIR / "tables"
GRAPH_DIR = RESULT_DIR / "graphs"
LOG_DIR = RESULT_DIR / "logs"
MODEL_DIR = ROOT.parent / "models" / "baseline_p1" / "b4"  # checkpoints (Git LFS), repository-level models/

# Candidate paths (P1 Section 5.1): 4 shortest paths per node pair.
NUM_PATHS = 4
PATH_METRIC = "min-hop"
EDGE_DISJOINT = False  # P1 says "4 shortest paths"; Yen k-shortest, not TEAL's edge-disjoint set.

# TopoFlowNet (P1 Section 5.3): 6 layers, output dim 7.
NUM_LAYERS = 6
OUTPUT_DIM = NUM_LAYERS + 1  # skip-concat with h^0

# A2C (P1 Section 5.3)
ACTOR_IN_DIM = NUM_PATHS * OUTPUT_DIM  # 28
ACTOR_OUT_DIM = NUM_PATHS  # 4
CRITIC_HIDDEN = 256
LOG_STD_MIN = -10.0
LOG_STD_MAX = 10.0
RAW_ACTION_CLAMP = 10.0

# Training (P1 Section 5.3)
EPISODES = 3
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
OPTIMIZER = "adam"
SEEDS = list(range(100, 1100, 100))

# ADMM (P1 Section 5.3)
RHO = 1.0
ADMM_ITERS = {
    "B4": 2,
    "UsCarrier": 25,
    "Cogentco": 25,
    "Kdl": 25,
    "ASN": 25,
    "GEANT": 100,
    "GÉANT": 100,
}

# TEAL-style time split of the 36 public B4 matrices (P1 does not publish the split).
# train [0, 20), val [20, 28), test [28, 36)
B4_SPLITS = {
    "train": (0, 20),
    "val": (20, 28),
    "test": (28, 36),
}

# Constraint-adjustment numerical tolerance (P1 eq. 23).
DEMAND_EPS = 1e-6

# Published P1 demand-satisfaction numbers (Table 6) for later comparison.
P1_PUBLISHED = {
    "B4": {"LP": 99.83, "NCFlow": 98.90, "Teal": 84.77, "Figret": 99.24, "GRL-TE": 92.39},
    "GEANT": {"LP": 88.40, "NCFlow": 88.99, "GRL-TE": 86.41},
    "UsCarrier": {"LP": 92.83, "NCFlow": 72.15, "GRL-TE": 91.82},
    "Cogentco": {"LP": 89.65, "NCFlow": 88.87, "GRL-TE": 87.20},
    "Kdl": {"LP": 95.40, "NCFlow": 65.21, "Figret": 68.89, "GRL-TE": 90.54},
    "ASN": {"LP": None, "NCFlow": 44.78, "GRL-TE": 87.77},
    "average": {"LP": 93.19, "NCFlow": 76.48, "Teal": 82.04, "Figret": 82.20, "GRL-TE": 89.36},
}
