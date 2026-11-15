"""Improved Safety Gate stack (isolated from ``safety.gate``).

Supports original and improved Digital Twin backends and paired
mean-only vs calibrated uncertainty-aware filtering. Does not modify
the original gate, twin, ablation, or RL checkpoints.
"""

from .config import DEFAULT_CONFIG_PATH, load_improvements_config
from .decision import VARIANT_IDS, decide_variant
from .gate_logic import ImprovedGateConfig, ImprovedGateVerdict, inspect_utilisation_ucb
from .graph_cache import TopologyGraphCache
from .oracle import capacity_feasible, physics_violations
from .policy_backend import PolicyBackend, policy_backend_from_config

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "VARIANT_IDS",
    "ImprovedGateConfig",
    "ImprovedGateVerdict",
    "PolicyBackend",
    "TopologyGraphCache",
    "capacity_feasible",
    "decide_variant",
    "inspect_utilisation_ucb",
    "load_improvements_config",
    "physics_violations",
    "policy_backend_from_config",
]
