"""Utilisation-space UCB Safety Gate (improved stack).

Works entirely in utilisation units:

    mean_util(e) = mean_load(e) / max(capacity(e), min_capacity)
    std_util(e)  = std_load(e)  / max(capacity(e), min_capacity)
    UCB(e)       = mean_util(e) + k * std_util(e)

Accept only when every finite link satisfies UCB(e) <= utilisation_limit.

Absolute load predictions from the Twin are converted to utilisation before
the bound is formed — absolute and utilisation uncertainties are never mixed.

Optional ``max_relative_std`` (std_util) is off by default so uncertain but
well-headed actions are not rejected solely for large std.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

REASON_INVALID = "invalid_prediction"
REASON_UNCERTAIN = "excessive_uncertainty"
REASON_UCB = "ucb_exceeds_utilisation_limit"


@dataclass
class ImprovedGateConfig:
    k: float = 1.0
    utilisation_limit: float = 1.0
    max_relative_std: float | None = None
    min_capacity: float = 1e-12

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ImprovedGateConfig":
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


@dataclass
class ImprovedGateVerdict:
    accepted: bool
    reasons: list[str]
    mean_util: np.ndarray
    std_util: np.ndarray
    ucb: np.ndarray
    relative_std: np.ndarray
    worst_link: int
    mlu_mean: float
    mlu_ucb: float
    max_relative_std_obs: float
    n_links_over_limit: int
    k: float
    utilisation_limit: float

    def summary(self) -> dict:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "worst_link": int(self.worst_link),
            "mlu_mean": self.mlu_mean,
            "mlu_ucb": self.mlu_ucb,
            "max_relative_std": self.max_relative_std_obs,
            "n_links_over_limit": int(self.n_links_over_limit),
            "k": self.k,
            "utilisation_limit": self.utilisation_limit,
        }


def _empty_verdict(n: int, reasons: list[str], config: ImprovedGateConfig) -> ImprovedGateVerdict:
    zeros = np.zeros(max(n, 0), dtype=np.float64)
    return ImprovedGateVerdict(
        accepted=False,
        reasons=reasons,
        mean_util=zeros,
        std_util=zeros,
        ucb=zeros,
        relative_std=zeros,
        worst_link=-1,
        mlu_mean=float("nan"),
        mlu_ucb=float("nan"),
        max_relative_std_obs=float("nan"),
        n_links_over_limit=n,
        k=float(config.k),
        utilisation_limit=float(config.utilisation_limit),
    )


def inspect_utilisation_ucb(
    mean_load: np.ndarray,
    std_load: np.ndarray,
    capacity: np.ndarray,
    config: ImprovedGateConfig | None = None,
) -> ImprovedGateVerdict:
    """Decide from absolute twin loads after converting to utilisation."""
    config = config or ImprovedGateConfig()
    mean = np.asarray(mean_load, dtype=np.float64)
    std = np.asarray(std_load, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)

    if mean.shape != std.shape or mean.shape != cap.shape or mean.ndim != 1:
        return _empty_verdict(int(cap.size) if cap.ndim == 1 else 0, [REASON_INVALID], config)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not np.isfinite(cap).all():
        return _empty_verdict(int(mean.size), [REASON_INVALID], config)
    if np.any(std < -1e-12):
        return _empty_verdict(int(mean.size), [REASON_INVALID], config)
    # Negative capacities or mean loads are not physically meaningful for this gate;
    # reject rather than risk accepting via a soft UCB under underflowed predictions.
    if np.any(cap < 0.0) or np.any(mean < -1e-12):
        return _empty_verdict(int(mean.size), [REASON_INVALID], config)

    floor = np.maximum(cap, float(config.min_capacity))
    std_nonneg = np.maximum(std, 0.0)
    mean_util = mean / floor
    std_util = std_nonneg / floor
    ucb = mean_util + float(config.k) * std_util
    rel_std = std_util

    reasons: list[str] = []
    if config.max_relative_std is not None and float(np.max(rel_std, initial=0.0)) > float(
        config.max_relative_std
    ):
        reasons.append(REASON_UNCERTAIN)

    over = ucb > float(config.utilisation_limit)
    n_over = int(np.count_nonzero(over))
    if n_over:
        reasons.append(REASON_UCB)

    worst = int(np.argmax(ucb)) if ucb.size else -1
    accepted = not reasons
    return ImprovedGateVerdict(
        accepted=accepted,
        reasons=reasons,
        mean_util=mean_util,
        std_util=std_util,
        ucb=ucb,
        relative_std=rel_std,
        worst_link=worst,
        mlu_mean=float(np.max(mean_util)) if mean_util.size else 0.0,
        mlu_ucb=float(np.max(ucb)) if ucb.size else 0.0,
        max_relative_std_obs=float(np.max(rel_std)) if rel_std.size else 0.0,
        n_links_over_limit=n_over,
        k=float(config.k),
        utilisation_limit=float(config.utilisation_limit),
    )


@dataclass
class MeanOnlyConfig:
    """Convenience: mean-only filtering is UCB with k=0 and no relative-std veto."""

    utilisation_limit: float = 1.0
    min_capacity: float = 1e-12

    def to_improved(self) -> ImprovedGateConfig:
        return ImprovedGateConfig(
            k=0.0,
            utilisation_limit=self.utilisation_limit,
            max_relative_std=None,
            min_capacity=self.min_capacity,
        )
