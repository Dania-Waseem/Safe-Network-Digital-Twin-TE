"""Accept or reject a routing action from twin mean, twin std, and capacities.

Phase 2 rule: accept only if predicted load plus uncertainty stays below
capacity on every link. Otherwise reject and use a fallback.

Upper bound on link e:

    upper(e) = mean(e) + k * std(e)

The action is accepted only when

    upper(e) <= utilisation_limit * capacity(e)   for every e

and the relative uncertainty is not extreme:

    max_e std(e) / capacity(e)  <=  max_relative_std
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

REASON_INVALID = "invalid_prediction"
REASON_UNCERTAIN = "excessive_uncertainty"
REASON_OVERLOAD = "upper_bound_exceeds_capacity"


@dataclass
class GateConfig:
    k: float = 2.0
    utilisation_limit: float = 1.0
    max_relative_std: float | None = 0.5
    min_capacity: float = 1e-12
    apply_shortest_if_all_fail: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "GateConfig":
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)


@dataclass
class GateVerdict:
    accepted: bool
    reasons: list[str]
    upper: np.ndarray
    utilisation_mean: np.ndarray
    utilisation_upper: np.ndarray
    relative_std: np.ndarray
    worst_link: int
    mlu_mean: float
    mlu_upper: float
    max_relative_std: float
    n_links_over_limit: int
    k: float
    utilisation_limit: float

    def summary(self) -> dict:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "worst_link": int(self.worst_link),
            "mlu_mean": self.mlu_mean,
            "mlu_upper": self.mlu_upper,
            "max_relative_std": self.max_relative_std,
            "n_links_over_limit": int(self.n_links_over_limit),
            "k": self.k,
            "utilisation_limit": self.utilisation_limit,
        }


def _empty_verdict(n: int, reasons: list[str], config: GateConfig) -> GateVerdict:
    zeros = np.zeros(n, dtype=np.float64)
    return GateVerdict(
        accepted=False,
        reasons=reasons,
        upper=zeros,
        utilisation_mean=zeros,
        utilisation_upper=zeros,
        relative_std=zeros,
        worst_link=-1,
        mlu_mean=float("nan"),
        mlu_upper=float("nan"),
        max_relative_std=float("nan"),
        n_links_over_limit=n,
        k=config.k,
        utilisation_limit=config.utilisation_limit,
    )


def inspect_prediction(
    mean_load: np.ndarray,
    std_load: np.ndarray,
    capacity: np.ndarray,
    config: GateConfig | None = None,
) -> GateVerdict:
    """Decide from arrays only. Topology-agnostic."""
    config = config or GateConfig()
    mean = np.asarray(mean_load, dtype=np.float64)
    std = np.asarray(std_load, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    if mean.shape != std.shape or mean.shape != cap.shape or mean.ndim != 1:
        return _empty_verdict(
            int(cap.size) if cap.ndim == 1 else 0,
            [REASON_INVALID],
            config,
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        return _empty_verdict(int(mean.size), [REASON_INVALID], config)
    if np.any(std < -1e-12):
        return _empty_verdict(int(mean.size), [REASON_INVALID], config)

    floor = np.maximum(cap, config.min_capacity)
    upper = mean + float(config.k) * np.maximum(std, 0.0)
    util_mean = mean / floor
    util_upper = upper / floor
    rel_std = np.maximum(std, 0.0) / floor
    worst = int(np.argmax(util_upper)) if util_upper.size else -1
    reasons: list[str] = []
    if config.max_relative_std is not None and float(np.max(rel_std, initial=0.0)) > float(
        config.max_relative_std
    ):
        reasons.append(REASON_UNCERTAIN)
    over = util_upper > float(config.utilisation_limit)
    n_over = int(np.count_nonzero(over))
    if n_over:
        reasons.append(REASON_OVERLOAD)
    accepted = not reasons
    return GateVerdict(
        accepted=accepted,
        reasons=reasons,
        upper=upper,
        utilisation_mean=util_mean,
        utilisation_upper=util_upper,
        relative_std=rel_std,
        worst_link=worst,
        mlu_mean=float(np.max(util_mean)) if util_mean.size else 0.0,
        mlu_upper=float(np.max(util_upper)) if util_upper.size else 0.0,
        max_relative_std=float(np.max(rel_std)) if rel_std.size else 0.0,
        n_links_over_limit=n_over,
        k=float(config.k),
        utilisation_limit=float(config.utilisation_limit),
    )


@dataclass
class AppliedAction:
    name: str
    action: np.ndarray | None
    verdict: GateVerdict
    source: str
    proposed_accepted: bool
    fallback: str | None
    reasons: list[str] = field(default_factory=list)


def select_fallback(
    proposed_name: str,
    proposed_action: np.ndarray,
    proposed_verdict: GateVerdict,
    *,
    last_name: str | None,
    last_action: np.ndarray | None,
    last_verdict: GateVerdict | None,
    shortest_name: str,
    shortest_action: np.ndarray,
    shortest_verdict: GateVerdict,
    config: GateConfig,
) -> AppliedAction:
    """Choose the action that is actually applied after a proposed verdict."""
    if proposed_verdict.accepted:
        return AppliedAction(
            name=proposed_name,
            action=proposed_action,
            verdict=proposed_verdict,
            source="proposed",
            proposed_accepted=True,
            fallback=None,
            reasons=[],
        )

    if last_action is not None and last_verdict is not None and last_verdict.accepted:
        return AppliedAction(
            name=last_name or "last_accepted",
            action=last_action,
            verdict=last_verdict,
            source="last_accepted",
            proposed_accepted=False,
            fallback="last_accepted",
            reasons=list(proposed_verdict.reasons),
        )

    if shortest_verdict.accepted:
        return AppliedAction(
            name=shortest_name,
            action=shortest_action,
            verdict=shortest_verdict,
            source="shortest",
            proposed_accepted=False,
            fallback="shortest",
            reasons=list(proposed_verdict.reasons),
        )

    if config.apply_shortest_if_all_fail:
        extra = list(proposed_verdict.reasons) + ["fallback_also_unsafe"]
        return AppliedAction(
            name=shortest_name,
            action=shortest_action,
            verdict=shortest_verdict,
            source="shortest_last_resort",
            proposed_accepted=False,
            fallback="shortest_last_resort",
            reasons=extra,
        )

    return AppliedAction(
        name=proposed_name,
        action=None,
        verdict=proposed_verdict,
        source="rejected_no_fallback",
        proposed_accepted=False,
        fallback=None,
        reasons=list(proposed_verdict.reasons) + ["no_safe_fallback"],
    )
