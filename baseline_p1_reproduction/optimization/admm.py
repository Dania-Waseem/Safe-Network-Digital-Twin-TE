"""ADMM path-flow refinement (P1 Section 4.4, eqs. 40-52).

Two solvers
-----------
1. ``admm_refine_p1``  — derived from P1 Lagrangian (48), update order
   F → z → ξ → λ as in (49)–(52). **Preferred for P1 fidelity.**
2. ``admm_refine_teal`` — public Teal closed forms, order s → λ → z → x.
   Kept as the Phase-2 baseline for A/B comparison.

``admm_refine`` aliases ``admm_refine_p1``.

---------------------------------------------------------------------------
P1 formulation (Lu et al. 2025)
---------------------------------------------------------------------------
Variables (Table 4):
  F_d(p)   path split fraction
  z_dpe    aux. flow of demand d on path p at link e
  ξ1d ≥ 0  unallocated demand slack
  ξ3e ≥ 0  residual capacity slack
  λ        multipliers for G = (G1, G3, G4)

Equalities (45)–(47):
  G1d    = Σ_p F_d(p) + ξ1d − 1
  G3e    = Σ_{d,p:e∈p} z_dpe + ξ3e − c(e)
  G4dpe  = F_d(p)·d − z_dpe

Augmented Lagrangian (48):
  L = −Σ_{d,p} F_d(p)·d  +  λᵀG  +  (ρ/2)||G||²

Iteration order (49)–(52): F → z → ξ → λ
Hyperparameters (§5.3): ρ = 1; B4 uses 2 fixed iterations (no residual stop).
Warm start: policy softmax fractions; z ← F·d on incidences; ξ,λ ← 0.
Non-negativity: ReLU on path/link flows (paper §4.4.1); ξ projected to ≥0.

---------------------------------------------------------------------------
Derived closed forms (NOT printed in P1 — derived from ∇L = 0)
---------------------------------------------------------------------------
F-update (49), per demand d (float64; hops n_p = |{e∈p}|):
  Stationarity ∂L/∂F_p = 0 yields the linear system
    ( 1 1ᵀ + d² diag(n) ) F  =  rhs / ρ
  with
    rhs_p = d − λ1 − ρ(ξ1 − 1)
            − d Σ_{e∈p} λ4_pe
            + ρ d Σ_{e∈p} z_pe
  Then F ← ReLU(F).  If d = 0, set F = 0 (11ᵀ alone is singular).

z-update (50), per edge e (incidences i on that edge):
  Stationarity ∂L/∂z_i = 0 yields
    ( I + 11ᵀ ) z  =  b,   b_i = F_i d_i + (λ4_i − λ3)/ρ − ξ3 + c
  Closed form:  S = (1ᵀb)/(n_e+1),  z = b − S·1,  then z ← ReLU(z).

  Ambiguity documented: P1 only names “ReLU projection”, not the exact
  Euclidean projection of the quadratic onto z≥0. We use unconstrained
  stationarity + ReLU (matches the paper’s stated mechanism).

ξ-update (51), separable with ξ≥0:
  ξ1* = 1 − Σ_p F_p − λ1/ρ,   ξ1 ← max(0, ξ1*)
  ξ3* = c − Σ z − λ3/ρ,       ξ3 ← max(0, ξ3*)

λ-update (52):
  λ ← λ + ρ G(F, z, ξ)

Assumptions
-----------
- No F∈[0,1] box beyond ReLU(F)≥0; ΣF is handled by G1/ξ1.
- Exact NNQP for F≥0 or z≥0 is not derived (paper does not specify it).
- Empty paths: n_p clamped to ≥1 for numerical safety (should not occur
  on valid Yen paths).

Known numerical limitation (Phase B finding)
--------------------------------------------
With ρ=1 and B4's published 2 iterations, the 2nd F-update can drive
Σ_p F_d(p) ≫ 1 for small-demand commodities: G4/λ4 live in flow units
while the F-Hessian scales with d², so small d is weakly regularized.
This is a consequence of applying the derived unconstrained argmins +
ReLU under P1's hyperparameters — not patched here with non-P1
simplex projections or Teal's update order. Eq. (23)–(28) still restore
hard feasibility after ADMM. See PHASE_B_REPORT.md.
"""

from __future__ import annotations

from typing import Literal

import torch

from config import RHO
from preprocessing.build_bipartite_graph import TEProblem


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, device=values.device, dtype=values.dtype)
    out.index_add_(0, index, values)
    return out


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    sums = _scatter_sum(values, index, size)
    counts = _scatter_sum(torch.ones_like(values), index, size).clamp_min(1.0)
    return sums / counts


# ---------------------------------------------------------------------------
# P1 ADMM (order F → z → ξ → λ)
# ---------------------------------------------------------------------------


class _P1ADMMState:
    """ADMM state with P1 variables F, z, ξ, λ and residuals G1,G3,G4."""

    def __init__(self, problem: TEProblem, rho: float, device, dtype):
        self.problem = problem
        self.rho = float(rho)
        self.device = device
        self.dtype = dtype
        self.p2e = problem.p2e
        self.num_path = problem.num_paths
        self.num_demands = problem.num_demands
        self.num_path_node = problem.num_demands * problem.num_paths
        self.num_edge_node = problem.num_edges
        self.num_inc = int(self.p2e.size(1))

        hops = problem.hops.to(dtype).reshape(-1, self.num_path).clamp_min(1.0)
        self.hops = hops  # [D, k]
        self.update_order: list[str] = []

    def init_from_fractions(self, fractions: torch.Tensor, demands: torch.Tensor):
        k = self.num_path
        self.d = demands.to(device=self.device, dtype=self.dtype).clamp_min(0.0)
        self.c = self.problem.capacities.to(device=self.device, dtype=self.dtype)
        F = fractions.reshape(-1, k).to(device=self.device, dtype=self.dtype).clamp_min(0.0)
        # Zero-demand rows: keep F = 0
        F = torch.where(self.d.unsqueeze(-1) > 0, F, torch.zeros_like(F))
        self.F = F  # [D, k]
        # Warm-start G4 ≈ 0: z_dpe = F_d(p) · d
        path_flow = (self.F * self.d.unsqueeze(-1)).reshape(-1)  # [D*k]
        self.z = path_flow[self.p2e[0]].clone()  # [num_inc]
        self.xi1 = torch.zeros(self.num_demands, device=self.device, dtype=self.dtype)
        self.xi3 = torch.zeros(self.num_edge_node, device=self.device, dtype=self.dtype)
        self.l1 = torch.zeros(self.num_demands, device=self.device, dtype=self.dtype)
        self.l3 = torch.zeros(self.num_edge_node, device=self.device, dtype=self.dtype)
        self.l4 = torch.zeros(self.num_inc, device=self.device, dtype=self.dtype)
        self.update_order = []

    def _G(self):
        g1 = self.F.sum(dim=-1) + self.xi1 - 1.0
        load_z = _scatter_sum(self.z, self.p2e[1], self.num_edge_node)
        g3 = load_z + self.xi3 - self.c
        path_flow = (self.F * self.d.unsqueeze(-1)).reshape(-1)
        g4 = path_flow[self.p2e[0]] - self.z
        return g1, g3, g4

    def update_F(self):
        """(49) F ← argmin_F L(F, z^k, ξ^k, λ^k); then ReLU."""
        self.update_order.append("F")
        rho = self.rho
        k = self.num_path
        d = self.d
        # Σ_{e∈p} λ4 and Σ_{e∈p} z  → [D*k] then [D,k]
        sum_l4 = _scatter_sum(self.l4, self.p2e[0], self.num_path_node).reshape(-1, k)
        sum_z = _scatter_sum(self.z, self.p2e[0], self.num_path_node).reshape(-1, k)

        # rhs_p = d − λ1 − ρ(ξ1 − 1) − d Σλ4 + ρ d Σz
        rhs = (
            d.unsqueeze(-1)
            - self.l1.unsqueeze(-1)
            - rho * (self.xi1 - 1.0).unsqueeze(-1)
            - d.unsqueeze(-1) * sum_l4
            + rho * d.unsqueeze(-1) * sum_z
        )  # [D, k]

        # A = 11ᵀ + d² diag(n);  F = (1/ρ) A^{-1} rhs
        ones = torch.ones(k, k, device=self.device, dtype=self.dtype)
        # Build A per demand: [D, k, k]
        d2 = (d * d).clamp_min(0.0)  # [D]
        A = ones.unsqueeze(0) + torch.diag_embed(self.hops * d2.unsqueeze(-1))
        # Solve A F = rhs/ρ  for rows with d > 0
        rhs_scaled = rhs / rho
        F_new = torch.linalg.solve(A, rhs_scaled)
        zero = d <= 0
        F_new = torch.where(zero.unsqueeze(-1), torch.zeros_like(F_new), F_new)
        self.F = F_new.relu()

    def update_z(self):
        """(50) z ← argmin_z L(F^{k+1}, z, ξ^k, λ^k); then ReLU."""
        self.update_order.append("z")
        rho = self.rho
        path_flow = (self.F * self.d.unsqueeze(-1)).reshape(-1)
        # b_i = F_i d_i + (λ4_i − λ3_e)/ρ − ξ3_e + c_e
        e = self.p2e[1]
        b = (
            path_flow[self.p2e[0]]
            + (self.l4 - self.l3[e]) / rho
            - self.xi3[e]
            + self.c[e]
        )
        # S_e = (Σ_{i on e} b_i) / (n_e + 1);  z_i = b_i − S_e
        # Equivalent: mean over incidences plus a phantom 0 → denom n_e+1
        b_extra = torch.zeros(self.num_edge_node, device=self.device, dtype=self.dtype)
        idx_extra = torch.arange(self.num_edge_node, device=self.device)
        b_mean = _scatter_mean(
            torch.cat([b, b_extra]),
            torch.cat([e, idx_extra]),
            self.num_edge_node,
        )
        # scatter_mean of b∪{0} with index e∪{0..E-1} gives Σb/(n_e+1) = S
        self.z = (b - b_mean[e]).relu()

    def update_xi(self):
        """(51) ξ ← argmin_{ξ≥0} L(F^{k+1}, z^{k+1}, ξ, λ^k)."""
        self.update_order.append("xi")
        rho = self.rho
        self.xi1 = (1.0 - self.F.sum(dim=-1) - self.l1 / rho).relu()
        load_z = _scatter_sum(self.z, self.p2e[1], self.num_edge_node)
        self.xi3 = (self.c - load_z - self.l3 / rho).relu()

    def update_lambda(self):
        """(52) λ ← λ + ρ G(F^{k+1}, z^{k+1}, ξ^{k+1})."""
        self.update_order.append("lambda")
        g1, g3, g4 = self._G()
        self.l1 = self.l1 + self.rho * g1
        self.l3 = self.l3 + self.rho * g3
        self.l4 = self.l4 + self.rho * g4

    def step(self):
        """One P1 iteration: F → z → ξ → λ."""
        self.update_F()
        self.update_z()
        self.update_xi()
        self.update_lambda()

    def residuals(self) -> dict:
        g1, g3, g4 = self._G()
        path_flow = (self.F * self.d.unsqueeze(-1)).reshape(-1)
        total_flow = float(path_flow.sum())
        total_demand = float(self.d.sum())
        return {
            "g1_abs_mean": float(g1.abs().mean()),
            "g1_abs_max": float(g1.abs().max()),
            "g3_abs_mean": float(g3.abs().mean()),
            "g3_abs_max": float(g3.abs().max()),
            "g4_abs_mean": float(g4.abs().mean()),
            "g4_abs_max": float(g4.abs().max()),
            "total_flow": total_flow,
            "total_demand": total_demand,
            "flow_over_demand": total_flow / max(total_demand, 1e-9),
            "F_sum_max": float(self.F.sum(dim=-1).max()),
            "F_nonneg": bool((self.F >= -1e-9).all()),
            "z_nonneg": bool((self.z >= -1e-9).all()),
            "xi_nonneg": bool((self.xi1 >= -1e-9).all() and (self.xi3 >= -1e-9).all()),
            "update_order": list(self.update_order),
        }


def admm_refine_p1(
    fractions: torch.Tensor,
    demands: torch.Tensor,
    problem: TEProblem,
    num_iters: int,
    rho: float = RHO,
    return_trace: bool = False,
):
    """P1 ADMM (49)–(52). Returns refined path flows x = F·d  [D*k]."""
    device = fractions.device
    dtype = torch.float64
    state = _P1ADMMState(problem, rho=rho, device=device, dtype=dtype)
    state.init_from_fractions(fractions, demands)

    trace: list[dict] = []
    # Reset per-step order log each iteration for clarity in traces
    for it in range(num_iters):
        state.update_order = []
        state.step()
        if return_trace:
            rec = {"iter": it, **state.residuals()}
            trace.append(rec)

    x_out = (state.F * state.d.unsqueeze(-1)).reshape(-1).to(dtype=fractions.dtype)
    if return_trace:
        # Attach full-run order from last iter (always F,z,xi,lambda)
        return x_out, trace
    return x_out


# ---------------------------------------------------------------------------
# Teal ADMM baseline (order s → λ → z → x) — Phase-2 preserved
# ---------------------------------------------------------------------------


def _teal_residuals(x, z, s1, s3, demands, problem) -> dict:
    k = problem.num_paths
    g1 = demands - x.reshape(-1, k).sum(-1) - s1
    load_z = _scatter_sum(z, problem.p2e[1], problem.num_edges)
    g3 = problem.capacities.to(x.dtype) - load_z - s3
    g4 = x[problem.p2e[0]] - z
    return {
        "g1_abs_mean": float(g1.abs().mean()),
        "g1_abs_max": float(g1.abs().max()),
        "g3_abs_mean": float(g3.abs().mean()),
        "g3_abs_max": float(g3.abs().max()),
        "g4_abs_mean": float(g4.abs().mean()),
        "g4_abs_max": float(g4.abs().max()),
        "total_flow": float(x.sum()),
        "total_demand": float(demands.sum()),
        "flow_over_demand": float(x.sum() / demands.sum().clamp_min(1e-9)),
        "demand_overflow_max": float(
            (x.reshape(-1, k).sum(-1) / demands.clamp_min(1e-9)).max()
        ),
        "x_nonneg": bool((x >= -1e-9).all()),
        "z_nonneg": bool((z >= -1e-9).all()),
        "update_order": ["s", "lambda", "z", "x"],
    }


class _TealADMMState:
    """Teal-style ADMM state (Phase-2 baseline)."""

    def __init__(self, problem: TEProblem, rho: float, device, dtype):
        self.problem = problem
        self.rho = rho
        self.device = device
        self.dtype = dtype
        self.p2e = problem.p2e
        self.num_path = problem.num_paths
        self.num_path_node = problem.num_demands * problem.num_paths
        self.num_edge_node = problem.num_edges

        hops = problem.hops.to(dtype).reshape(-1, self.num_path).clamp_min(1.0)
        ones = torch.ones(self.num_path, self.num_path, device=device, dtype=dtype)
        self.A_d_inv = torch.linalg.inv(torch.diag_embed(hops) + ones)
        self.b_extra = torch.zeros(self.num_edge_node, device=device, dtype=dtype)
        self.p2e_1_extra = torch.arange(self.num_edge_node, device=device)
        self.update_order: list[str] = []

    def init_from_action(self, obs_cap, obs_demand_path, action_flow):
        self.d = obs_demand_path[:: self.num_path].to(self.dtype)
        self.c = obs_cap.to(self.dtype)
        self.x = action_flow.to(self.dtype)
        self.z = self.x[self.p2e[0]].clone()
        self.s1 = torch.zeros(self.problem.num_demands, device=self.device, dtype=self.dtype)
        self.s3 = torch.zeros(self.num_edge_node, device=self.device, dtype=self.dtype)
        self.l1 = torch.zeros(self.problem.num_demands, device=self.device, dtype=self.dtype)
        self.l3 = torch.zeros(self.num_edge_node, device=self.device, dtype=self.dtype)
        self.l4 = torch.zeros(self.p2e.size(1), device=self.device, dtype=self.dtype)
        self.update_order = []

    def update_s(self):
        self.update_order.append("s")
        self.s1 = self.l1 / self.rho + (
            self.d - self.x.reshape(-1, self.num_path).sum(1)
        )
        self.s3 = self.l3 / self.rho + (
            self.c - _scatter_sum(self.z, self.p2e[1], self.num_edge_node)
        )
        self.s1 = self.s1.relu()
        self.s3 = self.s3.relu()

    def update_lambda(self):
        self.update_order.append("lambda")
        self.l1 = self.l1 + self.rho * (
            self.d - self.x.reshape(-1, self.num_path).sum(1) - self.s1
        )
        self.l3 = self.l3 + self.rho * (
            self.c - _scatter_sum(self.z, self.p2e[1], self.num_edge_node) - self.s3
        )
        self.l4 = self.l4 + self.rho * (self.x[self.p2e[0]] - self.z)

    def update_z(self, num_approx: int = 1):
        self.update_order.append("z")
        p2e_1 = self.p2e[1].clone()
        b = (
            self.rho
            * (-self.c[self.p2e[1]] + self.s3[self.p2e[1]] - self.x[self.p2e[0]])
            - self.l3[self.p2e[1]]
            - self.l4
        )
        b_mean = _scatter_mean(
            torch.cat([b, self.b_extra]),
            torch.cat([p2e_1, self.p2e_1_extra]),
            self.num_edge_node,
        )
        self.z = (b_mean[self.p2e[1]] - b) / self.rho
        for _ in range(num_approx):
            p2e_1 = self.p2e[1].clone()
            p2e_1[self.z < 0] = self.num_edge_node
            b_mean = _scatter_mean(
                torch.cat([b, self.b_extra]),
                torch.cat([p2e_1, self.p2e_1_extra]),
                self.num_edge_node + 1,
            )
            self.z = (b_mean[self.p2e[1]] - b) / self.rho
        self.z = self.z.relu()

    def update_x(self):
        self.update_order.append("x")
        b = (
            -1.0
            - self.l1[:, None]
            + self.rho * (-self.d + self.s1)[:, None]
            + _scatter_sum(self.l4 - self.rho * self.z, self.p2e[0], self.num_path_node).reshape(
                -1, self.num_path
            )
        )
        self.x = -torch.einsum("nab,nb->na", self.A_d_inv / self.rho, b).reshape(-1)
        self.x = self.x.relu()

    def step(self):
        self.update_s()
        self.update_lambda()
        self.update_z()
        self.update_x()


def admm_refine_teal(
    fractions: torch.Tensor,
    demands: torch.Tensor,
    problem: TEProblem,
    num_iters: int,
    rho: float = RHO,
    return_trace: bool = False,
):
    """Teal-derived ADMM (Phase-2 baseline). Returns path flows [D*k]."""
    device = fractions.device
    dtype = torch.float64
    k = problem.num_paths
    d = demands.to(device=device, dtype=dtype).clamp_min(0.0)
    F = fractions.reshape(-1, k).to(dtype).clamp_min(0.0)
    F = torch.where(d.unsqueeze(-1) > 0, F, torch.zeros_like(F))
    x0 = (F * d.unsqueeze(-1)).reshape(-1)

    state = _TealADMMState(problem, rho=rho, device=device, dtype=dtype)
    path_demand = d.repeat_interleave(k)
    state.init_from_action(problem.capacities, path_demand, x0)

    trace: list[dict] = []
    for it in range(num_iters):
        state.update_order = []
        state.step()
        if return_trace:
            trace.append(
                {
                    "iter": it,
                    **_teal_residuals(state.x, state.z, state.s1, state.s3, state.d, problem),
                    "update_order": list(state.update_order),
                }
            )

    x_out = state.x.to(dtype=fractions.dtype)
    if return_trace:
        return x_out, trace
    return x_out


def admm_refine(
    fractions: torch.Tensor,
    demands: torch.Tensor,
    problem: TEProblem,
    num_iters: int,
    rho: float = RHO,
    return_trace: bool = False,
    variant: Literal["p1", "teal"] = "p1",
):
    """Dispatch ADMM. Default ``p1`` (Phase B). Pass ``variant='teal'`` for baseline."""
    if variant == "teal":
        return admm_refine_teal(
            fractions, demands, problem, num_iters, rho=rho, return_trace=return_trace
        )
    return admm_refine_p1(
        fractions, demands, problem, num_iters, rho=rho, return_trace=return_trace
    )
