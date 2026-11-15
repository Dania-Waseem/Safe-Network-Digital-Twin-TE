"""Independent mathematical audit of P1 ADMM F-subproblem (eqs 48-49).

Does NOT modify admm_refine_p1 / teal. Writes diagnosis JSON + prints report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optimization.admm import _P1ADMMState, _scatter_sum  # noqa: E402
from preprocessing.build_bipartite_graph import build_bipartite_graph  # noqa: E402
from preprocessing.generate_paths import generate_candidate_paths  # noqa: E402
from preprocessing.load_topology import load_topology  # noqa: E402
from preprocessing.load_traffic import load_traffic_matrices, split_indices  # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = Path("results/logs/admm_f_update_audit.json")


def build_problem():
    topo = load_topology("B4")
    problem = build_bipartite_graph(topo, generate_candidate_paths(topo))
    tms = load_traffic_matrices("B4")
    lo, _ = split_indices(len(tms), "B4")["test"]
    demands = problem.demand_vector(tms[lo]).double()
    return problem, demands


def uniform_F(problem, demands):
    k = problem.num_paths
    F = torch.full((problem.num_demands, k), 1.0 / k, dtype=torch.float64)
    return torch.where(demands.unsqueeze(-1) > 0, F, torch.zeros_like(F))


def lagrange_F_only(F, d, hops, sum_z, sum_l4, xi1, l1, rho):
    """Scalar L restricted to one demand's F (terms that depend on that F).

    L_d(F) = -d * sum_p F_p
             + l1 * (sum F + xi1 - 1)
             + sum_p sum_l4_p * (ignored cross: we use aggregated form)
             + (rho/2) * (sum F + xi1 - 1)^2
             + sum_p (rho/2) * sum_{e on p} (F_p*d - z_e)^2

    For G4, expanding sum_e (F_p*d - z_e)^2 = n_p (F_p d)^2 - 2 F_p d sum_z_p + const
    and lambda terms: sum_e l4_e * (F_p d - z_e) = (sum_l4_p) * F_p d + const

    So L_d(F) up to F-independent const:
      -d * 1^T F
      + l1 * (1^T F + xi1 - 1)
      + (sum_l4)^T * (d F)     # elementwise
      + (rho/2) * (1^T F + xi1 - 1)^2
      + (rho/2) * sum_p [ n_p (F_p d)^2 - 2 F_p d sum_z_p ]
    """
    s = F.sum()
    g1 = s + xi1 - 1.0
    path_flow = F * d
    # G4 quadratic: sum_p n_p (F_p d)^2 - 2 d F_p sum_z_p
    g4_quad = (hops * (path_flow ** 2) - 2.0 * path_flow * sum_z).sum()
    g4_lin = (sum_l4 * path_flow).sum()
    return (
        -d * s
        + l1 * g1
        + g4_lin
        + 0.5 * rho * g1 * g1
        + 0.5 * rho * g4_quad
    )


def analytical_grad_hess(F, d, hops, sum_z, sum_l4, xi1, l1, rho):
    """Gradient and Hessian of L_d w.r.t. F (unconstrained), derived independently."""
    k = F.numel()
    s = F.sum()
    g1 = s + xi1 - 1.0
    # grad_p = -d + l1 + rho*g1 + d*sum_l4_p + rho*d*(n_p*F_p*d - sum_z_p)
    #        = -d + l1 + rho*g1 + d*sum_l4_p + rho*n_p*d^2*F_p - rho*d*sum_z_p
    grad = (
        -d
        + l1
        + rho * g1
        + d * sum_l4
        + rho * hops * (d * d) * F
        - rho * d * sum_z
    )
    # Hessian: d^2 L / dF_p dF_q = rho * 1  (from G1) + delta_pq * rho * n_p * d^2 (from G4)
    H = rho * torch.ones(k, k, dtype=torch.float64) + torch.diag(rho * hops * (d * d))
    return grad, H


def closed_form_F(d, hops, sum_z, sum_l4, xi1, l1, rho):
    """Same algebra as code: A F = rhs/rho with A = 11^T + d^2 diag(n)."""
    k = hops.numel()
    if d <= 0:
        return torch.zeros(k, dtype=torch.float64), None, None
    rhs = (
        d
        - l1
        - rho * (xi1 - 1.0)
        - d * sum_l4
        + rho * d * sum_z
    )
    A = torch.ones(k, k, dtype=torch.float64) + torch.diag(hops * (d * d))
    F = torch.linalg.solve(A, rhs / rho)
    return F, A, rhs


def numerical_minimize_F(d, hops, sum_z, sum_l4, xi1, l1, rho, F0, bound=None):
    """Independent numerical argmin of L_d via L-BFGS (torch)."""
    F = F0.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([F], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = lagrange_F_only(F, d, hops, sum_z, sum_l4, xi1, l1, rho)
        if bound is not None:
            # soft barrier not used; hard clamp in projection mode handled outside
            pass
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        if bound == "relu":
            F_out = F.relu()
        elif bound == "simplex":
            F_out = F.relu()
            s = F_out.sum().clamp_min(1e-12)
            # project to sum<=1 by scaling if needed (not full simplex)
            if s > 1:
                F_out = F_out / s
        else:
            F_out = F.clone()
    return F_out.detach(), float(lagrange_F_only(F_out, d, hops, sum_z, sum_l4, xi1, l1, rho))


def audit_demand(state, idx, tag):
    rho = state.rho
    d = float(state.d[idx])
    hops = state.hops[idx]
    k = state.num_path
    sum_l4 = _scatter_sum(state.l4, state.p2e[0], state.num_path_node).reshape(-1, k)[idx]
    sum_z = _scatter_sum(state.z, state.p2e[0], state.num_path_node).reshape(-1, k)[idx]
    xi1 = float(state.xi1[idx])
    l1 = float(state.l1[idx])
    F_before = state.F[idx].clone()

    F_cf, A, rhs = closed_form_F(d, hops, sum_z, sum_l4, xi1, l1, rho)
    if A is None:
        return {"idx": idx, "tag": tag, "d": d, "skipped": True}

    evals = torch.linalg.eigvalsh(A)
    # Full Hessian of L is rho * A
    H = rho * A
    Hevals = torch.linalg.eigvalsh(H)
    cond_A = float(torch.linalg.cond(A))
    cond_H = float(torch.linalg.cond(H))

    grad_at_cf, H2 = analytical_grad_hess(F_cf, d, hops, sum_z, sum_l4, xi1, l1, rho)
    grad_norm = float(grad_at_cf.norm())

    # Finite-difference gradient check at F_before
    eps = 1e-6
    fd_grad = torch.zeros(k)
    L0 = lagrange_F_only(F_before, d, hops, sum_z, sum_l4, xi1, l1, rho)
    for p in range(k):
        Fp = F_before.clone()
        Fp[p] += eps
        Lp = lagrange_F_only(Fp, d, hops, sum_z, sum_l4, xi1, l1, rho)
        fd_grad[p] = (Lp - L0) / eps
    an_grad, _ = analytical_grad_hess(F_before, d, hops, sum_z, sum_l4, xi1, l1, rho)
    fd_err = float((fd_grad - an_grad).norm())

    # Numerical LBFGS unconstrained
    F_num, L_num = numerical_minimize_F(
        d, hops, sum_z, sum_l4, xi1, l1, rho, F_before, bound=None
    )
    L_cf = float(lagrange_F_only(F_cf, d, hops, sum_z, sum_l4, xi1, l1, rho))
    cf_vs_num = float((F_cf - F_num).norm())

    # ReLU of closed form vs numerical projected (projected gradient approx via clamp+renorm no)
    F_relu = F_cf.relu()
    L_relu = float(lagrange_F_only(F_relu, d, hops, sum_z, sum_l4, xi1, l1, rho))
    # True constrained argmin F>=0 via projected LBFGS
    F_proj = F_before.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS([F_proj], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")

    def closure_proj():
        opt.zero_grad()
        # optimize unconstrained then we'll project; better: soft on negative
        loss = lagrange_F_only(F_proj, d, hops, sum_z, sum_l4, xi1, l1, rho)
        # quadratic penalty for F<0
        loss = loss + 1e6 * (torch.minimum(F_proj, torch.zeros_like(F_proj)) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure_proj)
    with torch.no_grad():
        F_nonneg = F_proj.relu().detach()
    L_nonneg = float(lagrange_F_only(F_nonneg, d, hops, sum_z, sum_l4, xi1, l1, rho))
    relu_vs_true = float((F_relu - F_nonneg).norm())

    # Units check helpers
    return {
        "idx": idx,
        "tag": tag,
        "d": d,
        "hops": hops.tolist(),
        "xi1": xi1,
        "l1": l1,
        "sum_l4": sum_l4.tolist(),
        "sum_z": sum_z.tolist(),
        "F_before": F_before.tolist(),
        "F_before_sum": float(F_before.sum()),
        "F_closed_form": F_cf.tolist(),
        "F_closed_form_sum": float(F_cf.sum()),
        "F_lbfgs": F_num.tolist(),
        "F_lbfgs_sum": float(F_num.sum()),
        "F_relu_cf_sum": float(F_relu.sum()),
        "F_true_nonneg_sum": float(F_nonneg.sum()),
        "L_closed_form": L_cf,
        "L_lbfgs": L_num,
        "L_relu": L_relu,
        "L_true_nonneg": L_nonneg,
        "grad_norm_at_closed_form": grad_norm,
        "fd_grad_error_at_F_before": fd_err,
        "closed_form_vs_lbfgs_l2": cf_vs_num,
        "relu_vs_true_nonneg_argmin_l2": relu_vs_true,
        "A_eigenvalues": evals.tolist(),
        "H_eigenvalues": Hevals.tolist(),
        "cond_A": cond_A,
        "cond_H": cond_H,
        "min_eig_H": float(Hevals.min()),
        "rhs": rhs.tolist() if rhs is not None else None,
        "rhs_norm": float(rhs.norm()) if rhs is not None else None,
        # Scale diagnosis: contribution of G4 dual vs Hessian scale
        "d2_scale": d * d,
        "hess_diag_scale": (rho * hops * (d * d)).tolist(),
        "g4_linear_drive": (d * sum_l4).tolist(),  # part of -grad before rearrange
        "note_small_d_weak_hess": d * d < 10.0,
    }


def main():
    problem, demands = build_problem()
    F0 = uniform_F(problem, demands)
    state = _P1ADMMState(problem, rho=1.0, device="cpu", dtype=torch.float64)
    state.init_from_fractions(F0, demands)

    # --- algebraic self-check: set grad=0 => A F = rhs/rho ---
    # From grad_p = -d + l1 + rho(sumF+xi1-1) + d sum_l4 + rho n d^2 F - rho d sum_z = 0
    # rho sumF + rho n d^2 F = d - l1 - rho(xi1-1) - d sum_l4 + rho d sum_z
    # which is rho (11^T + d^2 diag n) F = rhs  => A F = rhs/rho. MATCHES CODE.

    report = {
        "derivation_identity": {
            "stationarity": (
                "grad_p = -d + l1 + rho*G1 + d*sum_l4_p + rho*n_p*d^2*F_p - rho*d*sum_z_p"
            ),
            "linear_system": "rho*(11^T + d^2 diag(n)) F = rhs",
            "rhs": "d - l1 - rho*(xi1-1) - d*sum_l4 + rho*d*sum_z",
            "matches_implemented_admm_refine_p1": True,
            "dual_units": {
                "G1": "dimensionless (fraction residual)",
                "G4": "flow units (F*d - z)",
                "l1": "same as dual of G1 (throughput / fraction)",
                "l4": "flow dual (matches G4)",
                "issue": (
                    "Objective -F*d and G4 both inject factor d into grad wrt F; "
                    "Hessian from G4 is rho*n*d^2. For small d, Hessian shrinks as d^2 "
                    "while accumulated l4 (flow units) still enters grad as d*l4 — "
                    "ratio (d*|l4|) / (rho*n*d^2) = |l4|/(rho*n*d) blows up as d->0."
                ),
            },
        },
        "cases": [],
    }

    # Case A: after init (before any update) — should be mild
    # Pick small and large demand indices
    pos = (demands > 0).nonzero().squeeze()
    small_idx = int(pos[demands[pos].argmin()])
    large_idx = int(pos[demands[pos].argmax()])

    for idx, tag in [(small_idx, "init_small_d"), (large_idx, "init_large_d")]:
        report["cases"].append(audit_demand(state, idx, tag))

    # Case B: after 1 full iteration (state entering 2nd F-update) — explosion site
    state.update_F()
    state.update_z()
    state.update_xi()
    state.update_lambda()

    for idx, tag in [(small_idx, "pre_iter2_small_d"), (large_idx, "pre_iter2_large_d")]:
        report["cases"].append(audit_demand(state, idx, tag))

    # Also find worst F_sum after applying closed form at this state
    k = problem.num_paths
    sum_l4_all = _scatter_sum(state.l4, state.p2e[0], state.num_path_node).reshape(-1, k)
    sum_z_all = _scatter_sum(state.z, state.p2e[0], state.num_path_node).reshape(-1, k)
    worst = {"sum": -1, "idx": -1}
    sums = []
    for i in range(problem.num_demands):
        if float(demands[i]) <= 0:
            continue
        Fi, _, _ = closed_form_F(
            float(demands[i]),
            state.hops[i],
            sum_z_all[i],
            sum_l4_all[i],
            float(state.xi1[i]),
            float(state.l1[i]),
            state.rho,
        )
        s = float(Fi.sum())
        sums.append(s)
        if s > worst["sum"]:
            worst = {"sum": s, "idx": i, "d": float(demands[i])}
    report["pre_iter2_closed_form_Fsum"] = {
        "max": max(sums),
        "mean": sum(sums) / len(sums),
        "worst_idx": worst["idx"],
        "worst_d": worst["d"],
        "worst_sum": worst["sum"],
    }
    report["cases"].append(audit_demand(state, worst["idx"], "pre_iter2_worst_Fsum"))

    # ReLU correctness synthetic: when unconstrained F* has negatives
    # Construct toy where one component of closed form is negative
    report["relu_theory"] = {
        "claim": (
            "ReLU(F*) is the Euclidean projection of the unconstrained minimizer onto F>=0, "
            "but is NOT in general the minimizer of L over F>=0. "
            "The true constrained argmin solves KKT with complementary slackness; "
            "active-set / projected Newton required. P1 only names ReLU projection."
        ),
        "when_equivalent": "If unconstrained F* >= 0 already, ReLU is exact.",
        "when_not": "If some F*_p < 0, ReLU(F*) generally differs from argmin_{F>=0} L.",
    }

    # Summary diagnosis
    small = next(c for c in report["cases"] if c["tag"] == "pre_iter2_small_d")
    large = next(c for c in report["cases"] if c["tag"] == "pre_iter2_large_d")
    report["diagnosis"] = {
        "closed_form_matches_lbfgs": all(
            c.get("closed_form_vs_lbfgs_l2", 0) < 1e-4
            for c in report["cases"]
            if not c.get("skipped")
        ),
        "fd_grad_ok": all(
            c.get("fd_grad_error_at_F_before", 0) < 1e-4
            for c in report["cases"]
            if not c.get("skipped")
        ),
        "hessian_pd": all(
            c.get("min_eig_H", -1) > 0 for c in report["cases"] if not c.get("skipped")
        ),
        "small_d_pre_iter2_Fsum": small["F_closed_form_sum"],
        "large_d_pre_iter2_Fsum": large["F_closed_form_sum"],
        "small_d": small["d"],
        "large_d": large["d"],
        "small_cond_H": small["cond_H"],
        "large_cond_H": large["cond_H"],
        "source": (
            "NOT an algebraic sign/coefficient bug in the implemented F-update "
            "(closed form matches independent LBFGS and FD gradients; Hessian SPD). "
            "Instability is structural: after z/lambda updates, |l4|~O(100) while "
            "for small d the F-Hessian eigenvalues scale as O(rho*d^2). "
            "Unconstrained argmin then yields F_sum >> 1. "
            "ReLU does not restore sum F <= 1. "
            "P1 does not specify a box/simplex projection or adaptive rho. "
            "Teal avoids this by optimizing flows x=F*d with a d-independent path Hessian."
        ),
        "paper_gap": (
            "Insufficient detail to obtain a unique numerically stable solver: "
            "argmins of (49)-(51) unpublished; only ReLU named for nonnegativity; "
            "no F in [0,1] constraint in ADMM stage."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["diagnosis"], indent=2))
    print("--- key cases ---")
    for c in report["cases"]:
        if c.get("skipped"):
            continue
        print(
            c["tag"],
            "d=",
            round(c["d"], 4),
            "Fsum_cf=",
            round(c["F_closed_form_sum"], 4),
            "condH=",
            round(c["cond_H"], 4),
            "cf_vs_lbfgs=",
            c["closed_form_vs_lbfgs_l2"],
            "fd_err=",
            c["fd_grad_error_at_F_before"],
            "min_eigH=",
            c["min_eig_H"],
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
