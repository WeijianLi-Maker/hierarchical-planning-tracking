from dataclasses import dataclass

import numpy as np

try:
    import cvxpy as cp
except ModuleNotFoundError:
    cp = None


@dataclass(frozen=True)
class QuadraticProgramResult:
    x: np.ndarray
    cost: float
    status: str
    success: bool


def solve_quadratic_program(
    P,
    q,
    G=None,
    h=None,
    A_eq=None,
    b_eq=None,
    lb=None,
    ub=None,
    solver="OSQP",
    **solve_kwargs,
):
    """
    Solve a convex quadratic program in standard form:

        minimize   0.5 x^T P x + q^T x
        subject to G x <= h
                   A_eq x = b_eq
                   lb <= x <= ub

    Parameters
    ----------
    P : array-like, shape (n, n)
        Symmetric positive semidefinite cost matrix.
    q : array-like, shape (n,)
        Linear cost term.
    G : array-like, shape (m_ineq, n), optional
        Inequality constraint matrix.
    h : array-like, shape (m_ineq,), optional
        Inequality constraint upper bounds.
    A_eq : array-like, shape (m_eq, n), optional
        Equality constraint matrix.
    b_eq : array-like, shape (m_eq,), optional
        Equality constraint right-hand side.
    lb : array-like, shape (n,), optional
        Variable lower bounds.
    ub : array-like, shape (n,), optional
        Variable upper bounds.
    solver : str, optional
        CVXPY solver name. Defaults to OSQP.
    **solve_kwargs
        Extra keyword arguments passed to ``problem.solve``.

    Returns
    -------
    QuadraticProgramResult
        Solution vector, optimal cost, solver status, and success flag.
    """
    if cp is None:
        raise ModuleNotFoundError(
            "cvxpy is required to solve quadratic programs. "
            "Install it with: pip install cvxpy osqp"
        )

    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float).reshape(-1)
    n = q.shape[0]

    if P.shape != (n, n):
        raise ValueError(f"Expected P shape ({n}, {n}), got {P.shape}")

    x = cp.Variable(n)
    cost = 0.5 * cp.quad_form(x, P) + q @ x
    constraints = []

    if G is not None:
        G = np.asarray(G, dtype=float)
        h = np.asarray(h, dtype=float).reshape(-1)
        if G.shape[0] != h.shape[0]:
            raise ValueError("Expected G and h to have the same number of rows")
        if G.shape[1] != n:
            raise ValueError(f"Expected G shape (m, {n}), got {G.shape}")
        constraints.append(G @ x <= h)

    if A_eq is not None:
        A_eq = np.asarray(A_eq, dtype=float)
        b_eq = np.asarray(b_eq, dtype=float).reshape(-1)
        if A_eq.shape[0] != b_eq.shape[0]:
            raise ValueError("Expected A_eq and b_eq to have the same number of rows")
        if A_eq.shape[1] != n:
            raise ValueError(f"Expected A_eq shape (m, {n}), got {A_eq.shape}")
        constraints.append(A_eq @ x == b_eq)

    if lb is not None:
        lb = np.asarray(lb, dtype=float).reshape(-1)
        if lb.shape != (n,):
            raise ValueError(f"Expected lb shape ({n},), got {lb.shape}")
        finite = np.isfinite(lb)
        if np.any(finite):
            constraints.append(x[finite] >= lb[finite])

    if ub is not None:
        ub = np.asarray(ub, dtype=float).reshape(-1)
        if ub.shape != (n,):
            raise ValueError(f"Expected ub shape ({n},), got {ub.shape}")
        finite = np.isfinite(ub)
        if np.any(finite):
            constraints.append(x[finite] <= ub[finite])

    problem = cp.Problem(cp.Minimize(cost), constraints)

    options = dict(solve_kwargs)
    if str(solver).upper() == "OSQP" and "eps_abs" not in options:
        options.update({"eps_abs": 1e-7, "eps_rel": 1e-7, "polishing": True})

    try:
        problem.solve(solver=solver, verbose=False, **options)
    except cp.error.SolverError as exc:
        return QuadraticProgramResult(
            x=np.full(n, np.nan),
            cost=np.inf,
            status=str(exc),
            success=False,
        )

    success = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    x_value = np.full(n, np.nan) if x.value is None else np.asarray(x.value, dtype=float).reshape(-1)
    cost_value = np.inf if problem.value is None else float(problem.value)

    return QuadraticProgramResult(
        x=x_value,
        cost=cost_value,
        status=str(problem.status),
        success=success,
    )
