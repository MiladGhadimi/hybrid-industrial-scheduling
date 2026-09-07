from __future__ import annotations

import math
from itertools import permutations
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .heuristic import solve_heuristic
from .model import ScheduleResult, SchedulingInstance

MAX_QAOA_SIZE = 8


def _permutation_basis(n: int) -> tuple[tuple[int, ...], ...]:
    return tuple(permutations(range(n)))


def _adjacent_swap_mixer(basis: tuple[tuple[int, ...], ...]) -> csr_matrix:
    """Adjacency matrix of the Cayley graph of S_n under adjacent transpositions.

    Every edge joins two permutations, so evolution under this mixer never leaves
    the feasible set. The graph is undirected (each adjacent swap is an involution),
    hence the matrix is real symmetric and therefore a valid Hermitian generator.
    """
    index = {state: i for i, state in enumerate(basis)}
    rows, cols = [], []
    for i, state in enumerate(basis):
        for k in range(len(state) - 1):
            nxt = list(state)
            nxt[k], nxt[k + 1] = nxt[k + 1], nxt[k]
            rows.append(i)
            cols.append(index[tuple(nxt)])
    data = np.ones(len(rows), dtype=np.complex128)
    return csr_matrix((data, (rows, cols)), shape=(len(basis), len(basis)))


def warm_start_state(
    basis: tuple[tuple[int, ...], ...],
    reference: tuple[int, ...],
    mass: float = 0.7,
) -> np.ndarray:
    """Amplitudes placing ``mass`` on ``reference`` and the rest on its swap neighbours."""
    n = len(reference)
    index = {state: i for i, state in enumerate(basis)}
    amplitudes = np.zeros(len(basis), dtype=np.complex128)
    amplitudes[index[reference]] = math.sqrt(mass)
    neighbours = set()
    for k in range(n - 1):
        neighbour = list(reference)
        neighbour[k], neighbour[k + 1] = neighbour[k + 1], neighbour[k]
        neighbours.add(tuple(neighbour))
    neighbours.discard(reference)
    if neighbours:
        amplitude = math.sqrt((1.0 - mass) / len(neighbours))
        for state in neighbours:
            amplitudes[index[state]] = amplitude
    return amplitudes


def _cvar(probabilities: np.ndarray, costs: np.ndarray, alpha: float) -> float:
    """Conditional value at risk: mean cost of the cheapest ``alpha`` tail.

    With alpha = 1 this is exactly the expectation. Smaller alpha rewards a
    distribution that puts *any* mass on very good permutations, which the plain
    expectation does not: from a concentrated warm start, expectation is minimised
    by applying no mixing at all.
    """
    if alpha >= 1.0:
        return float(probabilities @ costs)
    order = np.argsort(costs, kind="stable")
    sorted_probs = probabilities[order]
    sorted_costs = costs[order]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, alpha)) + 1
    cutoff = min(cutoff, len(sorted_probs))
    weights = sorted_probs[:cutoff].copy()
    overshoot = cumulative[cutoff - 1] - alpha
    if overshoot > 0:
        weights[-1] -= overshoot
    total = weights.sum()
    if total <= 0:
        return float(sorted_costs[0])
    return float(weights @ sorted_costs[:cutoff] / total)


def _best_of_shots(
    probabilities: np.ndarray,
    raw_costs: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> tuple[int, float]:
    """Sample ``shots`` permutations and keep the cheapest, as hardware would.

    Returns (basis index, expected best-of-K cost under the sampled distribution).
    The expectation is computed in closed form rather than by repeated sampling:
    P(best cost > c) = (1 - P(cost <= c))^shots.
    """
    draws = rng.choice(len(probabilities), size=shots, p=probabilities)
    best_index = int(draws[np.argmin(raw_costs[draws])])
    order = np.argsort(raw_costs, kind="stable")
    tail = np.cumsum(probabilities[order])
    survival = np.clip(1.0 - tail, 0.0, 1.0) ** shots
    # E[min] = c_(0) + sum_k (c_(k+1) - c_(k)) * P(all shots worse than c_(k))
    increments = np.diff(raw_costs[order])
    expected_best = float(raw_costs[order][0] + increments @ survival[:-1])
    return best_index, expected_best


def solve_qaoa(
    instance: SchedulingInstance,
    depth: int = 1,
    seed: int = 0,
    maxiter: int = 45,
    warm_start: bool = True,
    alpha: float = 1.0,
    shots: int = 128,
    warm_start_mass: float = 0.7,
) -> ScheduleResult:
    """Statevector simulation of constraint-preserving QAOA on permutation space.

    The adjacent-swap mixer never leaves the feasible set. This is a simulator
    benchmark, not a claim of quantum advantage or hardware readiness.

    Parameters
    ----------
    depth:
        Number of cost/mixer layers, ``p``. ``depth=0`` applies no circuit at all
        and reports the warm-start distribution itself, which is the honest
        baseline any p >= 1 result must beat.
    alpha:
        Tail fraction for the CVaR merit function. ``alpha=1.0`` recovers the plain
        expectation. See :func:`_cvar` for why expectation alone is a poor choice
        when the initial state is already concentrated.
    shots:
        Measurement budget for the best-of-K readout, which is how a sampler would
        actually be used. Reported alongside the argmax readout, never instead of it.

    Notes
    -----
    ``raw_costs`` enumerates all ``n!`` permutations, so this routine already knows
    the exact optimum before the circuit runs. That is required to report success
    probability, but it means the reported runtime is optimizer overhead on top of
    an already-solved problem and must not be read as a solve time.
    """
    start = perf_counter()
    n = instance.size
    if n > MAX_QAOA_SIZE:
        raise ValueError(f"Permutation-state simulation is restricted to n <= {MAX_QAOA_SIZE}.")
    if depth < 0:
        raise ValueError("depth must be non-negative")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    if shots < 1:
        raise ValueError("shots must be at least 1")

    basis = _permutation_basis(n)
    raw_costs = np.array([instance.evaluate(order)["objective"] for order in basis])
    shift = float(raw_costs.min())
    scale = float(max(raw_costs.std(), 1e-9))
    costs = (raw_costs - shift) / scale
    mixer = _adjacent_swap_mixer(basis)

    if warm_start:
        reference = solve_heuristic(instance).order
        initial = warm_start_state(basis, reference, mass=warm_start_mass)
    else:
        initial = np.ones(len(basis), dtype=np.complex128) / math.sqrt(len(basis))

    def probabilities(params: np.ndarray) -> np.ndarray:
        state = initial.copy()
        gammas, betas = params[:depth], params[depth:]
        for gamma, beta in zip(gammas, betas, strict=True):
            state = state * np.exp(-1j * gamma * costs)
            state = expm_multiply((-1j * beta) * mixer, state)
        probs = np.abs(state) ** 2
        return probs / probs.sum()

    def merit(params: np.ndarray) -> float:
        return _cvar(probabilities(params), costs, alpha)

    if depth == 0:
        params = np.zeros(0)
        evaluations = 0
    else:
        bounds = [(0.0, 2 * math.pi)] * depth + [(0.0, math.pi)] * depth
        opt = differential_evolution(
            merit, bounds=bounds, seed=seed, maxiter=maxiter,
            popsize=7, polish=True, updating="immediate", workers=1,
        )
        params = opt.x
        evaluations = int(opt.nfev)

    probs = probabilities(params)
    optimum_cost = float(raw_costs.min())

    argmax_index = int(np.argmax(probs))
    shot_index, expected_best_of_shots = _best_of_shots(
        probs, raw_costs, shots, np.random.default_rng(seed)
    )
    # The reported schedule is the best-of-shots readout; argmax is kept for
    # comparison because it is what v0.1.0 reported.
    order = basis[shot_index]
    parts = instance.evaluate(order)

    success_probability = float(probs[raw_costs <= optimum_cost + 1e-9].sum())
    probability_optimum_with_shots = 1.0 - (1.0 - success_probability) ** shots
    expected_objective = float(probs @ raw_costs)
    name = "warm_start_" if warm_start else "cold_start_"
    method = f"{name}permutation_qaoa_p{depth}"
    if alpha < 1.0:
        method += f"_cvar{alpha:g}"

    return ScheduleResult(
        method=method,
        order=order,
        runtime_seconds=perf_counter() - start,
        optimal=bool(parts["objective"] <= optimum_cost + 1e-9),
        metadata={
            "states": len(basis),
            "depth": depth,
            "alpha": alpha,
            "shots": shots,
            "optimizer_evaluations": evaluations,
            "expected_objective": expected_objective,
            "success_probability": success_probability,
            "probability_optimum_with_shots": probability_optimum_with_shots,
            "argmax_objective": float(instance.evaluate(basis[argmax_index])["objective"]),
            "argmax_optimal": bool(raw_costs[argmax_index] <= optimum_cost + 1e-9),
            "best_of_shots_objective": float(parts["objective"]),
            "expected_best_of_shots_objective": expected_best_of_shots,
            "gamma": [float(x) for x in params[:depth]],
            "beta": [float(x) for x in params[depth:]],
            "simulator_only": True,
            "warm_start": warm_start,
        },
        **parts,
    )
