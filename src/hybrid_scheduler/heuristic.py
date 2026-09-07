from __future__ import annotations

import random
from itertools import permutations
from time import perf_counter

import numpy as np

from .model import ScheduleResult, SchedulingInstance


def solve_random_sampling(
    instance: SchedulingInstance,
    shots: int = 128,
    seed: int = 0,
    analyse_distribution: bool = False,
) -> ScheduleResult:
    """Draw ``shots`` uniform random permutations and keep the cheapest.

    This is the null model for any sampler reported under a best-of-K readout.
    A quantum arm that measures K times and keeps its best schedule must beat
    *this*, not merely beat greedy search: with the same shot budget, uniform
    sampling already solves small instances outright (n=4 has only 24 feasible
    permutations). Reporting best-of-K without this control overstates the
    sampler's contribution.
    """
    if shots < 1:
        raise ValueError("shots must be at least 1")

    start = perf_counter()
    rng = random.Random(seed)
    indices = list(range(instance.size))
    best_order: tuple[int, ...] | None = None
    best_cost = float("inf")
    for _ in range(shots):
        rng.shuffle(indices)
        candidate = tuple(indices)
        value = instance.evaluate(candidate)["objective"]
        if value < best_cost:
            best_order, best_cost = candidate, value
    assert best_order is not None
    parts = instance.evaluate(best_order)
    metadata: dict = {"shots": shots, "seed": seed}

    # On the small common scope we can calculate the uniform sampler's expected
    # best-of-K result exactly. This is more reproducible than promoting one
    # lucky sampled readout to the primary benchmark result.
    if analyse_distribution:
        costs = np.array([
            instance.evaluate(order)["objective"]
            for order in permutations(range(instance.size))
        ])
        order = np.argsort(costs, kind="stable")
        sorted_costs = costs[order]
        cumulative = np.arange(1, len(costs) + 1, dtype=float) / len(costs)
        survival = np.clip(1.0 - cumulative, 0.0, 1.0) ** shots
        expected_best = float(
            sorted_costs[0] + np.diff(sorted_costs) @ survival[:-1]
        )
        optimum = float(sorted_costs[0])
        optimum_count = int(np.count_nonzero(costs <= optimum + 1e-9))
        single_shot_success = optimum_count / len(costs)
        metadata.update({
            "expected_best_of_shots_objective": expected_best,
            "success_probability": single_shot_success,
            "probability_optimum_with_shots": 1.0 - (1.0 - single_shot_success) ** shots,
        })
    return ScheduleResult(
        method="random_sampling", order=best_order, runtime_seconds=perf_counter() - start,
        optimal=False, metadata=metadata, **parts
    )


def _greedy_order(instance: SchedulingInstance) -> tuple[int, ...]:
    remaining = set(range(instance.size))
    order: list[int] = []
    for slot in range(instance.size):
        def score(j: int, slot: int = slot) -> float:
            # slot bound as a default argument: the closure is consumed within this
            # iteration, but binding it makes that explicit rather than incidental.
            transition = 0.0 if not order else instance.transition_cost(order[-1], j)
            urgency = 0.12 * instance.jobs[j].priority * max(0, slot - instance.jobs[j].due_slot + 1)
            return instance.placement_cost(j, slot) + transition + urgency

        chosen = min(remaining, key=lambda j: (score(j), instance.jobs[j].due_slot, instance.jobs[j].id))
        order.append(chosen)
        remaining.remove(chosen)
    return tuple(order)


def _local_search(instance: SchedulingInstance, order: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    current = list(order)
    current_cost = instance.evaluate(current)["objective"]
    improvements = 0
    changed = True
    while changed:
        changed = False
        best_order = current
        best_cost = current_cost
        for i in range(len(current) - 1):
            for j in range(i + 1, len(current)):
                candidate = current.copy()
                candidate[i], candidate[j] = candidate[j], candidate[i]
                value = instance.evaluate(candidate)["objective"]
                if value < best_cost - 1e-10:
                    best_order, best_cost = candidate, value
        if best_cost < current_cost - 1e-10:
            current, current_cost = best_order, best_cost
            improvements += 1
            changed = True
    return tuple(current), improvements


def solve_heuristic(instance: SchedulingInstance) -> ScheduleResult:
    start = perf_counter()
    initial = _greedy_order(instance)
    order, improvements = _local_search(instance, initial)
    parts = instance.evaluate(order)
    return ScheduleResult(
        method="greedy_local_search", order=order, runtime_seconds=perf_counter() - start,
        optimal=False, metadata={"local_search_rounds": improvements}, **parts
    )
