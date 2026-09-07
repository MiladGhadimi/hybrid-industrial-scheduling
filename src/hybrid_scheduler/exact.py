from __future__ import annotations

from math import inf
from time import perf_counter

from .model import ScheduleResult, SchedulingInstance


def solve_exact(instance: SchedulingInstance) -> ScheduleResult:
    """Held-Karp dynamic program; O(n^2 2^n) time and O(n 2^n) memory."""
    start = perf_counter()
    n = instance.size
    if n > 20:
        raise ValueError("Exact dynamic programming is restricted to n <= 20.")

    # State (mask, last) stores best cost and predecessor.
    costs: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int | None] = {}
    for j in range(n):
        costs[(1 << j, j)] = instance.placement_cost(j, 0)
        parent[(1 << j, j)] = None

    for mask in range(1, 1 << n):
        slot = mask.bit_count() - 1
        if slot == 0:
            continue
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            prev_mask = mask ^ (1 << last)
            best = inf
            best_prev = None
            place = instance.placement_cost(last, slot)
            for prev in range(n):
                key = (prev_mask, prev)
                if key not in costs:
                    continue
                candidate = costs[key] + instance.transition_cost(prev, last) + place
                if candidate < best:
                    best = candidate
                    best_prev = prev
            costs[(mask, last)] = best
            parent[(mask, last)] = best_prev

    full = (1 << n) - 1
    last = min(range(n), key=lambda j: costs[(full, j)])
    rev = []
    mask = full
    while last is not None:
        rev.append(last)
        previous = parent[(mask, last)]
        mask ^= 1 << last
        last = previous
    order = tuple(reversed(rev))
    parts = instance.evaluate(order)
    return ScheduleResult(
        method="exact_dp", order=order, runtime_seconds=perf_counter() - start,
        optimal=True, metadata={"states": len(costs), "complexity": "O(n^2 2^n)"}, **parts
    )

