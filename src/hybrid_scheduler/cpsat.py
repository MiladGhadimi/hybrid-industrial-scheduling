"""OR-Tools CP-SAT baseline.

This is the strong classical arm. Greedy + pair-swap is a fast baseline, not a
serious solver; without CP-SAT the benchmark has no credible classical competitor
above the reach of the exact dynamic program.

The model is an assignment formulation: ``x[j][t] = 1`` iff job ``j`` occupies slot
``t``. Placement costs are linear in ``x``. Changeover costs need the pair
(slot t, slot t+1) family transition, which is linearised with one indicator per
(family a, family b, slot t) triple.

Install with ``pip install -e ".[cpsat]"``.
"""

from __future__ import annotations

from time import perf_counter

from .model import ScheduleResult, SchedulingInstance

try:  # pragma: no cover - optional dependency
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError('CP-SAT baseline requires OR-Tools: pip install -e ".[cpsat]"') from exc

# CP-SAT is integral, so costs are scaled to integers. 10^6 keeps six decimals,
# comfortably below the precision at which the benchmark compares objectives (1e-7).
SCALE = 1_000_000


def solve_cpsat(
    instance: SchedulingInstance,
    max_seconds: float = 30.0,
    workers: int = 8,
) -> ScheduleResult:
    """Solve the scaled-integer sequencing model with CP-SAT, subject to a time limit.

    ``optimal`` is set only when CP-SAT itself proves optimality for the scaled
    model; a run that hits the time limit reports the incumbent and
    ``optimal=False``. The generated benchmark instances are represented exactly
    at ``SCALE``. Higher-precision external inputs are rounded to six decimals.
    """
    start = perf_counter()
    n = instance.size
    families = sorted({job.family for job in instance.jobs})

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"x_{j}_{t}") for t in range(n)] for j in range(n)]
    for j in range(n):
        model.AddExactlyOne(x[j])
    for t in range(n):
        model.AddExactlyOne(x[j][t] for j in range(n))

    terms = []
    for j in range(n):
        for t in range(n):
            cost = round(instance.placement_cost(j, t) * SCALE)
            if cost:
                terms.append(cost * x[j][t])

    # in_family[a][t] == 1 iff the job in slot t belongs to family a.
    in_family = {
        a: [model.NewBoolVar(f"f_{a}_{t}") for t in range(n)] for a in families
    }
    for a in families:
        members = [j for j in range(n) if instance.jobs[j].family == a]
        for t in range(n):
            model.Add(sum(x[j][t] for j in members) == in_family[a][t])

    for t in range(n - 1):
        for a in families:
            for b in families:
                cost = round(instance.changeover_weight * instance.changeover_costs[a][b] * SCALE)
                if not cost:
                    continue
                pair = model.NewBoolVar(f"c_{a}_{b}_{t}")
                # pair == in_family[a][t] AND in_family[b][t+1]
                model.AddBoolOr([in_family[a][t].Not(), in_family[b][t + 1].Not(), pair])
                model.Add(pair <= in_family[a][t])
                model.Add(pair <= in_family[b][t + 1])
                terms.append(cost * pair)

    model.Minimize(sum(terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no feasible schedule (status {solver.StatusName(status)}).")

    slot_of = {}
    for j in range(n):
        for t in range(n):
            if solver.Value(x[j][t]):
                slot_of[t] = j
    order = tuple(slot_of[t] for t in range(n))
    parts = instance.evaluate(order)
    return ScheduleResult(
        method="cpsat",
        order=order,
        runtime_seconds=perf_counter() - start,
        optimal=status == cp_model.OPTIMAL,
        metadata={
            "status": solver.StatusName(status),
            "proved_optimal": status == cp_model.OPTIMAL,
            "best_bound": solver.BestObjectiveBound() / SCALE,
            "wall_time": solver.WallTime(),
            "scale": SCALE,
        },
        **parts,
    )
