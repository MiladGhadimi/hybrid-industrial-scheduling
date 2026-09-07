from __future__ import annotations

import random

from .model import Job, SchedulingInstance


def generate_instance(size: int, seed: int = 0, name: str | None = None) -> SchedulingInstance:
    if size < 2:
        raise ValueError("size must be at least 2")
    rng = random.Random(seed)
    family_count = min(4, max(2, round(size ** 0.5)))
    families = [chr(ord("A") + i) for i in range(family_count)]
    def _make_job(i: int) -> Job:
        family = rng.choice(families)
        # Seed-compatibility draw. Instances up to v0.1.0 carried a per-job
        # ``processing_time`` that never entered any cost term (the model assigns
        # jobs to unit-length slots). The field is gone, but the draw is retained
        # so that a given (size, seed) still produces the identical instance.
        # Delete this line together with a version bump if you want a clean stream.
        rng.uniform(0.8, 4.5)
        return Job(
            id=f"J{i + 1:02d}",
            family=family,
            due_slot=rng.randrange(size),
            priority=round(rng.uniform(0.8, 3.0), 2),
            energy=round(rng.uniform(0.7, 2.5), 2),
        )

    jobs = tuple(_make_job(i) for i in range(size))
    # A reproducible time-of-use price curve with a mid-horizon peak.
    energy_prices = tuple(
        round(0.65 + 0.75 * (1 - abs(2 * t / max(1, size - 1) - 1)) + rng.uniform(0, 0.18), 3)
        for t in range(size)
    )
    changeovers: dict[str, dict[str, float]] = {}
    for a in families:
        changeovers[a] = {}
        for b in families:
            changeovers[a][b] = 0.0 if a == b else round(rng.uniform(0.7, 3.2), 2)
    return SchedulingInstance(
        jobs=jobs,
        energy_prices=energy_prices,
        changeover_costs=changeovers,
        name=name or f"production_n{size}_s{seed}",
    )

