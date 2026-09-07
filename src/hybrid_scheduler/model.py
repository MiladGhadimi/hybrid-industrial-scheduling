from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    """A job in the sequencing model.

    The model is a pure *slot-assignment* model: every job occupies exactly one
    unit-length slot, so no processing-time field appears here. Sequence-dependent
    durations are a documented roadmap extension, not a silent part of the objective.
    """

    id: str
    family: str
    due_slot: int
    priority: float
    energy: float

    _KNOWN = ("id", "family", "due_slot", "priority", "energy")
    _LEGACY = ("processing_time",)

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        """Build a job while accepting the one legacy field from v0.1.0.

        Instances written by hybrid-scheduling <= 0.1.0 carry a ``processing_time``
        field that never entered any cost term; such files still load.
        """
        unknown = set(data) - set(cls._KNOWN) - set(cls._LEGACY)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown job field(s): {names}")
        return cls(**{k: data[k] for k in cls._KNOWN if k in data})


@dataclass(frozen=True)
class SchedulingInstance:
    jobs: tuple[Job, ...]
    energy_prices: tuple[float, ...]
    changeover_costs: dict[str, dict[str, float]]
    lateness_weight: float = 1.0
    energy_weight: float = 0.35
    changeover_weight: float = 0.7
    name: str = "instance"

    def __post_init__(self) -> None:
        n = len(self.jobs)
        if n < 2:
            raise ValueError("An instance needs at least two jobs.")
        if len(self.energy_prices) != n:
            raise ValueError("energy_prices must contain one value per slot.")
        ids = [job.id for job in self.jobs]
        if len(ids) != len(set(ids)):
            raise ValueError("Job IDs must be unique.")
        families = {job.family for job in self.jobs}
        if not families.issubset(self.changeover_costs):
            raise ValueError("changeover_costs must contain every job family.")
        for family in families:
            if not families.issubset(self.changeover_costs[family]):
                raise ValueError("Every changeover row must contain every family.")

    @property
    def size(self) -> int:
        return len(self.jobs)

    def placement_cost(self, job_index: int, slot: int) -> float:
        job = self.jobs[job_index]
        lateness = job.priority * max(0, slot - job.due_slot)
        energy = job.energy * self.energy_prices[slot]
        return self.lateness_weight * lateness + self.energy_weight * energy

    def transition_cost(self, first: int, second: int) -> float:
        a = self.jobs[first].family
        b = self.jobs[second].family
        return self.changeover_weight * self.changeover_costs[a][b]

    def evaluate(self, order: Iterable[int]) -> dict[str, float]:
        seq = tuple(order)
        if sorted(seq) != list(range(self.size)):
            raise ValueError("order must be a permutation of all job indices.")
        placement = sum(self.placement_cost(j, slot) for slot, j in enumerate(seq))
        transition = sum(self.transition_cost(seq[t], seq[t + 1]) for t in range(self.size - 1))
        return {
            "placement_cost": float(placement),
            "changeover_cost": float(transition),
            "objective": float(placement + transition),
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "jobs": [asdict(job) for job in self.jobs],
            "energy_prices": list(self.energy_prices),
            "changeover_costs": self.changeover_costs,
            "weights": {
                "lateness": self.lateness_weight,
                "energy": self.energy_weight,
                "changeover": self.changeover_weight,
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SchedulingInstance:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = data.get("weights", {})
        return cls(
            jobs=tuple(Job.from_dict(item) for item in data["jobs"]),
            energy_prices=tuple(float(x) for x in data["energy_prices"]),
            changeover_costs=data["changeover_costs"],
            lateness_weight=float(weights.get("lateness", 1.0)),
            energy_weight=float(weights.get("energy", 0.35)),
            changeover_weight=float(weights.get("changeover", 0.7)),
            name=data.get("name", Path(path).stem),
        )


@dataclass(frozen=True)
class ScheduleResult:
    method: str
    order: tuple[int, ...]
    objective: float
    placement_cost: float
    changeover_cost: float
    runtime_seconds: float
    optimal: bool
    metadata: dict

    def to_dict(self, instance: SchedulingInstance | None = None) -> dict:
        data = asdict(self)
        data["order"] = list(self.order)
        if instance is not None:
            data["job_order"] = [instance.jobs[i].id for i in self.order]
        return data
