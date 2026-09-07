"""Optional FastAPI adapter. Install the project with the ``api`` extra."""

from __future__ import annotations

from .exact import solve_exact
from .heuristic import solve_heuristic
from .model import Job, SchedulingInstance

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("Install with: pip install -e '.[api]'") from exc


class InstancePayload(BaseModel):
    name: str = "api-instance"
    jobs: list[dict]
    energy_prices: list[float]
    changeover_costs: dict[str, dict[str, float]]
    weights: dict[str, float] = Field(
        default_factory=lambda: {"lateness": 1.0, "energy": 0.35, "changeover": 0.7}
    )


app = FastAPI(title="Hybrid Production Sequencing API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/solve/{method}")
def solve(method: str, payload: InstancePayload) -> dict:
    try:
        instance = SchedulingInstance(
            jobs=tuple(Job.from_dict(item) for item in payload.jobs),
            energy_prices=tuple(payload.energy_prices),
            changeover_costs=payload.changeover_costs,
            lateness_weight=payload.weights.get("lateness", 1.0),
            energy_weight=payload.weights.get("energy", 0.35),
            changeover_weight=payload.weights.get("changeover", 0.7),
            name=payload.name,
        )
        if method == "exact":
            result = solve_exact(instance)
        elif method == "heuristic":
            result = solve_heuristic(instance)
        else:
            raise HTTPException(status_code=400, detail="method must be exact or heuristic")
        return result.to_dict(instance)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
