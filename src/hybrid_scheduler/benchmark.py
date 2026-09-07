from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, median

from .exact import solve_exact
from .generator import generate_instance
from .heuristic import solve_heuristic, solve_random_sampling
from .qaoa import solve_qaoa

FIELDS = [
    "instance", "size", "seed", "method", "family", "depth", "alpha", "warm_start",
    "objective", "placement_cost", "changeover_cost", "runtime_seconds", "optimal",
    "gap_percent", "success_probability", "expected_objective",
    "argmax_objective", "argmax_gap_percent", "argmax_optimal",
    "expected_best_of_shots_objective", "expected_best_of_shots_gap_percent",
    "probability_optimum_with_shots",
]

# (depth, alpha, warm_start). Every arm answers one question:
#   p=0 warm            -- the warm start alone; any p>=1 arm must beat this
#   p=1 alpha=1.0 warm  -- the v0.1.0 configuration, kept for comparability
#   p=1 alpha=0.10 warm -- does a tail-sensitive merit function unstick beta?
#   p=3 alpha=0.05 warm -- does depth plus a tail objective reach the optimum?
#   p=1 alpha=1.0 cold  -- ablation isolating the warm start's contribution
#   p=3 alpha=0.05 cold -- the same deep/tail arm without any classical hint
QAOA_ARMS: tuple[tuple[int, float, bool], ...] = (
    (0, 1.0, True),
    (1, 1.0, True),
    (1, 0.10, True),
    (3, 0.05, True),
    (1, 1.0, False),
    (3, 0.05, False),
)


def _row(instance, size, seed, result, reference_objective) -> dict:
    meta = result.metadata
    gap = 100 * (result.objective - reference_objective) / max(abs(reference_objective), 1e-12)
    argmax_objective = meta.get("argmax_objective")
    argmax_gap = (
        100 * (argmax_objective - reference_objective) / max(abs(reference_objective), 1e-12)
        if argmax_objective is not None else None
    )
    expected_best = meta.get("expected_best_of_shots_objective")
    expected_best_gap = (
        100 * (expected_best - reference_objective) / max(abs(reference_objective), 1e-12)
        if expected_best is not None else None
    )
    family = result.method.split("_p")[0] if "permutation_qaoa" in result.method else result.method
    return {
        "instance": instance.name,
        "size": size,
        "seed": seed,
        "method": result.method,
        "family": family,
        "depth": meta.get("depth", ""),
        "alpha": meta.get("alpha", ""),
        "warm_start": meta.get("warm_start", ""),
        "objective": round(result.objective, 6),
        "placement_cost": round(result.placement_cost, 6),
        "changeover_cost": round(result.changeover_cost, 6),
        "runtime_seconds": round(result.runtime_seconds, 6),
        "optimal": bool(abs(gap) <= 1e-7),
        "gap_percent": round(gap, 4),
        "success_probability": round(float(meta.get("success_probability", 0.0)), 6),
        "expected_objective": round(float(meta.get("expected_objective", result.objective)), 6),
        "argmax_objective": "" if argmax_objective is None else round(argmax_objective, 6),
        "argmax_gap_percent": "" if argmax_gap is None else round(argmax_gap, 4),
        "argmax_optimal": meta.get("argmax_optimal", ""),
        "expected_best_of_shots_objective": (
            "" if expected_best is None else round(float(expected_best), 6)
        ),
        "expected_best_of_shots_gap_percent": (
            "" if expected_best_gap is None else round(float(expected_best_gap), 4)
        ),
        "probability_optimum_with_shots": (
            "" if "probability_optimum_with_shots" not in meta
            else round(float(meta["probability_optimum_with_shots"]), 6)
        ),
    }


def run_benchmark(
    sizes: tuple[int, ...] = (4, 5, 6, 8, 10, 12, 14, 16),
    seeds: tuple[int, ...] = (11, 23, 37),
    qaoa_max_size: int = 6,
    qaoa_arms: tuple[tuple[int, float, bool], ...] = QAOA_ARMS,
    qaoa_maxiter: int = 28,
    shots: int = 128,
    with_cpsat: bool = True,
    output_dir: str | Path = "results",
) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    solve_cpsat = None
    if with_cpsat:
        try:
            from .cpsat import solve_cpsat
        except ImportError as exc:
            raise ImportError(
                'CP-SAT is enabled but OR-Tools is unavailable. Install with '
                '`pip install -e ".[cpsat]"` or pass --no-cpsat.'
            ) from exc

    rows: list[dict] = []
    for size in sizes:
        for seed in seeds:
            instance = generate_instance(size, seed)
            exact = solve_exact(instance)
            results = [exact]
            if solve_cpsat is not None:
                results.append(solve_cpsat(instance))
            results.append(solve_heuristic(instance))
            results.append(solve_random_sampling(
                instance, shots=shots, seed=seed,
                analyse_distribution=size <= qaoa_max_size,
            ))
            if size <= qaoa_max_size:
                for depth, alpha, warm in qaoa_arms:
                    results.append(solve_qaoa(
                        instance, depth=depth, seed=seed, maxiter=qaoa_maxiter,
                        warm_start=warm, alpha=alpha, shots=shots,
                    ))
            for result in results:
                rows.append(_row(instance, size, seed, result, exact.objective))

    csv_path = output / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows


def _aggregate(group: list[dict]) -> dict:
    def as_bool(value) -> bool:
        return value if isinstance(value, bool) else str(value).lower() == "true"

    stats = {
        "runs": len(group),
        "mean_gap_percent": round(mean(float(x["gap_percent"]) for x in group), 4),
        "max_gap_percent": round(max(float(x["gap_percent"]) for x in group), 4),
        "median_runtime_seconds": round(median(float(x["runtime_seconds"]) for x in group), 6),
        "optimal_rate": round(mean(as_bool(x["optimal"]) for x in group), 4),
    }
    if any("permutation_qaoa" in x["method"] for x in group):
        stats["mean_success_probability"] = round(
            mean(float(x["success_probability"]) for x in group), 6
        )
        argmax = [x for x in group if x["argmax_gap_percent"] != ""]
        if argmax:
            stats["argmax_mean_gap_percent"] = round(
                mean(float(x["argmax_gap_percent"]) for x in argmax), 4
            )
            stats["argmax_optimal_rate"] = round(
                mean(as_bool(x["argmax_optimal"]) for x in argmax), 4
            )
    expected = [x for x in group if x.get("expected_best_of_shots_gap_percent", "") != ""]
    if expected:
        stats["mean_expected_best_of_shots_gap_percent"] = round(
            mean(float(x["expected_best_of_shots_gap_percent"]) for x in expected), 4
        )
        stats["mean_probability_optimum_with_shots"] = round(
            mean(float(x["probability_optimum_with_shots"]) for x in expected), 6
        )
    return stats


def summarize(rows: list[dict]) -> dict:
    """Aggregate results, reporting every method on the scope it actually ran.

    ``common_scope`` restricts every method to the instances on which *all* methods
    produced a result. Comparing a quantum arm evaluated on n<=6 against a classical
    arm evaluated on n<=16 is not a comparison, and the top-level ``full_scope``
    block must never be read as one.
    """
    methods = sorted({row["method"] for row in rows})
    per_method_instances = {
        m: {row["instance"] for row in rows if row["method"] == m} for m in methods
    }
    common = set.intersection(*per_method_instances.values()) if methods else set()

    result = {
        "instances": len({row["instance"] for row in rows}),
        "sizes": sorted({int(row["size"]) for row in rows}),
        "common_scope_instances": sorted(common),
        "note": (
            "full_scope aggregates each method over the instances it ran on and is "
            "NOT a like-for-like comparison across methods. Use common_scope for that."
        ),
        "full_scope": {},
        "common_scope": {},
    }
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        result["full_scope"][method] = _aggregate(group)
        restricted = [row for row in group if row["instance"] in common]
        if restricted:
            result["common_scope"][method] = _aggregate(restricted)
    return result


def paired_comparison(rows: list[dict], baseline: str, challenger: str) -> dict:
    """Per-instance win/loss/tie of ``challenger`` against ``baseline``.

    A mean-gap table cannot tell you whether a method ever changes the answer.
    This can: v0.1.0's QAOA arm was identical to its own warm start on 9/9 instances.
    """
    base = {r["instance"]: float(r["objective"]) for r in rows if r["method"] == baseline}
    chal = {r["instance"]: float(r["objective"]) for r in rows if r["method"] == challenger}
    shared = sorted(set(base) & set(chal))
    wins = sum(chal[k] < base[k] - 1e-9 for k in shared)
    losses = sum(chal[k] > base[k] + 1e-9 for k in shared)
    return {
        "baseline": baseline,
        "challenger": challenger,
        "instances": len(shared),
        "challenger_better": wins,
        "challenger_worse": losses,
        "identical": len(shared) - wins - losses,
    }


def plot_results(csv_path: str | Path, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.read_csv(csv_path)
    grouped = frame.groupby(["size", "method"], as_index=False).agg(
        gap_percent=("gap_percent", "mean"), runtime_seconds=("runtime_seconds", "median")
    )
    labels = {
        "exact_dp": "Exact DP",
        "cpsat": "CP-SAT",
        "greedy_local_search": "Greedy + local search",
        "random_sampling": "Uniform random, best of K shots",
        "warm_start_permutation_qaoa_p0": "Warm start only (p=0)",
        "warm_start_permutation_qaoa_p1": "Warm QAOA p=1, expectation",
        "warm_start_permutation_qaoa_p1_cvar0.1": "Warm QAOA p=1, CVaR 0.10",
        "warm_start_permutation_qaoa_p3_cvar0.05": "Warm QAOA p=3, CVaR 0.05",
        "cold_start_permutation_qaoa_p1": "Cold QAOA p=1, expectation",
        "cold_start_permutation_qaoa_p3_cvar0.05": "Cold QAOA p=3, CVaR 0.05",
    }
    colors = {
        "exact_dp": "#102A43",
        "cpsat": "#6B7A8F",
        "greedy_local_search": "#1B8A8F",
        "random_sampling": "#A0AEC0",
        "warm_start_permutation_qaoa_p0": "#B0B7C3",
        "warm_start_permutation_qaoa_p1": "#D17A22",
        "warm_start_permutation_qaoa_p1_cvar0.1": "#C05621",
        "warm_start_permutation_qaoa_p3_cvar0.05": "#7B341E",
        "cold_start_permutation_qaoa_p1": "#9F7AEA",
        "cold_start_permutation_qaoa_p3_cvar0.05": "#553C9A",
    }
    # Quality is split across two panels on purpose. Every quantum arm lives at
    # n <= 6, where the classical spread is a few percent; the random baseline
    # reaches 130% at n = 16. On one shared axis the second flattens the first into
    # an unreadable line at zero, which is exactly the region the benchmark is about.
    quantum_sizes = sorted(
        grouped.loc[grouped["method"].str.contains("permutation_qaoa"), "size"].unique()
    )
    classical = [m for m in grouped["method"].unique() if "permutation_qaoa" not in m]

    def _style(method: str) -> dict:
        style = {"marker": "o", "label": labels.get(method, method), "color": colors.get(method)}
        if method.startswith("cold_start"):
            style["linestyle"] = "--"
        if method == "warm_start_permutation_qaoa_p0":
            style["linestyle"] = ":"
            style["linewidth"] = 3.0
        if method == "warm_start_permutation_qaoa_p1":
            style["linewidth"] = 1.4
        return style

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    common = grouped[grouped["size"].isin(quantum_sizes)]
    for method, part in common.groupby("method"):
        axes[0].plot(part.sort_values("size")["size"],
                     part.sort_values("size")["gap_percent"], **_style(method))
    for method, part in grouped[grouped["method"].isin(classical)].groupby("method"):
        axes[1].plot(part.sort_values("size")["size"],
                     part.sort_values("size")["gap_percent"], **_style(method))
    for method, part in grouped.groupby("method"):
        axes[2].plot(part.sort_values("size")["size"],
                     part.sort_values("size")["runtime_seconds"], **_style(method))

    axes[0].set(title="Quality, common scope (all methods)", xlabel="Jobs",
                ylabel="Mean optimality gap (%)")
    axes[0].set_xticks(quantum_sizes)
    axes[1].set(title="Quality, full range (classical only)", xlabel="Jobs",
                ylabel="Mean optimality gap (%)")
    axes[2].set(title="Runtime scaling", xlabel="Jobs", ylabel="Median runtime (s)", yscale="log")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(
        "Hybrid Production Sequencing Benchmark  -  best-of-shots readout, gap vs proven optimum",
        fontsize=13, fontweight="bold", color="#102A43",
    )
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
