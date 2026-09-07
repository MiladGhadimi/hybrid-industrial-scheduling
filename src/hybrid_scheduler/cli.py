from __future__ import annotations

import argparse
import json

from .benchmark import paired_comparison, plot_results, run_benchmark
from .exact import solve_exact
from .generator import generate_instance
from .heuristic import solve_heuristic
from .model import SchedulingInstance
from .qaoa import solve_qaoa


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.replace(",", " ").split())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid production-sequencing benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a reproducible JSON instance")
    gen.add_argument("--size", type=int, default=6)
    gen.add_argument("--seed", type=int, default=11)
    gen.add_argument("--output", default="examples/generated_instance.json")

    solve = sub.add_parser("solve", help="Solve one JSON instance")
    solve.add_argument("instance")
    solve.add_argument("--method", choices=["exact", "heuristic", "qaoa", "cpsat"],
                       default="heuristic")
    solve.add_argument("--depth", type=int, default=1,
                       help="QAOA layers p; 0 reports the warm start with no circuit")
    solve.add_argument("--seed", type=int, default=0, help="QAOA optimizer seed")
    solve.add_argument("--alpha", type=float, default=1.0,
                       help="CVaR tail fraction; 1.0 is the plain expectation")
    solve.add_argument("--shots", type=int, default=128,
                       help="Measurement budget for the best-of-shots readout")
    solve.add_argument("--no-warm-start", action="store_true",
                       help="Start from the uniform superposition instead of the heuristic")
    solve.add_argument("--maxiter", type=int, default=45)

    bench = sub.add_parser("benchmark", help="Run the reproducible benchmark suite")
    bench.add_argument("--output-dir", default="results")
    bench.add_argument("--sizes", type=_int_list, default=(4, 5, 6, 8, 10, 12, 14, 16))
    bench.add_argument("--seeds", type=_int_list, default=(11, 23, 37))
    bench.add_argument("--qaoa-max-size", type=int, default=6)
    bench.add_argument("--qaoa-maxiter", type=int, default=28)
    bench.add_argument("--shots", type=int, default=128)
    bench.add_argument("--no-cpsat", action="store_true",
                       help="Skip the OR-Tools CP-SAT baseline")
    bench.add_argument("--no-plot", action="store_true")

    compare = sub.add_parser("compare", help="Per-instance head-to-head between two methods")
    compare.add_argument("csv")
    compare.add_argument("baseline")
    compare.add_argument("challenger")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        instance = generate_instance(args.size, args.seed)
        instance.save(args.output)
        print(args.output)
    elif args.command == "solve":
        instance = SchedulingInstance.load(args.instance)
        if args.method == "exact":
            result = solve_exact(instance)
        elif args.method == "cpsat":
            from .cpsat import solve_cpsat
            result = solve_cpsat(instance)
        elif args.method == "qaoa":
            result = solve_qaoa(
                instance, depth=args.depth, seed=args.seed, maxiter=args.maxiter,
                warm_start=not args.no_warm_start, alpha=args.alpha, shots=args.shots,
            )
        else:
            result = solve_heuristic(instance)
        print(json.dumps(result.to_dict(instance), indent=2))
    elif args.command == "compare":
        import csv as _csv
        with open(args.csv, encoding="utf-8") as handle:
            rows = list(_csv.DictReader(handle))
        print(json.dumps(paired_comparison(rows, args.baseline, args.challenger), indent=2))
    else:
        run_benchmark(
            sizes=args.sizes, seeds=args.seeds, qaoa_max_size=args.qaoa_max_size,
            qaoa_maxiter=args.qaoa_maxiter, shots=args.shots,
            with_cpsat=not args.no_cpsat, output_dir=args.output_dir,
        )
        if not args.no_plot:
            plot_results(f"{args.output_dir}/benchmark.csv",
                         f"{args.output_dir}/figures/performance.png")
        print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
