import itertools
import unittest

import numpy as np
from scipy.sparse.linalg import expm_multiply

from hybrid_scheduler.benchmark import paired_comparison, summarize
from hybrid_scheduler.exact import solve_exact
from hybrid_scheduler.generator import generate_instance
from hybrid_scheduler.heuristic import solve_heuristic, solve_random_sampling
from hybrid_scheduler.qaoa import (
    _adjacent_swap_mixer,
    _cvar,
    _permutation_basis,
    solve_qaoa,
    warm_start_state,
)

try:
    from hybrid_scheduler.cpsat import solve_cpsat
except ImportError:  # optional locally; installed by the CI dev extra
    solve_cpsat = None


class SolverTests(unittest.TestCase):
    def test_exact_matches_brute_force(self):
        for size in (4, 5, 6):
            for seed in (13, 21):
                instance = generate_instance(size, seed)
                result = solve_exact(instance)
                brute = min(
                    instance.evaluate(order)["objective"]
                    for order in itertools.permutations(range(size))
                )
                self.assertAlmostEqual(result.objective, brute, places=9)
                # the returned order must actually achieve the reported objective
                self.assertAlmostEqual(
                    instance.evaluate(result.order)["objective"], result.objective, places=9
                )

    def test_heuristic_is_feasible_and_not_better_than_optimum(self):
        instance = generate_instance(8, 19)
        exact = solve_exact(instance)
        heuristic = solve_heuristic(instance)
        self.assertEqual(sorted(heuristic.order), list(range(instance.size)))
        self.assertGreaterEqual(heuristic.objective + 1e-9, exact.objective)

    @unittest.skipIf(solve_cpsat is None, "OR-Tools is not installed")
    def test_cpsat_matches_brute_force(self):
        for size in (4, 5, 6):
            for seed in (13, 21):
                instance = generate_instance(size, seed)
                result = solve_cpsat(instance, workers=1)
                brute = min(
                    instance.evaluate(order)["objective"]
                    for order in itertools.permutations(range(size))
                )
                self.assertTrue(result.metadata["proved_optimal"])
                self.assertAlmostEqual(result.objective, brute, places=7)

    def test_random_sampler_uses_exact_shot_budget(self):
        instance = generate_instance(4, 11)
        with self.assertRaises(ValueError):
            solve_random_sampling(instance, shots=0)
        result = solve_random_sampling(instance, shots=1, seed=3)
        self.assertEqual(result.metadata["shots"], 1)

    def test_uniform_expected_best_is_bounded_by_optimum(self):
        instance = generate_instance(4, 11)
        exact = solve_exact(instance)
        result = solve_random_sampling(
            instance, shots=128, seed=11, analyse_distribution=True
        )
        self.assertGreaterEqual(
            result.metadata["expected_best_of_shots_objective"] + 1e-9,
            exact.objective,
        )


class QaoaNumericsTests(unittest.TestCase):
    """The physics must be right even when the variational result is not."""

    def test_mixer_is_hermitian_and_connected(self):
        from scipy.sparse.csgraph import connected_components

        for n in (3, 4, 5):
            mixer = _adjacent_swap_mixer(_permutation_basis(n))
            self.assertAlmostEqual(abs(mixer - mixer.getH()).max(), 0.0, places=12)
            # adjacent transpositions generate S_n, so the feasible graph is connected
            self.assertEqual(connected_components(mixer.real, directed=False)[0], 1)

    def test_mixer_evolution_is_norm_preserving(self):
        mixer = _adjacent_swap_mixer(_permutation_basis(5))
        rng = np.random.default_rng(0)
        state = rng.normal(size=mixer.shape[0]) + 1j * rng.normal(size=mixer.shape[0])
        state /= np.linalg.norm(state)
        for beta in (0.1, 0.9, 2.7):
            evolved = expm_multiply((-1j * beta) * mixer, state)
            self.assertAlmostEqual(np.linalg.norm(evolved), 1.0, places=10)

    def test_warm_start_state_is_normalised_and_feasible(self):
        basis = _permutation_basis(5)
        reference = tuple(range(5))
        amplitudes = warm_start_state(basis, reference, mass=0.7)
        probs = np.abs(amplitudes) ** 2
        self.assertAlmostEqual(probs.sum(), 1.0, places=12)
        self.assertAlmostEqual(probs[basis.index(reference)], 0.7, places=12)
        # support is the reference plus its n-1 adjacent-swap neighbours
        self.assertEqual(int((probs > 1e-12).sum()), 5)

    def test_cvar_reduces_to_expectation_at_alpha_one(self):
        rng = np.random.default_rng(1)
        probs = rng.random(50)
        probs /= probs.sum()
        costs = rng.random(50) * 10
        self.assertAlmostEqual(_cvar(probs, costs, 1.0), float(probs @ costs), places=12)

    def test_cvar_is_monotone_in_alpha(self):
        rng = np.random.default_rng(2)
        probs = rng.random(200)
        probs /= probs.sum()
        costs = np.sort(rng.random(200) * 10)
        values = [_cvar(probs, costs, a) for a in (0.05, 0.2, 0.5, 1.0)]
        self.assertEqual(values, sorted(values))


class QaoaBehaviourTests(unittest.TestCase):
    def test_depth_zero_reports_the_warm_start_itself(self):
        instance = generate_instance(5, 5)
        result = solve_qaoa(instance, depth=0, seed=5)
        heuristic = solve_heuristic(instance)
        # With no circuit applied, 0.7 of the mass sits on the heuristic schedule.
        self.assertEqual(result.metadata["optimizer_evaluations"], 0)
        self.assertEqual(result.metadata["gamma"], [])
        self.assertEqual(result.metadata["argmax_objective"], heuristic.objective)

    def test_qaoa_output_is_feasible_and_metadata_bounded(self):
        instance = generate_instance(4, 5)
        result = solve_qaoa(instance, depth=1, seed=5, maxiter=3)
        self.assertEqual(sorted(result.order), list(range(4)))
        self.assertGreaterEqual(result.metadata["success_probability"], 0.0)
        self.assertLessEqual(result.metadata["success_probability"], 1.0)
        self.assertTrue(result.metadata["simulator_only"])

    def test_best_of_shots_cannot_be_better_than_exact_optimum(self):
        instance = generate_instance(5, 11)
        result = solve_qaoa(instance, depth=1, seed=11, maxiter=5, alpha=0.1, shots=256)
        # sampling K times and keeping the cheapest cannot beat the true optimum
        exact = solve_exact(instance)
        self.assertGreaterEqual(result.objective + 1e-9, exact.objective)
        self.assertGreaterEqual(
            result.metadata["expected_best_of_shots_objective"] + 1e-9, exact.objective
        )

    def test_cold_start_is_labelled_distinctly(self):
        instance = generate_instance(4, 7)
        warm = solve_qaoa(instance, depth=1, seed=7, maxiter=3, warm_start=True)
        cold = solve_qaoa(instance, depth=1, seed=7, maxiter=3, warm_start=False)
        self.assertTrue(warm.method.startswith("warm_start_"))
        self.assertTrue(cold.method.startswith("cold_start_"))
        self.assertNotEqual(warm.method, cold.method)

    def test_rejects_invalid_arguments(self):
        instance = generate_instance(4, 1)
        for kwargs in ({"depth": -1}, {"alpha": 0.0}, {"alpha": 1.5}, {"shots": 0}):
            with self.assertRaises(ValueError):
                solve_qaoa(instance, **kwargs)


class ReportingTests(unittest.TestCase):
    """Regression guards for the reporting defects found in v0.1.0."""

    ROWS = [
        {"instance": "a", "size": "4", "method": "m1", "objective": "1.0",
         "gap_percent": "0.0", "runtime_seconds": "0.1", "optimal": True,
         "success_probability": "0.0", "argmax_gap_percent": "", "argmax_optimal": ""},
        {"instance": "b", "size": "9", "method": "m1", "objective": "3.0",
         "gap_percent": "10.0", "runtime_seconds": "0.1", "optimal": False,
         "success_probability": "0.0", "argmax_gap_percent": "", "argmax_optimal": ""},
        {"instance": "a", "size": "4", "method": "m2", "objective": "1.0",
         "gap_percent": "0.0", "runtime_seconds": "0.2", "optimal": True,
         "success_probability": "0.0", "argmax_gap_percent": "", "argmax_optimal": ""},
    ]

    def test_common_scope_excludes_instances_a_method_did_not_run(self):
        summary = summarize(self.ROWS)
        self.assertEqual(summary["common_scope_instances"], ["a"])
        # m1 looks worse than m2 only because it ran on a harder instance
        self.assertEqual(summary["full_scope"]["m1"]["mean_gap_percent"], 5.0)
        self.assertEqual(summary["common_scope"]["m1"]["mean_gap_percent"], 0.0)
        self.assertEqual(summary["common_scope"]["m2"]["mean_gap_percent"], 0.0)

    def test_summary_parses_csv_boolean_strings(self):
        rows = [dict(row) for row in self.ROWS]
        rows[0]["optimal"] = "False"
        summary = summarize(rows)
        self.assertEqual(summary["full_scope"]["m1"]["optimal_rate"], 0.0)

    def test_paired_comparison_detects_an_identical_method(self):
        report = paired_comparison(self.ROWS, "m1", "m2")
        self.assertEqual(report["instances"], 1)
        self.assertEqual(report["identical"], 1)
        self.assertEqual(report["challenger_better"], 0)
        self.assertEqual(report["challenger_worse"], 0)


if __name__ == "__main__":
    unittest.main()
