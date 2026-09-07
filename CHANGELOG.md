# Changelog

## 0.2.0

Audit of v0.1.0 found that its quantum arm returned the warm-start heuristic schedule on 9 of 9
benchmark instances - never better, never worse. The release addresses the cause rather than the
symptom.

### Findings this release responds to

- **The p=1 expectation QAOA did nothing.** Minimising expected cost from a concentrated warm start
  is minimised by not mixing, so the optimiser drove `beta` to 0 (measured: 0.023, 0.000, 0.138,
  0.013). On the two instances where the heuristic was suboptimal, success probability was 0.000.
- **The reported success probability of 55.3% measured the warm start, not the circuit.** The
  warm start alone scores 54.4%.
- **The README table compared a heuristic on n=4-16 against a quantum arm on n=4-6.** Restricted to
  n<=6 the two were identical to four decimals, because they returned the same schedules.
- **CI was red.** `ruff check src tests` failed with three E402 errors in `api.py`.

### Added

- CVaR merit function (`--alpha`), which removes the incentive to leave the state alone. Both CVaR
  arms reach 0.00% mean gap and 9/9 optimal on the common scope.
- Best-of-K-shots readout (`--shots`), with the old argmax readout retained in `argmax_*` columns.
- `p=0` arm reporting the warm-start distribution with no circuit applied - the baseline any p>=1
  result must clear.
- Cold-start arms (`--no-warm-start`) isolating the circuit's contribution from the classical hint.
- Uniform random sampler at the same shot budget, the control that keeps a best-of-K readout honest.
- Exact expected best-of-K gap and optimum-return probability on the common scope, avoiding
  conclusions based only on one sampled readout seed.
- OR-Tools CP-SAT baseline (`pip install -e ".[cpsat]"`), agreeing with the exact DP on all 24
  instances.
- `hybrid-schedule compare`, reporting per-instance wins/losses/ties between any two methods.
- `common_scope` / `full_scope` split in `summary.json`; only the former is a comparison.
- Tests for mixer Hermiticity, connectivity, norm preservation, CVaR correctness, and the two
  reporting guards. 7 tests -> 25.

### Changed

- `Job.processing_time` removed. It was never read by any cost term. Instances written by v0.1.0
  still load (`Job.from_dict` ignores it), and the generator retains the corresponding RNG draw so
  that a given `(size, seed)` still produces the identical instance.
- Ruff's rule set is now pinned in `pyproject.toml`. Without `select`, the active rules were
  whatever the installed ruff defaulted to, so CI could turn red on a lint upgrade alone.
- `benchmark` and `solve` expose sizes, seeds, depth, alpha, shots, seed and maxiter.
- Figure split into three panels; one shared y-axis flattened the n<=6 region into a line at zero.
- Random sampling now consumes exactly K samples rather than evaluating an extra identity schedule.
- Unknown job fields are rejected; only the documented legacy `processing_time` key is ignored.

### Unchanged

No claim of quantum advantage, and none is supported. Every quantum arm runs at n<=6, where brute
force takes about 3 ms, and is dominated by CP-SAT everywhere.
