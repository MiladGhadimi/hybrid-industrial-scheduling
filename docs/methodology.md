# Methodology

## 1. Scope

This benchmark studies a single-machine production-sequencing layer. Every job must appear exactly once. The model captures slot-dependent lateness and electricity costs plus sequence-dependent setup costs between product families.

The scope is intentionally narrower than a job-shop scheduling system. Machine eligibility, maintenance windows, workforce constraints, and stochastic processing times are left for extensions so that the exact, heuristic, and quantum formulations remain independently auditable.

## 2. Objective

Let `J={0,...,n-1}` be the jobs and let `pi` be a permutation. Job `pi[t]` occupies slot `t`. Each job has due slot `d_j`, priority `q_j`, energy demand `e_j`, and family `f_j`. Each slot has price `p_t`, while `s(a,b)` is the changeover cost from family `a` to `b`.

Slots are unit length. There is no processing-time term, and jobs carry no duration field: a
duration that does not enter the objective would suggest the model captures something it does not.
Sequence-dependent durations are a named roadmap item, not a silent omission.

```text
C(pi) = sum_t [lambda_L q_pi[t] max(0, t-d_pi[t])
                 + lambda_E e_pi[t] p_t]
        + sum_(t=0)^(n-2) lambda_C s(f_pi[t], f_pi[t+1]).
```

The default weights are `lambda_L=1.0`, `lambda_E=0.35`, and `lambda_C=0.7`. They are stored in every instance and can be changed without modifying solver code.

## 3. Exact baseline

For subset `S` and final job `j in S`, define `D[S,j]` as the cheapest sequence containing exactly the jobs in `S` and ending in `j`. The slot of `j` is `|S|-1`.

```text
D[{j},j] = placement(j,0)
D[S,j] = placement(j,|S|-1)
         + min_(i in S\{j}) [D[S\{j},i] + transition(i,j)].
```

Backpointers recover the optimal permutation. Complexity is `O(n^2 2^n)` time and `O(n 2^n)` memory.

## 3b. CP-SAT baseline

Greedy + pair-swap is a fast baseline, not a serious solver, and a benchmark whose only strong
classical arm is the one defining optimality cannot be checked against anything. The CP-SAT model
uses assignment variables `x[j][t]`, one `ExactlyOne` per job and per slot, and family indicators
`in_family[a][t]` that linearise the sequence-dependent changeover term through one Boolean per
(a, b, t) triple. Costs are scaled by 10^6 to integers.

CP-SAT proves optimality on all 24 benchmark instances and agrees with the dynamic program on every
one. Because the two solvers share only the `SchedulingInstance` cost primitives and not their
search, that agreement is meaningful evidence that the objective is implemented consistently.

## 4. Classical heuristic

The construction phase chooses the next job using slot placement cost, transition cost from the current last job, and an urgency correction. A deterministic tie-break makes runs reproducible. Pair-swap local search then accepts the best improving exchange until reaching a local optimum.

The implementation favours transparency over highly tuned metaheuristics. It provides a credible fast baseline while keeping the decision process easy to inspect.

## 5. Constraint-preserving QAOA

The feasible computational basis contains the `n!` job permutations, not all binary strings of a one-hot encoding. The cost Hamiltonian is diagonal in this basis. The mixer is the adjacency matrix of the graph connecting permutations that differ by one adjacent swap.

Because every mixer edge connects two permutations, the state remains feasible. This avoids selecting a penalty coefficient for one-hot constraints.

The warm-start state places probability 0.7 on the schedule returned by greedy local search and spreads probability 0.3 uniformly across its adjacent-swap neighbours. A `--no-warm-start` variant uses the uniform superposition instead, which is what isolates the circuit's contribution from the classical hint's. For depth `p`, alternating cost and mixer evolutions are applied:

```text
|psi(gamma,beta)> = product_l exp(-i beta_l H_M) exp(-i gamma_l H_C) |psi_0>.
```

At `p=0` no evolution is applied and the reported distribution is `|psi_0>` itself. This arm is run
in the benchmark deliberately: it is the baseline that separates "the circuit helped" from "the
warm start was already good".

SciPy's sparse `expm_multiply` applies the mixer evolution without forming a dense matrix exponential.

### 5b. Choice of merit function

Differential evolution optimises CVaR at level `alpha`, the mean cost of the cheapest `alpha` tail
of the measured distribution. `alpha = 1` recovers the plain expectation used in v0.1.0.

Expectation is a poor choice for this warm-start configuration on these instances. If `|psi_0>`
already places most of its mass on a
near-optimal permutation, then any mixing redistributes probability toward the bulk of the
spectrum, which is worse on average. The expectation therefore *increases* under mixing, and the
optimiser's best move is `beta -> 0`: the variational problem is minimised by not running the
circuit. Measured `beta` values on the benchmark instances were 0.023, 0.000 (exactly at the
boundary), 0.138 and 0.013, and the arm returned its own warm-start schedule on 9 of 9 instances.

CVaR at small `alpha` is insensitive to the bulk and rewards any mass placed on excellent
permutations, so it does not penalise mixing. Measured `beta` values under CVaR are an order of
magnitude larger (0.49-1.04).

This is an observation about the merit function on these instances. It is not a proof that
expectation-based warm-start QAOA cannot help in general, and no claim of quantum speedup follows
from any arm here.

### 5c. Readout

The reported schedule is the **best of K measured samples**, which is how a sampler is used in
practice. v0.1.0 reported the single most probable permutation (argmax); that readout is retained
in the `argmax_*` columns because it is not merely a weaker choice but an actively misleading one
here. A distribution that spreads mass onto the optimum scores well on best-of-K and badly on
argmax, so the CVaR arms register 0.00% mean gap under best-of-shots and 11-40% under argmax. The
argmax readout cannot observe the improvement it is meant to measure.

For the same reason, success probability is informative but not sufficient as a standalone
quality score. In these runs it decreases (0.553 -> 0.369 -> 0.146) while the expected best-of-128
result improves, because concentration and optimum-finding are different things. Under 128 shots,
a success probability of 0.146 yields the
optimum with probability `1 - (1 - 0.146)^128 > 0.9999`.

A best-of-K readout also needs its own control, or the shot budget does the work and the sampler
takes the credit. The benchmark therefore includes a uniform random sampler at the same budget.
With 128 shots this baseline solves n=4 with very high probability (24 feasible permutations) and
has an exact expected 4.75% mean gap with a 60.5% probability of returning an optimum across the
nine common-scope instances, while
collapsing to 47.8% mean gap over the full n=4-16 range.

## 6. Metrics

All aggregate tables are reported on two scopes. `full_scope` aggregates each method over the
instances it actually ran on; `common_scope` restricts every method to the instances *all* methods
solved. Only the second is a comparison. v0.1.0's summary table placed a heuristic evaluated on
n=4-16 (3.61% mean gap) beside a quantum arm evaluated on n=4-6 (4.11%), which invited the reading
that the two arms had different quality profiles; restricted to n<=6 the heuristic's mean gap is
4.108%, identical to the quantum arm's to four decimals, because they returned the same schedules.

`paired_comparison` reports per-instance wins, losses and ties between any two methods. A mean-gap
table cannot show that one method never changes another's answer; a paired count can, and does.

- **Objective:** evaluated by the shared cost model.
- **Optimality gap:** `100 * (candidate - optimum) / |optimum|`.
- **Optimal rate:** fraction of runs whose reported schedule reaches the exact objective within numerical tolerance.
- **Runtime:** wall-clock solver time; environment-dependent and reported only as reference evidence.
- **Success probability:** QAOA probability mass on all globally optimal permutations.
- **Expected objective:** expectation of the original, unnormalized scheduling cost.

## 7. Reproducibility

Instances use deterministic local random generators. The QAOA optimizer receives the same seed as the corresponding instance. Benchmark rows preserve size, seed, method, objective decomposition, runtime, gap, success probability, and expected objective.

Correctness tests verify:

- generated-instance reproducibility;
- permutation validation and cost decomposition;
- exact-DP agreement with brute force for n = 4, 5, 6 across two seeds, including that the
  returned order achieves the reported objective;
- heuristic feasibility and consistency with the exact lower bound;
- QAOA feasibility and probability metadata bounds;
- mixer Hermiticity (`max|M - M^dagger| = 0`) and connectivity of the permutation graph;
- norm preservation of the mixer evolution, so that the renormalisation in `probabilities()` is
  numerical hygiene rather than a patch over a non-unitary step;
- CVaR reducing exactly to the expectation at `alpha = 1`, and monotonicity in `alpha`;
- `p=0` reproducing the warm-start distribution with zero optimizer evaluations;
- backward-compatible loading of v0.1.0 instances carrying the removed `processing_time` field;
- the two reporting guards: that `common_scope` excludes instances a method skipped, and that
  `paired_comparison` detects a method identical to its baseline.

## 8. Interpretation and limitations

The benchmark does not establish quantum advantage, and nothing in it should be read as evidence
toward one. Concretely:

- Every quantum arm runs at n <= 6, where enumerating all `n!` permutations costs about 3 ms and
  yields the exact optimum. The quantum arms are strictly dominated by brute force at the sizes at
  which they run, and by CP-SAT everywhere.
- `solve_qaoa` itself enumerates all `n!` permutation costs, because reporting success probability
  requires knowing the optimum. Its reported runtime is optimiser overhead on top of an
  already-solved problem and is not a solve time. No runtime here supports a scaling claim.
- The CVaR arms reaching 9/9 optimal is a statement about a 9-instance, n <= 6 sample under a
  128-shot readout. It shows that the v0.1.0 result was limited by the merit function and the
  readout rather than by the formulation. It does not show that the approach scales, and the
  cold-start arms (2.15% mean gap at p=3) indicate that most of the warm arms' performance comes
  from the classical schedule they start from.
- Statevector simulation assumes noiseless evolution and exact expectation values. A device
  implementing this mixer would need a Cayley-graph walk on `S_n`; no resource estimate is offered
  and none should be inferred.

Warm-start QAOA depends on a classical schedule and its results should be interpreted as hybrid
refinement evidence at small scale.

The synthetic instances are controlled experiments, not customer data. Industrial deployment would require calibration of weights and setup matrices, additional operational constraints, larger classical solvers, uncertainty handling, and validation with domain experts.
