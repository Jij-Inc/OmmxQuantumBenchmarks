# Discrete blackbox optimization

The `blackbox` collection adds the discrete suite developed in
[JijZeptWorkflow](https://github.com/Jij-Inc/JijZeptWorkflow/tree/5e2766138c4a84e37fd506aebc28ee16d62cd181/benchmarks/discrete).
It supplies finite domains, candidate evaluators, pinned instance inputs and shared
initial observations. It does not require JijZeptWorkflow or any optimizer package.

## Collection

| Family | Settings | Domain / objective | Distribution |
| --- | ---: | --- | --- |
| LABS | 6 | 16/32/64 bits; energy and negative merit factor | Included |
| QUBO controls | 9 | 16/32/64 bits × instance seeds 0/1/2 | Included; OMMX export |
| Contamination | 9 | 25 binary stages × 3 seeds × 3 penalties | Included |
| Pest control | 3 | 25 variables, 5 categories; reset RNG per evaluation | Included |
| Ising sparsification | 9 | 24 binary edge choices on a 4×4 lattice × 3 seeds × 3 penalties | Included |
| IOH/PBO | 150 | F1–F25 × instances 1/2/3 × dimensions 25/100 | Requires `pbo` extra |
| FMIRA fitness tables | 3 | Fluorescence 13 bits; Ribozyme and Influenza_fluB 16 bits | Explicit local acquisition |

The catalog contains **189 settings**: 36 work offline with the base package,
150 use `ioh==0.3.22`, and 3 require external tables. `available_instances` lists
all catalog IDs, including optional and external settings. It does not indicate
that their dependencies or tables are installed.

## Evaluate a candidate

All evaluators minimize a scalar objective. Fitness and PBO native scores are
negated; other scores keep their original direction. Candidates must be finite
integer vectors within the declared cardinalities; invalid values are rejected.

```python
from ommx_quantum_benchmarks.blackbox import DiscreteBlackbox, initial_design

benchmarks = DiscreteBlackbox()
material = benchmarks("labs-16-negative-merit")
print(material.entry["cardinalities"])
print(material.evaluate([0] * 16))

# The same 20 observed candidates can initialize every compared method.
data = benchmarks.initial_data("labs-16-negative-merit", seed=0)
assert data["count"] == 20
for candidate, score in zip(data["candidates"], data["objectives"], strict=True):
    assert abs(material.evaluate(candidate) - score) < 1e-10

# Generate a different reproducible design without consulting unobserved scores.
fresh = initial_design(material, count=20, seed=7)
assert len(fresh["candidates"]) == 20
```

`Material.entry` records the domain, instance seed, variant, source, optimum
status, and asset/evaluator hashes. `Material.identity` identifies those exact
settings. Initial design seeds 0/1/2 are independent of the instance and optimizer
seeds. Count the shared 20 observations toward the total evaluation budget.
Keep full tables and optima on the evaluator/scoring side; optimizers receive
only queried observations. Numerical replay across library/platform versions is
not guaranteed to be bit identical; record the NumPy/IOH versions and identities.

## QUBO in OMMX

QUBO controls can also be evaluated by ordinary OMMX tools:

```python
qubo = benchmarks("qubo-16-s0")
instance = qubo.to_ommx()
x = [0, 1] * 8
solution = instance.evaluate({i: float(value) for i, value in enumerate(x)})
assert abs(solution.objective - qubo.evaluate(x)) < 1e-10
```

The blackbox collection is distinct from the QOBLIB `Instance, Solution` API.
Simulation and lookup scores are accessed through `evaluate`; `to_ommx()` supports
only QUBO controls. No known optimum is claimed for the synthetic, COMBO or PBO
settings. FMIRA optima are exact only over their supplied finite tables.

## Optional PBO

```bash
pip install 'ommx-quantum-benchmarks[pbo]'
```

```python
pbo = benchmarks("pbo-f1-i1-d25")
assert pbo.evaluate([1] * 25) == -25.0
```

F4–F17 include W-model transformations. Native IOH optimum metadata is not
presented as a verified exact optimum for every problem.

## Explicit FMIRA acquisition

The pinned FMIRA snapshot has no repository license recorded by this collection.
Its raw tables, normalized fitness arrays and initial observations are excluded
from the distribution. The following explicit operation downloads and verifies
one table pair into a user-selected directory, without executing downloaded code:

```python
from pathlib import Path
from ommx_quantum_benchmarks.blackbox import fetch_fitness

fitness = fetch_fitness(Path("benchmark_data/fmira"), "fitness-Fluorescence")
score = fitness.evaluate([0] * 13)
local_design = initial_design(fitness, count=20, seed=0)
```

Acquisition does not grant redistribution permission. Row pairing is preserved
before sorting by MSB-first binary code, including the shuffled Ribozyme and
Influenza rows. Values are negated without rescaling, imputation or jitter.
Influenza has a large lowest-value plateau; measurement units, censoring and
experimental uncertainty are not independently established by this snapshot.

## Sources and variants

- [COMBO](https://github.com/QUVA-Lab/COMBO/tree/9529eabb86365ce3a2ca44fff08291a09a853ca2):
  the BSD notice is included in `blackbox/COMBO-LICENSE.txt`. Contamination retains
  the 25-stage, 100-simulation soft penalty score; it is not a hard constraint.
  Pest resets a local RandomState for each call and is a deterministic variant,
  not a recreation of the original noisy experiment. Ising retains the KL score
  and edge mapping but freezes NumPy-generated interactions rather than Torch RNG
  instances with the same numbered seed.
- [FMIRA](https://github.com/tucs7/FMIRA/tree/7d20bb6754aed252adb5343d0552a8b791c83e98):
  three biological fitness landscapes; source URLs, sizes and SHA-256 hashes are
  pinned in `blackbox/sources.lock.json`.
- [IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter): `ioh==0.3.22`;
  [PBO definitions](https://iohprofiler.github.io/IOHproblem/PBO).
- [LABS FMA example](https://arxiv.org/abs/2507.21024): energy and negative merit
  factor have separate IDs and must be reported separately.

For comparisons, report best-so-far scores, unique evaluations, total and fit /
acquisition time, multiple instance and optimizer seeds, and gaps only when an
optimum is verified. A smoke test is not evidence for ranking optimizers.
