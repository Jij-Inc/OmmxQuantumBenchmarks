# JijModeling 1 ↔ 2 equivalence verification

This directory contains the harness used to confirm that the JijModeling 2
migration of every `qoblib/*/models/*/model.py` produces a semantically
equivalent OMMX instance to the original JijModeling 1 implementation.

## How it works

1. A git worktree of the pre-migration `main` (commit `de75506`) is checked
   out at `/tmp/jm1_baseline` with `jijmodeling<2` + `minto<2` installed.
2. `equiv_runner.py` loads a `model.py` from a given root, builds the
   `Problem`, evaluates it against a synthetic `instance_data`, and writes a
   JSON summary plus the raw OMMX bytes.
3. `equiv_drive.py` runs the runner under both environments for each of the
   13 model variants, then diffs the JSON summaries.

The driver distinguishes:

- **Hard differences**: changes to the actual mathematical content (sense,
  variable set, constraint count per name, evaluated objective, feasibility
  flag).
- **Soft differences**: changes only to OMMX metadata annotations such as
  the `subscripts` tags attached to constraint families.

The migration is considered correct if the harness reports **zero hard
differences** across all models. A `SOFT-DIFF` row is acceptable: it means
the constraint counts and evaluations match but JijModeling 2 attaches
richer metadata.

## Reproducing the run

Requires `uv` and a git worktree set up at `/tmp/jm1_baseline`:

```bash
git worktree add /tmp/jm1_baseline de75506
cd /tmp/jm1_baseline
sed -i.bak 's/"jijmodeling>=1.13.0"/"jijmodeling>=1.13.0,<2.0.0"/' pyproject.toml
sed -i.bak2 's/"minto>=2.0.0"/"minto>=1.3.0,<2.0.0"/' pyproject.toml
uv sync   # picks up jijmodeling 1.14.x and minto 1.x
```

Then from the migrated repo root:

```bash
uv run --project . python verification/equiv_drive.py
```

The driver exits 0 if and only if there are no hard differences.

## Result on `feat/support-jijmodeling-2`

```text
EQUIV      labs_integer
EQUIV      labs_qubo
EQUIV      marketsplit_bl
EQUIV      marketsplit_bu
EQUIV      birkhoff
SOFT-DIFF  mis_bl
  - constraint subscript annotations differ ...  (semantic check passed)
EQUIV      mis_bu
EQUIV      network
EQUIV      routing
EQUIV      topo_flow
EQUIV      topo_sq
EQUIV      topo_sl
EQUIV      steiner

0 hard differences across 13 models
```

The `mis_bl` soft difference reflects that JijModeling 1 emits
`('no_adjacent', ())` for all 3 edge constraints (no subscript info), while
JijModeling 2 emits `('no_adjacent', (0,))`, `('no_adjacent', (1,))`,
`('no_adjacent', (2,))`. Both formulations encode the same three
inequalities `x[u] + x[v] <= 1`, as confirmed by the sample-solution check
(objective `2.0`, feasible `True` on both sides).

## Registry comparison: deployed `qoblib_v2` ↔ jm 2 regeneration

The scripts `compare_marketsplit.py`, `compare_network.py`,
`compare_independentset.py`, and `compare_birkhoff.py` go one step further.
Instead of running JijModeling 1 ourselves, they pull the deployed OMMX
instance for each `(model, instance)` pair directly from the live
`qoblib_v2` registry via `<Dataset>().__call__(model, instance)`, then:

1. Rebuild `instance_data` from the upstream qoblib data files at
   `/tmp/qoblib_upstream/...` (sparse-cloned from the original repository).
2. Build the migrated jm 2 `Problem` and evaluate it.
3. Compare the variable set (by `(name, subscripts)`), the constraint count
   per name, and the evaluated objective + feasibility of the deployed
   reference solution after re-mapping its variable IDs.

Each script can be run with `uv run --project . python
verification/compare_<dataset>.py` from the repo root once `/tmp/qoblib_upstream`
is set up (see `Reproducing the run` above, but pointing at the relevant
subdirectory of <https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library>).

### Phase 1 + Phase 2 results

| Dataset                                               | Comparisons                                              | Result                |
| ----------------------------------------------------- | -------------------------------------------------------- | --------------------- |
| marketsplit (binary_linear + binary_unconstrained)    | 312                                                      | 312 EQUIV             |
| network (integer_lp)                                  | 20                                                       | 20 EQUIV              |
| independentset (binary_linear + binary_unconstrained) | 83                                                       | 83 EQUIV              |
| labs (integer + quadratic_unconstrained)              | 198                                                      | 198 EQUIV             |
| routing (integer_linear)                              | 55                                                       | 55 EQUIV              |
| birkhoff (integer_linear)                             | 80 (out of 800 declared; rest are not in upstream 1.1.0) | data drift, see below |

For each EQUIV case the deployed reference solution evaluates to the same
objective and the same `feasible` flag against the jm 2 instance, confirming
that the migrated model is semantically equivalent to whatever the original
jm 1 model produced when `qoblib_v2` was uploaded.

### Birkhoff: upstream data drift

The deployed `qoblib_v2` artifacts for birkhoff were generated from an
earlier upstream snapshot whose `qbench_*.json` content differs from the
1.1.0 release at <https://git.zib.de/qopt/qoblib-quantum-optimization-benchmarking-library>.
For example, `bhD-3-001` in the registry encodes a doubly stochastic matrix
with first row `(197, 275, 528)`, whereas the corresponding entry in
upstream 1.1.0 has first row `(156, 580, 264)`. Because the constraint
constants are baked into the OMMX from `instance_data["A"]`, the
registry instance and the freshly-evaluated jm 2 instance disagree on the
right-hand sides of the `c2` family — even though both formulations are
algebraically identical.

The migration itself is unaffected: the offline equivalence harness above
already shows that running jm 1 and jm 2 against the *same* synthetic
birkhoff data yields EQUIV results (no hard differences). Reproducing
end-to-end equivalence against the registry would require regenerating
`qoblib_v2` from the current upstream snapshot.
