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

The scripts under `verification/compare_<dataset>.py` go one step further.
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

### Phase 1 + Phase 2 + Phase 3 results

| Dataset                                                | Comparisons                                                                       | Result                                          |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- | ----------------------------------------------- |
| marketsplit (binary_linear + binary_unconstrained)     | 312                                                                               | 312 EQUIV                                       |
| network (integer_lp)                                   | 20                                                                                | 20 EQUIV                                        |
| independentset (binary_linear + binary_unconstrained)  | 83                                                                                | 83 EQUIV                                        |
| labs (integer + quadratic_unconstrained)               | 198                                                                               | 198 EQUIV                                       |
| routing (integer_linear)                               | 55                                                                                | 55 EQUIV                                        |
| topology (flow_mip + seidel_linear + seidel_quadratic) | 48                                                                                | 48 EQUIV                                        |
| steiner (integer_linear)                               | 23 (out of 31 declared; stopped early due to eval cost — see below)               | 13 EQUIV, 7 VAR_MISMATCH (drift), 3 MISSING_DAT |
| birkhoff (integer_linear)                              | 80 (out of 800 declared; rest are not in upstream 1.1.0)                          | data drift, see below                           |

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

### Steiner: same data-drift pattern + early-stop

Steiner is the most expensive comparison in this suite. The driver ran
all 14 s020 instances plus 9 s030 instances (23 total) before being
stopped manually; the remaining 4 s030 instances and all 4 s040 instances
were not evaluated. At the time of stop, an s020 evaluation took ~5 min
and the larger s030 evaluations were running 1–2 hours each
(`stp_s030_l4_t3_h1_rs97531` finished at 8395 s ≈ 2.3 h, and
`stp_s030_l4_t4_h0_rs97531` was still running after 6+ hours of
compute).

Of the 23 instances actually compared:

- **13 EQUIV** — variables, constraints, evaluated objective, and
  feasibility all match the deployed `qoblib_v2` reference.
- **7 VAR_MISMATCH** — variable count differs from the registry.
- **3 MISSING_DAT** — declared in `available_instances` but the per-instance
  directory does not exist in upstream 1.1.0 at all.

`verification/investigate_var_mismatch.py` decodes the deployed jm 1
instance's variable subscripts to infer `(nA, nL, nR, nT)` and compares
them to the upstream `.dat` dimensions for every problem case. **Every
single one — including the three MISSING_DAT cases — is upstream data
drift, not a migration bug.** A representative slice:

| Instance                       | upstream nA / nL / nR / nT | jm 1 registry nA / nL / nR / nT | Drifted axes  |
| ------------------------------ | -------------------------- | ------------------------------- | ------------- |
| stp_s020_l2_t3_h3_rs97531      | 3660 / 12 / 12 / 24        | 3660 / 10 / 10 / 16             | nL, nR, nT    |
| stp_s020_l2_t4_h0_rs24098      | 3840 / 6 / 6 / 18          | 3840 / 8 / 8 / 18               | nL, nR        |
| stp_s020_l3_t4_h0_rs24098      | 6160 / 11 / 11 / 27        | 6160 / 9 / 9 / 17               | nL, nR, nT    |
| stp_s020_l3_t4_h2_rs97531      | 6040 / 8 / 8 / 18          | 6040 / 8 / 8 / 14               | nT            |
| stp_s020_l4_t4_h0_rs24098      | 8480 / 7 / 7 / 19          | 8480 / 9 / 9 / 20               | nL, nR, nT    |
| stp_s020_l5_t3_h3_rs24098      | 10596 / 11 / 11 / 14       | 10596 / 11 / 11 / 19            | nT            |
| stp_s020_l5_t4_h0_rs24098      | 10800 / 8 / 8 / 20         | 10800 / 9 / 9 / 20              | nL, nR        |
| stp_s020_l4_t3_h3_rs97531      | upstream missing           | 8300 / 11 / 11 / 18             | (missing)     |
| stp_s020_l5_t4_h3_rs24098      | upstream missing           | 10596 / 10 / 10 / 25            | (missing)     |
| stp_s030_l3_t4_h0_rs97531      | upstream missing           | 14040 / 10 / 10 / 25            | (missing)     |

The crucial observation: **nA (arc count) matches upstream for every
instance**. Only the auxiliary metadata derived from `terms.dat`,
`roots.dat`, and `param.dat` (i.e., the nets / roots / terminals
assignments) differs. The underlying graph topology is identical between
upstream 1.1.0 and the deployed snapshot — only the supplementary node
labelling was regenerated. The migration is correct; the gap is purely
in the deployed dataset's provenance.
