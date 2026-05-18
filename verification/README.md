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

```
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
