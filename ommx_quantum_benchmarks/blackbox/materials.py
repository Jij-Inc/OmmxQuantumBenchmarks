"""Discrete blackbox objectives and reproducible initial designs.

COMBO-derived scores retain the BSD notice in COMBO-LICENSE.txt.
"""

from __future__ import annotations

import hashlib
import json
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# COMBO-derived scores below retain the BSD notice in COMBO-LICENSE.txt.
# Upstream mapping/score semantics are pinned in sources.lock.json.


def validate_candidate(values: Any, sizes: list[int]) -> Any:
    x = np.asarray(values)
    if x.shape != (len(sizes),) or x.dtype.kind not in "biuf":
        raise ValueError("candidate must be a numeric vector of the declared dimension")
    if not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise ValueError("candidate values must be finite integers")
    if np.any(x < 0) or np.any(x >= np.asarray(sizes)):
        raise ValueError("candidate is outside the semantic domain")
    return x.astype(np.int64)


def labs(x: Any, mode: str) -> float:
    spins = 2 * x - 1
    energy = sum(int(spins[:-k] @ spins[k:]) ** 2 for k in range(1, len(x)))
    if mode == "energy":
        return float(energy)
    if mode == "negative-merit":
        return -(len(x) ** 2) / (2 * energy)
    raise ValueError(f"unknown LABS mode {mode!r}")


def contamination(x: Any, arrays: dict[str, Any], penalty: float) -> float:
    spread, restore = arrays["spread"], arrays["restore"]
    z = arrays["initial"]
    below = []
    for i, control in enumerate(x):
        z = spread[i] * (1 - control) * (1 - z) + (1 - restore[i] * control) * z
        below.append(np.mean(z < 0.1))
    return float(np.sum(x - (np.asarray(below) - 0.95)) + penalty * x.sum())


def pest_control(x: Any, seed: int) -> float:
    """Upstream score with a local reset stream; no global RNG mutation."""
    rng = np.random.RandomState(seed)
    beta = {1: 2 / 7, 2: 3 / 7, 3: 3 / 7, 4: 5 / 7}
    tolerance = {1: 1 / 7, 2: 2.5 / 7, 3: 2 / 7, 4: 0.5 / 7}
    price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    pest = rng.beta(1, 30, 100)
    total = 0.0
    for choice in x:
        spread = rng.beta(1, 17 / 3, 100)
        total += float(np.mean(pest > 0.1))  # upstream measures BEFORE this intervention
        if choice:
            control = rng.beta(1, beta[choice], 100)
            pest = (1 - control) * pest
            beta[choice] += tolerance[choice] / len(x)
            total += price[choice] * (1 - discount[choice] * np.sum(x == choice) / len(x))
        else:
            pest = spread * (1 - pest) + pest
    return float(total)


def _logsumexp(values: Any) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.exp(values - maximum).sum()))


@lru_cache(maxsize=1)
def _ising_features() -> Any:
    codes = np.arange(1 << 16, dtype=np.int64)
    spins = (2 * ((codes[:, None] >> np.arange(15, -1, -1)) & 1) - 1).reshape(-1, 4, 4)
    horizontal = (spins[:, :, :-1] * spins[:, :, 1:]).reshape(-1, 12)
    vertical = (spins[:, :-1, :] * spins[:, 1:, :]).reshape(-1, 12)
    features = np.concatenate([horizontal, vertical], axis=1).astype(float)
    features.flags.writeable = False
    return features


class Material:
    """One identified minimization objective and its finite candidate domain.

    Load instances through :class:`DiscreteBlackbox`. ``entry`` contains domain,
    source, variant and optional optimum metadata; ``evaluate`` returns a float.
    """

    def __init__(self, root: Path, entry: dict[str, Any]):
        self.root = root
        self.entry = json.loads(json.dumps(entry, allow_nan=False))
        expected = self.entry.get("implementation_sha256")
        actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if expected is not None and expected != actual:
            raise ValueError("evaluator implementation differs from the catalog")

    @property
    def identity(self) -> str:
        return hashlib.sha256(json.dumps(self.entry, sort_keys=True).encode()).hexdigest()

    @cached_property
    def arrays(self) -> dict[str, Any]:
        if "asset" not in self.entry:
            return {}
        path = (self.root / self.entry["asset"]).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("asset path escapes the dataset root")
        if not path.exists():
            raise FileNotFoundError(
                f"Asset for {self.entry['id']} is not installed. "
                "For FMIRA, explicitly use fetch_fitness(root, identifier)."
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.entry["asset_sha256"]:
            raise ValueError(f"prepared asset hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for array in arrays.values():
            array.flags.writeable = False
        return arrays

    @cached_property
    def _ising_original(self) -> tuple[Any, float]:
        energy = 2 * (_ising_features() @ self.arrays["weights"])
        log_z = _logsumexp(energy)
        expected = np.exp(energy - log_z) @ _ising_features()
        return expected, log_z

    @cached_property
    def _pbo(self) -> Any:
        import importlib
        import importlib.metadata

        try:
            ioh: Any = importlib.import_module("ioh")
        except ImportError as error:
            raise ImportError("PBO requires pip install 'ommx-quantum-benchmarks[pbo]'") from error

        if importlib.metadata.version("ioh") != self.entry["ioh_version"]:
            raise ValueError("ioh version differs from prepared catalog")
        return ioh.get_problem(
            self.entry["function_id"],
            self.entry["instance"],
            len(self.entry["cardinalities"]),
            problem_class=ioh.ProblemClass.PBO,
        )

    def evaluate(self, values: Any) -> float:
        x = validate_candidate(values, self.entry["cardinalities"])
        score = self._score(x)
        if not np.isfinite(score):
            raise ValueError("evaluator returned a nonfinite objective")
        return score

    def _score(self, x: Any) -> float:
        family = self.entry["family"]
        if family == "fitness":
            code = int(x @ (1 << np.arange(len(x) - 1, -1, -1)))
            return -float(self.arrays["fitness"][code])
        if family == "labs":
            return labs(x, self.entry["mode"])
        if family == "qubo":
            a = self.arrays
            return float(x @ a["q"] @ x + a["linear"] @ x + a["constant"])
        if family == "contamination":
            return contamination(x, self.arrays, self.entry["penalty"])
        if family == "pest":
            return pest_control(x, self.entry["instance_seed"])
        if family == "ising":
            horizontal_indices = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 22, 23]
            vertical_indices = [i for i in range(24) if i not in horizontal_indices]
            mask = np.concatenate([x[horizontal_indices], x[vertical_indices]])
            weights = self.arrays["weights"]
            expected, original_log_z = self._ising_original
            log_z = _logsumexp(2 * (_ising_features() @ (weights * mask)))
            kl = 2 * ((weights * (1 - mask)) @ expected) + log_z - original_log_z
            return float(kl + self.entry["penalty"] * x.sum())
        if family == "pbo":
            self._pbo.reset()  # evaluation bookkeeping never changes the static objective
            return -float(self._pbo(x.tolist()))
        raise ValueError(f"unknown benchmark family {family!r}")

    def to_ommx(self) -> Any:
        """Export a QUBO control as an OMMX instance with the same objective.

        Other families require an external evaluator and cannot be exported by
        this API. No placeholder polynomial is substituted for a blackbox score.
        """
        if self.entry["family"] != "qubo":
            raise ValueError("OMMX export is available only for QUBO controls")
        from ommx.v1 import DecisionVariable, Instance

        x = [
            DecisionVariable.binary(i, name="x", subscripts=[i])
            for i in range(len(self.entry["cardinalities"]))
        ]
        arrays = self.arrays
        objective = float(arrays["constant"])
        for i, variable in enumerate(x):
            objective += float(arrays["linear"][i]) * variable
            for j, other in enumerate(x):
                objective += float(arrays["q"][i, j]) * variable * other
        return Instance.from_components(
            decision_variables=x,
            objective=objective,
            constraints=[],
            sense=Instance.MINIMIZE,
        )


def load_material(root: Path, identifier: str) -> Material:
    catalog = json.loads((root / "catalog.json").read_text())
    if catalog["schema_version"] != 1:
        raise ValueError("unsupported catalog schema")
    entries = [entry for entry in catalog["entries"] if entry["id"] == identifier]
    if len(entries) != 1:
        raise KeyError(f"material id must match exactly one entry: {identifier}")
    return Material(root, entries[0])


def initial_design(material: Material, *, count: int, seed: int) -> dict[str, Any]:
    """Sample unique semantic candidates without consulting unobserved fitness."""
    import math

    sizes = material.entry["cardinalities"]
    if type(count) is not int or not 0 <= count <= math.prod(sizes):
        raise ValueError("initial count exceeds the finite domain or is negative")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("initial design seed must be a uint32")
    rng = np.random.default_rng(seed)
    seen: set[tuple[int, ...]] = set()
    candidates: list[list[int]] = []
    while len(candidates) < count:
        candidate = tuple(int(rng.integers(size)) for size in sizes)
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(list(candidate))
    return {
        "id": material.entry["id"],
        "material_identity": material.identity,
        "design_seed": seed,
        "candidates": candidates,
        "objectives": [material.evaluate(x) for x in candidates],
        "count": count,
        "sampling": "NumPy default_rng, uniform per coordinate, without replacement",
        "numpy_version": np.__version__,
    }
