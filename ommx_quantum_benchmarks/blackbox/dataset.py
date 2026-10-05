"""Packaged catalog and explicit acquisition of external fitness tables."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .materials import Material, load_material

DATA_ROOT = Path(__file__).resolve().parent / "data"


class DiscreteBlackbox:
    """189 discrete minimization settings identified by stable instance names.

    36 synthetic/COMBO settings are available offline; 150 PBO settings require
    the ``pbo`` extra. Three FMIRA fitness tables require explicit acquisition
    with :func:`fetch_fitness`; their data and observations are not redistributed.
    """

    def __init__(self) -> None:
        self._catalog = json.loads((DATA_ROOT / "catalog.json").read_text())
        if self._catalog["schema_version"] != 1:
            raise ValueError("unsupported catalog schema")

    @property
    def available_instances(self) -> dict[str, list[str]]:
        """Catalog IDs grouped by family, including optional/external settings."""
        result: dict[str, list[str]] = {}
        for entry in self._catalog["entries"]:
            result.setdefault(entry["family"], []).append(entry["id"])
        return result

    def __call__(self, identifier: str) -> Material:
        """Load an identified setting; fitness data must be acquired separately."""
        material = load_material(DATA_ROOT, identifier)
        if material.entry["family"] == "fitness":
            raise FileNotFoundError(
                "FMIRA tables are not redistributed; use fetch_fitness(root, identifier) explicitly"
            )
        return material

    def initial_data(self, identifier: str, *, seed: int = 0) -> dict[str, Any]:
        """Return 20 shared candidate/score pairs for design seed 0, 1 or 2.

        These are observations, not additional free evaluations. Include them in
        the total budget. FMIRA observations are generated locally instead.
        """
        material = self(identifier)
        if type(seed) is not int or seed not in (0, 1, 2):
            raise ValueError("packaged design seed must be 0, 1 or 2")
        data = json.loads((DATA_ROOT / "initial_designs.json").read_text())
        matches = [
            item
            for item in data["designs"]
            if item["id"] == identifier and item["design_seed"] == seed
        ]
        if len(matches) != 1 or matches[0]["material_identity"] != material.identity:
            raise ValueError("initial design does not match the material identity")
        return matches[0]


def fetch_fitness(root: Path, identifier: str) -> Material:
    """Download one pinned FMIRA table into a user-selected directory.

    This is an explicit network operation. Source hashes and full binary-domain
    coverage are verified before normalization. No source code is executed.
    Acquisition does not grant permission to redistribute the tables.
    """
    catalog = json.loads((DATA_ROOT / "catalog.json").read_text())
    matches = [
        entry
        for entry in catalog["entries"]
        if entry["id"] == identifier and entry["family"] == "fitness"
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown fitness setting: {identifier}")
    entry = matches[0]
    name = identifier.removeprefix("fitness-")
    root = Path(root).resolve()
    lock = json.loads((DATA_ROOT.parent / "sources.lock.json").read_text())
    for kind in ("vectors", "fitness"):
        source_path = f"data/{kind}_{name}.txt"
        item = next(
            item
            for item in lock["files"]
            if item["source"] == "fmira" and item["path"] == source_path
        )
        path = root / "raw" / source_path
        if path.exists():
            raw = path.read_bytes()
        else:
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                raw = response.read()
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"source hash mismatch: {source_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    dimension = len(entry["cardinalities"])
    x = np.loadtxt(root / "raw/data" / f"vectors_{name}.txt", ndmin=2)
    y = np.loadtxt(root / "raw/data" / f"fitness_{name}.txt", ndmin=1)
    expected = 1 << dimension
    if x.shape != (expected, dimension) or y.shape != (expected,):
        raise ValueError("unexpected fitness table shapes")
    if not np.isfinite(x).all() or not np.isin(x, (0, 1)).all() or not np.isfinite(y).all():
        raise ValueError("fitness table must contain finite binary vectors and finite scores")
    codes = x.astype(np.int64) @ (1 << np.arange(dimension - 1, -1, -1))
    if np.unique(codes).size != expected:
        raise ValueError("duplicate vectors or incomplete fitness table")
    ordered = np.empty(expected, dtype=np.float64)
    ordered[codes] = y
    path = root / entry["asset"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fitness=ordered)
    # The pinned input table defines content; zip encodings can vary by NumPy version.
    entry["asset_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    material = Material(root, entry)
    if -float(ordered.max()) != entry["optimum"]:
        raise ValueError("fitness table optimum differs from the catalog")
    return material
