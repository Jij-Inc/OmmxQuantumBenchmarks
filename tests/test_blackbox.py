from __future__ import annotations

import hashlib
import importlib.metadata
import json

import numpy as np
import pytest
from ommx_quantum_benchmarks.blackbox import (
    DiscreteBlackbox,
    fetch_fitness,
    initial_design,
)
from ommx_quantum_benchmarks.blackbox.dataset import DATA_ROOT
from ommx_quantum_benchmarks.blackbox.materials import Material, labs


def test_catalog_and_excluded_fitness_data():
    dataset = DiscreteBlackbox()
    assert {key: len(ids) for key, ids in dataset.available_instances.items()} == {
        "fitness": 3,
        "labs": 6,
        "qubo": 9,
        "contamination": 9,
        "pest": 3,
        "ising": 9,
        "pbo": 150,
    }
    for identifier in dataset.available_instances["fitness"]:
        assert not (DATA_ROOT / "prepared" / f"{identifier}.npz").exists()
        with pytest.raises(FileNotFoundError, match="not redistributed"):
            dataset(identifier)
    designs = json.loads((DATA_ROOT / "initial_designs.json").read_text())["designs"]
    assert len(designs) == 186 * 3
    assert not any(item["id"].startswith("fitness-") for item in designs)
    with pytest.raises(KeyError):
        dataset("missing")


@pytest.mark.parametrize(
    "candidate",
    [[0] * 15, [0.5] * 16, [2] * 16, [np.nan] * 16, [np.inf] * 16, [[0] * 16], ["0"] * 16],
)
def test_invalid_candidates_are_rejected(candidate):
    with pytest.raises(ValueError):
        DiscreteBlackbox()("labs-16-energy").evaluate(candidate)


def test_labs_known_scores_and_initial_design():
    assert labs(np.array([1, 1, 1, 0]), "energy") == 2
    assert labs(np.array([1, 1, 1, 0]), "negative-merit") == -4
    material = DiscreteBlackbox()("labs-16-negative-merit")
    first = initial_design(material, count=20, seed=0)
    assert first == initial_design(material, count=20, seed=0)
    assert len({tuple(x) for x in first["candidates"]}) == 20
    with pytest.raises(ValueError):
        initial_design(material, count=2**16 + 1, seed=0)


def test_bundled_scores_and_shared_designs_replay():
    dataset = DiscreteBlackbox()
    checks = json.loads((DATA_ROOT / "quality.json").read_text())["evaluator_checks"]
    for check in checks:
        if check["id"].startswith("pbo-"):
            continue
        material = dataset(check["id"])
        assert material.evaluate(check["candidate"]) == pytest.approx(check["objective"], abs=1e-11)
        for seed in (0, 1, 2):
            design = dataset.initial_data(check["id"], seed=seed)
            assert len({tuple(x) for x in design["candidates"]}) == design["count"] == 20
            assert design["material_identity"] == material.identity
            for x, score in zip(design["candidates"], design["objectives"], strict=True):
                assert material.evaluate(x) == pytest.approx(score, abs=1e-11)
    with pytest.raises(ValueError, match="design seed"):
        dataset.initial_data("labs-16-energy", seed=True)


@pytest.mark.parametrize("identifier", DiscreteBlackbox().available_instances["qubo"])
def test_qubo_ommx_objective_matches_blackbox(identifier):
    material = DiscreteBlackbox()(identifier)
    instance = material.to_ommx()
    rng = np.random.default_rng(19)
    for x in [
        np.zeros(len(material.entry["cardinalities"]), dtype=int),
        np.ones(len(material.entry["cardinalities"]), dtype=int),
        rng.integers(0, 2, len(material.entry["cardinalities"])),
    ]:
        solution = instance.evaluate({i: float(value) for i, value in enumerate(x)})
        assert solution.feasible
        assert solution.objective == pytest.approx(material.evaluate(x), abs=1e-10)
    with pytest.raises(ValueError, match="only for QUBO"):
        DiscreteBlackbox()("labs-16-energy").to_ommx()


def test_pest_is_deterministic_without_global_rng_mutation():
    material = DiscreteBlackbox()("pest-control-reset-rng-s2")
    before = np.random.get_state()
    x = np.arange(25) % 5
    a = material.evaluate(x)
    material.evaluate(np.zeros(25))
    assert material.evaluate(x) == a
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_asset_corruption_and_escaping_path_are_rejected(tmp_path):
    material = DiscreteBlackbox()("qubo-16-s0")
    entry = dict(material.entry)
    path = tmp_path / entry["asset"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        Material(tmp_path, entry).evaluate([0] * 16)
    entry["asset"] = "../outside.npz"
    with pytest.raises(ValueError, match="escapes"):
        Material(tmp_path, entry).evaluate([0] * 16)


def test_optional_pbo_replays_every_setting_and_shared_design():
    pytest.importorskip("ioh")
    assert importlib.metadata.version("ioh") == "0.3.22"
    dataset = DiscreteBlackbox()
    checks = json.loads((DATA_ROOT / "quality.json").read_text())["evaluator_checks"]
    for check in checks:
        if not check["id"].startswith("pbo-"):
            continue
        material = dataset(check["id"])
        assert material.evaluate(check["candidate"]) == pytest.approx(check["objective"])
        for seed in (0, 1, 2):
            data = dataset.initial_data(check["id"], seed=seed)
            for x, score in zip(data["candidates"], data["objectives"], strict=True):
                assert material.evaluate(x) == pytest.approx(score)


def _fitness_fixture(tmp_path, monkeypatch):
    import ommx_quantum_benchmarks.blackbox.dataset as module

    package = tmp_path / "package"
    data = package / "data"
    data.mkdir(parents=True)
    raw = tmp_path / "raw/data"
    raw.mkdir(parents=True)
    (raw / "vectors_test.txt").write_text("1 0\n0 0\n1 1\n0 1\n")
    (raw / "fitness_test.txt").write_text("20\n0\n30\n10\n")
    entry = {
        "id": "fitness-test",
        "family": "fitness",
        "cardinalities": [2, 2],
        "sense": "min",
        "optimum": -30,
        "asset": "prepared/fitness-test.npz",
    }
    (data / "catalog.json").write_text(json.dumps({"entries": [entry]}))
    lock = {
        "files": [
            {
                "source": "fmira",
                "path": f"data/{kind}_test.txt",
                "url": "https://example.invalid/unused",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for kind in ("vectors", "fitness")
            for path in [raw / f"{kind}_test.txt"]
        ]
    }
    (package / "sources.lock.json").write_text(json.dumps(lock))
    monkeypatch.setattr(module, "DATA_ROOT", data)
    return raw


def test_explicit_fitness_acquisition_preserves_row_pairing(tmp_path, monkeypatch):
    _fitness_fixture(tmp_path, monkeypatch)
    material = fetch_fitness(tmp_path, "fitness-test")
    assert [material.evaluate(x) for x in [[0, 0], [0, 1], [1, 0], [1, 1]]] == [0, -10, -20, -30]
    with pytest.raises(KeyError):
        fetch_fitness(tmp_path, "unknown")


def test_explicit_fitness_acquisition_rejects_corrupt_source(tmp_path, monkeypatch):
    raw = _fitness_fixture(tmp_path, monkeypatch)
    (raw / "fitness_test.txt").write_text("30\n0\n20\n10\n")
    with pytest.raises(ValueError, match="source hash"):
        fetch_fitness(tmp_path, "fitness-test")
