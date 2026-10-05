"""Import the prepared JijZeptWorkflow discrete suite without FMIRA data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Reviewed source implementation; an upstream change requires an explicit review.
SOURCE_IMPLEMENTATION_SHA256 = "271f4fe90986fc00782a80f5c9196c4d4f5bfedca4edfe390629281347d66fbf"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="prepared directory containing catalog.json")
    args = parser.parse_args()
    source = args.source.resolve()
    package = Path(__file__).resolve().parents[1] / "ommx_quantum_benchmarks/blackbox"
    target = package / "data"
    target.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((source / "catalog.json").read_text())
    if catalog["schema_version"] != 1:
        raise ValueError("unsupported source schema")
    if digest(package / "sources.lock.json") != catalog["sources_lock_sha256"]:
        raise ValueError("source lock differs from the prepared catalog")
    implementation = digest(package / "materials.py")
    identities = {}
    for entry in catalog["entries"]:
        if entry["implementation_sha256"] != SOURCE_IMPLEMENTATION_SHA256:
            raise ValueError("source evaluator differs from the reviewed implementation")
        if "asset" in entry and entry["family"] != "fitness":
            path = (source / entry["asset"]).resolve()
            if not path.is_relative_to(source) or digest(path) != entry["asset_sha256"]:
                raise ValueError("source asset mismatch")
            destination = target / entry["asset"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        entry["implementation_sha256"] = implementation
        entry["distribution"] = (
            "external-acquisition"
            if entry["family"] == "fitness"
            else "optional-ioh"
            if entry["family"] == "pbo"
            else "bundled"
        )
        identities[entry["id"]] = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()
    # Keep the full-table metadata and all FMIRA observations out of the distribution.
    designs = json.loads((source / "initial_designs.json").read_text())
    designs["designs"] = [
        item for item in designs["designs"] if not item["id"].startswith("fitness-")
    ]
    for design in designs["designs"]:
        design["material_identity"] = identities[design["id"]]
    quality = json.loads((source / "quality.json").read_text())
    quality.pop("landscapes", None)
    quality["evaluator_checks"] = [
        item for item in quality["evaluator_checks"] if not item["id"].startswith("fitness-")
    ]
    catalog["source"] = {
        "repository": "https://github.com/Jij-Inc/JijZeptWorkflow",
        "commit": "5e2766138c4a84e37fd506aebc28ee16d62cd181",
        "path": "benchmarks/discrete",
        "original_catalog_sha256": digest(source / "catalog.json"),
        "original_implementation_sha256": SOURCE_IMPLEMENTATION_SHA256,
    }
    for name, data in [
        ("catalog.json", catalog),
        ("initial_designs.json", designs),
        ("quality.json", quality),
    ]:
        (target / name).write_text(
            json.dumps(data, indent=None if name == "initial_designs.json" else 2, allow_nan=False)
            + "\n"
        )
    print(f"Imported {len(catalog['entries'])} settings; FMIRA tables and observations excluded")


if __name__ == "__main__":
    main()
