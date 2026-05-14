"""Migrate qoblib (minto 1.x layout) on GHCR to qoblib_v2 (minto 2.x layout).

For every (dataset, model, instance) listed in the package's
``available_instances``, this script:

1. Pulls the existing artifact from
   ``ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib:<dataset>-<model>-<instance>``
   via :meth:`minto.Experiment.load_from_registry`.
2. Reads ``instance`` and ``solution`` out of the loaded experiment's
   ``experiment_datastore`` (where minto 1.x had placed them).
3. Builds a fresh :class:`minto.Experiment` with the 2.x layout
   (instance at experiment level, solution inside a run).
4. Pushes that experiment to
   ``ghcr.io/jij-inc/ommxquantumbenchmarks/qoblib_v2:<same-tag>``.

No upstream qoblib source / ``ommx_create.py`` / jijmodeling are needed.

Authentication uses the standard ommx env vars::

    OMMX_BASIC_AUTH_DOMAIN   (auto-set to ghcr.io)
    OMMX_BASIC_AUTH_USERNAME
    OMMX_BASIC_AUTH_PASSWORD

In a GitHub Actions context, ``OMMX_BASIC_AUTH_USERNAME=${{ github.actor }}``
and ``OMMX_BASIC_AUTH_PASSWORD=${{ secrets.GITHUB_TOKEN }}`` with
``permissions: packages: write`` is sufficient.

Usage examples::

    uv run python scripts/upload_qoblib.py --datasets 02_labs --max-workers 8
    uv run python scripts/upload_qoblib.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import minto

from ommx_quantum_benchmarks.qoblib import (
    BaseDataset,
    Birkhoff,
    IndependentSet,
    Labs,
    Marketsplit,
    Network,
    Routing,
    Steiner,
    Topology,
)
from ommx_quantum_benchmarks.qoblib.definitions import (
    BASE_URL,
    IMAGE_NAME,
    get_instance_tag,
)
from ommx_quantum_benchmarks.uploader import Uploader as BaseUploader


# The migration source lives at the legacy image name on the same registry.
LEGACY_IMAGE_NAME = "qoblib"
LEGACY_BASE_URL = BASE_URL.replace(f"/{IMAGE_NAME}", f"/{LEGACY_IMAGE_NAME}")


DATASET_CLASSES: list[type[BaseDataset]] = [
    Marketsplit,
    Labs,
    Birkhoff,
    Steiner,
    IndependentSet,
    Network,
    Routing,
    Topology,
]


@dataclass(frozen=True)
class MigrationEntry:
    dataset: str
    model: str
    instance: str

    def label(self) -> str:
        return f"{self.dataset}/{self.model}/{self.instance}"

    @property
    def tag(self) -> str:
        return get_instance_tag(self.dataset, self.model, self.instance)

    @property
    def source_url(self) -> str:
        return f"{LEGACY_BASE_URL}:{self.tag}"


def discover_entries(datasets: list[str] | None) -> list[MigrationEntry]:
    entries: list[MigrationEntry] = []
    for cls in DATASET_CLASSES:
        d = cls()
        if datasets and d.name not in datasets:
            continue
        for model_name, instance_names in d.available_instances.items():
            for instance_name in instance_names:
                entries.append(MigrationEntry(d.name, model_name, instance_name))
    return entries


def _migrate(entry: MigrationEntry) -> None:
    old = minto.Experiment.load_from_registry(entry.source_url)
    ds = old.dataspace.experiment_datastore
    if entry.instance not in ds.instances:
        raise RuntimeError(
            f"Instance {entry.instance!r} not found in {entry.source_url}"
        )
    instance = ds.instances[entry.instance]
    solution = ds.solutions.get(entry.instance)

    new = minto.Experiment(
        name=IMAGE_NAME,
        auto_saving=False,
        verbose_logging=False,
        collect_environment=False,
    )
    new.log_global_instance(instance_name=entry.instance, instance=instance)
    if solution is not None:
        with new.run() as run:
            run.log_solution(solution_name=entry.instance, solution=solution)

    new.push_github(
        org=BaseUploader.ORG,
        repo=BaseUploader.REPO,
        name=IMAGE_NAME,
        tag=entry.tag,
    )


def migrate_one(entry: MigrationEntry, max_retries: int) -> MigrationEntry:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            _migrate(entry)
            return entry
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s, ...
                time.sleep(2 ** attempt)
    assert last_exc is not None
    raise last_exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to filter (e.g., 02_labs 03_birkhoff). Default: all.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel worker threads (default: 8).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts per artifact on transient failures (default: 3).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned migrations without pushing.",
    )
    args = parser.parse_args()

    # The Uploader.__init__ side effect (setting OMMX_BASIC_AUTH_DOMAIN) is
    # the convention this project follows, replicate it here without forcing
    # consumers to import the Uploader.
    os.environ.setdefault("OMMX_BASIC_AUTH_DOMAIN", "ghcr.io")

    entries = discover_entries(args.datasets)
    print(f"Discovered {len(entries)} entries to migrate.", flush=True)
    if not entries:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        for e in entries:
            print(f"  {e.label()}  source: {e.source_url}")
        return 0

    successes: list[MigrationEntry] = []
    failures: list[tuple[MigrationEntry, str]] = []
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(migrate_one, e, args.max_retries): e for e in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            entry = futures[fut]
            try:
                fut.result()
                successes.append(entry)
                print(
                    f"[{i}/{len(entries)}] OK   {entry.label()}",
                    flush=True,
                )
            except Exception as exc:
                failures.append((entry, repr(exc)))
                print(
                    f"[{i}/{len(entries)}] FAIL {entry.label()}: {exc}",
                    flush=True,
                )

    elapsed = time.monotonic() - start
    print()
    print(
        f"Done in {elapsed:.1f}s. "
        f"Success: {len(successes)}/{len(entries)}, "
        f"Failures: {len(failures)}"
    )
    if failures:
        print()
        print("Failed entries:")
        for entry, err in failures:
            print(f"  {entry.label()}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
