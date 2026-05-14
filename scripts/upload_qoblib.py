"""Bulk-upload generated ``.ommx`` files under
``ommx_quantum_benchmarks/qoblib/*/models/*/ommx_output/`` to GHCR via the
``qoblib`` :class:`Uploader`, in parallel.

The script enumerates every ``<dataset>/models/<model>/ommx_output/<instance>.ommx``
file (optionally filtered by ``--datasets``), then pushes each one through the
project's :class:`ommx_quantum_benchmarks.qoblib.uploader.Uploader` using a
thread pool. Each failure is captured and reported at the end so a single bad
instance does not stop the rest of the run.

Authentication relies on the standard ommx env vars::

    OMMX_BASIC_AUTH_DOMAIN   (auto-set to ghcr.io by Uploader.__init__)
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
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ommx_quantum_benchmarks.qoblib.uploader import Uploader


REPO_ROOT = Path(__file__).resolve().parents[1]
QOBLIB_DIR = REPO_ROOT / "ommx_quantum_benchmarks" / "qoblib"


@dataclass(frozen=True)
class PushEntry:
    dataset: str
    model: str
    instance: str
    path: Path

    def label(self) -> str:
        return f"{self.dataset}/{self.model}/{self.instance}"


def discover_ommx_files(datasets: list[str] | None) -> list[PushEntry]:
    """Walk the repo and collect every generated ``.ommx`` file as a PushEntry."""
    entries: list[PushEntry] = []
    for dataset_dir in sorted(QOBLIB_DIR.iterdir()):
        if not dataset_dir.is_dir() or not dataset_dir.name[:2].isdigit():
            continue
        if datasets and dataset_dir.name not in datasets:
            continue
        models_dir = dataset_dir / "models"
        if not models_dir.is_dir():
            continue
        for model_dir in sorted(models_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            ommx_output = model_dir / "ommx_output"
            if not ommx_output.is_dir():
                continue
            for ommx_file in sorted(ommx_output.glob("*.ommx")):
                entries.append(
                    PushEntry(
                        dataset=dataset_dir.name,
                        model=model_dir.name,
                        instance=ommx_file.stem,
                        path=ommx_file,
                    )
                )
    return entries


def push_one(entry: PushEntry, verification: bool, max_retries: int) -> PushEntry:
    """Push a single artifact, retrying on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            Uploader().push_ommx(
                dataset_name=entry.dataset,
                model_name=entry.model,
                instance_name=entry.instance,
                ommx_filepath=str(entry.path),
                verification=verification,
            )
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
        "--verification",
        action="store_true",
        help="Enable per-artifact local verification before push (slower).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned uploads without pushing.",
    )
    args = parser.parse_args()

    entries = discover_ommx_files(args.datasets)
    print(f"Discovered {len(entries)} .ommx files to push.", flush=True)
    if not entries:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        for e in entries:
            print(f"  {e.label()} -> {e.path}")
        return 0

    successes: list[PushEntry] = []
    failures: list[tuple[PushEntry, str]] = []
    start = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(push_one, e, args.verification, args.max_retries): e
            for e in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            entry = futures[fut]
            try:
                fut.result()
                successes.append(entry)
                print(f"[{i}/{len(entries)}] OK   {entry.label()}", flush=True)
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
        f"Success: {len(successes)}/{len(entries)}, Failures: {len(failures)}"
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
