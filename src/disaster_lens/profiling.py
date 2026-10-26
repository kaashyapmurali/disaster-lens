"""Wall-clock stage records and batch-level timing for training and evaluation loops."""

import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader


class StageTimer:
    """Emit start/end progress and preserve interrupted/failed stage visibility.

    Callers own CUDA synchronization where needed. Stages may nest, so their
    durations must not be summed without checking the hierarchy.
    """

    def __init__(self, on_change: Callable[[], None] | None = None) -> None:
        self.started = time.perf_counter()
        self.records: list[dict] = []
        self.on_change = on_change

    def snapshot(self) -> dict:
        return {"clock": "perf_counter wall seconds", "stages": self.records,
                "note": "Stages may overlap/nest; not an additive kernel-time profile."}

    def _publish(self, record: dict) -> None:
        print(f"Timing: {record['name']} {record['state']}"
              + (f" {record['elapsed_seconds']:.3f}s" if "elapsed_seconds" in record else ""), flush=True)
        if self.on_change is not None:
            self.on_change()

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not name or any(record["name"] == name for record in self.records):
            raise ValueError("Timing stage names must be nonempty and unique.")
        started = time.perf_counter()
        record = {"name": name, "state": "running", "start_offset_seconds": started - self.started}
        self.records.append(record)
        self._publish(record)
        try:
            yield
        except BaseException:
            record["state"] = "failed"
            raise
        else:
            record["state"] = "complete"
        finally:
            record["elapsed_seconds"] = time.perf_counter() - started
            self._publish(record)


# Batch-level timing helpers shared by detector training and evaluation.


def start_measurement(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def measurements(device: torch.device, started: float, scenes: int) -> dict:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed, "scenes_per_second": scenes / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profiled_batches(loader: DataLoader, profile: dict) -> Iterable[tuple[tuple, dict]]:
    started = time.perf_counter()
    iterator = iter(loader)
    profile["iterator_startup_seconds"] = time.perf_counter() - started
    profile["batches"] = []
    ready = time.perf_counter()
    for batch in iterator:
        timing = {"loader_wait_seconds": time.perf_counter() - ready}
        profile["batches"].append(timing)
        yield batch, timing
        ready = time.perf_counter()


def batch_timing_summary(profile: dict) -> dict:
    batches = profile["batches"]
    keys = sorted({key for batch in batches for key in batch if key.endswith("_seconds")})
    later = batches[1:]
    return {
        "iterator_startup_seconds": profile["iterator_startup_seconds"],
        "batches": len(batches), "first_batch": batches[0] if batches else None,
        "phase_totals_seconds": {key: sum(batch.get(key, 0.) for batch in batches) for key in keys},
        "after_first_batch_count": len(later),
        "after_first_batch_mean_seconds": {
            key: sum(batch.get(key, 0.) for batch in later) / len(later) if later else None for key in keys},
        "note": "Exposed loader wait includes reads/decoding/worker coordination not hidden by prefetch. "
                "CUDA transfer/compute phases synchronize; these are wall times, not kernel profiles. "
                "First versus later batches is not a controlled cold/warm benchmark.",
    }
