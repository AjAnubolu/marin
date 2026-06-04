# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pluggable shard execution strategies for ZephyrWorker.

A ``StageRunner`` is the strategy a worker uses to execute one ``ShardTask``.
Two implementations ship here:

* ``InlineRunner`` (default) — runs the stage in the worker actor's own
  process. Cheapest; appropriate for tests and pipelines whose user code is
  trusted not to corrupt the worker.
* ``SubprocessRunner`` — runs the stage in a fresh ``python -m zephyr.runners``
  subprocess. Each shard gets a clean Python heap, Arrow pool, and file
  descriptors; native crashes (SIGSEGV from Arrow/JAX, OOM kill) surface as
  deterministic ``returncode != 0`` task errors instead of bringing down the
  worker actor. Slower (~700ms of cold-import overhead per task).

Pick the runner pipeline-wide via ``ZephyrContext(stage_runner_factory=...)``.
"""

from __future__ import annotations

import logging
import os
import re
import signal
import subprocess as sp
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator
from contextlib import suppress
from typing import Any, TypeVar

import cloudpickle
import psutil
import pyarrow as pa
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

from zephyr import counters
from zephyr.plan import Scatter, StageContext, run_stage
from zephyr.stage_io import (
    ShardTask,
    StageRunner,
    TaskResult,
    _ensure_picklable_exception,
    _shared_data_path,
    _stage_throughput,
    _write_stage_output,
)
from zephyr.stats import (
    ZEPHYR_STAGE_BYTES_PROCESSED_KEY,
    ZEPHYR_STAGE_ITEM_COUNT_KEY,
    ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY,
    ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY,
    ZEPHYR_WORKER_CPU_TIME_KEY,
    ZEPHYR_WORKER_MEM_AVERAGE_KEY,
    ZEPHYR_WORKER_MEM_CURRENT_KEY,
    ZEPHYR_WORKER_MEM_PEAK_KEY,
    StatsWriter,
    ZephyrWorkerStatStatus,
)
from zephyr.worker_context import Aggregation, CounterEntry, CounterSnapshot, _worker_ctx_var

logger = logging.getLogger(__name__)


__all__ = ["InlineRunner", "StageRunner", "SubprocessRunner"]


SUBPROCESS_STATS_INTERVAL = 5.0
"""How often the subprocess child samples and emits its stats to finelog and
flushes its counters.

Matches the parent's heartbeat cadence so each beat reads at most one stale
snapshot before a fresh flush lands.
"""


# ---------------------------------------------------------------------------
# Shared worker context + stats wrapping (used by both runners)
# ---------------------------------------------------------------------------


class _InProcessWorkerContext:
    """WorkerContext satisfied by an in-memory counter dict.

    Used both by ``InlineRunner`` (in the worker actor process) and by the
    ``SubprocessRunner`` child (in the forked subprocess). Loads shared data
    lazily from the chunk store on first access and caches it for the rest
    of the task.
    """

    def __init__(self, chunk_prefix: str, execution_id: str, stage_name: str, num_workers: int = 1):
        self._chunk_prefix = chunk_prefix
        self._execution_id = execution_id
        self._stage_name = stage_name
        self._shared_data_cache: dict[str, Any] = {}
        self._counters: dict[str, CounterEntry] = {}
        self._generation = 0
        self.num_workers = num_workers

    def get_shared(self, name: str) -> Any:
        if name not in self._shared_data_cache:
            path = _shared_data_path(self._chunk_prefix, self._execution_id, name)
            logger.info("Loading shared data '%s' from %s", name, path)
            with open_url(path, "rb") as f:
                self._shared_data_cache[name] = cloudpickle.loads(f.read())
        return self._shared_data_cache[name]

    def increment_counter(self, name: str, value: int = 1, stage: str | None = None) -> None:
        if name in self._counters:
            self._counters[name].value += value
        else:
            self._counters[name] = CounterEntry(value, stage=stage)

    def set_counter(self, name: str, value: int | float, stage: str | None = None) -> None:
        if name in self._counters:
            entry = self._counters[name]
            entry.value = value
            entry.count = 1
            entry.stage = stage
        else:
            self._counters[name] = CounterEntry(value, stage=stage)

    def update_counter(self, name: str, value: int | float, stage: str | None = None) -> None:
        entry = self._counters.get(name)
        if entry is None or entry.count == 0:
            # First real observation: initialise the value regardless of aggregation.
            if entry is None:
                self._counters[name] = CounterEntry(value, stage=stage)
            else:
                entry.value = value
                entry.stage = stage
                entry.count = 1
            return
        entry.merge(CounterEntry(value, entry.aggregation, stage, count=1))

    def set_aggregation(self, name: str, agg: Aggregation) -> None:
        if name in self._counters:
            self._counters[name].aggregation = agg
        else:
            # count=0 marks the entry as uninitialised so update_counter sets the
            # first value directly rather than applying MIN/MAX/AVERAGE to 0.
            self._counters[name] = CounterEntry(0, aggregation=agg, count=0)

    def current_stage_name(self) -> str:
        return self._stage_name

    def get_counters(self, stage: str | None = None) -> dict[str, int | float]:
        """Flat view of counter values, for use by stats emission code."""
        return {k: e.value for k, e in self._counters.items() if stage is None or e.stage == stage}

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(
            counters={k: CounterEntry(e.value, e.aggregation, e.stage, e.count) for k, e in self._counters.items()},
            generation=self._generation,
        )


_T = TypeVar("_T")


def _wrap_stage_stats(gen: Iterator[_T], stage_name: str, ctx: _InProcessWorkerContext) -> Iterator[_T]:
    """Yield items from ``gen`` while recording item count and byte size into ``ctx``."""
    stage_counters = counters.current_stage()
    for item in gen:
        stage_counters.update_counter(ZEPHYR_STAGE_ITEM_COUNT_KEY, 1)
        stage_counters.update_counter(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, sys.getsizeof(item))
        yield item


def _sample_process_stats(cpu_s_at_start: float, proc: psutil.Process) -> None:
    """Sample the current process's resource usage into ``ctx`` counters.

    Uses set_counter (not increment) because these are point-in-time metrics.
    Peak memory is tracked as a monotonically increasing max across calls.
    IO counters are cumulative totals from the OS; unavailable on some platforms.
    ``cpu_s_at_start`` is subtracted from cumulative CPU time to give per-shard delta.
    ``proc`` must be the same object across calls so cpu_percent() has a
    prior measurement to diff against; prime it once before the first sample.
    """
    rss = proc.memory_info().rss
    cpu_times = proc.cpu_times()
    cpu_pct = proc.cpu_percent()
    stage_counters = counters.current_stage()
    stage_counters.set_counter(ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, cpu_pct)
    stage_counters.update_counter(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, cpu_pct)
    stage_counters.set_counter(ZEPHYR_WORKER_CPU_TIME_KEY, cpu_times.user + cpu_times.system - cpu_s_at_start)
    stage_counters.set_counter(ZEPHYR_WORKER_MEM_CURRENT_KEY, rss)
    stage_counters.update_counter(ZEPHYR_WORKER_MEM_AVERAGE_KEY, rss)
    stage_counters.update_counter(ZEPHYR_WORKER_MEM_PEAK_KEY, rss)


def _set_counter_aggregations() -> None:
    """Register aggregation modes for resource-usage counters on the current stage.

    Must be called once per task before the first ``_sample_process_stats``
    so that AVERAGE/MAX counters are reduced correctly.  SUM is the default
    and listed only for documentation.
    """
    sc = counters.current_stage()
    sc.set_aggregation(ZEPHYR_STAGE_ITEM_COUNT_KEY, Aggregation.SUM)
    sc.set_aggregation(ZEPHYR_STAGE_BYTES_PROCESSED_KEY, Aggregation.SUM)
    sc.set_aggregation(ZEPHYR_WORKER_CPU_PCT_AVERAGE_KEY, Aggregation.AVERAGE)
    sc.set_aggregation(ZEPHYR_WORKER_CPU_TIME_KEY, Aggregation.SUM)
    sc.set_aggregation(ZEPHYR_WORKER_MEM_AVERAGE_KEY, Aggregation.AVERAGE)
    sc.set_aggregation(ZEPHYR_WORKER_MEM_PEAK_KEY, Aggregation.MAX)


def _periodic_sampler(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    interval: float,
    *,
    cpu_s_at_start: float = 0.0,
    stats_writer: StatsWriter | None = None,
    task: ShardTask | None = None,
    execution_id: str = "",
    start_time: float = 0.0,
    proc: psutil.Process | None = None,
) -> None:
    """Periodically sample process stats and optionally emit RUNNING rows to finelog."""

    while not stop_event.wait(timeout=interval):
        try:
            if task is not None and proc is not None:
                _sample_process_stats(cpu_s_at_start, proc)

            if stats_writer is not None and task is not None and proc is not None:
                stats_writer.emit_worker_stat(
                    task.stage_name,
                    task.shard_idx,
                    execution_id,
                    ZephyrWorkerStatStatus.RUNNING,
                    start_time,
                    ctx.get_counters(),
                )
        except Exception:
            logger.warning("Failed to sample/emit process stats", exc_info=True)


def _run_task_with_ctx(
    task: ShardTask,
    chunk_prefix: str,
    execution_id: str,
    ctx: _InProcessWorkerContext,
) -> TaskResult:
    """Run one ShardTask inside the given worker context, writing stage output to disk.

    Shared between ``InlineRunner.execute`` and the subprocess child entry —
    once the right ctx is in place (and ``_worker_ctx_var`` is set), the
    actual per-shard work is identical.
    """
    stage_ctx = StageContext(
        shard=task.shard,
        shard_idx=task.shard_idx,
        total_shards=task.total_shards,
        aux_shards=task.aux_shards,
    )
    output_stage_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task.stage_name).strip("-")
    stage_dir = f"{chunk_prefix}/{execution_id}/{output_stage_name}"
    external_sort_dir = f"{stage_dir}-external-sort/shard-{task.shard_idx:04d}"
    scatter_op = next((op for op in task.operations if isinstance(op, Scatter)), None)
    return _write_stage_output(
        _wrap_stage_stats(
            run_stage(stage_ctx, task.operations, external_sort_dir=external_sort_dir),
            task.stage_name,
            ctx,
        ),
        source_shard=task.shard_idx,
        stage_dir=stage_dir,
        shard_idx=task.shard_idx,
        scatter_op=scatter_op,
        total_shards=task.total_shards,
    )


# ---------------------------------------------------------------------------
# InlineRunner — default
# ---------------------------------------------------------------------------


class InlineRunner:
    """Run shard work in the worker actor's own process.

    Cheap and observable (counters live in shared memory; the heartbeat just
    reads them) but does not isolate native crashes or per-shard memory
    growth. Default for ``ZephyrContext`` because most pipelines are fine
    here, and tests run dramatically faster than under ``SubprocessRunner``.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self._num_workers = num_workers
        self._ctx: _InProcessWorkerContext | None = None

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, task.stage_name, num_workers=self._num_workers)
        self._ctx = ctx
        worker_token = _worker_ctx_var.set(ctx)
        _set_counter_aggregations()
        stop_event = threading.Event()
        stats_writer = StatsWriter.connect()
        proc = psutil.Process()
        start_time = time.monotonic()
        cpu_times_at_start = proc.cpu_times()
        cpu_s_at_start = cpu_times_at_start.user + cpu_times_at_start.system
        proc.cpu_percent()  # prime so subsequent calls have a baseline
        stats_writer.emit_worker_stat(
            task.stage_name, task.shard_idx, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx.get_counters()
        )
        sampler = threading.Thread(
            target=_periodic_sampler,
            kwargs={
                "stop_event": stop_event,
                "ctx": ctx,
                "interval": SUBPROCESS_STATS_INTERVAL,
                "cpu_s_at_start": cpu_s_at_start,
                "stats_writer": stats_writer,
                "task": task,
                "execution_id": execution_id,
                "start_time": start_time,
                "proc": proc,
            },
            daemon=True,
            name="zephyr-inline-stats-sampler",
        )
        sampler.start()
        _task_failed = False
        try:
            result = _run_task_with_ctx(task, chunk_prefix, execution_id, ctx)
        except Exception:
            _task_failed = True
            raise
        finally:
            stop_event.set()
            sampler.join(timeout=2.0)
            if not sampler.is_alive():
                _sample_process_stats(cpu_s_at_start, proc)
            _status = ZephyrWorkerStatStatus.FAILED if _task_failed else ZephyrWorkerStatStatus.END
            stats_writer.emit_worker_stat(
                task.stage_name, task.shard_idx, execution_id, _status, start_time, ctx.get_counters()
            )
            stats_writer.close()
            _worker_ctx_var.reset(worker_token)
            self._ctx = None
        return result, dict(ctx._counters)

    def live_counters(self) -> dict[str, CounterEntry]:
        ctx = self._ctx
        return dict(ctx._counters) if ctx is not None else {}


# ---------------------------------------------------------------------------
# SubprocessRunner — opt-in isolation
# ---------------------------------------------------------------------------


def _periodic_counter_writer(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    counter_file: str,
    interval: float,
) -> None:
    """Atomic temp-write + rename so the parent never reads a half-written file."""
    while not stop_event.wait(timeout=interval):
        try:
            tmp_path = f"{counter_file}.tmp"
            with open(tmp_path, "wb") as f:
                cloudpickle.dump(dict(ctx._counters), f)
            os.rename(tmp_path, counter_file)
        except Exception:
            logger.warning("Failed to flush counter file to %s", counter_file, exc_info=True)


def _periodic_status_logger(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    stage_name: str,
    execution_id: str,
    shard_idx: int,
    total_shards: int,
    monotonic_start: float,
    interval: float,
) -> None:
    """Per-shard items/bytes rate log line (mirrors coordinator ``_log_status``)."""
    while not stop_event.wait(timeout=interval):
        if sys.is_finalizing():
            return
        elapsed = time.monotonic() - monotonic_start
        # Map-only stages never populate these counters; logging zeros is misleading.
        throughput = _stage_throughput(ctx.get_counters(stage_name), elapsed)
        if throughput is None:
            continue
        logger.info(
            "[%s] [%s] [%s] shard %d/%d; %s",
            execution_id,
            stage_name,
            threading.current_thread().name,
            shard_idx,
            total_shards,
            throughput,
        )


class SubprocessRunner:
    """Run each shard in a fresh ``python -m zephyr.runners`` subprocess.

    Provides full memory and crash isolation: native crashes (Arrow/JAX
    SIGSEGV, OOM) terminate only the child and surface as deterministic
    ``returncode != 0`` task errors. Costs ~700ms per task in cold Python
    imports plus pickle round-trip; reserve for stages with leak-prone or
    crash-prone user code.

    Args:
        num_workers: Total number of concurrent subprocess workers sharing this
            actor's RAM. Passed to child processes via ``ZEPHYR_NUM_WORKERS_PER_ACTOR``
            so each child scales its scatter-write buffer budget proportionally.
    """

    def __init__(self, num_workers: int = 1) -> None:
        self._num_workers = num_workers
        self._counter_file: str | None = None

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        finelog_url = StatsWriter.resolve_url()  # Requires Iris context, so called here and passed to subprocess
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            cloudpickle.dump((task, chunk_prefix, execution_id, finelog_url), f)
            task_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            result_file = f.name
        counter_file = f"{result_file}.counters"
        self._counter_file = counter_file

        try:
            # ``-u`` keeps the child's stdout/stderr unbuffered so any
            # faulthandler traceback reaches the parent's log before the
            # process dies.
            proc = sp.run(
                [sys.executable, "-u", "-m", "zephyr.runners", task_file, result_file, str(self._num_workers)],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

            if proc.returncode != 0:
                # Linux OOM-killer sends SIGKILL → returncode == -9. Distinguish
                # so callers/retries can react to memory pressure specifically.
                if proc.returncode == -signal.SIGKILL:
                    raise MemoryError(
                        f"Subprocess for shard {task.shard_idx} was killed by SIGKILL "
                        f"(returncode {proc.returncode}); most likely OOM-killed by the kernel."
                    )
                raise RuntimeError(
                    f"Subprocess for shard {task.shard_idx} exited with code {proc.returncode}; "
                    "see worker stderr above for the faulthandler traceback."
                )

            with open(result_file, "rb") as f:
                result_or_error, child_counters = cloudpickle.load(f)

            # Clear counter pointer BEFORE returning so a heartbeat racing
            # this and ``report_result`` reads {} rather than re-shipping
            # values the caller is about to send as final.
            self._counter_file = None

            if isinstance(result_or_error, Exception):
                raise result_or_error

            return result_or_error, dict(child_counters)
        finally:
            self._counter_file = None
            for p in (task_file, result_file, counter_file, f"{counter_file}.tmp"):
                with suppress(FileNotFoundError):
                    os.unlink(p)

    def live_counters(self) -> dict[str, CounterEntry]:
        cf = self._counter_file
        if cf is None:
            return {}
        try:
            with open(cf, "rb") as f:
                return cloudpickle.load(f)
        except (FileNotFoundError, EOFError):
            # Race against atomic rename, or task already cleaned up its file.
            return {}
        except Exception:
            logger.warning("Failed to read counter file %s", cf, exc_info=True)
            return {}


# ---------------------------------------------------------------------------
# Subprocess child entry point: `python -m zephyr.runners <task_file> <result_file>`
# ---------------------------------------------------------------------------


def _execute_shard_subprocess(task_file: str, result_file: str, num_workers: int) -> None:
    """Subprocess child body: runs one ShardTask and writes the result file."""
    # Each shard already runs in its own subprocess; redundant Arrow thread
    # pools just compete with the parent's shard-level parallelism.
    pa.set_io_thread_count(1)
    pa.set_cpu_count(1)

    # configure_logging installs faulthandler so SIGSEGV / SIGABRT / SIGBUS
    # / SIGFPE / SIGILL in a C extension produces a Python traceback on
    # stderr instead of a bare ``returncode < 0``.
    configure_logging(level=logging.INFO)

    counter_file = f"{result_file}.counters"
    stop_event = threading.Event()
    flusher: threading.Thread | None = None
    status_logger: threading.Thread | None = None
    sampler: threading.Thread | None = None
    result_or_error: Any
    ctx: _InProcessWorkerContext | None = None
    stats_writer: StatsWriter = StatsWriter(None)
    proc = psutil.Process()
    proc.cpu_percent()
    start_time = 0.0
    cpu_s_at_start = 0.0
    _task_failed = True  # cleared to False only on clean _run_task_with_ctx completion
    try:
        with open(task_file, "rb") as f:
            task, chunk_prefix, execution_id, finelog_url = cloudpickle.load(f)

        start_time = time.monotonic()
        _cpu = proc.cpu_times()
        cpu_s_at_start = _cpu.user + _cpu.system

        if finelog_url:
            stats_writer = StatsWriter.connect(finelog_url)

        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, task.stage_name, num_workers=num_workers)
        _worker_ctx_var.set(ctx)
        _set_counter_aggregations()

        shard_monotonic_start = time.monotonic()
        stats_writer.emit_worker_stat(
            task.stage_name, task.shard_idx, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx.get_counters()
        )

        flusher = threading.Thread(
            target=_periodic_counter_writer,
            args=(stop_event, ctx, counter_file, SUBPROCESS_STATS_INTERVAL),
            daemon=True,
            name="zephyr-subprocess-counter-flusher",
        )
        flusher.start()

        status_logger = threading.Thread(
            target=_periodic_status_logger,
            args=(
                stop_event,
                ctx,
                task.stage_name,
                execution_id,
                task.shard_idx,
                task.total_shards,
                shard_monotonic_start,
                SUBPROCESS_STATS_INTERVAL,
            ),
            daemon=True,
            name="zephyr-subprocess-status-logger",
        )
        status_logger.start()

        sampler = threading.Thread(
            target=_periodic_sampler,
            kwargs={
                "stop_event": stop_event,
                "ctx": ctx,
                "interval": SUBPROCESS_STATS_INTERVAL,
                "cpu_s_at_start": cpu_s_at_start,
                "stats_writer": stats_writer,
                "task": task,
                "execution_id": execution_id,
                "start_time": start_time,
                "proc": proc,
            },
            daemon=True,
            name="zephyr-subprocess-stats-sampler",
        )
        sampler.start()

        result_or_error = _run_task_with_ctx(task, chunk_prefix, execution_id, ctx)
        _task_failed = False
    except Exception as e:
        # Cloudpickling an exception drops ``__traceback__``, so a naive
        # parent re-raise would otherwise show only the parent stack at the
        # re-raise site. ``__notes__`` survives pickling and Python prints
        # it inline when the exception eventually propagates.
        logger.exception("Subprocess shard execution failed")
        e.add_note(f"--- subprocess traceback ---\n{traceback.format_exc().rstrip()}")
        # Normalize before handing the error to the parent: a subclass that
        # cannot round-trip through pickle (e.g. an __init__ whose signature
        # does not match its args) would otherwise revive into a TypeError when
        # the parent loads the result file, masking the real failure.
        result_or_error = _ensure_picklable_exception(e)
    finally:
        stop_event.set()
        if flusher is not None and flusher.is_alive():
            flusher.join(timeout=2.0)
        if status_logger is not None and status_logger.is_alive():
            status_logger.join(timeout=2.0)
        if sampler is not None and sampler.is_alive():
            sampler.join(timeout=2.0)
        if ctx is not None:
            if sampler is None or not sampler.is_alive():
                with suppress(Exception):
                    _sample_process_stats(cpu_s_at_start, proc)
            _status = ZephyrWorkerStatStatus.FAILED if _task_failed else ZephyrWorkerStatStatus.END
            with suppress(Exception):
                stats_writer.emit_worker_stat(
                    task.stage_name, task.shard_idx, execution_id, _status, start_time, ctx.get_counters()
                )
        stats_writer.close()

    with open(result_file, "wb") as f:
        counters_out = dict(ctx._counters) if ctx is not None else {}
        cloudpickle.dump((result_or_error, counters_out), f)


def _subprocess_main() -> None:
    if len(sys.argv) != 4:
        print("Usage: python -m zephyr.runners <task_file> <result_file> <num_workers>", file=sys.stderr)
        os._exit(1)
    # Bypass interpreter shutdown: PyArrow GCS/Azure filesystem background
    # threads can race with module GC and fire ``std::terminate`` → SIGABRT,
    # poisoning the parent's returncode check. The result file is already
    # on disk and the counter flusher has been joined, so nothing in this
    # one-shot child needs ``atexit`` / ``__del__`` to run.
    exit_code = 0
    try:
        _execute_shard_subprocess(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        with suppress(Exception):
            sys.stdout.flush()
        with suppress(Exception):
            sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    _subprocess_main()
