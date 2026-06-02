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
from datetime import datetime, timezone
from typing import Any, TypeVar

import cloudpickle
import psutil
import pyarrow as pa
from finelog.client import LogClient, Table
from iris.client import get_iris_ctx
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

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
    ZEPHYR_WORKER_CPU_MILLI_KEY,
    ZEPHYR_WORKER_CPU_TIME_MS_KEY,
    ZEPHYR_WORKER_IO_READ_KEY,
    ZEPHYR_WORKER_IO_WRITE_KEY,
    ZEPHYR_WORKER_MEM_CURRENT_KEY,
    ZEPHYR_WORKER_MEM_PEAK_KEY,
    ZEPHYR_WORKER_STATS_NAMESPACE,
    ZephyrWorkerStat,
    ZephyrWorkerStatStatus,
)
from zephyr.worker_context import CounterSnapshot, _worker_ctx_var

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

    def __init__(self, chunk_prefix: str, execution_id: str, num_workers: int = 1):
        self._chunk_prefix = chunk_prefix
        self._execution_id = execution_id
        self._shared_data_cache: dict[str, Any] = {}
        self._counters: dict[str, int] = {}
        self._generation = 0
        self.num_workers = num_workers

    def get_shared(self, name: str) -> Any:
        if name not in self._shared_data_cache:
            path = _shared_data_path(self._chunk_prefix, self._execution_id, name)
            logger.info("Loading shared data '%s' from %s", name, path)
            with open_url(path, "rb") as f:
                self._shared_data_cache[name] = cloudpickle.loads(f.read())
        return self._shared_data_cache[name]

    def increment_counter(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def set_counter(self, name: str, value: int) -> None:
        self._counters[name] = value

    def get_counter_snapshot(self) -> CounterSnapshot:
        self._generation += 1
        return CounterSnapshot(counters=dict(self._counters), generation=self._generation)


_T = TypeVar("_T")


def _wrap_stage_stats(gen: Iterator[_T], stage_name: str, ctx: _InProcessWorkerContext) -> Iterator[_T]:
    """Yield items from ``gen`` while recording item count and byte size into ``ctx``."""
    item_key = ZEPHYR_STAGE_ITEM_COUNT_KEY.format(stage_name=stage_name)
    byte_key = ZEPHYR_STAGE_BYTES_PROCESSED_KEY.format(stage_name=stage_name)
    for item in gen:
        ctx.increment_counter(item_key, 1)
        ctx.increment_counter(byte_key, sys.getsizeof(item))
        yield item


def _sample_process_stats(
    ctx: _InProcessWorkerContext, cpu_s_at_start: float, stage_name: str, proc: psutil.Process
) -> None:
    """Sample the current process's resource usage into ``ctx`` counters.

    Uses set_counter (not increment) because these are point-in-time metrics.
    Peak memory is tracked as a monotonically increasing max across calls.
    IO counters are cumulative totals from the OS; unavailable on some platforms.
    ``cpu_s_at_start`` is subtracted from cumulative CPU time to give per-shard delta.
    ``proc`` must be the same object across calls so cpu_percent() has a
    prior measurement to diff against; prime it once before the first sample.
    """
    mem = proc.memory_info()
    cpu_pct = proc.cpu_percent()
    cpu_times = proc.cpu_times()
    cpu_time_delta_ms = int(max(0, (cpu_times.user + cpu_times.system - cpu_s_at_start) * 1000))
    peak_key = ZEPHYR_WORKER_MEM_PEAK_KEY.format(stage_name=stage_name)
    ctx.set_counter(ZEPHYR_WORKER_CPU_MILLI_KEY.format(stage_name=stage_name), int(cpu_pct * 1000))
    ctx.set_counter(ZEPHYR_WORKER_CPU_TIME_MS_KEY.format(stage_name=stage_name), cpu_time_delta_ms)
    ctx.set_counter(ZEPHYR_WORKER_MEM_CURRENT_KEY.format(stage_name=stage_name), mem.rss)
    ctx.set_counter(peak_key, max(ctx._counters.get(peak_key, 0), mem.rss))
    with suppress(AttributeError, psutil.AccessDenied):
        io = proc.io_counters()
        ctx.set_counter(ZEPHYR_WORKER_IO_READ_KEY.format(stage_name=stage_name), io.read_bytes)
        ctx.set_counter(ZEPHYR_WORKER_IO_WRITE_KEY.format(stage_name=stage_name), io.write_bytes)


def _emit_runner_stat(
    log_table: Any,
    task: ShardTask,
    execution_id: str,
    status: ZephyrWorkerStatStatus,
    start_time: float,
    ctx: _InProcessWorkerContext,
    proc: psutil.Process,
    cpu_s_at_start: float,
) -> None:
    """Emit one ZephyrWorkerStat row to finelog from inside the runner."""
    elapsed = time.monotonic() - start_time
    counters = ctx._counters
    throughput = _stage_throughput(counters, task.stage_name, elapsed)
    try:
        current_cpu = proc.cpu_times()
        cumulative_cpu_s = max(0.0, (current_cpu.user + current_cpu.system) - cpu_s_at_start)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        cumulative_cpu_s = 0.0
    avg_cpu_pct = (cumulative_cpu_s / elapsed * 100) if elapsed > 0 else 0.0
    stat = ZephyrWorkerStat(
        execution_id=execution_id,
        stage_name=task.stage_name,
        shard_idx=task.shard_idx,
        status=status,
        ts=datetime.now(timezone.utc).replace(tzinfo=None),
        items=throughput.items if throughput else 0,
        bytes_processed=throughput.bytes_processed if throughput else 0,
        item_rate=throughput.item_rate if throughput else 0.0,
        byte_rate=throughput.byte_rate if throughput else 0.0,
        cumulative_cpu_s=cumulative_cpu_s,
        avg_cpu_pct=avg_cpu_pct,
        mem_current_bytes=counters.get(ZEPHYR_WORKER_MEM_CURRENT_KEY.format(stage_name=task.stage_name), 0),
        mem_peak_bytes=counters.get(ZEPHYR_WORKER_MEM_PEAK_KEY.format(stage_name=task.stage_name), 0),
        io_read_bytes=counters.get(ZEPHYR_WORKER_IO_READ_KEY.format(stage_name=task.stage_name), 0),
        io_write_bytes=counters.get(ZEPHYR_WORKER_IO_WRITE_KEY.format(stage_name=task.stage_name), 0),
    )
    try:
        log_table.write([stat])
    except Exception:
        logger.warning("Failed to write runner worker stat to finelog", exc_info=True)


def _periodic_sampler(
    stop_event: threading.Event,
    ctx: _InProcessWorkerContext,
    interval: float,
    *,
    cpu_s_at_start: float = 0.0,
    log_table: Any = None,
    task: ShardTask | None = None,
    execution_id: str = "",
    start_time: float = 0.0,
    proc: psutil.Process | None = None,
) -> None:
    """Periodically sample process stats and optionally emit RUNNING rows to finelog."""
    while not stop_event.wait(timeout=interval):
        try:
            if task is not None and proc is not None:
                _sample_process_stats(ctx, cpu_s_at_start, task.stage_name, proc)

            if log_table is not None and task is not None and proc is not None:
                _emit_runner_stat(
                    log_table,
                    task,
                    execution_id,
                    ZephyrWorkerStatStatus.RUNNING,
                    start_time,
                    ctx,
                    proc,
                    cpu_s_at_start,
                )
        except Exception:
            logger.warning("Failed to sample/emit process stats", exc_info=True)


def _resolve_finelog_url() -> str | None:
    """Resolve the finelog endpoint URL via the Iris controller registry."""
    iris_ctx = get_iris_ctx()
    if iris_ctx is None or iris_ctx.client is None:
        return None
    try:
        return iris_ctx.client.resolve_endpoint("/system/log-server")
    except Exception:
        logger.warning("Could not resolve finelog endpoint for runner stats", exc_info=True)
        return None


def _make_log_client() -> LogClient | None:
    """Return a LogClient connected to the finelog service.

    Returns ``None`` when not running inside an Iris context. Callers treat
    ``None`` as a signal to skip stats emission silently.
    """
    url = _resolve_finelog_url()
    if url is None:
        return None
    try:
        return LogClient.connect(url)
    except Exception:
        logger.warning("Could not connect to finelog stats service; runner stats disabled", exc_info=True)
        return None


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
        self._log_client: LogClient | None = None
        self._worker_stats_table: Table | None = None
        self._log_client_initialized: bool = False

    def _get_worker_stats_table(self) -> Any:
        if not self._log_client_initialized:
            self._log_client_initialized = True
            self._log_client = _make_log_client()
            if self._log_client is not None:
                try:
                    self._worker_stats_table = self._log_client.get_table(
                        ZEPHYR_WORKER_STATS_NAMESPACE, ZephyrWorkerStat
                    )
                except Exception:
                    logger.warning(
                        "Could not initialize finelog worker stats table; worker stats disabled", exc_info=True
                    )
                    self._log_client = None
        return self._worker_stats_table

    def execute(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
    ) -> tuple[TaskResult, dict[str, int]]:
        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, num_workers=self._num_workers)
        self._ctx = ctx
        worker_token = _worker_ctx_var.set(ctx)
        stop_event = threading.Event()
        log_table = self._get_worker_stats_table()
        proc = psutil.Process()
        start_time = time.monotonic()
        cpu_times_at_start = proc.cpu_times()
        cpu_s_at_start = cpu_times_at_start.user + cpu_times_at_start.system
        proc.cpu_percent()  # prime so subsequent calls have a baseline
        if log_table is not None:
            _emit_runner_stat(
                log_table, task, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx, proc, cpu_s_at_start
            )
        sampler = threading.Thread(
            target=_periodic_sampler,
            kwargs={
                "stop_event": stop_event,
                "ctx": ctx,
                "interval": SUBPROCESS_STATS_INTERVAL,
                "cpu_s_at_start": cpu_s_at_start,
                "log_table": log_table,
                "task": task,
                "execution_id": execution_id,
                "start_time": start_time,
                "proc": proc,
            },
            daemon=True,
            name="zephyr-inline-stats-sampler",
        )
        sampler.start()
        try:
            result = _run_task_with_ctx(task, chunk_prefix, execution_id, ctx)
        finally:
            stop_event.set()
            sampler.join(timeout=2.0)
            _sample_process_stats(ctx, cpu_s_at_start, task.stage_name, proc)
            if log_table is not None:
                _emit_runner_stat(
                    log_table, task, execution_id, ZephyrWorkerStatStatus.END, start_time, ctx, proc, cpu_s_at_start
                )
            _worker_ctx_var.reset(worker_token)
            self._ctx = None
        return result, dict(ctx._counters)

    def live_counters(self) -> dict[str, int]:
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
        throughput = _stage_throughput(ctx._counters, stage_name, elapsed)
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
    ) -> tuple[TaskResult, dict[str, int]]:
        finelog_url = _resolve_finelog_url()  # Requires Iris context, so called here and passed to subprocess
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

    def live_counters(self) -> dict[str, int]:
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
    log_client: Any = None
    log_table: Any = None
    proc = psutil.Process()
    start_time = time.monotonic()
    cpu_times_at_start = proc.cpu_times()
    cpu_s_at_start = cpu_times_at_start.user + cpu_times_at_start.system
    proc.cpu_percent()  # prime so subsequent calls have a baseline
    try:
        with open(task_file, "rb") as f:
            task, chunk_prefix, execution_id, finelog_url = cloudpickle.load(f)

        if finelog_url:
            try:
                log_client = LogClient.connect(finelog_url)
                log_table = log_client.get_table(ZEPHYR_WORKER_STATS_NAMESPACE, ZephyrWorkerStat)
            except Exception:
                logger.warning("Could not connect to finelog in subprocess; worker stats disabled", exc_info=True)
                log_client = None

        ctx = _InProcessWorkerContext(chunk_prefix, execution_id, num_workers=num_workers)
        _worker_ctx_var.set(ctx)

        shard_monotonic_start = time.monotonic()
        if log_table is not None:
            _emit_runner_stat(
                log_table, task, execution_id, ZephyrWorkerStatStatus.START, start_time, ctx, proc, cpu_s_at_start
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
                "log_table": log_table,
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
            with suppress(Exception):
                _sample_process_stats(ctx, cpu_s_at_start, task.stage_name, proc)
        if log_table is not None and ctx is not None:
            try:
                _emit_runner_stat(
                    log_table,
                    task,
                    execution_id,
                    ZephyrWorkerStatStatus.END,
                    start_time,
                    ctx,
                    proc,
                    cpu_s_at_start,
                )
            except Exception:
                logger.warning("Failed to emit END runner stat", exc_info=True)
        if log_client is not None:
            with suppress(Exception):
                log_client.close()

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
