# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU provisioning-outcome gauges, collected from the controller's own logs.

TPU creation resolves in two stages: the autoscaler logs ``Created slice <id> for
group tpu_<variant>_<size>-<zone>`` when the create POST is accepted, then later
exactly one of ``Bootstrap completed for TPU slice <id>`` (success) or ``Bootstrap
failed for slice <id>: <reason>`` (failure, almost always a zone stockout). Each
outcome is attributed to a (zone, tpu_type) pool by mapping its slice_id back to
the Created line's group — the slice_id string itself drops the zone letter
inconsistently, so the group is the only reliable source.

I/O (the bounded finelog query) is separated from the pure ``aggregate`` so the
attribution and windowing logic is unit-testable without a live controller.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from finelog.client.log_client import LogClient
from sample import Sample, sample

# The controller's logs live under this key in the finelog "log" namespace.
CONTROLLER_KEY = "/system/controller"
MS_PER_HOUR = 3_600_000
# Bootstrap can lag its Created line by up to the bootstrap timeout (~1h), so
# fetch this much extra history before the window to map edge outcomes to their
# (older) Created line. Mapping-only; outcomes are still counted within-window.
MAP_MARGIN_HOURS = 1.0
# Per-stream row ceiling for one query; a window's worth of one message type is
# far below this even in a dense stockout storm (~1.2k failures/hr).
MAX_ROWS = 500_000

# Substrings narrow the finelog scan to one message type (TPU only). marin-tpu-
# excludes CPU slices; "for TPU slice" is already TPU-specific for completions.
CREATED_LIKE = "%Created slice marin-tpu-%"
COMPLETED_LIKE = "%Bootstrap completed for TPU slice %"
FAILED_LIKE = "%Bootstrap failed for slice marin-tpu-%"

CREATED_RE = re.compile(r"Created slice (\S+) for group (\S+)")
COMPLETED_RE = re.compile(r"Bootstrap completed for TPU slice (\S+)")
FAILED_RE = re.compile(r"Bootstrap failed for slice (\S+?): (.*)")

# A failure is a capacity stockout (the dominant mode) iff its reason says so;
# everything else (internal error, queued-resource quota) is a real fault.
STOCKOUT_MARKER = "no more capacity"

METRIC_SUCCESS = "tpu_provision_success"
METRIC_FAILURE = "tpu_provision_failure"
METRIC_FAILURE_STOCKOUT = "tpu_provision_failure_stockout"
METRIC_FAILURE_ERROR = "tpu_provision_failure_error"
METRIC_OUTCOMES = "tpu_provision_outcomes"
METRIC_LATENCY_SECONDS = "tpu_provision_latency_seconds"
METRIC_SUCCESS_RATIO = "tpu_provision_success_ratio"
METRIC_CREATED = "tpu_provision_created"
METRIC_POOLS_PLACING = "tpu_provision_pools_placing"
METRIC_POOLS_STOCKOUT_DEAD = "tpu_provision_pools_stockout_dead"
METRIC_UNMAPPED = "tpu_provision_unmapped"
METRIC_WINDOW_HOURS = "tpu_provision_window_hours"

# Label value marking the fleet-wide aggregate series (no zone/tpu_type).
FLEET = "fleet"
# Latency quantiles emitted per pool and fleet-wide.
QUANTILES = {"p50": 0.50, "p95": 0.95}


@dataclass(frozen=True)
class Pool:
    tpu_type: str
    zone: str


def group_to_type_zone(group: str) -> Pool | None:
    """Parse ``tpu_<variant>_<size>-<zone>`` into a Pool.

    e.g. ``tpu_v6e-preemptible_8-europe-west4-a`` -> ``Pool("v6e-preemptible-8",
    "europe-west4-a")``. Returns None for non-TPU or malformed groups.
    """
    if not group.startswith("tpu_"):
        return None
    rest = group[len("tpu_") :]
    if "_" not in rest:
        return None
    variant, size_zone = rest.rsplit("_", 1)
    if "-" not in size_zone:
        return None
    size, zone = size_zone.split("-", 1)
    return Pool(tpu_type=f"{variant}-{size}", zone=zone)


@dataclass
class _Tally:
    success: int = 0
    failure: int = 0
    stockout: int = 0
    error: int = 0


def _percentile(values_sorted: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of a pre-sorted, non-empty sequence."""
    rank = max(0, min(len(values_sorted) - 1, math.ceil(q * len(values_sorted)) - 1))
    return values_sorted[rank]


def _latency_samples(metric_labels: dict[str, str], latencies: list[float]) -> list[Sample]:
    if not latencies:
        return []
    ordered = sorted(latencies)
    return [
        sample(METRIC_LATENCY_SECONDS, _percentile(ordered, q), quantile=name, **metric_labels)
        for name, q in QUANTILES.items()
    ]


def _pool_samples(pool: Pool, tally: _Tally, latencies: list[float]) -> list[Sample]:
    labels = {"zone": pool.zone, "tpu_type": pool.tpu_type}
    samples = [
        sample(METRIC_SUCCESS, tally.success, **labels),
        sample(METRIC_FAILURE, tally.failure, **labels),
        sample(METRIC_FAILURE_STOCKOUT, tally.stockout, **labels),
        sample(METRIC_FAILURE_ERROR, tally.error, **labels),
        sample(METRIC_OUTCOMES, tally.success + tally.failure, **labels),
    ]
    samples.extend(_latency_samples(labels, latencies))
    return samples


def aggregate(
    created: Sequence[tuple[int, str]],
    completed: Sequence[tuple[int, str]],
    failed: Sequence[tuple[int, str]],
    *,
    window_hours: float,
    now_ms: int,
) -> list[Sample]:
    """Attribute outcomes to pools and emit gauge samples for the trailing window.

    Each input is ``(epoch_ms, log_line)``. The slice->group map is built from all
    fetched Created lines (including the pre-window margin); only outcomes at/after
    the cutoff are counted. Latency is Created->Completed wall time, in seconds.
    """
    cutoff = now_ms - int(window_hours * MS_PER_HOUR)

    pool_of: dict[str, Pool] = {}
    created_ms: dict[str, int] = {}
    created_in_window = 0
    for ms, line in created:
        match = CREATED_RE.search(line)
        if not match:
            continue
        slice_id, group = match.group(1), match.group(2)
        pool = group_to_type_zone(group)
        if pool is None:
            continue
        pool_of[slice_id] = pool
        created_ms.setdefault(slice_id, ms)
        if ms >= cutoff:
            created_in_window += 1

    tallies: dict[Pool, _Tally] = defaultdict(_Tally)
    latencies: dict[Pool, list[float]] = defaultdict(list)
    unmapped = 0

    for ms, line in completed:
        if ms < cutoff:
            continue
        match = COMPLETED_RE.search(line)
        if not match:
            continue
        pool = pool_of.get(match.group(1))
        if pool is None:
            unmapped += 1
            continue
        tallies[pool].success += 1
        create_ms = created_ms.get(match.group(1))
        if create_ms is not None:
            latencies[pool].append((ms - create_ms) / 1000)

    for ms, line in failed:
        if ms < cutoff:
            continue
        match = FAILED_RE.search(line)
        if not match:
            continue
        pool = pool_of.get(match.group(1))
        if pool is None:
            unmapped += 1
            continue
        tally = tallies[pool]
        tally.failure += 1
        if STOCKOUT_MARKER in match.group(2).lower():
            tally.stockout += 1
        else:
            tally.error += 1

    samples: list[Sample] = []
    fleet = _Tally()
    fleet_latencies: list[float] = []
    pools_placing = 0
    pools_stockout_dead = 0
    for pool, tally in tallies.items():
        samples.extend(_pool_samples(pool, tally, latencies[pool]))
        fleet.success += tally.success
        fleet.failure += tally.failure
        fleet.stockout += tally.stockout
        fleet.error += tally.error
        fleet_latencies.extend(latencies[pool])
        if tally.success > 0:
            pools_placing += 1
        elif tally.failure > 0:
            pools_stockout_dead += 1

    fleet_outcomes = fleet.success + fleet.failure
    fleet_labels = {"scope": FLEET}
    samples.extend(
        [
            sample(METRIC_SUCCESS, fleet.success, **fleet_labels),
            sample(METRIC_FAILURE, fleet.failure, **fleet_labels),
            sample(METRIC_FAILURE_STOCKOUT, fleet.stockout, **fleet_labels),
            sample(METRIC_FAILURE_ERROR, fleet.error, **fleet_labels),
            sample(METRIC_OUTCOMES, fleet_outcomes, **fleet_labels),
            sample(METRIC_CREATED, created_in_window, **fleet_labels),
            sample(METRIC_POOLS_PLACING, pools_placing, **fleet_labels),
            sample(METRIC_POOLS_STOCKOUT_DEAD, pools_stockout_dead, **fleet_labels),
            sample(METRIC_UNMAPPED, unmapped, **fleet_labels),
            sample(METRIC_WINDOW_HOURS, window_hours, **fleet_labels),
        ]
    )
    if fleet_outcomes > 0:
        samples.append(sample(METRIC_SUCCESS_RATIO, fleet.success / fleet_outcomes, **fleet_labels))
    samples.extend(_latency_samples(fleet_labels, fleet_latencies))
    return samples


def _query(finelog: LogClient, since_ms: int, like: str) -> list[tuple[int, str]]:
    """Fetch ``(epoch_ms, data)`` for controller log lines matching ``like`` at or
    after ``since_ms``."""
    # The epoch_ms predicate prunes finelog's Parquet row-groups, so the window
    # returns without the deep-scan timeout that plagues newest-first tail reads.
    table = finelog.query(
        f'SELECT epoch_ms, data FROM "log" '
        f"WHERE key = '{CONTROLLER_KEY}' AND epoch_ms >= {since_ms} AND data LIKE '{like}' "
        f"ORDER BY epoch_ms",
        max_rows=MAX_ROWS,
    )
    return list(zip(table.column("epoch_ms").to_pylist(), table.column("data").to_pylist(), strict=True))


def collect_tpu_provisioning(
    finelog: LogClient, *, window_hours: float, now: Callable[[], float] = time.time
) -> list[Sample]:
    """Query the three outcome streams over the trailing window and aggregate them
    into gauge samples. ``now`` is injectable for testing."""
    now_ms = int(now() * 1000)
    since_ms = now_ms - int((window_hours + MAP_MARGIN_HOURS) * MS_PER_HOUR)
    created = _query(finelog, since_ms, CREATED_LIKE)
    completed = _query(finelog, since_ms, COMPLETED_LIKE)
    failed = _query(finelog, since_ms, FAILED_LIKE)
    return aggregate(created, completed, failed, window_hours=window_hours, now_ms=now_ms)
