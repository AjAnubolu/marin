# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TPU provisioning aggregation: slice->pool attribution, windowing, the
stockout/error split, and latency. These exercise the pure ``aggregate`` over
synthetic log lines — no finelog needed."""

from __future__ import annotations

import json

import pytest
from tpu_provisioning import (
    METRIC_CREATED,
    METRIC_FAILURE,
    METRIC_FAILURE_ERROR,
    METRIC_FAILURE_STOCKOUT,
    METRIC_LATENCY_SECONDS,
    METRIC_OUTCOMES,
    METRIC_POOLS_PLACING,
    METRIC_POOLS_STOCKOUT_DEAD,
    METRIC_SUCCESS,
    METRIC_SUCCESS_RATIO,
    METRIC_UNMAPPED,
    Pool,
    aggregate,
    group_to_type_zone,
)

MS_PER_HOUR = 3_600_000
NOW_MS = 1_700_000_000_000
CUTOFF = NOW_MS - 3 * MS_PER_HOUR  # 3h window
T0 = CUTOFF + 60_000  # 1 min inside the window


def _created(sid: str, group: str) -> str:
    return f"2026-06-15 00:00:00,000 autoscaler.runtime Created slice {sid} for group {group}"


def _completed(sid: str) -> str:
    return f"2026-06-15 00:00:00,000 gcp.workers Bootstrap completed for TPU slice {sid} (1 workers)"


def _failed(sid: str, reason: str) -> str:
    return f"2026-06-15 00:00:00,000 gcp.workers Bootstrap failed for slice {sid}: {reason}"


STOCKOUT = 'There is no more capacity in the zone "europe-west4-b"; you can try ...'
INTERNAL = "TPU operation failed: an internal error [EID: 0xabc]"


def _find(samples, metric, **labels):
    want = json.dumps(labels, sort_keys=True)
    vals = [s.value for s in samples if s.metric == metric and s.labels == want]
    assert len(vals) == 1, f"{metric} {labels}: {vals}"
    return vals[0]


@pytest.mark.parametrize(
    "group,expected",
    [
        ("tpu_v6e-preemptible_8-europe-west4-a", Pool("v6e-preemptible-8", "europe-west4-a")),
        ("tpu_v5e-serving_4-europe-west4-b", Pool("v5e-serving-4", "europe-west4-b")),
        ("tpu_v5p-preemptible_128-us-central1-a", Pool("v5p-preemptible-128", "us-central1-a")),
        ("cpu_vm_e2_highmem_2_ondemand-us-east1-b", None),  # non-TPU
        ("tpu_garbage", None),  # no size/zone
    ],
)
def test_group_to_type_zone(group, expected):
    assert group_to_type_zone(group) == expected


P1_GROUP = "tpu_v6e-preemptible_8-us-east5-b"
P2_GROUP = "tpu_v5e-serving_8-europe-west4-b"
P1 = {"zone": "us-east5-b", "tpu_type": "v6e-preemptible-8"}
P2 = {"zone": "europe-west4-b", "tpu_type": "v5e-serving-8"}
FLEET = {"scope": "fleet"}


@pytest.fixture
def samples():
    created = [
        (T0, _created("sliceA", P1_GROUP)),
        (T0, _created("sliceB", P1_GROUP)),
        (T0, _created("sliceC", P2_GROUP)),
        (T0, _created("sliceD", P2_GROUP)),
        # created + resolved before the window: must not count anywhere
        (CUTOFF - 100_000, _created("sliceF", P1_GROUP)),
    ]
    completed = [
        (T0 + 300_000, _completed("sliceA")),  # success, 300s provision latency
        (T0 + 10_000, _completed("sliceE")),  # unmapped: no Created line
    ]
    failed = [
        (T0 + 10_000, _failed("sliceB", STOCKOUT)),  # P1 stockout
        (T0 + 10_000, _failed("sliceC", STOCKOUT)),  # P2 stockout
        (T0 + 10_000, _failed("sliceD", INTERNAL)),  # P2 real error
        (CUTOFF - 50_000, _failed("sliceF", STOCKOUT)),  # before window: excluded
    ]
    return aggregate(created, completed, failed, window_hours=3.0, now_ms=NOW_MS)


def test_per_pool_counts(samples):
    assert _find(samples, METRIC_SUCCESS, **P1) == 1
    assert _find(samples, METRIC_FAILURE, **P1) == 1
    assert _find(samples, METRIC_FAILURE_STOCKOUT, **P1) == 1
    assert _find(samples, METRIC_FAILURE_ERROR, **P1) == 0
    assert _find(samples, METRIC_OUTCOMES, **P1) == 2

    assert _find(samples, METRIC_SUCCESS, **P2) == 0
    assert _find(samples, METRIC_FAILURE, **P2) == 2
    assert _find(samples, METRIC_FAILURE_STOCKOUT, **P2) == 1
    assert _find(samples, METRIC_FAILURE_ERROR, **P2) == 1


def test_fleet_rollup(samples):
    assert _find(samples, METRIC_SUCCESS, **FLEET) == 1
    assert _find(samples, METRIC_FAILURE, **FLEET) == 3
    assert _find(samples, METRIC_FAILURE_STOCKOUT, **FLEET) == 2
    assert _find(samples, METRIC_FAILURE_ERROR, **FLEET) == 1
    assert _find(samples, METRIC_OUTCOMES, **FLEET) == 4
    assert _find(samples, METRIC_SUCCESS_RATIO, **FLEET) == pytest.approx(0.25)
    assert _find(samples, METRIC_POOLS_PLACING, **FLEET) == 1  # P1
    assert _find(samples, METRIC_POOLS_STOCKOUT_DEAD, **FLEET) == 1  # P2
    assert _find(samples, METRIC_UNMAPPED, **FLEET) == 1  # sliceE
    # sliceF created+failed before the window contributes nothing
    assert _find(samples, METRIC_CREATED, **FLEET) == 4


def test_latency_from_created_to_completed(samples):
    assert _find(samples, METRIC_LATENCY_SECONDS, quantile="p50", **P1) == pytest.approx(300.0)
    assert _find(samples, METRIC_LATENCY_SECONDS, quantile="p95", **FLEET) == pytest.approx(300.0)


def test_success_ratio_is_fleet_scoped_only(samples):
    ratios = [s for s in samples if s.metric == METRIC_SUCCESS_RATIO]
    assert ratios and all(json.loads(s.labels) == FLEET for s in ratios)


def test_empty_inputs_emit_fleet_zeros_no_ratio():
    samples = aggregate([], [], [], window_hours=3.0, now_ms=NOW_MS)
    assert _find(samples, METRIC_OUTCOMES, **FLEET) == 0
    assert _find(samples, METRIC_POOLS_PLACING, **FLEET) == 0
    # ratio is undefined with no outcomes, so it is omitted rather than NaN
    assert not [s for s in samples if s.metric == METRIC_SUCCESS_RATIO]
