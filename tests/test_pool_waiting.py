"""Regression coverage for queued shared allocations and deferred scheduler IDs."""

import uuid
from types import SimpleNamespace

import pytest
from aiida.schedulers import SchedulerError
from aiida.schedulers.datastructures import JobInfo, JobState

from aiida_flux_scheduler.flux import FluxScheduler
from aiida_flux_scheduler.pools import (
    create_or_update_pool_group,
    get_pool_runtime,
    update_pool_runtime,
)


@pytest.fixture
def pool(aiida_profile_clean, aiida_computer_local):
    """Create an isolated pool definition for allocation-state tests."""
    computer = aiida_computer_local(label="queued-pool", configure=True)
    group, _ = create_or_update_pool_group(
        "shared",
        computer.label,
        {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
            "max_wallclock_seconds": 600,
        },
    )
    return group, SimpleNamespace(user=group.user, computer=computer)


def test_three_submitters_share_queued_allocation(pool, monkeypatch):
    """Queue delays do not hold the pool lock or trigger another allocation."""
    group, node = pool
    lock_held = []
    creations = []
    job = JobInfo()
    job.job_state = JobState.QUEUED
    job.job_id = "allocation"
    for parent in (1, 2, 3):
        scheduler = FluxScheduler()
        scheduler.flux_values = {"t": 120}
        monkeypatch.setattr(
            scheduler, "_acquire_pool_lock", lambda *a, **kw: lock_held.append(True)
        )
        monkeypatch.setattr(scheduler, "_release_pool_lock", lambda *a: lock_held.pop())
        monkeypatch.setattr(scheduler, "_get_top_level_job", lambda *a: job)

        def create(config, pk):
            """Record one immediate allocation submission while locked."""
            assert lock_held
            creations.append(config["generation"])
            return "allocation"

        monkeypatch.setattr(scheduler, "_start_allocation", create)
        assert (
            scheduler._get_or_create_pooled_allocation(node, parent, {"pool": "shared"})
            == "allocation"
        )
        assert not lock_held
    assert len(creations) == 1
    assert get_pool_runtime(group)["state"] == "queued"


def test_unknown_allocation_preserves_reference(pool, monkeypatch):
    """An unavailable observation is not evidence that replacement is safe."""
    group, node = pool
    update_pool_runtime(group, current_flux_id="allocation")
    scheduler = FluxScheduler()
    monkeypatch.setattr(scheduler, "_acquire_pool_lock", lambda *a, **kw: None)
    monkeypatch.setattr(scheduler, "_release_pool_lock", lambda *a: None)
    monkeypatch.setattr(scheduler, "_get_top_level_job", lambda *a: None)
    monkeypatch.setattr(
        scheduler, "_start_allocation", lambda *a: pytest.fail("duplicate allocation")
    )
    with pytest.raises(SchedulerError, match="preserving its reference"):
        scheduler._get_or_create_pooled_allocation(node, 1, {"pool": "shared"})
    assert get_pool_runtime(group)["current_flux_id"] == "allocation"


def test_generation_survives_lost_allocation_response(pool, monkeypatch):
    """A retry uses the same allocation reconciliation identity."""
    group, node = pool
    scheduler = FluxScheduler()
    generations = []
    monkeypatch.setattr(scheduler, "_acquire_pool_lock", lambda *a, **kw: None)
    monkeypatch.setattr(scheduler, "_release_pool_lock", lambda *a: None)

    def create(config, parent):
        """Simulate a lost response followed by receipt reconciliation."""
        generations.append(config["generation"])
        if len(generations) == 1:
            raise SchedulerError("response lost")
        return "allocation"

    monkeypatch.setattr(scheduler, "_start_allocation", create)
    with pytest.raises(SchedulerError, match="response lost"):
        scheduler._get_or_create_pooled_allocation(node, 1, {"pool": "shared"})
    scheduler._get_or_create_pooled_allocation(node, 2, {"pool": "shared"})
    assert generations[0] == generations[1]
    assert get_pool_runtime(group)["current_flux_id"] == "allocation"


def test_deferred_poll_stays_queued_without_submit_retry(monkeypatch):
    """Repeated readiness polling keeps the accepted intent in scheduler waiting."""
    scheduler = FluxScheduler()
    job_id = f"allocation:pending-{uuid.uuid4()}"
    monkeypatch.setattr(
        scheduler, "_run_deferred_request", lambda request: {"state": "queued"}
    )
    for _ in range(100):
        assert (
            scheduler.get_jobs([job_id], as_dict=True)[job_id].job_state
            == JobState.QUEUED
        )


@pytest.mark.parametrize("state", ["done", "cancelled"])
def test_terminal_alias_does_not_query_shutdown_pool(monkeypatch, state):
    """Durably terminal aliases stay terminal after their enclosing broker exits."""
    scheduler = FluxScheduler()
    job_id = f"123:pending-{uuid.uuid4()}"
    monkeypatch.setattr(
        scheduler,
        "_run_deferred_request",
        lambda request: {"state": state, "child_id": "456"},
    )
    assert scheduler.get_jobs([job_id], as_dict=True)[job_id].job_state == JobState.DONE


def test_deferred_retry_returns_original_allocation(monkeypatch):
    """An existing intent survives allocation replacement and pool disabling."""
    scheduler = FluxScheduler()
    node = SimpleNamespace(
        uuid=str(uuid.uuid4()),
        get_metadata_inputs=lambda: {
            "metadata": {"options": {"resources": {"flux_pool": "shared"}}},
        },
    )
    monkeypatch.setattr(scheduler, "_parse_pk", lambda *a: 1)
    monkeypatch.setattr("aiida_flux_scheduler.flux.load_node", lambda *a: node)
    monkeypatch.setattr(
        scheduler,
        "_run_deferred_request",
        lambda request: {"allocation_id": "original"},
    )
    monkeypatch.setattr(
        scheduler,
        "_flux_allocation_with_ownership",
        lambda *a: pytest.fail("resolve again"),
    )
    assert (
        scheduler.submit_job("/remote", "submit.sh") == f"original:pending-{node.uuid}"
    )
