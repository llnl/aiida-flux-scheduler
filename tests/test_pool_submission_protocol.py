"""Check scheduler integration with the exact deferred submission protocol."""

import uuid
from types import SimpleNamespace

import pytest
from aiida.schedulers import SchedulerError

from aiida_flux_scheduler.deferred import validate_request
from aiida_flux_scheduler.flux import FluxScheduler


@pytest.mark.parametrize(
    "walltime, seconds", [("120s", 120), ("2m", 120), ("1.5h", 5400), ("2", 120)]
)
def test_parse_walltime_is_integer_seconds(walltime, seconds):
    """Normalize script durations before pool checks and intent registration."""
    scheduler = FluxScheduler()
    scheduler._transport = SimpleNamespace(
        exec_command_wait=lambda command: (
            0,
            f"#flux: --job-name=aiida-1\n#flux: -t {walltime}\n",
            "",
        )
    )
    assert scheduler._parse_pk("/remote", "submit.sh") == 1
    assert scheduler.flux_values["t"] == seconds
    assert type(scheduler.flux_values["t"]) is int


def test_allocation_submission_is_nonblocking_and_validated(monkeypatch):
    """Publish an immediate jobspec submission with an idle-grace watcher."""
    scheduler = FluxScheduler()
    generation = str(uuid.uuid4())
    requests = []

    def request(value):
        """Validate the same protocol executed by the remote helper."""
        requests.append(validate_request(value))
        return {"state": "submitted", "allocation_id": "123"}

    monkeypatch.setattr(scheduler, "_run_deferred_request", request)
    assert (
        scheduler._start_allocation(
            {
                "pool": "shared",
                "generation": generation,
                "job_name": f"aiida-allocation-{generation}",
                "resources": {
                    "num_machines": 1,
                    "num_mpiprocs_per_machine": 1,
                    "max_wallclock_seconds": 600,
                },
            },
            1,
        )
        == "123"
    )
    assert "--dry-run" in requests[0]["command"]
    assert "--bg" not in requests[0]["command"]
    assert "idle_since" in requests[0]["command"]


def test_ambiguous_allocation_never_falls_back_to_direct(monkeypatch):
    """An uncertain accepted allocation cannot silently become a direct job."""
    scheduler = FluxScheduler()
    monkeypatch.setattr(
        scheduler,
        "_run_deferred_request",
        lambda value: {
            "state": "ambiguous",
            "allocation_id": None,
        },
    )
    with pytest.raises(SchedulerError, match="refusing a duplicate or direct fallback"):
        scheduler._start_allocation(
            {"pool": "shared", "generation": str(uuid.uuid4())}, 1
        )
