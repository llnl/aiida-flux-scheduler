"""Check scheduler integration with the exact deferred submission protocol."""

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiida.schedulers import SchedulerError
from aiida.schedulers.datastructures import JobTemplate

from aiida_flux_scheduler.deferred import handle_request, validate_request
from aiida_flux_scheduler.flux import FluxJobResource, FluxScheduler


def _submission_request(tmp_path):
    """Build a durable intent used by the isolated protocol tests."""
    return {
        "op": "register",
        "allocation_id": "123",
        "calculation_uuid": str(uuid.uuid4()),
        "working_directory": str(Path(tmp_path)),
        "submit_script": "submit.sh",
    }


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


def test_generated_short_walltime_is_registered_in_seconds(monkeypatch):
    """A generated 60-second pooled job keeps seconds through registration."""
    scheduler = FluxScheduler()
    template = JobTemplate()
    template.job_name = "aiida-1"
    template.job_resource = FluxJobResource(
        num_machines=1,
        num_mpiprocs_per_machine=1,
        flux_pool="shared",
    )
    template.max_wallclock_seconds = 60
    header = scheduler._get_submit_script_header(template)
    scheduler._transport = SimpleNamespace(
        exec_command_wait=lambda command: (0, header, "")
    )
    node = SimpleNamespace(
        uuid=str(uuid.uuid4()),
        get_metadata_inputs=lambda: {
            "metadata": {"options": {"resources": {"flux_pool": "shared"}}}
        },
    )
    scheduler._validate_job_fits_pool(
        node,
        {
            "pool": "shared",
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 1,
                "max_wallclock_seconds": 600,
            },
        },
    )
    monkeypatch.setattr("aiida_flux_scheduler.flux.load_node", lambda _: node)
    monkeypatch.setattr(
        scheduler, "_flux_allocation_with_ownership", lambda _: ("123", False)
    )
    monkeypatch.setattr(scheduler, "_maybe_cleanup_journals", lambda _: None)
    requests = []

    def request(value):
        """Record only protocol input; no remote journal is touched."""
        requests.append(value)
        return {} if value["op"] == "lookup" else {"state": "queued"}

    monkeypatch.setattr(scheduler, "_run_deferred_request", request)

    result = scheduler.submit_job("/remote", "submit.sh")

    assert result == f"123:pending-{node.uuid}"
    assert requests[1]["requested_wallclock_seconds"] == 60


def test_short_walltime_can_run_in_longer_pool(tmp_path):
    """An exact 60-second intent can submit within a 600-second allocation."""
    calls = []

    def runner(args, cwd=None):
        """Model a running 600-second parent and ready nested broker."""
        calls.append(args)
        if args[:2] == ["flux", "jobs"]:
            return json.dumps(
                {"id": 123, "state": "RUN", "expiration": time.time() + 600}
            )
        if args[4:5] == ["python"]:
            return json.dumps({"state": "run"})
        if args[4:6] == ["jobs", "-a"]:
            return json.dumps({"jobs": []})
        if args[4:5] == ["batch"]:
            return "456\n"
        raise AssertionError(args)

    submission = _submission_request(tmp_path)
    submission["requested_wallclock_seconds"] = 60
    handle_request(submission, home=tmp_path)

    result = handle_request(
        {
            "op": "poll",
            "allocation_id": submission["allocation_id"],
            "calculation_uuid": submission["calculation_uuid"],
        },
        home=tmp_path,
        runner=runner,
    )

    assert result["state"] == "submitted"
    assert any(args[4:5] == ["batch"] for args in calls)


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
