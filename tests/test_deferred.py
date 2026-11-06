"""Isolated tests for durable deferred submission and the embedded remote helper."""

import fcntl
import json
import os
import shlex
import subprocess
import sys
from uuid import uuid4

import pytest

from aiida_flux_scheduler.deferred import build_deferred_command, handle_request


@pytest.fixture
def submission(tmp_path):
    """Return an immutable register request with shell-sensitive paths."""
    return {
        "op": "register",
        "allocation_id": "123",
        "calculation_uuid": str(uuid4()),
        "working_directory": str(tmp_path),
        "submit_script": "script with ' quotes.sh",
    }


class FakeFlux:
    """Record bounded CLI calls and simulate persistent Flux job state."""

    def __init__(self):
        """Initialize a queued allocation with no child."""
        self.calls = []
        self.parent = {"id": 123, "state": "SCHED"}
        self.ready = "run"
        self.jobs = []
        self.submit_error = None
        self.lookup_error = None
        self.proxy_error = None

    def __call__(self, args, cwd=None):
        """Simulate the exact commands used by the remote protocol."""
        self.calls.append((args, cwd))
        if args[:2] == ["flux", "jobs"]:
            return json.dumps(self.parent)
        assert args[:4] == ["flux", "proxy", "123", "flux"]
        if self.proxy_error:
            raise self.proxy_error
        command = args[4:]
        if command[0] == "python":
            return json.dumps({"state": self.ready})
        if command[:2] == ["jobs", "-a"]:
            if self.lookup_error:
                raise self.lookup_error
            assert "--count=0" in command
            return json.dumps({"jobs": self.jobs})
        if command[0] == "jobs":
            return json.dumps(self.jobs[0])
        if command[0] == "batch":
            self.jobs = [{"id": 456, "state": "SCHED"}]
            if self.submit_error:
                raise self.submit_error
            return "456\n"
        if command[:1] == ["cancel"]:
            assert command[1] == "456"
            self.jobs[0]["state"] = "INACTIVE"
            return ""
        if command[:2] == ["job", "info"]:
            return '{"attributes": {"system": {}}}'
        raise AssertionError(args)


def operate(request, tmp_path, flux, op):
    """Apply a non-register operation using only the stable identity."""
    return handle_request(
        {
            k: v
            for k, v in {**request, "op": op}.items()
            if k
            not in ("working_directory", "submit_script", "requested_wallclock_seconds")
        },
        home=tmp_path,
        runner=flux,
    )


def journal(request, tmp_path):
    """Return the immutable identity's journal path."""
    return (
        tmp_path
        / ".aiida-flux-scheduler"
        / "submissions"
        / (request["calculation_uuid"] + ".json")
    )


def test_register_is_idempotent_and_immutable(submission, tmp_path):
    """Repeated registration never submits or changes an existing intent."""
    request = submission
    flux = FakeFlux()
    first = handle_request(request, home=tmp_path, runner=flux)
    assert first["state"] == "queued"
    assert first["alias"] == "pending-" + request["calculation_uuid"]
    assert handle_request(request, home=tmp_path, runner=flux) == first
    assert flux.calls == []
    with pytest.raises(ValueError, match="Conflicting"):
        handle_request({**request, "submit_script": "other.sh"}, home=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        {"extra": 1},
        {"calculation_uuid": "../escape"},
        {"allocation_id": "-123"},
        {"allocation_id": "123; touch /tmp/unsafe"},
        {"working_directory": "relative"},
        {"submit_script": "\x00"},
        {"op": "allocate"},
    ],
)
def test_request_validation(submission, change):
    """Reject unsafe or unknown request fields before constructing a command."""
    request = submission
    with pytest.raises(ValueError):
        build_deferred_command({**request, **change})


@pytest.mark.parametrize("state", ["SCHED", "PRIORITY", "DEPEND", 8])
def test_queued_parent_does_not_connect_or_submit(submission, tmp_path, state):
    """Pending allocation polls are bounded top-level queries only."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = state
    handle_request(request, home=tmp_path)
    assert operate(request, tmp_path, flux, "poll")["state"] == "queued"
    assert len(flux.calls) == 1


@pytest.mark.parametrize("state", ["INACTIVE", 64])
def test_dead_parent_reports_failure_without_child(submission, tmp_path, state):
    """An ended allocation never looks like a successful queued calculation."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = state
    handle_request(request, home=tmp_path)
    result = operate(request, tmp_path, flux, "poll")
    assert result["state"] == "done"
    assert "reason" in result and "child_id" not in result
    assert len(flux.calls) == 1


def test_parent_cleanup_waits_for_inactive(submission, tmp_path):
    """Do not retrieve files while the enclosing allocation is still cleaning up."""
    flux = FakeFlux()
    flux.parent["state"] = "CLEANUP"
    handle_request(submission, home=tmp_path)
    assert operate(submission, tmp_path, flux, "poll")["state"] == "queued"


def test_terminal_receipt_details_survive_pool_shutdown(submission, tmp_path):
    """A terminal receipt remains readable without querying the dead broker."""
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(submission, home=tmp_path)
    operate(submission, tmp_path, flux, "poll")
    flux.jobs[0]["state"] = "INACTIVE"
    assert operate(submission, tmp_path, flux, "poll")["state"] == "done"

    def unavailable(*args, **kwargs):
        """Reject every broker call after the terminal receipt was persisted."""
        pytest.fail("terminal receipt queried shutdown broker")

    assert operate(submission, tmp_path, unavailable, "poll")["state"] == "done"
    assert "intent" in operate(submission, tmp_path, unavailable, "detail")


def test_ready_submission_and_reconciliation(submission, tmp_path):
    """Ready polls submit once and reconcile the unique child including completion."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)
    assert operate(request, tmp_path, flux, "poll")["child_id"] == "456"
    batch = next(call for call in flux.calls if call[0][4:5] == ["batch"])
    assert batch[1] == request["working_directory"]
    assert batch[0][-2:] == ["--", request["submit_script"]]
    assert "--job-name=aiida-deferred-" + request["calculation_uuid"] in batch[0]
    assert operate(request, tmp_path, flux, "poll")["state"] == "submitted"
    flux.jobs[0].update(state=64, returncode=0)
    assert operate(request, tmp_path, flux, "poll")["job"]["returncode"] == 0
    assert sum(call[0][4:5] == ["batch"] for call in flux.calls) == 1


def test_not_ready_does_not_submit(submission, tmp_path):
    """RUN allocation state alone is insufficient while the broker initializes."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    flux.ready = "quorum"
    handle_request(request, home=tmp_path)
    assert operate(request, tmp_path, flux, "poll")["state"] == "queued"
    assert len(flux.calls) == 2


def test_queries_raise_instead_of_marking_done(submission, tmp_path):
    """Readiness and lookup failures preserve the queued intent."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    flux.lookup_error = RuntimeError("broker unavailable")
    handle_request(request, home=tmp_path)
    with pytest.raises(RuntimeError, match="unavailable"):
        operate(request, tmp_path, flux, "poll")
    assert json.loads(journal(request, tmp_path).read_text())["state"] == "queued"


def test_uncertain_submission_never_retries(submission, tmp_path):
    """Lost submission responses reconcile but never trigger a duplicate batch."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    flux.submit_error = subprocess.TimeoutExpired("flux batch", 5)
    handle_request(request, home=tmp_path)
    assert operate(request, tmp_path, flux, "poll")["state"] == "ambiguous"
    accepted = flux.jobs
    flux.jobs = []
    assert operate(request, tmp_path, flux, "poll")["state"] == "ambiguous"
    flux.jobs = accepted
    assert operate(request, tmp_path, flux, "poll")["state"] == "submitted"
    assert sum(call[0][4:5] == ["batch"] for call in flux.calls) == 1


def test_journal_precedes_remote_submit(submission, tmp_path):
    """A process dying during batch leaves a durable submitting fence."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)

    def crash(args, cwd=None):
        """Simulate process death after verifying the durable submitting intent."""
        if args[4:5] == ["batch"]:
            assert (
                json.loads(journal(request, tmp_path).read_text())["state"]
                == "submitting"
            )
            raise KeyboardInterrupt
        return flux(args, cwd=cwd)

    with pytest.raises(KeyboardInterrupt):
        operate(request, tmp_path, crash, "poll")
    assert operate(request, tmp_path, flux, "poll")["state"] == "ambiguous"
    assert not any(args[4:5] == ["batch"] for args, _ in flux.calls)


def test_cancel_queued_never_cancels_pool_or_submits(submission, tmp_path):
    """A cancellation tombstone permanently prohibits submission."""
    request = submission
    flux = FakeFlux()
    handle_request(request, home=tmp_path)
    assert operate(request, tmp_path, flux, "cancel")["state"] == "cancelled"
    assert operate(request, tmp_path, flux, "poll")["state"] == "cancelled"
    assert handle_request(request, home=tmp_path)["state"] == "cancelled"
    assert flux.calls == []


def test_cancel_submitted_only_cancels_child(submission, tmp_path):
    """Cancel the child id inside the allocation, never the allocation itself."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)
    operate(request, tmp_path, flux, "poll")
    assert operate(request, tmp_path, flux, "cancel")["child_id"] == "456"
    assert (
        ["flux", "proxy", "123", "flux", "cancel", "456"],
        None,
    ) in flux.calls
    count = len(flux.calls)
    operate(request, tmp_path, flux, "cancel")
    assert len(flux.calls) == count


def test_cancel_confirmed_done_journal_after_shutdown_is_idempotent(
    submission, tmp_path
):
    """Cancelling an already completed child does not poison its stable alias."""
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(submission, home=tmp_path)
    operate(submission, tmp_path, flux, "poll")
    flux.jobs[0]["state"] = "INACTIVE"
    assert operate(submission, tmp_path, flux, "poll")["state"] == "done"
    calls_before_shutdown = len(flux.calls)

    def unavailable(*args, **kwargs):
        """Fail if a terminal cancel attempts to contact the old broker."""
        pytest.fail("confirmed terminal cancellation queried shutdown broker")

    cancelled = operate(submission, tmp_path, unavailable, "cancel")
    polled = operate(submission, tmp_path, unavailable, "poll")

    assert cancelled["state"] == "done"
    assert polled["state"] == "done"
    assert len(flux.calls) == calls_before_shutdown


def test_cancel_submitted_child_after_parent_shutdown(submission, tmp_path):
    """Confirmed top-level inactivity completes cancellation without proxying."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)
    operate(request, tmp_path, flux, "poll")
    flux.parent["state"] = "INACTIVE"
    flux.proxy_error = RuntimeError("nested broker is gone")
    calls_before_cancel = len(flux.calls)

    result = operate(request, tmp_path, flux, "cancel")

    assert result["state"] == "cancelled"
    record = json.loads(journal(request, tmp_path).read_text())
    assert record["state"] == "cancelled"
    assert record["cancel_complete"] is True
    assert not any(
        args[:2] == ["flux", "proxy"]
        for args, _ in flux.calls[calls_before_cancel:]
    )
    assert operate(request, tmp_path, flux, "poll")["state"] == "cancelled"


def test_incomplete_cancel_resolves_after_parent_shutdown(submission, tmp_path):
    """A failed child query stays uncertain until top-level inactivity is known."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)
    operate(request, tmp_path, flux, "poll")
    flux.proxy_error = RuntimeError("nested broker is gone")
    with pytest.raises(RuntimeError, match="nested broker is gone"):
        operate(request, tmp_path, flux, "cancel")
    record = json.loads(journal(request, tmp_path).read_text())
    assert record["state"] == "cancelled"
    assert record["cancel_complete"] is False

    flux.parent["state"] = "INACTIVE"
    resolved = operate(request, tmp_path, flux, "poll")

    assert resolved["state"] == "cancelled"
    record = json.loads(journal(request, tmp_path).read_text())
    assert record["cancel_complete"] is True


def test_cancel_parent_query_failure_preserves_uncertainty(submission, tmp_path):
    """An unavailable top-level query is not evidence that the pool ended."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)
    operate(request, tmp_path, flux, "poll")

    def parent_unavailable(args, cwd=None):
        """Fail before returning any evidence about the parent allocation."""
        if args[:2] == ["flux", "jobs"]:
            raise RuntimeError("top-level broker unavailable")
        return flux(args, cwd=cwd)

    with pytest.raises(RuntimeError, match="top-level broker unavailable"):
        operate(request, tmp_path, parent_unavailable, "cancel")
    record = json.loads(journal(request, tmp_path).read_text())
    assert record["state"] == "cancelled"
    assert record["cancel_complete"] is False


def test_cancel_completed_child_after_watcher_stops_parent(submission, tmp_path):
    """A watcher may stop the pool after a child completes but before AiiDA polls."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(request, home=tmp_path)
    operate(request, tmp_path, flux, "poll")
    calls_before_shutdown = len(flux.calls)
    # The child has completed; the watcher shuts down before its terminal state
    # is recorded in the deferred journal.
    flux.jobs[0]["state"] = "INACTIVE"
    flux.parent["state"] = "INACTIVE"

    result = operate(request, tmp_path, flux, "cancel")

    assert result["state"] == "cancelled"
    record = json.loads(journal(request, tmp_path).read_text())
    assert record["cancel_complete"] is True
    assert not any(
        args[:2] == ["flux", "proxy"]
        for args, _ in flux.calls[calls_before_shutdown:]
    )


def test_cancel_ambiguous_reconciles_after_empty_lookup(submission, tmp_path):
    """Cancellation preserves uncertainty and later cancels a discovered child."""
    request = submission
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    flux.submit_error = RuntimeError("lost response")
    handle_request(request, home=tmp_path)
    operate(request, tmp_path, flux, "poll")
    accepted = flux.jobs
    flux.jobs = []
    with pytest.raises(RuntimeError, match="awaiting reconciliation"):
        operate(request, tmp_path, flux, "cancel")
    with pytest.raises(RuntimeError, match="needs reconciliation"):
        operate(request, tmp_path, flux, "poll")
    assert (
        handle_request(
            {"op": "lookup", "calculation_uuid": request["calculation_uuid"]},
            home=tmp_path,
        )["state"]
        == "cancelling"
    )
    flux.jobs = accepted
    result = operate(request, tmp_path, flux, "cancel")
    assert result["state"] == "cancelled" and result["child_id"] == "456"
    assert flux.jobs[0]["state"] == "INACTIVE"


def test_detail_returns_intent_or_child_jobspec(submission, tmp_path):
    """Detail is read-only and exposes the intent until a child is known."""
    request = submission
    flux = FakeFlux()
    handle_request(request, home=tmp_path)
    assert (
        operate(request, tmp_path, flux, "detail")["intent"]["submit_script"]
        == request["submit_script"]
    )
    assert not flux.calls
    flux.parent["state"] = "RUN"
    operate(request, tmp_path, flux, "poll")
    assert "attributes" in operate(request, tmp_path, flux, "detail")["jobspec"]


@pytest.mark.parametrize("version", [0, 2, True, "1"])
def test_journal_version_is_exact(submission, tmp_path, version):
    """Persisted records reject silent coercion and unsupported schemas."""
    request = submission
    handle_request(request, home=tmp_path)
    path = journal(request, tmp_path)
    record = json.loads(path.read_text())
    record["version"] = version
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="version"):
        operate(request, tmp_path, FakeFlux(), "poll")


def test_guard_is_stable_and_nonblocking(submission, tmp_path):
    """Atomic journal replacement never replaces the flock identity."""
    request = submission
    handle_request(request, home=tmp_path)
    with journal(request, tmp_path).with_suffix(".guard").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy"):
            operate(request, tmp_path, FakeFlux(), "poll")


def test_embedded_remote_command_with_fake_flux(submission, tmp_path):
    """Execute the shell-quoted helper with no installed AiiDA or real Flux."""
    request = submission
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    executable = bin_path / "flux"
    executable.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, sys\n"
        'if sys.argv[1] == "python":\n'
        ' os.execv(sys.executable, [sys.executable, "-S", *sys.argv[2:]])\n'
        'elif sys.argv[1] == "jobs":\n'
        ' print(json.dumps({"id": 123, "state": "SCHED"}))\n'
        "else:\n"
        " sys.exit(99)\n"
    )
    executable.chmod(0o700)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PATH": str(bin_path) + ":" + os.environ["PATH"],
    }
    command = build_deferred_command(request)
    assert json.loads(shlex.split(command)[-1]) == request
    result = subprocess.run(
        command, shell=True, env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["state"] == "queued"
    poll = {
        k: v
        for k, v in {**request, "op": "poll"}.items()
        if k not in ("working_directory", "submit_script")
    }
    result = subprocess.run(
        build_deferred_command(poll),
        shell=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["state"] == "queued"
    poll["allocation_id"] = "999"
    result = subprocess.run(
        build_deferred_command(poll),
        shell=True,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "error" in json.loads(result.stdout)


def test_lookup_original_allocation(submission, tmp_path):
    """Lookup finds the durable alias without resolving a new allocation."""
    query = {"op": "lookup", "calculation_uuid": submission["calculation_uuid"]}
    assert handle_request(query, home=tmp_path) == {}
    registered = handle_request(submission, home=tmp_path)
    assert handle_request(query, home=tmp_path) == registered


@pytest.mark.parametrize("expiration, state", [(0, "submitted"), (1, "done")])
def test_walltime_check(submission, tmp_path, expiration, state):
    """Unlimited pools permit children while expired pools fail without submitting."""
    flux = FakeFlux()
    flux.parent.update(state="RUN", expiration=expiration)
    submission["requested_wallclock_seconds"] = 60
    handle_request(submission, home=tmp_path)
    assert operate(submission, tmp_path, flux, "poll")["state"] == state
    assert any(args[4:5] == ["batch"] for args, _ in flux.calls) == (
        state == "submitted"
    )


def test_ended_parent_marker_with_child(submission, tmp_path):
    """An ended pool suppresses later proxy queries even when a child is recorded."""
    flux = FakeFlux()
    flux.parent["state"] = "RUN"
    handle_request(submission, home=tmp_path)
    operate(submission, tmp_path, flux, "poll")
    flux.parent["state"] = "INACTIVE"
    result = operate(submission, tmp_path, flux, "poll")
    assert result["allocation_ended"] is True and result["child_id"] == "456"


def test_allocation_success_normalizes_f58_and_does_not_repeat(tmp_path):
    """A confirmed receipt is reusable without another query or allocation."""
    generation = str(uuid4())
    request = {
        "op": "allocate",
        "generation": generation,
        "job_name": "aiida-allocation-" + generation,
        "command": "flux alloc --dry-run | flux job submit",
    }
    calls = []

    def runner(args):
        """Return one successful F58 allocation id."""
        calls.append(args)
        return "ƒ2\n" if args[0] == "bash" else '{"jobs": []}'

    assert handle_request(request, home=tmp_path, runner=runner)["allocation_id"] == "1"
    assert handle_request(request, home=tmp_path, runner=runner)["allocation_id"] == "1"
    assert len(calls) == 2


def test_allocation_receipt_fails_closed_and_reconciles(tmp_path):
    """Lost allocation responses are fenced until the named allocation appears."""
    generation = str(uuid4())
    allocation = {
        "op": "allocate",
        "generation": generation,
        "job_name": "aiida-allocation-" + generation,
        "command": "flux alloc --dry-run | flux job submit",
    }
    calls = []
    jobs = []

    def runner(args, cwd=None):
        """Simulate a lost allocation receipt and delayed Flux job-list visibility."""
        calls.append(args)
        if args[0] == "bash":
            path = (
                tmp_path
                / ".aiida-flux-scheduler"
                / "allocations"
                / (generation + ".json")
            )
            assert json.loads(path.read_text())["state"] == "submitting"
            assert args[:2] == ["bash", "-c"] and args[2].startswith("set -o pipefail;")
            raise RuntimeError("lost receipt")
        assert "--name=" + allocation["job_name"] in args
        return json.dumps({"jobs": jobs})

    assert (
        handle_request(allocation, home=tmp_path, runner=runner)["state"] == "ambiguous"
    )
    assert (
        handle_request(allocation, home=tmp_path, runner=runner)["state"] == "ambiguous"
    )
    assert sum(args[0] == "bash" for args in calls) == 1
    jobs.append({"id": 999, "state": "SCHED"})
    assert (
        handle_request(allocation, home=tmp_path, runner=runner)["allocation_id"]
        == "999"
    )
    count = len(calls)
    assert (
        handle_request(allocation, home=tmp_path, runner=runner)["allocation_id"]
        == "999"
    )
    assert len(calls) == count
    with pytest.raises(ValueError, match="Conflicting"):
        handle_request({**allocation, "command": "other"}, home=tmp_path, runner=runner)
