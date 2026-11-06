"""Regression tests for bounded, best-effort automatic journal maintenance."""

import fcntl
import json
import os
import shlex
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from uuid import uuid4

import pytest
from aiida import orm

from aiida_flux_scheduler import deferred
from aiida_flux_scheduler import flux as flux_module
from aiida_flux_scheduler.flux import FluxScheduler

NOW = 2_000_000_000
RETENTION = 30 * 86400
INTERVAL = 3600


def no_flux(*args, **kwargs):
    """Reject scheduler commands from journal-only maintenance."""
    pytest.fail("Automatic journal cleanup contacted Flux")


@pytest.fixture
def clock(monkeypatch):
    """Expose a deterministic clock that can advance between cleanup claims."""
    value = [NOW]
    monkeypatch.setattr(deferred.time, "time", lambda: value[0])
    return value


@pytest.fixture
def make_journal(tmp_path, clock):
    """Create old version-one records without invoking Flux."""

    def create(state="done", cancel_complete=False, age=RETENTION + 1):
        """Persist an intent with the requested lifecycle state and age."""
        request = {
            "op": "register",
            "allocation_id": "123",
            "calculation_uuid": str(uuid4()),
            "working_directory": str(tmp_path / "remote work"),
            "submit_script": "submit.sh",
            "requested_wallclock_seconds": 60,
        }
        deferred.handle_request(request, home=tmp_path, runner=no_flux)
        path = (
            tmp_path
            / ".aiida-flux-scheduler"
            / "submissions"
            / (request["calculation_uuid"] + ".json")
        )
        record = json.loads(path.read_text())
        record.update(
            state=state,
            child_id="456" if state in ("submitted", "done", "cancelled") else None,
            cancel_complete=cancel_complete,
            reason="terminal" if state in ("done", "cancelled") else None,
        )
        path.write_text(json.dumps(record))
        os.utime(path, (clock[0] - age, clock[0] - age))
        return request["calculation_uuid"], path

    return create


def begin(home, batch_size=2):
    """Claim one remote maintenance batch through the public protocol."""
    return deferred.handle_request(
        {
            "op": "auto_cleanup_begin",
            "retention_seconds": RETENTION,
            "interval_seconds": INTERVAL,
            "batch_size": batch_size,
        },
        home=home,
        runner=no_flux,
    )


def finish(home, claim, approved=None):
    """Finish the exact reservation, approving eligible rows by default."""
    if approved is None:
        approved = {
            row["calculation_uuid"]: row["fingerprint"]
            for row in claim["records"]
            if row["action"] == "eligible"
        }
    return deferred.handle_request(
        {"op": "auto_cleanup_finish", "token": claim["token"], "approved": approved},
        home=home,
        runner=no_flux,
    )


def test_concurrent_claims_are_throttled(make_journal, tmp_path):
    """Concurrent clients share one remote reservation and interval throttle."""
    make_journal()
    barrier = Barrier(8)

    def claim():
        """Synchronize clients before their independent helper requests."""
        barrier.wait(timeout=10)
        return begin(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(lambda _: claim(), range(8)))
    winners = [response for response in responses if response.get("token")]
    assert len(winners) == 1
    assert all(
        response.get("skipped") for response in responses if not response.get("token")
    )
    finish(tmp_path, winners[0], {})
    assert begin(tmp_path).get("skipped")


def test_bounded_batches_rotate_past_retained_records(make_journal, tmp_path, clock):
    """Repeated scans cover the journal tree rather than retrying its first page."""
    journals = dict(make_journal() for _ in range(7))
    visited = set()
    for _ in range(4):
        claim = begin(tmp_path, batch_size=2)
        assert claim["token"]
        assert 0 < len(claim["records"]) <= 2
        visited.update(row["calculation_uuid"] for row in claim["records"])
        finish(tmp_path, claim, {})
        clock[0] += INTERVAL + 1
    assert visited == set(journals)
    assert all(
        json.loads(path.read_text())["version"] == 1 for path in journals.values()
    )


def test_stale_token_cannot_finish_replacement(make_journal, tmp_path, clock):
    """An expired client cannot compact records or consume its successor's claim."""
    _, path = make_journal()
    original = begin(tmp_path)
    clock[0] += INTERVAL + 1
    replacement = begin(tmp_path)
    assert replacement["token"] != original["token"]
    before = path.read_bytes()
    result = finish(tmp_path, original)
    assert result.get("skipped")
    assert path.read_bytes() == before
    finish(tmp_path, replacement)
    assert json.loads(path.read_text())["version"] == 2


@pytest.mark.parametrize(
    "state,cancel_complete,age,eligible",
    [
        ("done", False, RETENTION + 1, True),
        ("cancelled", True, RETENTION + 1, True),
        ("cancelled", False, RETENTION + 1, False),
        ("queued", False, RETENTION + 1, False),
        ("submitting", False, RETENTION + 1, False),
        ("submitted", False, RETENTION + 1, False),
        ("ambiguous", False, RETENTION + 1, False),
        ("done", False, RETENTION - 1, False),
    ],
)
def test_automatic_compaction_reuses_terminal_and_retention_checks(
    make_journal, tmp_path, state, cancel_complete, age, eligible
):
    """Only sufficiently old confirmed terminal journals become compact receipts."""
    uuid, path = make_journal(state, cancel_complete, age)
    before = path.read_bytes()
    claim = begin(tmp_path)
    row = next(row for row in claim["records"] if row["calculation_uuid"] == uuid)
    assert (row["action"] == "eligible") is eligible
    finish(tmp_path, claim)
    assert json.loads(path.read_text())["version"] == (2 if eligible else 1)
    if not eligible:
        assert path.read_bytes() == before


def test_changed_record_invalidates_automatic_approval(make_journal, tmp_path, clock):
    """Reservation ownership cannot bypass the existing record fingerprint guard."""
    _, path = make_journal()
    claim = begin(tmp_path)
    record = json.loads(path.read_text())
    record["reason"] = "changed after scan"
    path.write_text(json.dumps(record))
    os.utime(path, (clock[0] - RETENTION - 1, clock[0] - RETENTION - 1))
    before = path.read_bytes()
    finish(tmp_path, claim)
    assert path.read_bytes() == before


def test_malformed_uuid_filename_does_not_poison_cursor(make_journal, tmp_path, clock):
    """Malformed UUID-like names never enter persisted automatic cleanup state."""
    uuid, path = make_journal()
    malformed = path.parent / ("-" * 36 + ".json")
    malformed.write_text("opaque malformed filename")
    before = malformed.read_bytes()
    state_path = tmp_path / ".aiida-flux-scheduler" / "cleanup.json"
    for _ in range(2):
        claim = begin(tmp_path)
        state = json.loads(state_path.read_text())
        assert state["cursor"] == uuid
        assert state["names"] == [uuid]
        assert [row["calculation_uuid"] for row in claim["records"]] == [uuid]
        finish(tmp_path, claim, {})
        clock[0] += INTERVAL + 1
    assert malformed.read_bytes() == before


@pytest.mark.parametrize("existing_state", [False, True])
def test_discovery_budget_preserves_state(
    make_journal, tmp_path, clock, monkeypatch, existing_state
):
    """An oversized directory cannot publish a partial cursor or reservation."""
    make_journal()
    state_path = tmp_path / ".aiida-flux-scheduler" / "cleanup.json"
    if existing_state:
        finish(tmp_path, begin(tmp_path), {})
        clock[0] += INTERVAL + 1
        before = state_path.read_bytes(), state_path.stat().st_mtime_ns

    class OversizedDirectory:
        """Supply excessive entries without creating thousands of test files."""

        def __enter__(self):
            """Return a lazy directory iterator."""
            return (
                SimpleNamespace(name=f"entry-{index}.guard") for index in range(10001)
            )

        def __exit__(self, *args):
            """Close the fake iterator without suppressing discovery failures."""
            return False

    monkeypatch.setattr(deferred.os, "scandir", lambda directory: OversizedDirectory())
    monkeypatch.setattr(deferred.time, "monotonic", lambda: 0)
    with pytest.raises(RuntimeError, match="discovery budget exceeded"):
        begin(tmp_path)
    if existing_state:
        assert (state_path.read_bytes(), state_path.stat().st_mtime_ns) == before
    else:
        assert not state_path.exists()


@pytest.mark.parametrize("op", ["auto_cleanup_begin", "auto_cleanup_finish"])
def test_generated_automatic_command_has_remote_alarm(op):
    """Both automatic helper phases enforce their ten-second remote watchdog."""
    request = (
        {
            "op": op,
            "retention_seconds": RETENTION,
            "interval_seconds": INTERVAL,
            "batch_size": 2,
        }
        if op == "auto_cleanup_begin"
        else {"op": op, "token": str(uuid4()), "approved": {}}
    )
    command = shlex.split(deferred.build_deferred_command(request))
    assert "signal.alarm(10)" in command[4]
    assert json.loads(command[-1]) == request


def test_finish_cannot_expand_reserved_batch(make_journal, tmp_path):
    """Extra approvals cannot mutate journals outside the claimed page."""
    journals = dict(make_journal() for _ in range(4))
    scan = deferred.handle_request(
        {"op": "cleanup_scan", "retention_seconds": RETENTION},
        home=tmp_path,
        runner=no_flux,
    )
    approved = {row["calculation_uuid"]: row["fingerprint"] for row in scan["records"]}
    claim = begin(tmp_path, batch_size=1)
    reserved = claim["records"][0]["calculation_uuid"]
    finish(tmp_path, claim, approved)
    assert {
        uuid: json.loads(path.read_text())["version"] for uuid, path in journals.items()
    } == {uuid: 2 if uuid == reserved else 1 for uuid in journals}


def test_busy_journal_guard_does_not_block_batch(make_journal, tmp_path):
    """A guarded journal is retained while unrelated terminal records compact."""
    busy_uuid, busy_path = make_journal()
    other_uuid, other_path = make_journal()
    before = busy_path.read_bytes()
    guard_path = busy_path.with_suffix(".guard")
    inode = guard_path.stat().st_ino
    with guard_path.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        claim = begin(tmp_path)
        rows = {row["calculation_uuid"]: row for row in claim["records"]}
        assert rows[busy_uuid]["action"] == "retained"
        assert rows[other_uuid]["action"] == "eligible"
        finish(tmp_path, claim)
    assert busy_path.read_bytes() == before
    assert guard_path.stat().st_ino == inode
    assert json.loads(other_path.read_text())["version"] == 2


@pytest.mark.parametrize("existing_receipt", [False, True])
def test_cleanup_failure_does_not_fail_accepted_submission(
    monkeypatch, existing_receipt
):
    """Maintenance transport failures never invalidate accepted pending aliases."""
    scheduler = FluxScheduler()
    node = SimpleNamespace(uuid=str(uuid4()))
    scheduler.flux_values = {"t": 60}
    calls = []
    monkeypatch.setattr(flux_module, "load_node", lambda pk: node)
    monkeypatch.setattr(scheduler, "_parse_pk", lambda *args: 1)
    monkeypatch.setattr(
        scheduler,
        "_get_node_options",
        lambda node: {"resources": {"flux_pool": "shared"}},
    )
    monkeypatch.setattr(
        scheduler, "_flux_allocation_with_ownership", lambda pk: ("123", False)
    )

    def request(value):
        """Accept journal registration but reject only automatic maintenance."""
        calls.append(value)
        if value["op"] == "lookup":
            return {"allocation_id": "123"} if existing_receipt else {}
        if value["op"] == "register":
            return {"allocation_id": "123", "state": "queued"}
        raise RuntimeError("maintenance unavailable")

    monkeypatch.setattr(scheduler, "_run_deferred_request", request)
    assert scheduler.submit_job("/remote", "submit.sh") == f"123:pending-{node.uuid}"
    assert any(value["op"] == "auto_cleanup_begin" for value in calls)


def test_automatic_cleanup_never_acquires_pool_locks(monkeypatch):
    """The remote journal reservation is independent of allocation ownership locks."""
    scheduler = FluxScheduler()
    calls = []

    def forbidden(*args, **kwargs):
        """Reject pool-lock entry from maintenance."""
        pytest.fail("Automatic cleanup entered a pool ownership lock")

    def request(value):
        """Record a throttled journal request without real remote execution."""
        calls.append(value)
        return {"skipped": True, "records": []}

    monkeypatch.setattr(scheduler, "_acquire_pool_lock", forbidden)
    monkeypatch.setattr(scheduler, "_release_pool_lock", forbidden)
    monkeypatch.setattr(scheduler, "_run_deferred_request", request)
    scheduler._maybe_cleanup_journals(SimpleNamespace())
    assert [value["op"] for value in calls] == ["auto_cleanup_begin"]


def test_automatic_cleanup_finish_never_acquires_pool_locks(monkeypatch):
    """Both cleanup phases avoid allocation ownership locks."""
    scheduler = FluxScheduler()
    calls = []

    def forbidden(*args, **kwargs):
        """Reject pool-lock entry from maintenance."""
        pytest.fail("Automatic cleanup entered a pool ownership lock")

    def request(value):
        """Return an empty reserved batch to execute both phases."""
        calls.append(value)
        return {"token": str(uuid4()), "records": []}

    monkeypatch.setattr(scheduler, "_acquire_pool_lock", forbidden)
    monkeypatch.setattr(scheduler, "_release_pool_lock", forbidden)
    monkeypatch.setattr(scheduler, "_run_deferred_request", request)
    scheduler._maybe_cleanup_journals(SimpleNamespace())
    assert [value["op"] for value in calls] == [
        "auto_cleanup_begin",
        "auto_cleanup_finish",
    ]


@pytest.mark.parametrize(
    "case,approved",
    [
        ("sealed", True),
        ("unsealed", False),
        ("missing", False),
        ("not calcjob", False),
        ("other computer", False),
        ("other owner", False),
        ("wrong id", False),
        ("missing id", False),
        ("no computer", False),
    ],
)
def test_automatic_cleanup_approves_only_matching_sealed_calcjobs(
    aiida_profile_clean, aiida_computer_local, monkeypatch, case, approved
):
    """Automatic maintenance requires sealed provenance matching computer and owner."""
    computer = aiida_computer_local(label="automatic-cleanup", configure=True)
    current = orm.CalcJobNode(computer=computer).store()
    if case == "missing":
        uuid = str(uuid4())
    elif case == "not calcjob":
        uuid = orm.Dict(dict={}).store().uuid
    else:
        candidate_computer = computer
        if case == "no computer":
            candidate_computer = None
        if case == "other computer":
            candidate_computer = aiida_computer_local(
                label="automatic-other", configure=True
            )
        candidate_user = current.user
        if case == "other owner":
            candidate_user = orm.User(email="automatic-other@example.org").store()
        candidate = orm.CalcJobNode(computer=candidate_computer, user=candidate_user)
        if case != "missing id":
            candidate.base.attributes.set(
                "job_id",
                f"{'124' if case == 'wrong id' else '123'}:pending-{candidate.uuid}",
            )
        candidate.store()
        if case != "unsealed":
            candidate.seal()
        uuid = candidate.uuid
    scheduler = FluxScheduler()
    calls = []
    row = {
        "calculation_uuid": uuid,
        "allocation_id": "123",
        "action": "eligible",
        "fingerprint": "a" * 64,
    }

    def request(value):
        """Return one remote candidate and capture local approval decisions."""
        calls.append(value)
        if value["op"] == "auto_cleanup_begin":
            return {"token": str(uuid4()), "records": [row]}
        return {"records": [], "allocation_receipts": "retained"}

    monkeypatch.setattr(scheduler, "_run_deferred_request", request)
    scheduler._maybe_cleanup_journals(current)
    assert [value["op"] for value in calls] == [
        "auto_cleanup_begin",
        "auto_cleanup_finish",
    ]
    assert calls[1]["approved"] == ({uuid: "a" * 64} if approved else {})
