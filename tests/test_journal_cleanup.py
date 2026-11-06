"""Isolated remote compaction and sealed-node CLI approval regression tests."""

import fcntl
import hashlib
import json
import os
import shlex
from uuid import uuid4

import pytest
from aiida import orm
from click.testing import CliRunner

from aiida_flux_scheduler import cli as cli_module
from aiida_flux_scheduler import deferred

NOW = 2_000_000_000
RETENTION = 30 * 86400


def fingerprint(value):
    """Compute the protocol digest independently of the implementation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def no_flux(*args, **kwargs):
    """Fail if journal maintenance attempts to contact or mutate Flux."""
    pytest.fail("Journal cleanup or terminal receipt contacted Flux")


@pytest.fixture
def make_journal(tmp_path, monkeypatch):
    """Create realistic version-one journals in an isolated remote home."""
    monkeypatch.setattr(deferred.time, "time", lambda: NOW)

    def create(state="done", cancel_complete=False, age=RETENTION + 1, uuid=None):
        """Register an intent and persist the selected lifecycle state and age."""
        request = {
            "op": "register",
            "allocation_id": "123",
            "calculation_uuid": uuid or str(uuid4()),
            "working_directory": str(tmp_path / "work with spaces"),
            "submit_script": "submit 'quoted'.sh",
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
            reason="confirmed terminal" if state in ("done", "cancelled") else None,
        )
        path.write_text(json.dumps(record))
        os.utime(path, (NOW - age, NOW - age))
        return request, path, record

    return create


def cleanup(home, approved=None):
    """Run only the in-process helper against the test's remote home."""
    request = {"op": "cleanup_scan", "retention_seconds": RETENTION}
    if approved is not None:
        request.update(op="cleanup", approved=approved)
    return deferred.handle_request(request, home=home, runner=no_flux)


def snapshot(path):
    """Capture contents and inode metadata to detect unintended mutations."""
    stat = path.stat()
    return path.read_bytes(), stat.st_ino, stat.st_mtime_ns


@pytest.mark.parametrize(
    "state,cancel_complete", [("done", False), ("cancelled", True)]
)
def test_scan_old_confirmed_terminal_is_read_only(
    make_journal, tmp_path, state, cancel_complete
):
    """Dry scans identify terminal candidates without rewriting journals or guards."""
    request, path, record = make_journal(state, cancel_complete)
    guard = path.with_suffix(".guard")
    before = snapshot(path), snapshot(guard)

    report = cleanup(tmp_path)

    assert report["records"] == [
        {
            "calculation_uuid": request["calculation_uuid"],
            "allocation_id": "123",
            "state": state,
            "action": "eligible",
            "fingerprint": fingerprint(record),
            "reason": "requires sealed AiiDA CalcJob approval",
        }
    ]
    assert report["allocation_receipts"] == "retained"
    assert (snapshot(path), snapshot(guard)) == before


@pytest.mark.parametrize(
    "state,cancel_complete", [("done", False), ("cancelled", True)]
)
def test_apply_preserves_alias_and_immutable_intent_without_flux(
    make_journal, tmp_path, state, cancel_complete
):
    """Compaction removes the full intent but preserves all idempotent alias reads."""
    request, path, record = make_journal(state, cancel_complete)
    guard = path.with_suffix(".guard")
    guard_before = snapshot(guard)
    uuid = request["calculation_uuid"]

    report = cleanup(tmp_path, {uuid: fingerprint(record)})
    assert report["records"][0]["action"] == "compacted"
    receipt = json.loads(path.read_text())
    assert receipt == {
        "version": 2,
        "intent_sha256": fingerprint(record["intent"]),
        "calculation_uuid": uuid,
        "allocation_id": "123",
        "state": state,
        "child_id": "456",
        "reason": "confirmed terminal",
    }
    assert "working_directory" not in path.read_text()
    assert "submit_script" not in path.read_text()
    assert snapshot(guard) == guard_before
    receipt_before = snapshot(path)
    expected = {
        key: value
        for key, value in receipt.items()
        if key not in ("version", "intent_sha256")
    }
    expected.update(alias="pending-" + uuid, compacted=True)
    queries = [request, {"op": "lookup", "calculation_uuid": uuid}]
    queries.extend(
        {"op": op, "allocation_id": "123", "calculation_uuid": uuid}
        for op in ("poll", "detail", "cancel")
    )
    for query in queries:
        assert deferred.handle_request(query, home=tmp_path, runner=no_flux) == expected

    for change in (
        {"submit_script": "different.sh"},
        {"working_directory": str(tmp_path / "different")},
        {"requested_wallclock_seconds": 61},
    ):
        with pytest.raises(ValueError, match="Conflicting immutable submission intent"):
            deferred.handle_request(
                {**request, **change}, home=tmp_path, runner=no_flux
            )
    with pytest.raises(ValueError, match="Conflicting terminal receipt identity"):
        deferred.handle_request(
            {"op": "poll", "allocation_id": "124", "calculation_uuid": uuid},
            home=tmp_path,
            runner=no_flux,
        )
    assert (
        cleanup(tmp_path, {uuid: fingerprint(record)})["records"][0]["reason"]
        == "already compacted"
    )
    assert snapshot(path) == receipt_before
    assert snapshot(guard) == guard_before


@pytest.mark.parametrize(
    "state,cancel_complete,age,reason",
    [
        ("queued", False, RETENTION + 1, "not confirmed terminal"),
        ("submitting", False, RETENTION + 1, "not confirmed terminal"),
        ("submitted", False, RETENTION + 1, "not confirmed terminal"),
        ("ambiguous", False, RETENTION + 1, "not confirmed terminal"),
        ("cancelled", False, RETENTION + 1, "not confirmed terminal"),
        ("done", False, RETENTION - 1, "within retention period"),
        ("cancelled", True, RETENTION - 1, "within retention period"),
    ],
)
def test_active_uncertain_and_recent_journals_retained_even_if_approved(
    make_journal, tmp_path, state, cancel_complete, age, reason
):
    """Approval cannot bypass lifecycle confirmation or minimum retention."""
    request, path, record = make_journal(state, cancel_complete, age)
    before = snapshot(path)
    approved = {request["calculation_uuid"]: fingerprint(record)}
    for report in (cleanup(tmp_path), cleanup(tmp_path, approved)):
        row = report["records"][0]
        assert row["action"] == "retained"
        assert row["reason"] == reason
        assert "fingerprint" not in row
    assert snapshot(path) == before


def test_retention_boundary_is_eligible(make_journal, tmp_path):
    """A journal exactly at the minimum age may be approved for compaction."""
    make_journal(age=RETENTION)
    assert cleanup(tmp_path)["records"][0]["action"] == "eligible"


@pytest.mark.parametrize("approval", [{}, "stale"])
def test_missing_or_stale_approval_is_refused(make_journal, tmp_path, approval):
    """Apply requires the exact record fingerprint from the approval scan."""
    request, path, _ = make_journal()
    before = snapshot(path)
    if approval == "stale":
        approval = {request["calculation_uuid"]: "0" * 64}
    row = cleanup(tmp_path, approval)["records"][0]
    assert row["action"] == "retained"
    assert row["reason"] == "missing or stale sealed-node approval"
    assert snapshot(path) == before


def test_record_change_between_scan_and_apply_refuses_fingerprint(
    make_journal, tmp_path
):
    """A changed but still old terminal record invalidates the original approval."""
    request, path, record = make_journal()
    scanned = cleanup(tmp_path)["records"][0]
    record["reason"] = "updated after approval"
    path.write_text(json.dumps(record))
    os.utime(path, (NOW - RETENTION - 1, NOW - RETENTION - 1))
    before = snapshot(path)
    row = cleanup(tmp_path, {request["calculation_uuid"]: scanned["fingerprint"]})[
        "records"
    ][0]
    assert row["action"] == "retained"
    assert row["reason"] == "missing or stale sealed-node approval"
    assert row["fingerprint"] != scanned["fingerprint"]
    assert snapshot(path) == before


def test_busy_guard_retained_and_lock_inode_survives_apply(make_journal, tmp_path):
    """Cleanup neither waits for a held guard nor replaces its lock identity."""
    request, path, record = make_journal()
    before = snapshot(path)
    guard_path = path.with_suffix(".guard")
    inode = guard_path.stat().st_ino
    approved = {request["calculation_uuid"]: fingerprint(record)}
    with guard_path.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for report in (cleanup(tmp_path), cleanup(tmp_path, approved)):
            assert report["records"][0]["action"] == "retained"
        assert snapshot(path) == before
    assert cleanup(tmp_path, approved)["records"][0]["action"] == "compacted"
    assert guard_path.stat().st_ino == inode
    with guard_path.open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="busy"):
            deferred.handle_request(request, home=tmp_path, runner=no_flux)


@pytest.mark.parametrize(
    "damage",
    [
        "invalid JSON",
        "not object",
        "unknown version",
        "unknown field",
        "incomplete",
        "bad receipt",
    ],
)
def test_malformed_and_unknown_schema_journals_retained(make_journal, tmp_path, damage):
    """Unreadable or unrecognized persisted boundaries are never compacted."""
    request, path, record = make_journal()
    approved = {request["calculation_uuid"]: fingerprint(record)}
    if damage == "invalid JSON":
        content = "{broken"
    elif damage == "not object":
        content = "[]"
    else:
        if damage == "unknown version":
            record["version"] = 99
        elif damage == "unknown field":
            record["unexpected"] = True
        elif damage == "incomplete":
            del record["cancel_complete"]
        else:
            record["version"] = 2
        content = json.dumps(record)
    path.write_text(content)
    os.utime(path, (NOW - RETENTION - 1, NOW - RETENTION - 1))
    before = snapshot(path)
    for report in (cleanup(tmp_path), cleanup(tmp_path, approved)):
        row = report["records"][0]
        assert row["action"] == "retained"
        assert row["reason"]
    assert snapshot(path) == before


def test_allocation_receipts_and_guards_are_untouched(make_journal, tmp_path):
    """Submission maintenance never interprets or removes allocation artifacts."""
    request, _, record = make_journal()
    allocations = tmp_path / ".aiida-flux-scheduler" / "allocations"
    allocations.mkdir()
    paths = [allocations / (str(uuid4()) + suffix) for suffix in (".json", ".guard")]
    for path in paths:
        path.write_text("opaque allocation receipt or guard")
    before = {path: snapshot(path) for path in paths}
    assert cleanup(tmp_path)["allocation_receipts"] == "retained"
    assert (
        cleanup(tmp_path, {request["calculation_uuid"]: fingerprint(record)})[
            "allocation_receipts"
        ]
        == "retained"
    )
    assert {path: snapshot(path) for path in paths} == before


def test_empty_scan_does_not_create_remote_home(tmp_path):
    """Scanning an unused remote home is a no-op, including apply."""
    assert cleanup(tmp_path) == {"records": [], "allocation_receipts": "retained"}
    assert cleanup(tmp_path, {}) == {"records": [], "allocation_receipts": "retained"}
    assert not (tmp_path / ".aiida-flux-scheduler").exists()


@pytest.mark.parametrize(
    "case,expected",
    [
        ("sealed", None),
        ("missing", "CalcJob not found in the active profile"),
        ("not calcjob", "CalcJob is not sealed"),
        ("unsealed", "CalcJob is not sealed"),
        ("no computer", "CalcJob computer or owner does not match"),
        ("other computer", "CalcJob computer or owner does not match"),
        ("other owner", "CalcJob computer or owner does not match"),
        ("wrong id", "CalcJob scheduler ID does not match"),
        ("missing id", "CalcJob scheduler ID does not match"),
    ],
)
def test_cleanup_approval_real_orm(
    aiida_profile_clean, aiida_computer_local, case, expected
):
    """Only sealed matching CalcJobs belonging to the active user are approved."""
    computer = aiida_computer_local(label="cleanup-local", configure=True)
    if case == "missing":
        uuid = str(uuid4())
    elif case == "not calcjob":
        uuid = orm.Dict(dict={}).store().uuid
    else:
        node_computer = None if case == "no computer" else computer
        if case == "other computer":
            node_computer = aiida_computer_local(label="cleanup-other", configure=True)
        node_user = cli_module.get_current_user()
        if case == "other owner":
            node_user = orm.User(email="other-cleanup@example.org").store()
        node = orm.CalcJobNode(computer=node_computer, user=node_user)
        if case != "missing id":
            node.base.attributes.set(
                "job_id",
                "124:pending-" + node.uuid
                if case == "wrong id"
                else "123:pending-" + node.uuid,
            )
        node.store()
        if case != "unsealed":
            node.seal()
        uuid = node.uuid
    assert (
        cli_module._cleanup_approval(
            {"calculation_uuid": uuid, "allocation_id": "123"}, computer
        )
        == expected
    )


@pytest.mark.parametrize("apply", [False, True])
def test_cli_dry_run_and_apply_use_sealed_node_approvals(
    aiida_profile_clean,
    aiida_computer_local,
    make_journal,
    tmp_path,
    monkeypatch,
    apply,
):
    """The CLI sends only approved fingerprints and preserves rejection reasons."""
    monkeypatch.setattr(cli_module, "_load_profile", lambda: None)
    computer = aiida_computer_local(label="cleanup-cli", configure=True)
    node = orm.CalcJobNode(computer=computer)
    node.base.attributes.set("job_id", "123:pending-" + node.uuid)
    node.store().seal()
    request, path, record = make_journal(uuid=node.uuid)
    missing, missing_path, _ = make_journal()
    _, active_path, _ = make_journal(state="ambiguous")
    before = {item: snapshot(item) for item in (path, missing_path, active_path)}
    calls = []

    class FakeTransport:
        """Execute embedded requests in the isolated home instead of a remote host."""

        def __enter__(self):
            """Return the fake context-managed transport."""
            return self

        def __exit__(self, *args):
            """Close the fake transport without suppressing exceptions."""
            return False

        def exec_command_wait(self, command):
            """Decode the shell-safe protocol without launching its embedded source."""
            query = json.loads(shlex.split(command)[-1])
            calls.append(query)
            result = deferred.handle_request(query, home=tmp_path, runner=no_flux)
            return 0, json.dumps(result), ""

    monkeypatch.setattr(orm.Computer, "get_transport", lambda self: FakeTransport())
    args = ["journal", "cleanup", "--computer", computer.label]
    if apply:
        args.append("--apply")
    result = CliRunner().invoke(cli_module.cli, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dry_run"] is not apply
    assert report["allocation_receipts"] == "retained"
    rows = {row["calculation_uuid"]: row for row in report["records"]}
    assert rows[node.uuid]["action"] == ("compacted" if apply else "eligible")
    assert rows[missing["calculation_uuid"]]["action"] == "retained"
    assert (
        rows[missing["calculation_uuid"]]["reason"]
        == "CalcJob not found in the active profile"
    )
    assert calls[0] == {"op": "cleanup_scan", "retention_seconds": RETENTION}
    if apply:
        assert calls[1:] == [
            {
                "op": "cleanup",
                "retention_seconds": RETENTION,
                "approved": {request["calculation_uuid"]: fingerprint(record)},
            }
        ]
        assert json.loads(path.read_text())["version"] == 2
    else:
        assert len(calls) == 1
        assert snapshot(path) == before[path]
    assert snapshot(missing_path) == before[missing_path]
    assert snapshot(active_path) == before[active_path]
