"""Exercise pool ownership and guard races using an isolated local shell."""

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from aiida.schedulers import SchedulerError

from aiida_flux_scheduler.flux import FluxScheduler


class LocalShellTransport:
    """Run generated remote commands against a temporary home directory."""

    def __init__(self, home):
        """Set the isolated shell home."""
        self.home = home
        self.commands = []
        self.before_command = None

    def exec_command_wait(self, command):
        """Execute one shell command and return its captured result."""
        self.commands.append(command)
        if self.before_command is not None:
            self.before_command(command)
        result = subprocess.run(
            ["sh", "-c", command],
            env={**os.environ, "HOME": str(self.home)},
            text=True,
            capture_output=True,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr


@pytest.fixture
def scheduler(tmp_path):
    """Return a scheduler whose shell can touch only the test's lock tree."""
    result = FluxScheduler()
    result._transport = LocalShellTransport(tmp_path)
    return result


def metadata_path(scheduler):
    """Return the test pool's owner file."""
    return (
        Path(scheduler.transport.home)
        / ".aiida-flux-scheduler/locks/shared.lock/owner.json"
    )


def write_metadata(scheduler, metadata):
    """Replace metadata to emulate a prior process or a racing acquisition."""
    metadata_path(scheduler).write_text(json.dumps(metadata, sort_keys=True))


def abandon(scheduler):
    """Make the test lock refer to a terminated child process."""
    scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    metadata = json.loads(metadata_path(scheduler).read_text())
    process = subprocess.Popen(["sleep", "60"])
    try:
        start_time = (
            Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
    finally:
        process.terminate()
        process.wait()
    metadata["owner"]["pid"] = process.pid
    metadata["owner"]["start_time"] = start_time
    metadata["acquired_at_epoch"] = int(time.time()) - 1000
    write_metadata(scheduler, metadata)
    scheduler._pool_lock_owners.clear()
    return metadata


def test_acquire_release_preserves_guard_and_unique_token(scheduler):
    """Every acquisition has a token and releases without unlinking the guard."""
    scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    first = json.loads(metadata_path(scheduler).read_text())
    scheduler._release_pool_lock("shared")
    assert not metadata_path(scheduler).parent.exists()
    assert metadata_path(scheduler).parent.with_suffix(".lock.guard").is_file()
    assert (
        metadata_path(scheduler).parent.with_suffix(".lock.guard").stat().st_mode
        & 0o777
        == 0o600
    )
    scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    assert json.loads(metadata_path(scheduler).read_text())["token"] != first["token"]
    scheduler._release_pool_lock("shared")


def test_release_without_ownership_does_not_touch_remote(scheduler):
    """A scheduler that did not acquire must not remove another owner's lock."""
    scheduler._release_pool_lock("shared")
    assert scheduler.transport.commands == []


def test_release_does_not_delete_replacement(scheduler):
    """A delayed release cannot remove an acquisition with another token."""
    scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    replacement = json.loads(metadata_path(scheduler).read_text())
    replacement["token"] = "0" * 32
    write_metadata(scheduler, replacement)
    scheduler._release_pool_lock("shared")
    assert json.loads(metadata_path(scheduler).read_text()) == replacement


def test_old_live_owner_is_never_recovered(scheduler):
    """Elapsed time is not evidence that an owner stopped working."""
    scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    metadata = json.loads(metadata_path(scheduler).read_text())
    metadata["acquired_at_epoch"] = 1
    write_metadata(scheduler, metadata)
    assert not scheduler._recover_stale_pool_lock("shared", 0)
    assert metadata_path(scheduler).exists()


def test_dead_local_owner_is_recovered(scheduler):
    """Confirmed dead local owners can be reclaimed under the guard."""
    abandon(scheduler)
    scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    assert (
        json.loads(metadata_path(scheduler).read_text())["owner"]["pid"] == os.getpid()
    )
    scheduler._release_pool_lock("shared")


def test_young_dead_owner_is_not_recovered(scheduler):
    """Recovery preserves the configured age threshold in addition to ownership."""
    metadata = abandon(scheduler)
    metadata["acquired_at_epoch"] = int(time.time())
    write_metadata(scheduler, metadata)
    assert not scheduler._recover_stale_pool_lock("shared", 300)


def test_unreadable_process_identity_is_not_recovered(scheduler, monkeypatch):
    """Permission failures cannot be interpreted as owner death."""
    metadata = abandon(scheduler)
    original = Path.read_text

    def unreadable(path, *args, **kwargs):
        """Reject only the recorded owner's process inspection."""
        if str(path) == f"/proc/{metadata['owner']['pid']}/stat":
            raise PermissionError("not permitted")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert not scheduler._recover_stale_pool_lock("shared", 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("host", "foreign-host"),
        ("boot_id", "old-boot"),
        ("pid_namespace", -1),
        ("uid", -1),
    ],
)
def test_foreign_identity_is_not_recovered(scheduler, field, value):
    """Unverifiable process namespaces require explicit manual intervention."""
    metadata = abandon(scheduler)
    metadata["owner"][field] = value
    write_metadata(scheduler, metadata)
    assert not scheduler._recover_stale_pool_lock("shared", 0)
    assert metadata_path(scheduler).exists()


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        "{}",
        '{"acquired_at_epoch": 1}',
        '{"schema_version": 1, "owner": null}',
    ],
)
def test_legacy_and_malformed_metadata_fail_closed(scheduler, contents):
    """Unknown ownership never falls back to age-only reclamation."""
    path = metadata_path(scheduler)
    path.parent.mkdir(parents=True)
    path.write_text(contents)
    assert not scheduler._recover_stale_pool_lock("shared", 0)
    assert path.read_text() == contents


def test_missing_metadata_fail_closed(scheduler):
    """A crash before writing metadata must not authorize automatic deletion."""
    metadata_path(scheduler).parent.mkdir(parents=True)
    assert not scheduler._recover_stale_pool_lock("shared", 0)
    assert scheduler._get_pool_lock_age_seconds("shared") is None


def test_recovery_snapshot_cannot_remove_racing_replacement(scheduler):
    """Recovery rechecks ownership inside the remote guard."""
    metadata = abandon(scheduler)
    replacement = {**metadata, "token": "f" * 32}

    def replace_before_guard(command):
        """Emulate an acquisition between inspection and guarded deletion."""
        if "flock -x" in command:
            write_metadata(scheduler, replacement)

    scheduler.transport.before_command = replace_before_guard
    assert not scheduler._recover_stale_pool_lock("shared", 0)
    assert json.loads(metadata_path(scheduler).read_text()) == replacement


def test_reused_pid_with_different_start_time_is_recovered(scheduler):
    """A living unrelated process is not the recorded acquisition owner."""
    metadata = abandon(scheduler)
    metadata["owner"]["pid"] = os.getpid()
    metadata["owner"]["start_time"] = "0"
    write_metadata(scheduler, metadata)
    assert scheduler._recover_stale_pool_lock("shared", 0)


def test_competing_acquisitions_have_one_winner(tmp_path):
    """The persistent advisory guard serializes simultaneous contenders."""
    schedulers = [FluxScheduler(), FluxScheduler()]
    for candidate in schedulers:
        candidate._transport = LocalShellTransport(tmp_path)

    def acquire(candidate):
        """Return whether this contender won."""
        try:
            candidate._acquire_pool_lock("shared", timeout_seconds=0)
        except SchedulerError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        won = list(executor.map(acquire, schedulers))
    assert sum(won) == 1
    schedulers[won.index(True)]._release_pool_lock("shared")


def test_guard_failure_does_not_create_lock(scheduler, monkeypatch):
    """A missing or unusable flock utility must fail without unguarded fallback."""
    original = scheduler.transport.exec_command_wait

    def fail_guard(command):
        """Emulate an unavailable remote flock binary."""
        return (
            (127, "", "flock unavailable")
            if "flock -x" in command
            else original(command)
        )

    monkeypatch.setattr(scheduler.transport, "exec_command_wait", fail_guard)
    with pytest.raises(SchedulerError, match="Unable to write metadata"):
        scheduler._acquire_pool_lock("shared", timeout_seconds=0)
    assert not metadata_path(scheduler).parent.exists()
