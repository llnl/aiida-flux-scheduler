"""Durable, remote deferred submissions without a remote AiiDA dependency.

Responses always include ``state``, ``allocation_id``, ``calculation_uuid`` and
``alias``. A submitted child adds ``child_id`` and polling adds ``job`` (the
Flux jobs JSON object). Detail adds ``jobspec`` or ``intent``. Errors produce
``{"error": ...}`` and a nonzero helper exit status, never a fabricated completion.
"""

import fcntl
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

COMMAND_TIMEOUT = 5
Runner = Callable[..., str]


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact remote protocol and normalize its identity.

    Raises:
        ValueError: If a field is missing, unknown, or invalid.
    """
    if not isinstance(request, dict):
        raise ValueError("Request must be an object")
    op = request.get("op")
    if op not in ("register", "poll", "cancel", "detail", "lookup", "allocate"):
        raise ValueError("Unsupported deferred operation")
    keys = {"op", "allocation_id", "calculation_uuid"}
    if op == "lookup":
        keys = {"op", "calculation_uuid"}
    if op == "allocate":
        keys = {"op", "generation", "command", "job_name"}
    if op == "register":
        keys |= {"working_directory", "submit_script"}
    optional = {"requested_wallclock_seconds"} if op == "register" else set()
    if not keys <= set(request) or set(request) - keys - optional:
        raise ValueError("Incorrect request fields")
    result = dict(request)
    for key in keys:
        if not isinstance(result[key], str) or not result[key] or "\x00" in result[key]:
            raise ValueError(f"Invalid {key}")
    uuid_key = "generation" if op == "allocate" else "calculation_uuid"
    uuid = str(UUID(result[uuid_key]))
    if uuid != result[uuid_key]:
        raise ValueError("calculation_uuid must be a canonical UUID")
    if op == "allocate":
        if result["job_name"] != "aiida-allocation-" + uuid:
            raise ValueError("Allocation job name must match generation")
        return result
    if op == "lookup":
        return result
    if not re.fullmatch(
        r"(?:[0-9]+|[fƒ][1-9A-HJ-NP-Za-km-z]+)", result["allocation_id"]
    ):
        raise ValueError("allocation_id must be a decimal or F58 Flux job ID")
    if op == "register" and not os.path.isabs(result["working_directory"]):
        raise ValueError("working_directory must be absolute")
    if "requested_wallclock_seconds" in result:
        seconds = result["requested_wallclock_seconds"]
        if type(seconds) is not int or seconds <= 0:
            raise ValueError("requested_wallclock_seconds must be a positive integer")
    return result


def build_deferred_command(request: dict[str, Any]) -> str:
    """Build a shell-safe helper command requiring only Flux and Python remotely."""
    request = validate_request(request)
    # Embedding this stdlib-only module avoids version skew or remote AiiDA imports.
    source = Path(__file__).read_text(encoding="utf-8") + "\n_remote_main()\n"
    # Some sites expose Python 3.6 through `flux python`. The helper uses only
    # stdlib runtime features available there; strip annotations in that runtime
    # before compilation rather than imposing a remote AiiDA/Python installation.
    bootstrap = (
        "import ast, sys\n"
        "class StripAnnotations(ast.NodeTransformer):\n"
        "    def visit_FunctionDef(self, node):\n"
        "        node.returns = None\n"
        "        for arg in node.args.args + node.args.kwonlyargs:\n"
        "            arg.annotation = None\n"
        "        return self.generic_visit(node)\n"
        "    def visit_AnnAssign(self, node):\n"
        "        return ast.copy_location(ast.Assign(targets=[node.target], value=node.value or ast.Constant(value=None)), node)\n"
        "tree = StripAnnotations().visit(ast.parse(sys.argv.pop(1)))\n"
        "exec(compile(ast.fix_missing_locations(tree), '<flux-pool-helper>', 'exec'))\n"
    )
    return shlex.join(["flux", "python", "-c", bootstrap, source, json.dumps(request)])


def _run(args: list[str], cwd: str | None = None) -> str:
    """Run a bounded Flux command, killing the proxy process group on timeout."""
    if args[:2] == ["flux", "proxy"]:
        # Flux proxy interprets its command arguments as shell text, not argv.
        args = args[:3] + [" ".join(shlex.quote(argument) for argument in args[3:])]
    process = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(
            f"Flux command failed ({process.returncode}): {stderr.strip()}"
        )
    return stdout


def _write(path: Path, record: dict[str, Any]) -> None:
    """Atomically replace and fsync a journal while its stable guard is locked."""
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_record(record: dict[str, Any], request: dict[str, Any]) -> None:
    """Reject unknown versions, damaged journals, and conflicting identities."""
    keys = {"version", "intent", "state", "child_id", "reason", "cancel_complete"}
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("Invalid deferred journal fields")
    if type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("Unsupported deferred journal version")
    intent = validate_request({"op": "register", **record["intent"]})
    if set(record["intent"]) != set(intent) - {"op"}:
        raise ValueError("Invalid journal intent")
    for key in ("allocation_id", "calculation_uuid"):
        if key in request and intent[key] != request[key]:
            raise ValueError(f"Conflicting {key}")
    if request["op"] == "register" and intent != request:
        raise ValueError("Conflicting immutable submission intent")
    if record["state"] not in (
        "queued",
        "submitting",
        "submitted",
        "done",
        "cancelled",
        "ambiguous",
    ):
        raise ValueError("Invalid journal state")
    if record["child_id"] is not None:
        validate_request(
            {
                "op": "poll",
                "allocation_id": record["child_id"],
                "calculation_uuid": request["calculation_uuid"],
            }
        )
    if record["state"] == "submitted" and record["child_id"] is None:
        raise ValueError("Submitted journal has no child")
    if record["reason"] is not None and not isinstance(record["reason"], str):
        raise ValueError("Invalid journal reason")
    if type(record["cancel_complete"]) is not bool:
        raise ValueError("Invalid cancellation marker")


def _job_state(job: dict[str, Any]) -> str:
    """Normalize Flux jobs' string or numeric RFC 21 state."""
    state = job.get("state")
    if type(state) is int:
        state = {
            1: "NEW",
            2: "DEPEND",
            4: "PRIORITY",
            8: "SCHED",
            16: "RUN",
            32: "CLEANUP",
            64: "INACTIVE",
        }.get(state)
    if state not in (
        "NEW",
        "DEPEND",
        "PRIORITY",
        "SCHED",
        "RUN",
        "CLEANUP",
        "INACTIVE",
    ):
        raise ValueError("Invalid Flux job state")
    return state


def _decimal_id(value: str) -> str:
    """Normalize decimal and RFC 19 F58 job IDs for journal comparisons."""
    if re.fullmatch(r"[0-9]+", value):
        return str(int(value))
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    number = 0
    for character in value[1:]:
        number = number * 58 + alphabet.index(character)
    return str(number)


def _job(output: str) -> dict[str, Any]:
    """Read one explicitly selected Flux job, rejecting missing/invalid results."""
    value = json.loads(output)
    if isinstance(value, dict) and "jobs" in value:
        if len(value["jobs"]) != 1:
            raise ValueError("Expected exactly one Flux job")
        value = value["jobs"][0]
    if not isinstance(value, dict) or "id" not in value:
        raise ValueError("Invalid Flux job response")
    _job_state(value)
    return value


def _lookup(record: dict[str, Any], runner: Runner) -> dict[str, Any] | None:
    """Find the uniquely named child including inactive jobs before any submit."""
    intent = record["intent"]
    name = "aiida-deferred-" + intent["calculation_uuid"]
    output = runner(
        [
            "flux",
            "proxy",
            intent["allocation_id"],
            "flux",
            "jobs",
            "-a",
            "--count=0",
            "--name=" + name,
            "--json",
        ]
    )
    value = json.loads(output)
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise ValueError("Invalid child lookup response")
    jobs = value["jobs"]
    if len(jobs) > 1:
        raise ValueError("Multiple children match the immutable submission identity")
    return _job(json.dumps(jobs[0])) if jobs else None


def _response(record: dict[str, Any]) -> dict[str, Any]:
    """Build the stable response envelope without exposing internal transitions."""
    intent = record["intent"]
    value = {key: intent[key] for key in ("allocation_id", "calculation_uuid")}
    value.update(
        alias="pending-" + intent["calculation_uuid"],
        state="ambiguous" if record["state"] == "submitting" else record["state"],
    )
    if record["state"] == "cancelled" and not record["cancel_complete"]:
        value["state"] = "cancelling"
    if record["child_id"] is not None:
        value["child_id"] = record["child_id"]
    if record["reason"] is not None:
        value["reason"] = record["reason"]
    if record["reason"] == "Allocation ended before child completion":
        value["allocation_ended"] = True
    return value


def handle_request(
    request: dict[str, Any],
    *,
    home: Path | None = None,
    runner: Runner = _run,
) -> dict[str, Any]:
    """Apply one serialized, durable deferred operation.

    Args:
        request: Exact version-one protocol request.
        home: Journal home override for isolated tests.
        runner: Bounded command executor, injectable without a live Flux instance.

    Raises:
        ValueError: If protocol, journal, or Flux responses are invalid.
        RuntimeError: If a query fails or another operation holds the guard.
    """
    request = validate_request(request)
    if request["op"] == "allocate":
        return _allocate(request, home or Path.home(), runner)
    root = (home or Path.home()) / ".aiida-flux-scheduler" / "submissions"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / (request["calculation_uuid"] + ".json")
    with path.with_suffix(".guard").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exception:
            raise RuntimeError(
                "Deferred submission is busy; retry later"
            ) from exception
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            _validate_record(record, request)
        elif request["op"] == "lookup":
            return {}
        elif request["op"] == "register":
            record = {
                "version": 1,
                "intent": {k: v for k, v in request.items() if k != "op"},
                "state": "queued",
                "child_id": None,
                "reason": None,
                "cancel_complete": False,
            }
            _write(path, record)
        else:
            raise ValueError("Deferred submission is not registered")

        if request["op"] in ("register", "lookup"):
            return _response(record)
        allocation = request["allocation_id"]
        proxy = ["flux", "proxy", allocation, "flux"]
        if request["op"] == "detail":
            response = _response(record)
            if record["child_id"] is not None and record["state"] not in (
                "done",
                "cancelled",
            ):
                response["jobspec"] = json.loads(
                    runner(proxy + ["job", "info", record["child_id"], "jobspec"])
                )
            else:
                response["intent"] = record["intent"]
            return response

        if request["op"] == "cancel":
            # Persist the tombstone before any remote call, even a failed lookup.
            uncertain = record["state"] in ("submitting", "ambiguous")
            record["state"] = "cancelled"
            if uncertain:
                record["reason"] = "Cancellation awaiting reconciliation"
            _write(path, record)
            if record["cancel_complete"]:
                return _response(record)
            if record["child_id"] is None:
                if (
                    uncertain
                    or record["reason"] == "Cancellation awaiting reconciliation"
                ):
                    record["reason"] = "Cancellation awaiting reconciliation"
                    _write(path, record)
                    job = _lookup(record, runner)
                    if job is None:
                        raise RuntimeError(
                            "Cancellation awaiting reconciliation; child outcome is unknown"
                        )
                    record["child_id"] = _decimal_id(str(job["id"]))
                    _write(path, record)
                else:
                    record["cancel_complete"] = True
            if record["child_id"] is not None:
                job = _job(runner(proxy + ["jobs", record["child_id"], "--json"]))
                if _job_state(job) != "INACTIVE":
                    runner(proxy + ["cancel", record["child_id"]])
                record["cancel_complete"] = True
            record["reason"] = None
            _write(path, record)
            return _response(record)

        if record["state"] == "cancelled" and not record["cancel_complete"]:
            raise RuntimeError("Cancellation needs reconciliation; retry cancel")
        if record["state"] in ("cancelled", "done"):
            return _response(record)
        parent = _job(runner(["flux", "jobs", allocation, "--json"]))
        parent_state = _job_state(parent)
        if parent_state == "CLEANUP":
            return {**_response(record), "state": "queued"}
        if parent_state == "INACTIVE":
            record.update(
                state="done", reason="Allocation ended before child completion"
            )
            _write(path, record)
            return _response(record)
        if parent_state != "RUN":
            return _response(record)
        # Flux state-machine.get is a nonblocking broker RPC returning {state: "run"}.
        probe = 'import json, flux; print(json.dumps(flux.Flux().rpc("state-machine.get").get()))'
        readiness = json.loads(runner(proxy + ["python", "-c", probe]))
        if not isinstance(readiness, dict) or not isinstance(
            readiness.get("state"), str
        ):
            raise ValueError("Invalid broker readiness response")
        if readiness["state"] != "run":
            return _response(record)
        job = _lookup(record, runner)
        if job is not None:
            child_id = _decimal_id(str(job["id"]))
            if record["child_id"] is not None and record["child_id"] != child_id:
                raise ValueError("Reconciled child conflicts with journal")
            record.update(
                child_id=child_id,
                state="done" if _job_state(job) == "INACTIVE" else "submitted",
                reason=None,
            )
            _write(path, record)
            return {**_response(record), "job": job}
        if record["state"] != "queued":
            record.update(
                state="ambiguous",
                reason="Child submission cannot yet be reconciled; automatic retry disabled",
            )
            _write(path, record)
            return _response(record)
        requested = record["intent"].get("requested_wallclock_seconds")
        if requested is not None:
            expiration = parent.get("expiration")
            if (
                isinstance(expiration, bool)
                or not isinstance(expiration, (int, float))
                or not math.isfinite(expiration)
            ):
                raise ValueError("Allocation expiration is unavailable")
            if expiration != 0 and expiration - time.time() < requested:
                record.update(
                    state="done", reason="Insufficient allocation walltime for child"
                )
                _write(path, record)
                return _response(record)
        record["state"] = "submitting"
        _write(path, record)
        try:
            output = runner(
                proxy
                + [
                    "batch",
                    "--job-name=aiida-deferred-" + request["calculation_uuid"],
                    "--",
                    record["intent"]["submit_script"],
                ],
                cwd=record["intent"]["working_directory"],
            )
            child_id = output.strip()
            validate_request(
                {
                    "op": "poll",
                    "allocation_id": child_id,
                    "calculation_uuid": request["calculation_uuid"],
                }
            )
            child_id = _decimal_id(child_id)
        except Exception as exception:
            # A failure or timeout may happen AFTER Flux accepted the child.
            record.update(
                state="ambiguous", reason=f"Submission outcome unknown: {exception}"
            )
            _write(path, record)
            return _response(record)
        record.update(state="submitted", child_id=child_id)
        _write(path, record)
        return _response(record)


def _allocate(request: dict[str, Any], home: Path, runner: Runner) -> dict[str, Any]:
    """Submit or reconcile a generation-fenced allocation receipt exactly once."""
    root = home / ".aiida-flux-scheduler" / "allocations"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = root / (request["generation"] + ".json")
    with path.with_suffix(".guard").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exception:
            raise RuntimeError(
                "Allocation submission is busy; retry later"
            ) from exception
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(record, dict)
                or set(record)
                != {"version", "intent", "state", "allocation_id", "reason"}
                or type(record["version"]) is not int
                or record["version"] != 1
            ):
                raise ValueError("Invalid allocation receipt version or fields")
            if record["intent"] != request:
                raise ValueError("Conflicting immutable allocation intent")
            if record["state"] not in (
                "queued",
                "submitting",
                "submitted",
                "ambiguous",
            ):
                raise ValueError("Invalid allocation receipt state")
            if record["allocation_id"] is not None:
                validate_request(
                    {
                        "op": "poll",
                        "allocation_id": record["allocation_id"],
                        "calculation_uuid": request["generation"],
                    }
                )
                return {**record, "generation": request["generation"]}
            if record["state"] == "submitted":
                raise ValueError("Submitted allocation receipt has no allocation ID")
        else:
            record = {
                "version": 1,
                "intent": request,
                "state": "queued",
                "allocation_id": None,
                "reason": None,
            }
            _write(path, record)
        result = json.loads(
            runner(
                [
                    "flux",
                    "jobs",
                    "-a",
                    "--count=0",
                    "--name=" + request["job_name"],
                    "--json",
                ]
            )
        )
        if not isinstance(result, dict) or not isinstance(result.get("jobs"), list):
            raise ValueError("Invalid allocation lookup response")
        if len(result["jobs"]) > 1:
            raise ValueError("Multiple allocations match the generation")
        if result["jobs"]:
            job = _job(json.dumps(result["jobs"][0]))
            record.update(
                state="submitted",
                allocation_id=_decimal_id(str(job["id"])),
                reason=None,
            )
        elif record["state"] != "queued":
            record.update(
                state="ambiguous",
                reason="Allocation submission cannot be reconciled; automatic retry disabled",
            )
        else:
            record["state"] = "submitting"
            _write(path, record)
            try:
                output = runner(
                    ["bash", "-c", "set -o pipefail; " + request["command"]]
                )
                allocation_id = output.strip()
                validate_request(
                    {
                        "op": "poll",
                        "allocation_id": allocation_id,
                        "calculation_uuid": request["generation"],
                    }
                )
                record.update(
                    state="submitted", allocation_id=_decimal_id(allocation_id)
                )
            except Exception as exception:
                record.update(
                    state="ambiguous", reason=f"Allocation outcome unknown: {exception}"
                )
        _write(path, record)
        return {**record, "generation": request["generation"]}


def _remote_main() -> None:
    """Emit exactly one JSON response and use a nonzero status for query errors."""
    try:
        response = handle_request(json.loads(sys.argv[1]))
    except Exception as exception:
        print(json.dumps({"error": str(exception)}))
        sys.exit(1)
    print(json.dumps(response))
