"""
Plugin for Flux.
"""

import datetime
import hashlib
import json
import math
import re
import shlex
import time
import uuid
from collections import defaultdict, namedtuple
from typing import Any

from aiida.common import exceptions
from aiida.common.escaping import escape_for_bash
from aiida.common.extendeddicts import AttributeDict
from aiida.common.lang import type_check
from aiida.engine.processes.exit_code import ExitCode
from aiida.orm import CalcJobNode, load_node
from aiida.schedulers import Scheduler, SchedulerError
from aiida.schedulers.datastructures import (
    JobInfo,
    JobState,
    JobTemplate,
    NodeNumberJobResource,
)

from aiida_flux_scheduler.cleanup import cleanup_approval
from aiida_flux_scheduler.deferred import build_deferred_command
from aiida_flux_scheduler.pools import (
    POOL_EXTRA_KEY,
    clear_pool_runtime,
    get_pool_group,
    get_pool_runtime,
    update_pool_runtime,
)

_MAP_STATUS_FLUX = {
    "D": JobState.QUEUED,  # Depend
    "P": JobState.QUEUED,  # Priority
    "S": JobState.QUEUED,  # Scheduled
    "R": JobState.RUNNING,  # Running
    "C": JobState.RUNNING,  # Cleanup
    "CD": JobState.DONE,  # Completed
    "F": JobState.DONE,  # Failed
    "CA": JobState.DONE,  # Canceled
    "TO": JobState.DONE,  # Timeout
}


class FluxJobResource(NodeNumberJobResource):
    """
    Class for Flux job resources.
    """

    _default_fields = NodeNumberJobResource._default_fields + ("num_gpus_per_mpiproc",)

    @classmethod
    def validate_resources(cls, **kwargs: dict) -> AttributeDict:
        """
        Validate the resouces against the job resource class of this scheduler.

        :param kwargs: dictionary of values to define the job resources.
        :return: attribute dictionary with the parsed parameters populated.
        """

        mws = kwargs.pop("max_wallclock_seconds", None)
        flux_pool = kwargs.pop("flux_pool", None)
        allocation_lifecycle = kwargs.pop("flux_allocation_lifecycle", None)
        if mws is not None:
            try:
                mws = int(mws)
            except ValueError:
                raise ValueError("`max_wallclock_seconds` should be set as an int.")

        resources = super().validate_resources(**kwargs)
        if (
            resources.num_gpus_per_mpiproc is not None
            and resources.num_gpus_per_mpiproc < 1
        ):
            raise ValueError(
                "`num_gpus_per_mpiproc` must be greater than or equal to one."
            )
        resources["max_wallclock_seconds"] = mws
        resources["flux_pool"] = flux_pool
        if allocation_lifecycle not in (None, "parent", "child"):
            raise ValueError(
                "`flux_allocation_lifecycle` must be either `parent` or `child`."
            )
        resources["flux_allocation_lifecycle"] = allocation_lifecycle

        return resources


class FluxScheduler(Scheduler):
    """
    Flux scheduler.
    """

    _FIELD_SEPARATOR = "|"

    _features = {
        "can_query_by_user": False,
    }

    _job_resource_class = FluxJobResource

    fields = [
        ("id", "job_id"),  # job or job step id
        ("status_abbrev", "state_raw"),  # job state in compact form
        ("annotations", "annotation"),  # reason for the job being in its current state
        ("username", "username"),  # username
        ("nnodes", "number_nodes"),  # number of nodes allocated
        ("ncores", "number_cpus"),  # number of allocated cores (if already running)
        (
            "nodelist",
            "allocated_machines",
        ),  # list of allocated nodes when running, otherwise
        # reason within parenthesis
        ("queue", "partition"),  # partition (queue) of the job
        ("duration", "time_limit"),  # time limit in seconds
        ("runtime", "time_used"),  # Time used by the job in days-hours:minutes:seconds
        ("t_run", "dispatch_time"),  # actual or expected dispatch time (start time)
        ("name", "job_name"),  # job name (title)
        ("t_submit", "submission_time"),  # This is probably new, it exists in version
        # 14.03.7 and later
    ]

    def _get_joblist_command(
        self,
        jobs: list[str] | None = None,
        user: str | None = None,
        flux_id: str | None = None,
    ) -> str:
        """
        Command to report full information on an existing job.

        :param jobs: List of job ids.
        :param user: Username for the job queue.
        :param flux_id: Job ID of the active flux allocation to be used for the proxy command.
        :return comm: Command to retrieve full job information.
        """

        command = "flux jobs {job_id} {user} {format}"

        fields = defaultdict(str, {})

        if flux_id:
            fields["flux_id"] = flux_id

        if jobs:
            if flux_id:
                command = 'flux proxy {flux_id} "' + command + '"'
            joblist = []
            if isinstance(jobs, str):
                joblist = jobs
            else:
                if not isinstance(jobs, (tuple, list)):
                    raise TypeError(
                        "If provided, the 'jobs' variable must be a string or a list of strings"
                    )
                joblist = " ".join(jobs)
            fields["job_id"] = joblist

        if user:
            fields["user"] = f"-u {user}"

        fields["format"] = (
            f"--format '{self._FIELD_SEPARATOR.join(f'{{{field[0]}}}' for field in self.fields)}'"
        )
        comm = command.format_map(fields)

        self.logger.info(f"Checking joblist with {comm}")

        return comm

    def _get_detailed_job_info_command(self, job_id: str) -> dict[str, Any]:
        """
        Command to get detailed information on a job even after completion.

        :param job_id: Job id for the flux scheduler.
        :return comm: Command for detailed job info.
        """

        allocation_id, separator, child_id = job_id.partition(":")
        if separator and child_id.startswith("pending-"):
            return build_deferred_command(self._deferred_identity(job_id, "detail"))
        if separator:
            if not allocation_id or not child_id or ":" in child_id:
                raise ValueError(f"Invalid pooled Flux job ID: {job_id!r}")
            return (
                f"flux proxy {shlex.quote(allocation_id)} "
                f"flux job info {shlex.quote(child_id)} jobspec"
            )

        return f"flux job info {shlex.quote(job_id)} jobspec"

    def _get_submit_script_header(self, job_tmpl: JobTemplate) -> str:
        """
        Return the submit script header with the parameters from the job_tmpl.

        :param job_tmpl: JobTemplate instance with relevant parameters set.
        :return header: Job submission header as a string.
        """

        header = []

        if job_tmpl.job_name:
            header.append(f"#flux: --job-name={job_tmpl.job_name}")

        if job_tmpl.sched_output_path:
            header.append(f"#flux: --output={job_tmpl.sched_output_path}")

        if job_tmpl.sched_error_path:
            header.append(f"#flux: --error={job_tmpl.sched_error_path}")

        # Nested instances inherit the queue and bank from their allocation.
        nested = job_tmpl.job_resource and (
            job_tmpl.job_resource.get("flux_pool")
            or job_tmpl.job_resource.get("flux_allocation_lifecycle")
        )
        if not nested:
            if job_tmpl.queue_name:
                header.append(f"#flux: -q {shlex.quote(job_tmpl.queue_name)}")
            if job_tmpl.account:
                header.append(f"#flux: -B {shlex.quote(job_tmpl.account)}")

        if job_tmpl.priority:
            # Check that the specified value is within the appropriate range.
            # 0 - Hold
            # 16 - Default
            # 31 - Expedite
            priority = job_tmpl.priority
            if priority >= 31:
                priority = 31
            elif priority < 0:
                priority = 0
            header.append(f"#flux: --urgency={priority}")

        if not job_tmpl.job_resource:
            raise ValueError(
                "Job resources (number of nodes) are required for the Flux "
                "scheduler plugin."
            )

        header.append(f"#flux: -N {job_tmpl.job_resource.num_machines}")
        if job_tmpl.job_resource.num_mpiprocs_per_machine:
            header.append(
                f"#flux: -n "
                f"{job_tmpl.job_resource.num_mpiprocs_per_machine * job_tmpl.job_resource.num_machines}"
            )

        if job_tmpl.job_resource.num_cores_per_mpiproc:
            header.append(f"#flux: -c {job_tmpl.job_resource.num_cores_per_mpiproc}")

        if job_tmpl.job_resource.num_gpus_per_mpiproc:
            header.append(f"#flux: -g {job_tmpl.job_resource.num_gpus_per_mpiproc}")

        if job_tmpl.max_wallclock_seconds is not None:
            try:
                tot_secs = int(job_tmpl.max_wallclock_seconds)
                if tot_secs <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    "max_wallclock_seconds must be "
                    "a positive integer (in seconds)! It is instead '{}'"
                    "".format((job_tmpl.max_wallclock_seconds))
                )

            # Check if total time is larger than day, hour, or minutes
            # and convert to float.

            # Days
            if tot_secs > 86400:
                time = f"{tot_secs / 86400:.2f}d"
            # Hours
            elif tot_secs > 3600:
                time = f"{tot_secs / 3600:.2f}h"
            elif tot_secs > 60:
                time = f"{tot_secs / 60:.2f}m"
            else:
                time = tot_secs

            header.append(f"#flux: -t {time}")

        if (
            isinstance(job_tmpl.custom_scheduler_commands, str)
            and job_tmpl.custom_scheduler_commands is not None
        ):
            header.append(job_tmpl.custom_scheduler_commands)

        header = "\n".join(header)

        return header

    def _get_parent_pk(self, pk: int) -> int:
        """
        Return parent pk based on calculation pk retrieved.

        :param pk: AiiDA PK
        :return: The parent pk of the calculation pk.
        """

        parent = load_node(pk)
        while parent.caller:
            parent = parent.caller

        return parent.pk

    def _parse_pk(
        self,
        working_directory: str,
        submit_script: str,
    ) -> int:
        """
        Parse the AiiDA PK from the job submission script.

        :param working_directory: Path to the working directory on remote machine.
        :param submit_script: Name of the submission script.
        :returns: PK of the current calculation.
        """

        command = (
            f"cd {escape_for_bash(working_directory)} && "
            f'grep "#flux" {escape_for_bash(submit_script)}'
        )
        retval, stdout, stderr = self.transport.exec_command_wait(command)

        if retval != 0:
            self.logger.error(
                f"Error in _flux_allocation: {retval=}; {stdout=}; {stderr=}"
            )
            raise SchedulerError(
                f"Error during submission, {retval=}\n{stdout=}\n{stderr=}"
            )

        self.flux_values = self._parse_flux_directives(stdout)

        if "t" not in self.flux_values or "job-name" not in self.flux_values:
            raise SchedulerError(
                "The Flux submit script must contain time-limit and job-name directives."
            )

        # Convert walltime to seconds
        walltime = self.flux_values["t"]
        if "d" in walltime:
            walltime = 86400 * float(walltime.strip("d"))
        elif "h" in walltime:
            walltime = 3600 * float(walltime.strip("h"))
        elif "m" in walltime:
            walltime = 60 * float(walltime.strip("m"))
        elif walltime.endswith("s"):
            walltime = float(walltime[:-1])
        else:
            walltime = 60 * float(walltime)
        self.flux_values["t"] = math.ceil(walltime)

        pk = int(self.flux_values["job-name"].split("-")[-1])

        return pk

    @staticmethod
    def _parse_flux_directives(output: str) -> dict[str, str]:
        """Parse only the Flux directives needed to identify an AiiDA job."""

        values = {}
        for line in output.splitlines():
            _, _, directive = line.partition("#flux:")
            if not directive:
                continue
            try:
                tokens = shlex.split(directive, comments=True)
            except ValueError:
                continue

            for index, token in enumerate(tokens):
                if token.startswith("--job-name="):
                    values["job-name"] = token.split("=", maxsplit=1)[1]
                elif token == "--job-name" and index + 1 < len(tokens):
                    values["job-name"] = tokens[index + 1]
                elif token.startswith("--time-limit="):
                    values["t"] = token.split("=", maxsplit=1)[1]
                elif token in ("-t", "--time-limit") and index + 1 < len(tokens):
                    values["t"] = tokens[index + 1]
                elif token.startswith("-t") and token != "-t":
                    values["t"] = token[2:]

        return values

    def _flux_allocation(self, pk: int) -> str:
        """
        Start a flux allocation to submit jobs.

        :param pk: AiiDA PK of the current job submission.
        :returns: Flux job ID of the persistent allocation.
        """

        return self._flux_allocation_with_ownership(pk)[0]

    def _flux_allocation_with_ownership(self, pk: int) -> tuple[str, bool]:
        """Return an allocation ID and whether this submission created it.

        Direct allocations are child-owned unless callers explicitly select
        parent reuse. Shared pools own their allocation lifecycle independently
        and must never be cancelled because an individual child fails.
        """

        node = load_node(pk)

        if not isinstance(node, CalcJobNode):
            raise TypeError(f"{node} is not a recognized type for this scheduler.")

        allocation_config = self._resolve_allocation_config(node)
        if not allocation_config.get("pool") and not allocation_config["lifecycle"]:
            return "", False

        parent_pk = self._get_parent_pk(pk)
        if allocation_config.get("pool"):
            self._validate_job_fits_pool(node, allocation_config)
            flux_id = self._get_or_create_pooled_allocation(
                node,
                parent_pk,
                allocation_config,
            )
            self.logger.info(
                f"Flux pool `{allocation_config['pool']}` is using flux id: {flux_id}."
            )
            return flux_id, False

        if allocation_config["lifecycle"] == "child":
            flux_id = self._start_allocation(allocation_config, pk)
            self.logger.info(
                f"Created child-owned Flux allocation {flux_id} for <{pk}>."
            )
            return flux_id, True

        # Based on the parent_pk, see if there is an active flux allocation.
        state = self._check_allocation(parent_pk)

        if not state.active:
            self.logger.info(
                f"No flux allocation found for aiida-{parent_pk}. Starting one now."
            )
            flux_id = self._start_allocation(allocation_config, parent_pk)
            created = True
        elif state.active:
            if not self._direct_allocation_is_compatible(
                state.job_info, allocation_config
            ):
                raise SchedulerError(
                    f"Existing direct Flux allocation `{state.flux_id}` for aiida-{parent_pk} "
                    "is incompatible with this child resource request. Use the default "
                    "child lifecycle or a named Flux pool."
                )
            # Check if there is enough walltime left for the job.
            diff = state.walltime - self.flux_values["t"]
            if diff < 0:
                self.logger.info(
                    "Current job exceeds remaining time. Killing allocation "
                    "and requesting a new allocation."
                )
                # Kill allocation and start a new one
                kill_cmd = self._get_kill_command(state.flux_id)
                retval, stdout, stderr = self.transport.exec_command_wait(kill_cmd)
                if retval != 0:
                    self.logger.error(
                        f"Error in _flux_allocation: {retval=}; {stdout=}; {stderr=}"
                    )
                    raise SchedulerError(
                        f"Error during submission, {retval=}\n{stdout=}\n{stderr=}"
                    )
                flux_id = self._start_allocation(allocation_config, parent_pk)
                created = True
            else:
                flux_id = state.flux_id
                created = False

        self.logger.info(
            f"Flux instance for <{parent_pk}> is running with flux id: {flux_id}."
        )

        return flux_id, created

    def _check_allocation(self, parent_pk: int) -> namedtuple:
        """
        Given a parent_pk, check to see if Flux already has an active allocation.

        :param parent_pk: The parent PK of the job being submitted.
        :returns: State of the current parent allocation in Flux.
        """
        joblist = self.get_jobs()
        State = namedtuple(
            "State",
            ["active", "flux_id", "walltime", "job_info"],
            defaults=[False, "", 0, None],
        )

        state = State()
        for job in joblist:
            if getattr(job, "title", None) == f"aiida-{parent_pk}":
                flux_id = job.job_id
                total = job.requested_wallclock_time_seconds or 0
                used = job.wallclock_time_seconds or 0
                remaining = total - used
                state = State(True, flux_id, remaining, job)

        return state

    def _direct_allocation_is_compatible(
        self,
        job_info: JobInfo | None,
        allocation_config: dict[str, Any],
    ) -> bool:
        """Return whether a reusable direct allocation matches child resources."""

        if job_info is None:
            return False

        resources = allocation_config.get("resources", {})
        requested_nodes = self.recursive_dict_search("num_machines", resources)
        if requested_nodes is not None and job_info.num_machines != int(
            requested_nodes
        ):
            return False

        requested_slots = self.recursive_dict_search(
            "num_mpiprocs_per_machine", resources
        )
        if requested_slots is None:
            requested_slots = self.recursive_dict_search(
                "num_mpi_procs_per_machine", resources
            )
        requested_cores = self.recursive_dict_search("num_cores_per_mpiproc", resources)
        if requested_slots is not None:
            if job_info.num_mpiprocs is None:
                return False
            expected_cpus = int(requested_nodes or 1) * int(requested_slots)
            if requested_cores is not None:
                expected_cpus *= int(requested_cores)
            if job_info.num_mpiprocs != expected_cpus:
                return False

        requested_queue = self.recursive_dict_search("queue_name", resources)
        if requested_queue is not None and job_info.queue_name != requested_queue:
            return False

        # ``flux jobs`` does not expose the parent allocation's GPU shape or
        # bank. Reuse is unsafe if either has been requested but cannot be
        # verified from the scheduler record.
        if self.recursive_dict_search("num_gpus_per_mpiproc", resources) is not None:
            return False
        if self.recursive_dict_search("account", resources) is not None:
            return False

        return True

    def _start_allocation(self, allocation_config: dict[str, Any], parent_pk) -> str:
        """
        Start a flux allocation from a resolved allocation configuration.

        :param allocation_config: Resolved configuration for the allocation.
        :param parent_pk: AiiDA pk of parent.
        :return: Job ID of the Flux instance.
        """
        resources = allocation_config.get("resources", {})

        keys = (
            ("num_machines", "num_machines"),
            ("num_mpiprocs_per_machine", "num_mpiprocs_per_machine"),
            ("num_mpi_procs_per_machine", "num_mpiprocs_per_machine"),
            ("queue_name", "queue_name"),
            ("max_wallclock_seconds", "max_wallclock_seconds"),
            ("account", "account"),
            ("num_cores_per_mpiproc", "num_cores_per_mpiproc"),
            ("num_gpus_per_mpiproc", "num_gpus_per_mpiproc"),
        )

        allocation_resources = {}
        for search_key, target_key in keys:
            result = self.recursive_dict_search(search_key, resources)
            if result is not None:
                allocation_resources[target_key] = result

        values = defaultdict(str, {})

        if "num_machines" in allocation_resources:
            num_machines = int(allocation_resources["num_machines"])
            values["num_machines"] = f"--nodes={num_machines}"

            if "num_mpiprocs_per_machine" in allocation_resources:
                num_tasks = num_machines * int(
                    allocation_resources["num_mpiprocs_per_machine"]
                )
                values["num_tasks"] = f"-n {num_tasks}"

        if "num_cores_per_mpiproc" in allocation_resources:
            values["cores_per_task"] = (
                f"-c {int(allocation_resources['num_cores_per_mpiproc'])}"
            )

        if "num_gpus_per_mpiproc" in allocation_resources:
            values["gpus_per_task"] = (
                f"-g {int(allocation_resources['num_gpus_per_mpiproc'])}"
            )

        if "queue_name" in allocation_resources:
            values["queue_name"] = (
                f"-q {shlex.quote(str(allocation_resources['queue_name']))}"
            )

        if "max_wallclock_seconds" in allocation_resources:
            values["max_wallclock_seconds"] = (
                f"-t {int(allocation_resources['max_wallclock_seconds'])}s"
            )

        if "account" in allocation_resources:
            values["account"] = (
                f"-B {shlex.quote(str(allocation_resources['account']))}"
            )

        job_name = allocation_config.get("job_name", f"aiida-{parent_pk}")
        values["job_name"] = f"--job-name={shlex.quote(job_name)}"

        timeout = allocation_config.get("timeout", "5m")

        values["watcher"] = (
            f"bash -c 'while true; do sleep 60; if [ $(flux jobs --since=-{timeout} | wc -l) -gt 1 ]; then continue; else exit; fi; done'"
        )
        if allocation_config.get("pool"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", timeout)
            if match is None:
                raise SchedulerError(
                    "Pool idle timeout must be a positive duration ending in s, m, h, or d."
                )
            idle_seconds = (
                float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
            )
            if idle_seconds <= 0:
                raise SchedulerError("Pool idle timeout must be positive.")
            # Count active jobs structurally and allow a full initial grace
            # period for deferred intents to attach after broker startup.
            watcher = (
                "import json, subprocess, time\n"
                "idle_since = time.monotonic()\n"
                "while True:\n"
                "    time.sleep(5)\n"
                "    try:\n"
                "        jobs = json.loads(subprocess.check_output(['flux', 'jobs', '--json'], timeout=5))['jobs']\n"
                "    except Exception:\n"
                "        idle_since = time.monotonic()\n"
                "        continue\n"
                "    if jobs:\n"
                "        idle_since = time.monotonic()\n"
                f"    elif time.monotonic() - idle_since >= {idle_seconds!r}:\n"
                "        break\n"
            )
            values["watcher"] = shlex.join(["flux", "python", "-c", watcher])

        flux_submit = (
            "flux alloc {job_name} {num_machines} {num_tasks} "
            "{cores_per_task} {gpus_per_task} {queue_name} {account} "
            "{max_wallclock_seconds} -x --bg {watcher}"
        )

        flux_submit = flux_submit.format_map(values)

        if allocation_config.get("pool"):
            # --bg waits for resources. Submit its dry-run jobspec instead so
            # the pending allocation is published before releasing the lock.
            flux_submit = flux_submit.replace(" -x --bg ", " -x --dry-run ")
            request = {
                "op": "allocate",
                "generation": allocation_config["generation"],
                "job_name": job_name,
                "command": f"bash -c {shlex.quote('set -o pipefail; ' + flux_submit + ' | flux job submit')}",
            }
            response = self._run_deferred_request(request)
            if response.get("state") == "ambiguous" or not response.get(
                "allocation_id"
            ):
                raise SchedulerError(
                    "Pool allocation submission needs reconciliation; refusing a duplicate or direct fallback."
                )
            return response["allocation_id"]

        self.logger.info(
            f"Starting a flux allocation for parent workchain <{parent_pk}> with {flux_submit}"
        )

        retval, stdout, stderr = self.transport.exec_command_wait(flux_submit)

        if retval != 0:
            self.logger.error(
                f"Error in _start_allocation {retval=}; {stdout=}; {stderr=}"
            )

            raise SchedulerError(
                f"Error while starting a flux allocation, {retval=}\n{stdout=}\n{stderr=}"
            )
        else:
            flux_id = stdout

        flux_id = flux_id.strip("\n")

        return flux_id

    def _get_node_options(
        self,
        node: CalcJobNode,
    ) -> dict[str, Any]:
        """
        Return scheduler options from a CalcJobNode.
        """

        metadata_inputs = node.get_metadata_inputs() or {}
        metadata = metadata_inputs.get("metadata", {})

        return metadata.get("options", {})

    def _resolve_allocation_config(
        self,
        node: CalcJobNode,
    ) -> dict[str, Any]:
        """
        Resolve allocation configuration for a job submission.

        This is a temporary seam for the future pool registry. For now, the
        allocation envelope combines the resource shape with AiiDA's standard
        top-level scheduler options. A pool selector remains a resource key.
        """

        options = self._get_node_options(node)
        resources = dict(options.get("resources", {}))
        # AiiDA stores these scheduler options alongside ``resources``. Keep
        # them in the resolved envelope so ``flux alloc`` receives the same
        # queue, bank, and walltime requested by the CalcJob.
        for key in ("queue_name", "account", "max_wallclock_seconds"):
            if options.get(key) is not None:
                resources[key] = options[key]
        pool_name = self.recursive_dict_search("flux_pool", resources)

        if pool_name:
            if (
                self.recursive_dict_search("flux_allocation_lifecycle", resources)
                is not None
            ):
                raise SchedulerError(
                    "`flux_allocation_lifecycle` cannot be used with `flux_pool`; "
                    "named pools always retain their shared allocation lifecycle."
                )
            return self._load_pool_definition(node, pool_name)

        return {
            "pool": pool_name,
            "resources": resources,
            "lifecycle": self.recursive_dict_search(
                "flux_allocation_lifecycle", resources
            ),
        }

    def _validate_job_fits_pool(
        self,
        node: CalcJobNode,
        allocation_config: dict[str, Any],
    ) -> None:
        """Reject resource shapes that can never fit in the selected pool."""

        job_resources = self._get_node_options(node).get("resources", {})
        pool_resources = allocation_config.get("resources", {})

        def value(resources: dict[str, Any], key: str, default: int) -> int:
            result = self.recursive_dict_search(key, resources)
            return default if result is None else int(result)

        job_nodes = value(job_resources, "num_machines", 1)
        pool_nodes = value(pool_resources, "num_machines", 1)
        job_slots_per_node = value(job_resources, "num_mpiprocs_per_machine", 1)
        pool_slots_per_node = value(pool_resources, "num_mpiprocs_per_machine", 1)
        job_cores_per_slot = value(job_resources, "num_cores_per_mpiproc", 1)
        pool_cores_per_slot = value(pool_resources, "num_cores_per_mpiproc", 1)
        job_gpus_per_slot = value(job_resources, "num_gpus_per_mpiproc", 0)
        pool_gpus_per_slot = value(pool_resources, "num_gpus_per_mpiproc", 0)

        requested = {"nodes": job_nodes}
        available = {"nodes": pool_nodes}
        if (
            self.recursive_dict_search("num_cores_per_mpiproc", pool_resources)
            is not None
        ):
            requested["cores per node"] = job_slots_per_node * job_cores_per_slot
            available["cores per node"] = pool_slots_per_node * pool_cores_per_slot
        if job_gpus_per_slot or pool_gpus_per_slot:
            requested["GPUs per node"] = job_slots_per_node * job_gpus_per_slot
            available["GPUs per node"] = pool_slots_per_node * pool_gpus_per_slot
        exceeded = [
            name
            for name, requested_value in requested.items()
            if requested_value > available[name]
        ]
        if exceeded:
            details = ", ".join(
                f"{name}={requested[name]} (pool {available[name]})"
                for name in exceeded
            )
            raise SchedulerError(
                f"Job resource request cannot fit in Flux pool "
                f"`{allocation_config['pool']}`: {details}."
            )

    def _load_pool_definition(
        self,
        node: CalcJobNode,
        pool_name: str,
    ) -> dict[str, Any]:
        """
        Load a named pool definition from the AiiDA database.

        Pool definitions are stored in a namespaced group whose extras contain
        the plugin-managed configuration payload.
        """

        if node.computer is None:
            raise SchedulerError(f"CalcJobNode<{node.pk}> does not define a computer.")

        try:
            group = get_pool_group(
                node.user.email,
                node.computer.label,
                pool_name,
            )
        except exceptions.NotExistent as exception:
            raise SchedulerError(
                f"No Flux pool definition named `{pool_name}` exists for "
                f"user `{node.user.email}` on computer `{node.computer.label}`."
            ) from exception

        pool_definition = group.base.extras.all.get(POOL_EXTRA_KEY)
        if not isinstance(pool_definition, dict):
            raise SchedulerError(
                f"Flux pool `{pool_name}` is missing the `{POOL_EXTRA_KEY}` configuration payload."
            )

        if not pool_definition.get("enabled", True):
            raise SchedulerError(f"Flux pool `{pool_name}` is disabled.")

        pool_resources = pool_definition.get("resources", {})
        if not isinstance(pool_resources, dict):
            raise SchedulerError(
                f"Flux pool `{pool_name}` has an invalid `resources` definition."
            )

        return {
            "pool": pool_name,
            "group": group,
            "resources": pool_resources,
            "timeout": pool_definition.get("timeout", "5m"),
            "job_name": pool_definition.get("job_name", f"aiida-pool-{pool_name}"),
        }

    def _get_or_create_pooled_allocation(
        self,
        node: CalcJobNode,
        parent_pk: int,
        allocation_config: dict[str, Any],
    ) -> str:
        """
        Resolve a shared allocation from pool runtime state, creating it if
        necessary under a per-pool remote lock.
        """

        pool_name = allocation_config["pool"]
        lock_name = self._get_pool_lock_name(node, pool_name)
        lock_timeout_seconds = int(allocation_config.get("lock_timeout_seconds", 60))
        stale_lock_value = allocation_config.get("stale_lock_seconds", 300)
        stale_lock_seconds = None if stale_lock_value is None else int(stale_lock_value)

        self._acquire_pool_lock(
            lock_name,
            timeout_seconds=lock_timeout_seconds,
            stale_lock_seconds=stale_lock_seconds,
        )
        try:
            allocation_config = self._load_pool_definition(node, pool_name)
            group = get_pool_group(
                node.user.email,
                node.computer.label,
                pool_name,
            )
            runtime = get_pool_runtime(group)
            current_flux_id = runtime.get("current_flux_id")

            if current_flux_id:
                job_info = self._get_top_level_job(current_flux_id)
                expected_job_name = runtime.get("job_name") or allocation_config.get(
                    "job_name", f"aiida-pool-{pool_name}"
                )
                if (
                    job_info is not None
                    and job_info.title is not None
                    and job_info.title != expected_job_name
                ):
                    self.logger.warning(
                        f"Flux pool `{pool_name}` runtime id `{current_flux_id}` "
                        f"refers to job `{job_info.title}`, not `{expected_job_name}`."
                    )
                    raise SchedulerError(
                        "Flux pool allocation identity does not match its runtime record."
                    )
                if job_info is None or job_info.job_state == JobState.UNDETERMINED:
                    raise SchedulerError(
                        f"Cannot establish state of Flux pool allocation `{current_flux_id}`; "
                        "preserving its reference instead of creating a duplicate."
                    )
                if job_info.job_state in (JobState.QUEUED, JobState.QUEUED_HELD):
                    update_pool_runtime(
                        group,
                        state="queued",
                        last_used_by_parent_pk=parent_pk,
                        last_validated_at=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    )
                    return current_flux_id
                if job_info is not None and job_info.job_state == JobState.RUNNING:
                    remaining = self._get_remaining_walltime(job_info)
                    if remaining >= self.flux_values["t"]:
                        update_pool_runtime(
                            group,
                            current_flux_id=current_flux_id,
                            state="running",
                            last_used_by_parent_pk=parent_pk,
                            last_validated_at=datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat(),
                            remaining_seconds=remaining,
                        )
                        return current_flux_id

                    self.logger.info(
                        f"Flux pool `{pool_name}` allocation `{current_flux_id}` "
                        "does not have enough remaining walltime; allowing it to "
                        "drain while starting a replacement."
                    )
                elif job_info.job_state == JobState.DONE:
                    self.logger.info(
                        f"Flux pool `{pool_name}` allocation `{current_flux_id}` "
                        "is no longer running or could not be found; starting a "
                        "replacement allocation."
                    )

                else:
                    raise SchedulerError(
                        f"Unrecognized pool allocation state: {job_info.job_state}."
                    )

                clear_pool_runtime(group)
                runtime = {}

            # Persist the generation before contacting Flux. A retry can then
            # reconcile the remote receipt after a lost response or DB failure.
            generation = runtime.get("generation") or str(uuid.uuid4())
            job_name = runtime.get("job_name") or f"aiida-allocation-{generation}"
            allocation_config.update(generation=generation, job_name=job_name)
            update_pool_runtime(
                group, generation=generation, job_name=job_name, state="creating"
            )
            flux_id = self._start_allocation(allocation_config, parent_pk)
            update_pool_runtime(
                group,
                current_flux_id=flux_id,
                state="queued",
                last_used_by_parent_pk=parent_pk,
                last_validated_at=datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            )
            return flux_id
        finally:
            self._release_pool_lock(lock_name)

    @staticmethod
    def _get_pool_lock_name(node: CalcJobNode, pool_name: str) -> str:
        """Return a collision-resistant lock name scoped to user and computer."""

        identity = f"{node.user.email}\0{node.computer.label}\0{pool_name}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        safe_pool_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", pool_name)[:48]
        return f"{safe_pool_name}-{digest}"

    def _get_pool_lock_path(
        self,
        pool_name: str,
    ) -> str:
        """
        Return the remote lock directory path for a pool.
        """

        safe_pool_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", pool_name)
        return f"$HOME/.aiida-flux-scheduler/locks/{safe_pool_name}.lock"

    def _acquire_pool_lock(
        self,
        pool_name: str,
        timeout_seconds: int = 60,
        stale_lock_seconds: int | None = 300,
    ) -> None:
        """Acquire an owned lock, optionally recovering a provably dead local owner."""

        lock_path = self._get_pool_lock_path(pool_name)
        lock_root = lock_path.rsplit("/", maxsplit=1)[0]
        mkdir_root = f'umask 077; mkdir -p "{lock_root}" && chmod 700 "{lock_root}"'
        retval, stdout, stderr = self.transport.exec_command_wait(mkdir_root)
        if retval != 0:
            raise SchedulerError(
                f"Unable to prepare lock root for pool `{pool_name}`: {retval=}\n{stdout=}\n{stderr=}"
            )

        deadline = time.monotonic() + timeout_seconds

        while True:
            if self._write_pool_lock_metadata(pool_name):
                return

            if stale_lock_seconds is not None and self._recover_stale_pool_lock(
                pool_name,
                stale_lock_seconds=stale_lock_seconds,
            ):
                continue

            if time.monotonic() >= deadline:
                raise SchedulerError(
                    f"Timed out waiting for Flux pool lock `{pool_name}`; "
                    "active, foreign, and unverifiable owners are not reclaimed."
                )

            time.sleep(1)

    def _release_pool_lock(
        self,
        pool_name: str,
    ) -> None:
        """Release only this process's recorded acquisition, under the remote guard."""

        import os

        owners = getattr(self, "_pool_lock_owners", {})
        acquisition = owners.get(pool_name)
        if acquisition is None or acquisition["owner"]["pid"] != os.getpid():
            return
        lock_path = self._get_pool_lock_path(pool_name)
        metadata = json.dumps(acquisition, sort_keys=True)
        script = (
            f'chmod 600 "{lock_path}.guard" || exit 1; '
            f'current=$(cat "{self._get_pool_lock_metadata_path(pool_name)}") || exit 1; '
            f'[ "$current" = {shlex.quote(metadata)} ] || exit 1; '
            f'rm -f "{self._get_pool_lock_metadata_path(pool_name)}" && rmdir "{lock_path}"'
        )
        retval, stdout, stderr = self.transport.exec_command_wait(
            f'umask 077; flock -x -w 5 "{lock_path}.guard" sh -c {shlex.quote(script)}'
        )
        owners.pop(pool_name, None)
        if retval != 0:
            self.logger.warning(
                f"Unable to release Flux pool lock `{pool_name}`: {retval=}; {stdout=}; {stderr=}"
            )

    def _get_pool_lock_metadata_path(
        self,
        pool_name: str,
    ) -> str:
        """
        Return the metadata file path stored inside a lock directory.
        """

        return f"{self._get_pool_lock_path(pool_name)}/owner.json"

    def _write_pool_lock_metadata(
        self,
        pool_name: str,
    ) -> bool:
        """Atomically create the directory and owner metadata under a persistent guard."""

        import os
        import socket
        import uuid
        from pathlib import Path

        owners = getattr(self, "_pool_lock_owners", None)
        if owners is None:
            owners = self._pool_lock_owners = {}
        if pool_name in owners:
            raise SchedulerError(
                f"Flux pool lock `{pool_name}` is already held by this scheduler."
            )

        # A PID alone is not an identity: boot, namespace and start time prevent reuse.
        try:
            start_time = (
                Path(f"/proc/{os.getpid()}/stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()[19]
            )
            owner = {
                "host": socket.gethostname(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "pid_namespace": os.stat("/proc/self/ns/pid").st_ino,
                "uid": os.getuid(),
                "pid": os.getpid(),
                "start_time": start_time,
            }
        except (OSError, IndexError) as exc:
            raise SchedulerError(
                "Cannot establish a Linux process identity for the pool lock."
            ) from exc

        acquisition = {
            "schema_version": 1,
            "pool": pool_name,
            "token": uuid.uuid4().hex,
            "owner": owner,
            "acquired_at_epoch": int(time.time()),
        }
        metadata = json.dumps(acquisition, sort_keys=True)
        lock_path = self._get_pool_lock_path(pool_name)
        metadata_path = self._get_pool_lock_metadata_path(pool_name)
        script = (
            f'chmod 600 "{lock_path}.guard" || exit 1; '
            f'mkdir "{lock_path}" || {{ [ -d "{lock_path}" ] && exit 73; exit 1; }}; '
            f'printf %s {shlex.quote(metadata)} > "{metadata_path}" || '
            f'{{ rm -f "{metadata_path}"; rmdir "{lock_path}"; exit 1; }}'
        )
        retval, stdout, stderr = self.transport.exec_command_wait(
            f'umask 077; flock -x -w 5 "{lock_path}.guard" sh -c {shlex.quote(script)}'
        )
        if retval == 73:
            return False
        if retval != 0:
            raise SchedulerError(
                f"Unable to write metadata for Flux pool lock `{pool_name}`: "
                f"{retval=}\n{stdout=}\n{stderr=}"
            )
        owners[pool_name] = acquisition
        return True

    def _recover_stale_pool_lock(
        self,
        pool_name: str,
        stale_lock_seconds: int,
    ) -> bool:
        """Recover only aged locks whose same-host owner is provably dead.

        Foreign hosts, different PID namespaces, legacy metadata and inspection
        failures require manual intervention. The guard file must never be removed.
        """

        import os
        import socket
        from pathlib import Path

        lock_path = self._get_pool_lock_path(pool_name)
        metadata_path = self._get_pool_lock_metadata_path(pool_name)
        retval, stdout, _ = self.transport.exec_command_wait(f'cat "{metadata_path}"')
        if retval != 0:
            return False
        try:
            metadata = json.loads(stdout)
            if (
                not isinstance(metadata, dict)
                or type(metadata.get("schema_version")) is not int
                or metadata["schema_version"] != 1
            ):
                return False
            owner = metadata["owner"]
            epoch = metadata["acquired_at_epoch"]
            if (
                metadata.get("pool") != pool_name
                or not isinstance(metadata.get("token"), str)
                or re.fullmatch(r"[0-9a-f]{32}", metadata["token"]) is None
                or type(epoch) is not int
                or int(time.time()) - epoch < stale_lock_seconds
                or not isinstance(owner, dict)
                or type(owner.get("pid")) is not int
                or owner["pid"] <= 0
                or not isinstance(owner.get("start_time"), str)
                or not owner["start_time"].isdigit()
                or owner.get("host") != socket.gethostname()
                or owner.get("boot_id")
                != Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                or owner.get("pid_namespace") != os.stat("/proc/self/ns/pid").st_ino
                or owner.get("uid") != os.getuid()
            ):
                return False
            try:
                current_start = (
                    Path(f"/proc/{owner['pid']}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19]
                )
            except FileNotFoundError:
                # Hidden /proc entries are not evidence that the owner died.
                if not Path("/proc/self/stat").is_file():
                    return False
                try:
                    os.kill(owner["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    return False
            else:
                if current_start == owner["start_time"]:
                    return False
        except (KeyError, TypeError, ValueError, OSError, IndexError):
            return False

        # A replacement may have acquired between inspection and this command.
        # Compare the complete snapshot while holding the same mutation guard.
        script = (
            f'chmod 600 "{lock_path}.guard" || exit 1; '
            f'current=$(cat "{metadata_path}") || exit 1; '
            f'[ "$current" = {shlex.quote(stdout.rstrip(chr(10)))} ] || exit 1; '
            f'rm -f "{metadata_path}" && rmdir "{lock_path}"'
        )
        retval, stdout, stderr = self.transport.exec_command_wait(
            f'umask 077; flock -x -w 5 "{lock_path}.guard" sh -c {shlex.quote(script)}'
        )
        if retval == 0:
            self.logger.warning(
                f"Recovered Flux pool lock `{pool_name}` from a dead local owner."
            )
            return True

        self.logger.warning(
            f"Unable to recover stale Flux pool lock `{pool_name}`: "
            f"{retval=}; {stdout=}; {stderr=}"
        )
        return False

    def _get_pool_lock_age_seconds(
        self,
        pool_name: str,
    ) -> int | None:
        """Return metadata age for diagnostics, never as proof of abandonment."""

        metadata_path = self._get_pool_lock_metadata_path(pool_name)
        retval, stdout, _ = self.transport.exec_command_wait(f'cat "{metadata_path}"')
        if retval == 0:
            try:
                metadata = json.loads(stdout)
            except json.JSONDecodeError:
                return None
            else:
                if not isinstance(metadata, dict):
                    return None
                acquired_at_epoch = metadata.get("acquired_at_epoch")
                if type(acquired_at_epoch) is int:
                    return max(0, int(time.time()) - acquired_at_epoch)
        return None

    def _get_top_level_job(
        self,
        flux_id: str,
    ) -> JobInfo | None:
        """
        Return top-level Flux job information for a single allocation id.
        """

        fields = self._FIELD_SEPARATOR.join(f"{{{field[0]}}}" for field in self.fields)
        command = f"flux jobs {escape_for_bash(flux_id)} --format '{fields}'"
        retval, stdout, stderr = self.transport.exec_command_wait(command)

        if retval != 0:
            self.logger.info(
                f"Flux allocation lookup failed for `{flux_id}`: {retval=}; {stdout=}; {stderr=}"
            )
            return None

        jobs = self._parse_joblist_output(retval, stdout, stderr)
        if not jobs:
            return None

        return jobs[0]

    def _get_remaining_walltime(
        self,
        job_info: JobInfo,
    ) -> int:
        """
        Return remaining walltime in seconds for a Flux job.
        """

        total = job_info.requested_wallclock_time_seconds or 0
        used = job_info.wallclock_time_seconds or 0

        return total - used

    def _cancel_job_if_exists(
        self,
        job_id: str,
    ) -> None:
        """
        Attempt to cancel a Flux job, ignoring missing-job failures.
        """

        retval, stdout, stderr = self.transport.exec_command_wait(
            self._get_kill_command(job_id)
        )
        if retval != 0:
            self.logger.warning(
                f"Unable to cancel Flux job `{job_id}`: {retval=}; {stdout=}; {stderr=}"
            )

    def recursive_dict_search(self, key, dictionary: dict):
        """
        Take a dictionary and look for the first instance of the key.
        """

        if key in dictionary:
            return dictionary[key]
        for value in dictionary.values():
            if isinstance(value, dict):
                result = self.recursive_dict_search(key, value)
                if result is not None:
                    return result
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        result = self.recursive_dict_search(key, item)
                        if result is not None:
                            return result
        return None

    def _parse_seconds_field(
        self,
        value: str | None,
        field_name: str,
        job_id: str,
    ) -> int | None:
        """
        Parse Flux time fields that are reported as float seconds.
        """

        if value in (None, ""):
            return None

        try:
            return int(float(value))
        except ValueError:
            self.logger.warning(
                f"Error parsing {field_name} for job id {job_id}: {value!r}"
            )
            return None

    def get_jobs(
        self,
        jobs: list[str] | None = None,
        user: str | None = None,
        as_dict: bool = False,
    ) -> list[JobInfo] | dict[str, JobInfo]:
        """Return the list of currently active jobs.

        .. note:: typically, only either jobs or user can be specified. See
            also comments in `_get_joblist_command`.

        :param list jobs: a list of jobs to check; only these are checked
        :param str user: a string with a user: only jobs of this user are checked
        :param list as_dict: if False (default), a list of JobInfo objects
            is returned. If True, a dictionary is returned, having as key
            the job_id and as value the JobInfo object.
        :return: list of active jobs
        """

        if jobs:
            joblist = []
            for job in jobs:
                if ":pending-" in job:
                    joblist.append(self._get_deferred_job(job))
                    continue
                allocation_id, separator, child_id = job.partition(":")
                if separator and (not allocation_id or not child_id or ":" in child_id):
                    raise ValueError(f"Invalid pooled Flux job ID: {job!r}")
                flux_id = allocation_id if separator else None
                query_id = child_id if separator else allocation_id
                with self.transport:
                    retval, stdout, stderr = self.transport.exec_command_wait(
                        self._get_joblist_command(
                            jobs=[query_id], user=user, flux_id=flux_id
                        )
                    )

                for single_job in self._parse_joblist_output(retval, stdout, stderr):
                    if flux_id:
                        single_job.job_id = f"{flux_id}:{single_job.job_id}"
                    joblist.append(single_job)
        else:
            with self.transport:
                retval, stdout, stderr = self.transport.exec_command_wait(
                    self._get_joblist_command(jobs=jobs, user=user)
                )
            joblist = self._parse_joblist_output(retval, stdout, stderr)

        if as_dict:
            jobdict = {job.job_id: job for job in joblist}
            if None in jobdict:
                raise SchedulerError("Found at least one job without jobid")
            return jobdict

        return joblist

    def _get_submit_command(
        self,
        submit_script: str,
        flux_id: str | None = None,
        queue_name: str | None = None,
        account: str | None = None,
    ) -> str:
        """
        Return the string to execute the submission script.

        :param submit_script: Path to the submission script relative to the
            working directory.
        :return submit_command: Command used to submit the submission script.
        """

        submit_command = f"flux batch {submit_script}"
        if flux_id:
            submit_command = f"flux proxy {shlex.quote(flux_id)} {submit_command}"
        else:
            options = []
            if queue_name:
                options.append(f"-q {shlex.quote(queue_name)}")
            if account:
                options.append(f"-B {shlex.quote(account)}")
            submit_command = " ".join(["flux batch", *options, submit_script])

        self.logger.info(f"submitting with : {submit_command}")

        return submit_command

    def submit_job(self, working_directory: str, submit_script: str) -> str | ExitCode:
        """
        Submit a job to the Flux scheduler.

        :param working_directory: Absolute filepath to working directory of the job to be submitted.
        :param submit_script: Name of the submission script relative to the working directory.
        """

        pk = self._parse_pk(working_directory, submit_script)
        node = load_node(pk)
        pooled = bool(
            self._get_node_options(node).get("resources", {}).get("flux_pool")
        )
        if pooled:
            receipt = self._run_deferred_request(
                {"op": "lookup", "calculation_uuid": str(node.uuid)}
            )
            if receipt:
                self._maybe_cleanup_journals(node)
                return f"{receipt['allocation_id']}:pending-{node.uuid}"
        flux_id, created_direct_allocation = self._flux_allocation_with_ownership(pk)
        if flux_id:
            if pooled:
                self._run_deferred_request(
                    {
                        "op": "register",
                        "allocation_id": flux_id,
                        "calculation_uuid": str(node.uuid),
                        "working_directory": working_directory,
                        "submit_script": submit_script,
                        "requested_wallclock_seconds": int(
                            float(self.flux_values["t"])
                        ),
                    }
                )
                self._maybe_cleanup_journals(node)
                return f"{flux_id}:pending-{node.uuid}"
        # Already-uploaded scripts may predate direct queue/bank headers. Use
        # recorded options on the command line as well so resuming is safe.
        options = self._get_node_options(load_node(pk)) if not flux_id else {}
        # ``core.ssh_async`` transports do not implement ``chdir``. Submit
        # from the CalcJob work directory explicitly so both synchronous and
        # asynchronous AiiDA transports use the same scheduler path.
        command = self._get_submit_command(
            escape_for_bash(submit_script),
            flux_id,
            queue_name=options.get("queue_name"),
            account=options.get("account"),
        )
        result = self.transport.exec_command_wait(
            f"cd {escape_for_bash(working_directory)} && {command}"
        )
        try:
            child_job_id = self._parse_submit_output(*result)
        except SchedulerError:
            # Only a non-zero command status proves Flux rejected the child.
            # A transport exception or absent stdout can occur after Flux has
            # accepted it, so cancelling the allocation in those cases could
            # kill an unrecorded running child.
            if created_direct_allocation and result[0] != 0:
                self.logger.warning(
                    "Flux rejected child submission; cancelling newly created "
                    f"direct Flux allocation `{flux_id}`."
                )
                self._cancel_job_if_exists(flux_id)
            raise

        total_job_id = f"{flux_id}:{child_job_id}" if flux_id else child_job_id

        return total_job_id

    def _maybe_cleanup_journals(self, node: CalcJobNode) -> None:
        """Run best-effort, throttled maintenance only after a pooled intent is accepted."""
        try:
            report = self._run_deferred_request(
                {
                    "op": "auto_cleanup_begin",
                    "retention_seconds": 30 * 86400,
                    "interval_seconds": 86400,
                    "batch_size": 100,
                }
            )
            if not report.get("token"):
                return
            approved = {}
            for row in report["records"]:
                if (
                    row["action"] == "eligible"
                    and cleanup_approval(row, node.computer) is None
                ):
                    approved[row["calculation_uuid"]] = row["fingerprint"]
            result = self._run_deferred_request(
                {
                    "op": "auto_cleanup_finish",
                    "token": report["token"],
                    "approved": approved,
                }
            )
            count = sum(row["action"] == "compacted" for row in result["records"])
            if count:
                self.logger.info(
                    f"Compacted {count} old terminal Flux submission journals."
                )
        except Exception as exception:
            # Acceptance is already durable. Maintenance must not turn success
            # into an AiiDA transport retry or cause a duplicate submission.
            self.logger.warning(f"Skipped automatic Flux journal cleanup: {exception}")

    @staticmethod
    def _deferred_identity(job_id: str, operation: str) -> dict[str, str]:
        """Validate a deferred scheduler ID and build its journal request."""
        allocation_id, separator, child_id = job_id.partition(":")
        if not separator or not allocation_id or not child_id.startswith("pending-"):
            raise ValueError(f"Invalid deferred Flux job ID: {job_id!r}")
        calculation_uuid = str(uuid.UUID(child_id.removeprefix("pending-")))
        return {
            "op": operation,
            "allocation_id": allocation_id,
            "calculation_uuid": calculation_uuid,
        }

    def _run_deferred_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute a bounded remote journal operation and decode its response."""
        retval, stdout, stderr = self.transport.exec_command_wait(
            build_deferred_command(request)
        )
        if retval:
            raise SchedulerError(f"Flux pool journal operation failed: {stderr}")
        try:
            response = json.loads(stdout)
        except (ValueError, TypeError) as exception:
            raise SchedulerError("Invalid Flux pool journal response.") from exception
        if not isinstance(response, dict):
            raise SchedulerError("Flux pool journal response must be a mapping.")
        return response

    def _get_deferred_job(self, job_id: str) -> JobInfo:
        """Advance a pending intent without consuming CalcJob submission retries."""
        response = self._run_deferred_request(self._deferred_identity(job_id, "poll"))
        if response.get("state") == "ambiguous":
            raise SchedulerError(f"Flux submission needs reconciliation: {job_id}.")
        if response.get("state") in ("done", "cancelled"):
            info = JobInfo()
            info.job_id = job_id
            info.job_state = JobState.DONE
            info.annotation = response.get("reason")
            return info
        child_id = (
            None if response.get("allocation_ended") else response.get("child_id")
        )
        if child_id:
            allocation_id = job_id.partition(":")[0]
            retval, stdout, stderr = self.transport.exec_command_wait(
                self._get_joblist_command(jobs=[child_id], flux_id=allocation_id)
            )
            jobs = self._parse_joblist_output(retval, stdout, stderr)
            if not jobs:
                raise SchedulerError(
                    f"Cannot establish state of deferred child `{job_id}`."
                )
            info = jobs[0]
        else:
            info = JobInfo()
            info.job_state = (
                JobState.DONE
                if response.get("state") in ("done", "cancelled")
                else JobState.QUEUED
            )
            info.annotation = response.get(
                "reason", "Waiting for shared Flux allocation"
            )
        info.job_id = job_id
        return info

    def _parse_submit_output(
        self, retval: int, stdout: str, stderr: str
    ) -> str | ExitCode:
        """
        Parse the output from the submission command.

        :param retval: Return value of the submission command.
        :param stdout: Standard output.
        :param stderr: Error from standard output.
        :return job_id: Job ID from the submitted job.
        """

        if retval != 0:
            self.logger.error(
                f"Error in _parse_submit_output: {retval=}; {stdout=}; {stderr=}"
            )

            raise SchedulerError(
                f"Error during submission, {retval=}, {stdout=}, {stderr=}"
            )

        try:
            transport_string = f" for {self.transport}"
        except SchedulerError:
            transport_string = ""

        if stderr.strip():
            self.logger.warning(
                f"in _parse_submit_output{transport_string}: there was some text in stderr: {stderr}"
            )

        stdout = stdout.strip("\n")
        if stdout:
            return stdout
        self.logger.error(
            f" in _parse_submit_output{transport_string}: unable to find the job id: {stdout}"
        )
        raise SchedulerError(
            "Error during submission, could not retrieve the jobID from flux output; see log for more info."
        )

    def _parse_joblist_output(
        self, retval: int, stdout: str, stderr: str
    ) -> list[JobInfo]:
        """
        Parse the output from the job queue as returned by the
        _get_joblist_command command. The return is a list of lines, one for
        each job.

        :param retval: Return value from the command.
        :param stdout: Standard output from command.
        :param stderr: Standard error from command.
        :return job_list: List of JobInfo instances for each submitted job.
        """

        num_fields = len(self.fields)

        # See discussion in _get_joblist_command on how we ensure that AiiDA can expect exit code 0 here.
        if retval != 0:
            raise SchedulerError(
                f"""flux jobs returned exit code {retval} (_parse_joblist_output function)
                stdout='{stdout.strip()}'
                stderr='{stderr.strip()}'"""
            )
        if stderr.strip():
            self.logger.warning(
                f"flux jobs returned exit code 0 (_parse_joblist_output function) but non-empty stderr='{stderr.strip()}'"
            )

        stdout_split = stdout.splitlines()[
            1:
        ]  # Remove the first line which has the header names.
        jobdata_raw = [
            line.split(self._FIELD_SEPARATOR, maxsplit=num_fields)
            for line in stdout_split
        ]

        job_list = []

        for job in jobdata_raw:
            thisjob_dict = {k[1]: v for k, v in zip(self.fields, job)}

            this_job = JobInfo()
            try:
                this_job.job_id = thisjob_dict["job_id"]
                this_job.annotation = thisjob_dict["annotation"]
                job_state_raw = thisjob_dict["state_raw"]
            except KeyError:
                self.logger.error(f"Wrong line length in flux output! '{job}'")

            try:
                job_state_string = _MAP_STATUS_FLUX[job_state_raw]
            except KeyError:
                self.logger.warning(
                    f"Unrecognized job_state '{job_state_raw}' for job id {this_job.job_id}"
                )
                job_state_string = JobState.UNDETERMINED
            this_job.job_state = job_state_string

            if len(job) < num_fields:
                self.logger.warning(
                    f"Wrong line length in flux output! Skipping optional fields. Line: `{jobdata_raw}`"
                )
                job_list.append(this_job)
                continue

            this_job.job_owner = thisjob_dict["username"]

            try:
                this_job.num_machines = int(thisjob_dict["number_nodes"])
            except ValueError:
                self.logger.warning(
                    f"The number of allocated nodes is not an integer "
                    f"({thisjob_dict['number_nodes']}) for job id "
                    f"{this_job.job_id}!"
                )

            try:
                this_job.num_mpiprocs = int(thisjob_dict["number_cpus"])
            except ValueError:
                self.logger.warning(
                    f"The number of allocated cores is not an integer "
                    f"({thisjob_dict['number_cpus']}) for job id "
                    f"{this_job.job_id}!"
                )

            if this_job.job_state == JobState.RUNNING:
                this_job.allocated_machines_raw = thisjob_dict["allocated_machines"]

            this_job.queue_name = thisjob_dict["partition"]

            this_job.requested_wallclock_time_seconds = self._parse_seconds_field(
                thisjob_dict.get("time_limit"),
                "time_limit",
                this_job.job_id,
            )

            this_job.wallclock_time_seconds = self._parse_seconds_field(
                thisjob_dict.get("time_used"),
                "time_used",
                this_job.job_id,
            )

            if this_job.job_state == JobState.RUNNING:
                try:
                    dispatch_time = float(thisjob_dict["dispatch_time"])
                    dispatch_time = datetime.datetime.fromtimestamp(dispatch_time)
                    this_job.dispatch_time = dispatch_time
                except ValueError:
                    self.logger.warning(
                        f"Error parsing dispatch_time for job id {this_job.job_id}"
                    )

            try:
                submission_time = float(thisjob_dict["submission_time"])
                submission_time = datetime.datetime.fromtimestamp(submission_time)
                this_job.submission_time = submission_time
            except ValueError:
                self.logger.warning(
                    f"Error parsing submission_time for job id {this_job.job_id}"
                )

            this_job.title = thisjob_dict["job_name"]

            job_list.append(this_job)

        return job_list

    def kill_job(self, jobid) -> bool:
        """
        Function to kill a job on the Flux scheduler.

        :param jobid: Job ID within Flux
        :returns: True if executed correctly.
        """

        retval, stdout, stderr = self.transport.exec_command_wait(
            self._get_kill_command(jobid=jobid)
        )

        if retval != 0:
            self.logger.error(f"Error in kill_job {retval=}; {stdout=}; {stderr=}")

            raise RuntimeError(
                f"Error while kill Flux job, {retval=}\n{stdout=}\n{stderr=}"
            )

        return True

    def _get_kill_command(self, jobid: str) -> str:
        """
        Return the command to kill the job with the specified jobid.

        :param jobid: Job ID of the job within Flux.
        :return comm: Command to kill a job within Flux.
        """

        allocation_id, separator, child_id = jobid.partition(":")
        if separator and child_id.startswith("pending-"):
            return build_deferred_command(self._deferred_identity(jobid, "cancel"))
        if separator:
            if not allocation_id or not child_id or ":" in child_id:
                raise ValueError(f"Invalid pooled Flux job ID: {jobid!r}")
            return (
                f"flux proxy {shlex.quote(allocation_id)} "
                f"flux cancel {shlex.quote(child_id)}"
            )

        return f"flux cancel {shlex.quote(jobid)}"

    def _parse_kill_output(self, retval: int, stdout: str, stderr: str) -> bool:
        """
        Parse the output returned from the kill command.

        :param retval: Return value from the kill command.
        :param stdout: Standard output from the kill command.
        :param stderr: Standard error from the kill command.
        :return: True if everything is ok, False otherwise.
        """

        if retval != 0:
            self.logger.error(
                f"Error in _parse_kill_output: retval={retval}; "
                f"stdout={stdout}; stderr={stderr}"
            )
            return False

        try:
            transport_string = f" for {self.transport}"
        except SchedulerError:
            transport_string = ""

        if stderr.strip():
            self.logger.warning(
                f"in _parse_kill_output{transport_string}: there was some "
                f"text in stderr: {stderr}"
            )

        if stdout.strip():
            self.logger.warning(
                f"in _parse_kill_output{transport_string}: there was some "
                f"text in stdout: {stdout}"
            )

        return True

    def parse_output(
        self,
        detailed_job_info: dict[str, str | int] | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> ExitCode | None:
        """
        Parse the output of the scheduler.

        :param detailed_job_info: dictionary with the ouput returned by the
            `Scheduler.get_detailed_job_info` command. This should contain the
            keys `retval`, `stdout`, and `stderr` corresponding to the return
            value, stdout and stderr returned by the accounting command
            executed for a specfic job id.
        :param stdout: Standard output from the scheduler.
        :param stderr: Standard error from the scheduler.
        :return: Raise error otherwise None.
        """

        if detailed_job_info is not None:
            type_check(detailed_job_info, dict)

            try:
                detailed_stdout = json.loads(detailed_job_info["stdout"])
            except KeyError:
                raise ValueError(
                    "the `detailed_job_info` does not contain the "
                    "required key `stdout`."
                )

            # The format of the detailed job info should be a dictionary.
            type_check(detailed_stdout, dict)

            # data = dict(zip(fields, attributes))

            # if data['State'] == 'OUT_OF_MEMORY':
            #    return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY

            # if data['State'] == 'TO':
            #    return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME

            # if data['State'] == 'F':
            #    return CalcJob.exit_codes.ERROR_SCHEDULER_NODE_FAILURE

        return None
