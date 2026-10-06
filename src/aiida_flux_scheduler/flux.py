"""
Plugin for Flux.
"""

from typing import Any
from aiida.common.lang import type_check
from aiida.engine.processes.exit_code import ExitCode
from aiida.schedulers import Scheduler, SchedulerError
from aiida.schedulers.datastructures import JobInfo, JobState, JobTemplate, NodeNumberJobResource
from aiida.common.extendeddicts import AttributeDict
import re
import json
import string
import datetime

_MAP_STATUS_FLUX = {
    'D': JobState.QUEUED,  # Depend
    'P': JobState.QUEUED,  # Priority
    'S': JobState.QUEUED,  # Scheduled
    'R': JobState.RUNNING, # Running
    'C': JobState.RUNNING, # Cleanup
    'CD': JobState.DONE,   # Completed
    'F': JobState.DONE,    # Failed
    'CA': JobState.DONE,   # Canceled
    'TO': JobState.DONE,   # Timeout
}

class FluxJobResource(NodeNumberJobResource):
    """
    Class for Flux job resources.
    """

    @classmethod
    def validate_resources(
        cls, 
        **kwargs: dict
    ) -> AttributeDict:
        """
        Validate the resouces against the job resource class of this scheduler.

        :param kwargs: dictionary of values to define the job resources.
        :return: attribute dictionary with the parsed parameters populated.
        """

        resources = super().validate_resources(**kwargs)

        return resources
    
class FluxScheduler(Scheduler):
    """
    Flux scheduler.
    """

    _FIELD_SEPARATOR="|"

    _features = {
        'can_query_by_user': False,
    }

    _job_resource_class = FluxJobResource

    fields = [
        ('id', 'job_id'),  # job or job step id
        ('status_abbrev', 'state_raw'),  # job state in compact form
        ('annotations', 'annotation'),  # reason for the job being in its current state
        ('username', 'username'),  # username
        ('nnodes', 'number_nodes'),  # number of nodes allocated
        ('ncores', 'number_cpus'),  # number of allocated cores (if already running)
        ('nodelist', 'allocated_machines'),  # list of allocated nodes when running, otherwise
        # reason within parenthesis
        ('queue', 'partition'),  # partition (queue) of the job
        ('duration', 'time_limit'),  # time limit in seconds
        ('runtime', 'time_used'),  # Time used by the job in days-hours:minutes:seconds
        ('t_run', 'dispatch_time'),  # actual or expected dispatch time (start time)
        ('name', 'job_name'),  # job name (title)
        ('t_submit', 'submission_time'),  # This is probably new, it exists in version
        # 14.03.7 and later
    ]

    def _get_joblist_command(
        self, 
        jobs: list[str] | None = None, 
        user: str | None = None
    ) -> str:
        """
        Command to report full information on an existing job.

        :param jobs: List of job ids.
        :param user: Username for the job queue.
        :return comm: Command to retrieve full job information.
        """

        command = [ "flux", "jobs"]

        if jobs:
            joblist = []
            if isinstance(jobs, str):
                joblist = jobs
            else:
                if not isinstance(jobs, (tuple, list)):
                    raise TypeError("If provided, the 'jobs' variable must be a string or a list of strings")
                joblist = ' '.join(jobs)
            command.append(joblist)

        if user:
            command.append(f'-u {user}')

        command.append(f"--format '{self._FIELD_SEPARATOR.join(f'{{{field[0]}}}' for field in self.fields)}'")

        comm = ' '.join(command)

        return comm
    
    def _get_detailed_job_info_command(
        self, 
        job_id: str
    ) -> dict[str, Any]:
        """
        Command to get detailed information on a job even after completion.

        :param job_id: Job id for the flux scheduler.
        :return comm: Command for detailed job info.
        """

        return f"flux job info {job_id} jobspec"

    def submit_job(
        self,
        working_directory: str,
        submit_script: str,
    ) -> str | ExitCode:
        """Submit a job to Flux.

        :param working_directory: Absolute path to the job working directory.
        :param submit_script: Submission script relative to that directory.
        """

        self.transport.chdir(working_directory)
        with self.transport:
            result = self.transport.exec_command_wait(
                self._get_submit_command(submit_script)
            )

        return self._parse_submit_output(*result)

    def get_jobs(
        self,
        jobs: list[str] | None = None,
        user: str | None = None,
        as_dict: bool = False,
    ) -> list[JobInfo] | dict[str, JobInfo]:
        """Return currently active Flux jobs.

        :param jobs: Optional list of job IDs to query.
        :param user: Optional username used to filter jobs.
        :param as_dict: Return a mapping keyed by job ID when true.
        """

        with self.transport:
            result = self.transport.exec_command_wait(
                self._get_joblist_command(jobs=jobs, user=user)
            )

        job_list = self._parse_joblist_output(*result)
        if as_dict:
            job_dict = {job.job_id: job for job in job_list}
            if None in job_dict:
                raise SchedulerError('Found at least one job without jobid')
            return job_dict

        return job_list

    def kill_job(self, jobid: str) -> bool:
        """Cancel a Flux job and return whether cancellation succeeded."""

        with self.transport:
            result = self.transport.exec_command_wait(
                self._get_kill_command(jobid)
            )

        return self._parse_kill_output(*result)
        
    def _get_submit_script_header(
        self, 
        job_tmpl: JobTemplate
    ) -> str:
        """
        Return the submit script header with the parameters from the job_tmpl.

        :param job_tmpl: JobTemplate instance with relevant parameters set.
        :return header: Job submission header as a string.
        """

        header = []

        if job_tmpl.job_name:

            # Remove unwanted symbols leaving only letters, numbers, dots, 
            # and dashes.
            job_name = re.sub(r'[^a-zA-Z0-9_.-]+', '', job_tmpl.job_name)

            # prepend a 'j' (for 'job') before the string if the string
            # is now empty or does not start with a valid charachter
            if not job_name or (job_name[0] not in string.ascii_letters + string.digits):
                job_name = f'j{job_name}'

            # Truncate to the first 128 characters
            # Nothing is done if the string is shorter.
            job_name = job_name[:128]

            header.append(f'#flux: --job-name={job_name}')

        if job_tmpl.sched_output_path:
            header.append(f'#flux: --output={job_tmpl.sched_output_path}')

        if job_tmpl.sched_error_path:
            header.append(f'#flux: --error={job_tmpl.sched_error_path}')

        if job_tmpl.queue_name:
            header.append(f'#flux: -q {job_tmpl.queue_name}')

        if job_tmpl.account:
            header.append(f'#flux: -B {job_tmpl.account}')
        
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
            header.append(f'#flux: --urgency={priority}')

        if not job_tmpl.job_resource:
            raise ValueError(
                'Job resources (number of nodes) are required for the Flux '
                'scheduler plugin.'
            )

        header.append(f'#flux: -N {job_tmpl.job_resource.num_machines}')
        if job_tmpl.job_resource.num_mpiprocs_per_machine:
            header.append(f'#flux: -n '
                f'{job_tmpl.job_resource.num_mpiprocs_per_machine * job_tmpl.job_resource.num_machines}'
            )
        
        if job_tmpl.job_resource.num_cores_per_mpiproc:
            header.append(
                f'#flux: -c {job_tmpl.job_resource.num_cores_per_mpiproc}'
            )

        if job_tmpl.max_wallclock_seconds is not None:
            try:
                tot_secs = int(job_tmpl.max_wallclock_seconds)
                if tot_secs <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    'max_wallclock_seconds must be ' "a positive integer (in seconds)! It is instead '{}'" ''.format(
                        (job_tmpl.max_wallclock_seconds)
                    )
                )

            # Check if total time is larger than day, hour, or minutes 
            # and convert to float.

            # Days
            if tot_secs > 86400:
                time = f'{tot_secs / 86400:.2f}d'
            # Hours
            elif tot_secs > 3600:
                time = f'{tot_secs / 3600:.2f}h'
            elif tot_secs > 60:
                time = f'{tot_secs / 60:.2f}m'
            else:
                time = tot_secs

            header.append(f'#flux: -t {time}')
        
        if job_tmpl.custom_scheduler_commands:
            header.append(job_tmpl.custom_scheduler_commands)

        header = '\n'.join(header)         

        return header
    
    def _get_submit_command(
        self, 
        submit_script: str
    ) -> str:
        """
        Return the string to execute the submission script.

        :param submit_script: Path to the submission script relative to the 
            working directory.
        :return submit_command: Command used to submit the submission script.
        """

        submit_command = f"flux batch {submit_script}"

        self.logger.info(f'submitting with : {submit_command}')

        return submit_command
    
    def _parse_submit_output(
        self, 
        retval: int, 
        stdout: str, 
        stderr: str
    ) -> str | ExitCode:
        """
        Parse the output from the submission command.

        :param retval: Return value of the submission command.
        :param stdout: Standard output.
        :param stderr: Error from standard output.
        :return job_id: Job ID from the submitted job.
        """

        if retval != 0:
            self.logger.error(f'Error in _parse_submit_output: {retval=}; {stdout=}; {stderr=}')

            raise SchedulerError(
                f'Error during submission, {retval=}, {stdout=}, {stderr=}'
            )

        try:
            transport_string = f' for {self.transport}'
        except SchedulerError:
            transport_string = ''

        if stderr.strip():
            self.logger.warning(f'in _parse_submit_output{transport_string}: there was some text in stderr: {stderr}')

        stdout = stdout.strip('\n')
        if stdout:
            return stdout
        self.logger.error(f' in _parse_submit_output{transport_string}: unable to find the job id: {stdout}')
        raise SchedulerError(
            'Error during submission, cound not retrieve the jobID from flux output; see log for more info.'
        )
    
    def _parse_joblist_output(
        self, 
        retval: int, 
        stdout: str, 
        stderr: str
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

        stdout_split = stdout.splitlines()[1:] # Remove the first line which has the header names.
        jobdata_raw = [line.split(self._FIELD_SEPARATOR,maxsplit=num_fields) for line in stdout_split]

        job_list = []

        for job in jobdata_raw:
            thisjob_dict = {k[1]: v for k, v in zip(self.fields, job)}

            this_job = JobInfo()
            try:
                this_job.job_id = thisjob_dict['job_id']
                this_job.annotation = thisjob_dict['annotation']
                job_state_raw = thisjob_dict['state_raw']
            except KeyError:
                self.logger.error(f"Wrong line length in flux output! '{job}'")
            
            try:
                job_state_string = _MAP_STATUS_FLUX[job_state_raw]
            except KeyError:
                self.logger.warning(F"Unrecognized job_state '{job_state_raw}' for job id {this_job.job_id}")
                job_state_string = JobState.UNDETERMINED
            this_job.job_state = job_state_string

            if len(job) < num_fields:
                self.logger.warning(f'Wrong line length in flux output! Skipping optional fields. Line: `{jobdata_raw}`')
                job_list.append(this_job)
                continue

            this_job.job_owner = thisjob_dict['username']

            try:
                this_job.num_machines = int(thisjob_dict['number_nodes'])
            except ValueError:
                self.logger.warning(
                    f"The number of allocated nodes is not an integer "
                    f"({thisjob_dict['number_nodes']}) for job id "
                    f"{this_job.job_id}!"
                )

            try:
                this_job.num_mpiprocs = int(thisjob_dict['number_cpus'])
            except ValueError:
                self.logger.warning(
                    f"The number of allocated cores is not an integer "
                    f"({thisjob_dict['number_cpus']}) for job id "
                    f"{this_job.job_id}!"
                )

            if this_job.job_state == JobState.RUNNING:
                this_job.allocated_machines_raw = thisjob_dict['allocated_machines']

            this_job.queue_name = thisjob_dict['partition']

            try:
                walltime = thisjob_dict['time_limit']
                this_job.requested_wallclock_time_seconds = walltime
            except ValueError:
                self.logger.warning(f'Error parsing the time limit for job id {this_job.job_id}')

            if this_job.job_state == JobState.RUNNING:
                try:
                    this_job.wallclock_time_seconds = thisjob_dict['time_used']
                except ValueError:
                    self.logger.warning(f'Error parsing time_used for job id {this_job.job_id}')

                try:
                    dispatch_time = float(thisjob_dict['dispatch_time'])
                    dispatch_time = datetime.datetime.fromtimestamp(dispatch_time)
                    this_job.dispatch_time = dispatch_time
                except ValueError:
                    self.logger.warning(f'Error parsing dispatch_time for job id {this_job.job_id}')

            try:
                submission_time = float(thisjob_dict['submission_time'])
                submission_time = datetime.datetime.fromtimestamp(submission_time)
                this_job.submission_time = submission_time
            except ValueError:
                self.logger.warning(f'Error parsing submission_time for job id {this_job.job_id}')

            job_list.append(this_job)

        return job_list
    
    def _get_kill_command(
        self, 
        jobid: str
    ) -> str:
        """
        Return the command to kill the job with the specified jobid.

        :param jobid: Job ID of the job within Flux.
        :return comm: Command to kill a job within Flux.
        """

        return f"flux cancel {jobid}"
    
    def _parse_kill_output(
        self, 
        retval: int, 
        stdout: str, 
        stderr: str
    ) -> bool:
        """
        Parse the output returned from the kill command.

        :param retval: Return value from the kill command.
        :param stdout: Standard output from the kill command.
        :param stderr: Standard error from the kill command.
        :return: True if everything is ok, False otherwise.
        """

        if retval != 0:
            self.logger.error(
                f'Error in _parse_kill_output: retval={retval}; '
                f'stdout={stdout}; stderr={stderr}'
            )
            return False

        try:
            transport_string = f' for {self.transport}'
        except SchedulerError:
            transport_string = ''

        if stderr.strip():
            self.logger.warning(
                f'in _parse_kill_output{transport_string}: there was some '
                f'text in stderr: {stderr}'
            )

        if stdout.strip():
            self.logger.warning(
                f'in _parse_kill_output{transport_string}: there was some '
                f'text in stdout: {stdout}'
            )

        return True
    
    def parse_output(
        self, 
        detailed_job_info: dict[str, str | int] | None = None, 
        stdout: str | None = None, 
        stderr: str | None = None
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
                detailed_stdout = json.loads(detailed_job_info['stdout'])
                print(f'{detailed_stdout=}')
            except KeyError:
                raise ValueError(
                    'the `detailed_job_info` does not contain the '
                    'required key `stdout`.'
                )

            # The format of the detailed job info should be a dictionary.
            type_check(detailed_stdout, dict) 

            #data = dict(zip(fields, attributes))

            #if data['State'] == 'OUT_OF_MEMORY':
            #    return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_MEMORY

            #if data['State'] == 'TO':
            #    return CalcJob.exit_codes.ERROR_SCHEDULER_OUT_OF_WALLTIME

            #if data['State'] == 'F':
            #    return CalcJob.exit_codes.ERROR_SCHEDULER_NODE_FAILURE

        return None
