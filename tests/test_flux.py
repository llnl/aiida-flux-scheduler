from types import SimpleNamespace

import pytest
from aiida.schedulers.datastructures import JobInfo, JobState, JobTemplate

from aiida_flux_scheduler.flux import FluxJobResource, FluxScheduler
from aiida_flux_scheduler.pools import (
    create_or_update_pool_group,
    get_pool_runtime,
)


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def exec_command_wait(self, command):
        self.commands.append(command)
        return self.responses.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_submit_job_uses_shell_cd_for_async_transport(monkeypatch):
    """Submit from the work directory without requiring transport.chdir."""

    scheduler = FluxScheduler()
    transport = FakeTransport(
        [
            (0, '#flux: --job-name=aiida-42\n#flux: -t 5m\n', ''),
            (0, 'child-456\n', ''),
        ]
    )
    scheduler._transport = transport
    monkeypatch.setattr(
        scheduler, '_flux_allocation_with_ownership', lambda _: ('alloc-123', False)
    )

    assert scheduler.submit_job('/remote/work directory', '_aiidasubmit.sh') == 'alloc-123:child-456'
    assert transport.commands == [
        "cd '/remote/work directory' && grep \"#flux\" '_aiidasubmit.sh'",
        "cd '/remote/work directory' && flux proxy alloc-123 flux batch '_aiidasubmit.sh'"
    ]


def test_submit_job_cancels_new_direct_allocation_when_child_submit_fails(monkeypatch):
    """Do not leave a direct allocation running after its first child fails."""

    scheduler = FluxScheduler()
    scheduler._transport = FakeTransport(
        [(0, '#flux: --job-name=aiida-42\n#flux: -t 5m\n', ''), (1, '', 'batch failed')]
    )
    monkeypatch.setattr(
        scheduler, '_flux_allocation_with_ownership', lambda _: ('alloc-123', True)
    )
    cancelled = []
    monkeypatch.setattr(scheduler, '_cancel_job_if_exists', cancelled.append)

    with pytest.raises(Exception, match='Error during submission'):
        scheduler.submit_job('/remote/work', '_aiidasubmit.sh')

    assert cancelled == ['alloc-123']


def test_submit_job_does_not_cancel_reused_or_pooled_allocation_on_failure(monkeypatch):
    scheduler = FluxScheduler()
    scheduler._transport = FakeTransport(
        [(0, '#flux: --job-name=aiida-42\n#flux: -t 5m\n', ''), (1, '', 'batch failed')]
    )
    monkeypatch.setattr(
        scheduler, '_flux_allocation_with_ownership', lambda _: ('alloc-123', False)
    )
    monkeypatch.setattr(
        scheduler,
        '_cancel_job_if_exists',
        lambda job_id: pytest.fail(f'unexpected cancellation of {job_id}'),
    )

    with pytest.raises(Exception, match='Error during submission'):
        scheduler.submit_job('/remote/work', '_aiidasubmit.sh')


def test_submit_job_does_not_cancel_on_ambiguous_child_submission(monkeypatch):
    """Keep a new allocation when Flux may have accepted an unreported child."""

    scheduler = FluxScheduler()
    scheduler._transport = FakeTransport(
        [(0, '#flux: --job-name=aiida-42\n#flux: -t 5m\n', ''), (0, '', '')]
    )
    monkeypatch.setattr(
        scheduler, '_flux_allocation_with_ownership', lambda _: ('alloc-123', True)
    )
    monkeypatch.setattr(
        scheduler,
        '_cancel_job_if_exists',
        lambda job_id: pytest.fail(f'unexpected cancellation of {job_id}'),
    )

    with pytest.raises(Exception, match='could not retrieve'):
        scheduler.submit_job('/remote/work', '_aiidasubmit.sh')


def test_validate_resources_accepts_flux_pool():
    resources = FluxJobResource.validate_resources(
        num_machines=2,
        num_mpiprocs_per_machine=4,
        max_wallclock_seconds='60',
        flux_pool='shared',
    )

    assert resources['num_machines'] == 2
    assert resources['num_mpiprocs_per_machine'] == 4
    assert resources['max_wallclock_seconds'] == 60
    assert resources['flux_pool'] == 'shared'
    assert resources['flux_allocation_lifecycle'] is None


def test_validate_resources_accepts_child_owned_lifecycle():
    resources = FluxJobResource.validate_resources(
        num_machines=1,
        num_mpiprocs_per_machine=1,
        flux_allocation_lifecycle='child',
    )

    assert resources['flux_allocation_lifecycle'] == 'child'


def test_validate_resources_accepts_explicit_parent_lifecycle():
    """Allow root-workchain allocation reuse only when explicitly requested."""

    resources = FluxJobResource.validate_resources(
        num_machines=1,
        num_mpiprocs_per_machine=1,
        flux_allocation_lifecycle='parent',
    )

    assert resources['flux_allocation_lifecycle'] == 'parent'


def test_validate_resources_rejects_unknown_allocation_lifecycle():
    with pytest.raises(ValueError, match='flux_allocation_lifecycle'):
        FluxJobResource.validate_resources(
            num_machines=1,
            num_mpiprocs_per_machine=1,
            flux_allocation_lifecycle='pool',
        )


def test_validate_resources_accepts_gpu_count():
    resources = FluxJobResource.validate_resources(
        num_machines=2,
        num_mpiprocs_per_machine=4,
        num_gpus_per_mpiproc='1',
    )

    assert resources['num_gpus_per_mpiproc'] == 1


@pytest.mark.parametrize('value', [0, -1])
def test_validate_resources_rejects_non_positive_gpu_count(value):
    with pytest.raises(ValueError, match='num_gpus_per_mpiproc'):
        FluxJobResource.validate_resources(
            num_machines=1,
            num_mpiprocs_per_machine=1,
            num_gpus_per_mpiproc=value,
        )


def test_submit_header_requests_gpus_per_flux_slot():
    scheduler = FluxScheduler()
    template = JobTemplate()
    template.job_name = 'aiida-42'
    template.job_resource = FluxJobResource(
        num_machines=2,
        num_mpiprocs_per_machine=4,
        num_cores_per_mpiproc=8,
        num_gpus_per_mpiproc=1,
    )
    template.max_wallclock_seconds = 600

    header = scheduler._get_submit_script_header(template)

    assert '#flux: -N 2' in header
    assert '#flux: -n 8' in header
    assert '#flux: -c 8' in header
    assert '#flux: -g 1' in header


def test_submit_header_keeps_cpu_only_jobs_gpu_free():
    scheduler = FluxScheduler()
    template = JobTemplate()
    template.job_name = 'aiida-43'
    template.job_resource = FluxJobResource(
        num_machines=1,
        num_mpiprocs_per_machine=8,
        num_cores_per_mpiproc=2,
    )

    header = scheduler._get_submit_script_header(template)

    assert '#flux: -N 1' in header
    assert '#flux: -n 8' in header
    assert '#flux: -c 2' in header
    assert '#flux: -g' not in header


def test_parse_flux_directives_ignores_custom_flags():
    values = FluxScheduler._parse_flux_directives(
        '#flux: --job-name=aiida-42\n'
        '#flux: -t 10m\n'
        '#flux: --exclusive\n'
        '#flux: --setattr=user.note="GPU test"\n'
    )

    assert values == {'job-name': 'aiida-42', 't': '10m'}


def test_parse_seconds_field_handles_float_and_blank():
    scheduler = FluxScheduler()

    assert scheduler._parse_seconds_field('10.9', 'time_used', '123') == 10
    assert scheduler._parse_seconds_field('', 'time_used', '123') is None
    assert scheduler._parse_seconds_field(None, 'time_used', '123') is None


def test_parse_joblist_output_parses_flux_time_fields():
    scheduler = FluxScheduler()
    stdout = (
        'HEADER\n'
        '123|R||user|2|64|node[1-2]|debug|120.0|30.0|1710000000.0|aiida-42|1709990000.0\n'
    )

    jobs = scheduler._parse_joblist_output(0, stdout, '')

    assert len(jobs) == 1
    job = jobs[0]
    assert job.job_id == '123'
    assert job.job_state == JobState.RUNNING
    assert job.title == 'aiida-42'
    assert job.requested_wallclock_time_seconds == 120
    assert job.wallclock_time_seconds == 30
    assert job.queue_name == 'debug'


def test_start_allocation_builds_expected_flux_alloc_command():
    transport = FakeTransport([(0, 'flux-123\n', '')])
    scheduler = FluxScheduler()
    scheduler._transport = transport

    flux_id = scheduler._start_allocation(
        {
            'resources': {
                'num_machines': 2,
                'num_mpi_procs_per_machine': 4,
                'queue_name': 'debug',
                'max_wallclock_seconds': 600,
                'account': 'project-a',
                'num_cores_per_mpiproc': 8,
                'num_gpus_per_mpiproc': 1,
            },
            'job_name': 'aiida-pool-shared',
            'timeout': '10m',
        },
        parent_pk=42,
    )

    assert flux_id == 'flux-123'
    assert transport.commands == [
        "flux alloc --job-name=aiida-pool-shared --nodes=2 -n 8 -c 8 -g 1 -q debug "
        "-B project-a -t 600s -x --bg bash -c 'while true; do sleep 60; if "
        "[ $(flux jobs --since=-10m | wc -l) -gt 1 ]; then continue; else "
        "exit; fi; done'"
    ]


def test_start_allocation_keeps_cpu_only_pool_gpu_free():
    transport = FakeTransport([(0, 'flux-123\n', '')])
    scheduler = FluxScheduler()
    scheduler._transport = transport

    scheduler._start_allocation(
        {
            'resources': {
                'num_machines': 2,
                'num_mpiprocs_per_machine': 4,
                'max_wallclock_seconds': 600,
            },
            'job_name': 'aiida-pool-cpu',
        },
        parent_pk=42,
    )

    assert ' -g ' not in transport.commands[0]
    assert '--nodes=2 -n 8' in transport.commands[0]


def test_start_allocation_shell_quotes_queue_and_account():
    transport = FakeTransport([(0, 'flux-123\n', '')])
    scheduler = FluxScheduler()
    scheduler._transport = transport

    scheduler._start_allocation(
        {
            'resources': {
                'queue_name': 'debug; unexpected-command',
                'account': 'project name',
            },
        },
        parent_pk=42,
    )

    assert "-q 'debug; unexpected-command'" in transport.commands[0]
    assert "-B 'project name'" in transport.commands[0]


def test_kill_job_cancels_pooled_child_through_owning_allocation():
    transport = FakeTransport([(0, '', '')])
    scheduler = FluxScheduler()
    scheduler._transport = transport

    assert scheduler.kill_job('allocation-123:child-456') is True
    assert transport.commands == ['flux proxy allocation-123 flux cancel child-456']


def test_kill_command_quotes_job_ids():
    scheduler = FluxScheduler()

    assert scheduler._get_kill_command('top; level') == "flux cancel 'top; level'"
    assert scheduler._get_kill_command('outer id:child; id') == (
        "flux proxy 'outer id' flux cancel 'child; id'"
    )


@pytest.mark.parametrize('job_id', ['allocation-123:', ':child-456', 'outer:child:extra'])
def test_kill_command_rejects_malformed_pooled_job_id(job_id):
    scheduler = FluxScheduler()

    with pytest.raises(ValueError, match='Invalid pooled Flux job ID'):
        scheduler._get_kill_command(job_id)


def test_detailed_job_info_command_handles_pooled_and_direct_jobs():
    scheduler = FluxScheduler()

    assert scheduler._get_detailed_job_info_command('allocation-123:child-456') == (
        'flux proxy allocation-123 flux job info child-456 jobspec'
    )
    assert scheduler._get_detailed_job_info_command('top-level-123') == (
        'flux job info top-level-123 jobspec'
    )


@pytest.mark.parametrize('job_id', ['allocation-123:', ':child-456', 'outer:child:extra'])
def test_detailed_job_info_command_rejects_malformed_pooled_job_id(job_id):
    scheduler = FluxScheduler()

    with pytest.raises(ValueError, match='Invalid pooled Flux job ID'):
        scheduler._get_detailed_job_info_command(job_id)


def test_check_allocation_matches_exact_title_only():
    scheduler = FluxScheduler()

    fuzzy = JobInfo()
    fuzzy.job_id = '111'
    fuzzy.title = 'prefix-aiida-42-suffix'
    fuzzy.requested_wallclock_time_seconds = 100
    fuzzy.wallclock_time_seconds = 10

    exact = JobInfo()
    exact.job_id = '222'
    exact.title = 'aiida-42'
    exact.requested_wallclock_time_seconds = 90
    exact.wallclock_time_seconds = 15

    scheduler.get_jobs = lambda: [fuzzy, exact]

    state = scheduler._check_allocation(42)

    assert state.active is True
    assert state.flux_id == '222'
    assert state.walltime == 75


def test_direct_allocation_reuse_rejects_incompatible_resources():
    scheduler = FluxScheduler()
    job = JobInfo()
    job.num_machines = 1
    job.num_mpiprocs = 8
    job.queue_name = 'debug'

    assert scheduler._direct_allocation_is_compatible(
        job,
        {'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4, 'num_cores_per_mpiproc': 2, 'queue_name': 'debug'}},
    )
    assert not scheduler._direct_allocation_is_compatible(
        job,
        {'resources': {'num_machines': 2, 'num_mpiprocs_per_machine': 4, 'num_cores_per_mpiproc': 2}},
    )
    assert not scheduler._direct_allocation_is_compatible(
        job,
        {'resources': {'num_machines': 1, 'num_mpiprocs_per_machine': 4, 'num_cores_per_mpiproc': 2, 'queue_name': 'other'}},
    )
    assert not scheduler._direct_allocation_is_compatible(
        job,
        {'resources': {'num_machines': 1, 'num_gpus_per_mpiproc': 1}},
    )
    assert not scheduler._direct_allocation_is_compatible(
        job,
        {'resources': {'num_machines': 1, 'account': 'project-a'}},
    )


def test_explicit_child_allocation_uses_child_ownership(monkeypatch):
    """Create a calculation-owned allocation only when explicitly selected."""

    scheduler = FluxScheduler()
    node = SimpleNamespace()
    monkeypatch.setattr('aiida_flux_scheduler.flux.load_node', lambda _: node)
    monkeypatch.setattr('aiida_flux_scheduler.flux.CalcJobNode', SimpleNamespace)
    monkeypatch.setattr(scheduler, '_get_parent_pk', lambda _: 10)
    monkeypatch.setattr(
        scheduler,
        '_resolve_allocation_config',
        lambda _: {'pool': None, 'resources': {}, 'lifecycle': 'child'},
    )
    monkeypatch.setattr(scheduler, '_start_allocation', lambda _, pk: f'alloc-{pk}')
    monkeypatch.setattr(
        scheduler,
        '_check_allocation',
        lambda _: pytest.fail('default child ownership must not look up a parent allocation'),
    )

    assert scheduler._flux_allocation_with_ownership(42) == ('alloc-42', True)


def test_resolve_direct_allocation_merges_standard_scheduler_options():
    """Pass top-level AiiDA queue, account, and walltime to ``flux alloc``."""

    node = SimpleNamespace(
        get_metadata_inputs=lambda: {
            'metadata': {
                'options': {
                    'resources': {'num_machines': 1},
                    'queue_name': 'pdebug',
                    'account': 'qnoise',
                    'max_wallclock_seconds': 900,
                }
            }
        }
    )

    config = FluxScheduler()._resolve_allocation_config(node)

    assert config['pool'] is None
    assert config['lifecycle'] is None
    assert config['resources'] == {
        'num_machines': 1,
        'queue_name': 'pdebug',
        'account': 'qnoise',
        'max_wallclock_seconds': 900,
    }


@pytest.mark.parametrize('pool', [None, ''])
def test_submit_job_defaults_to_direct_batch(monkeypatch, pool):
    """Ordinary scheduler options must not start or proxy into an allocation."""
    scheduler = FluxScheduler()
    scheduler._transport = FakeTransport([
        (0, '#flux: --job-name=aiida-42\n#flux: -t 15m\n', ''),
        (0, 'direct-123\n', ''),
    ])
    node = SimpleNamespace(get_metadata_inputs=lambda: {
        'metadata': {'options': {
            'resources': {'num_machines': 1, 'flux_pool': pool},
            'queue_name': 'pdebug', 'account': 'qnoise',
            'max_wallclock_seconds': 900,
        }},
    })
    monkeypatch.setattr('aiida_flux_scheduler.flux.load_node', lambda _: node)
    monkeypatch.setattr('aiida_flux_scheduler.flux.CalcJobNode', SimpleNamespace)
    monkeypatch.setattr(scheduler, '_get_parent_pk', lambda _: pytest.fail('parent lookup'))
    monkeypatch.setattr(scheduler, '_start_allocation', lambda *args: pytest.fail('allocation'))
    assert scheduler.submit_job('/remote/work directory', '_aiidasubmit.sh') == 'direct-123'
    assert scheduler._transport.commands[-1] == (
        "cd '/remote/work directory' && flux batch -q pdebug -B qnoise '_aiidasubmit.sh'"
    )


@pytest.mark.parametrize('selector', [{}, {'flux_pool': 'shared'}, {'flux_allocation_lifecycle': 'child'}, {'flux_allocation_lifecycle': 'parent'}])
def test_submit_header_queue_and_bank_only_for_direct_jobs(selector):
    """Direct batch jobs request their queue and bank in the submit script."""
    template = JobTemplate()
    template.job_resource = FluxJobResource(num_machines=1, num_mpiprocs_per_machine=1, **selector)
    template.queue_name = 'pdebug'
    template.account = 'bank name'
    header = FluxScheduler()._get_submit_script_header(template)
    assert ('#flux: -q pdebug' in header) is (not selector)
    assert ("#flux: -B 'bank name'" in header) is (not selector)


@pytest.mark.parametrize('job_id', ['direct-123', 'alloc-123:child-456'])
def test_get_jobs_supports_direct_and_nested_ids(job_id):
    """Monitor both direct batch jobs and existing allocation-backed jobs."""
    scheduler = FluxScheduler()
    local_id = job_id.split(':')[-1]
    scheduler._transport = FakeTransport([(0, f'HEADER\n{local_id}|R||user|1|4|node|pdebug|900|1||test|\n', '')])
    jobs = scheduler.get_jobs(jobs=[job_id], as_dict=True)
    assert job_id in jobs
    command = scheduler._transport.commands[0]
    assert ('flux proxy' in command) is (':' in job_id)
    assert f'flux jobs {local_id}' in command


def test_get_jobs_handles_missing_direct_job():
    """Return an empty mapping when Flux no longer reports the requested job."""
    scheduler = FluxScheduler()
    scheduler._transport = FakeTransport([(0, '', '')])
    assert scheduler.get_jobs(jobs=['direct-123'], as_dict=True) == {}


def test_acquire_pool_lock_recovers_stale_lock(monkeypatch):
    transport = FakeTransport(
        [
            (0, '', ''),
            (1, '', 'exists'),
            (0, '', ''),
            (0, '', ''),
        ]
    )
    scheduler = FluxScheduler()
    scheduler._transport = transport

    written = []
    monkeypatch.setattr(scheduler, '_write_pool_lock_metadata', written.append)
    monkeypatch.setattr(scheduler, '_get_pool_lock_age_seconds', lambda pool_name: 600)
    monkeypatch.setattr('aiida_flux_scheduler.flux.time.sleep', lambda _: None)

    scheduler._acquire_pool_lock('shared', timeout_seconds=1, stale_lock_seconds=300)

    lock_path = scheduler._get_pool_lock_path('shared')
    metadata_path = scheduler._get_pool_lock_metadata_path('shared')
    assert transport.commands == [
        'mkdir -p "$HOME/.aiida-flux-scheduler/locks"',
        f'mkdir "{lock_path}"',
        f'rm -f "{metadata_path}" && rmdir "{lock_path}"',
        f'mkdir "{lock_path}"',
    ]
    assert written == ['shared']


def test_resolve_allocation_config_uses_pool_definition(
    aiida_profile_clean,
    aiida_computer_local,
):
    computer = aiida_computer_local(label='flux-config', configure=True)
    group, _ = create_or_update_pool_group(
        pool_name='shared',
        computer_label=computer.label,
        resources={
            'num_machines': 2,
            'num_mpiprocs_per_machine': 8,
            'max_wallclock_seconds': 600,
        },
        timeout='10m',
    )

    node = SimpleNamespace(
        user=group.user,
        computer=computer,
        get_metadata_inputs=lambda: {
            'metadata': {
                'options': {
                    'resources': {
                        'flux_pool': 'shared',
                    }
                }
            }
        },
    )

    config = FluxScheduler()._resolve_allocation_config(node)

    assert config['pool'] == 'shared'
    assert config['group'].pk == group.pk
    assert config['resources']['num_mpiprocs_per_machine'] == 8
    assert config['timeout'] == '10m'


def test_get_or_create_pooled_allocation_reuses_existing_runtime(
    aiida_profile_clean,
    aiida_computer_local,
    monkeypatch,
):
    computer = aiida_computer_local(label='flux-reuse', configure=True)
    group, _ = create_or_update_pool_group(
        pool_name='shared',
        computer_label=computer.label,
        resources={
            'num_machines': 2,
            'num_mpiprocs_per_machine': 8,
            'max_wallclock_seconds': 600,
        },
    )

    scheduler = FluxScheduler()
    scheduler.flux_values = {'t': 120}

    releases = []
    monkeypatch.setattr(scheduler, '_acquire_pool_lock', lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, '_release_pool_lock', releases.append)
    monkeypatch.setattr(scheduler, '_start_allocation', lambda *args, **kwargs: 'new-id')

    job = JobInfo()
    job.job_id = 'old-id'
    job.requested_wallclock_time_seconds = 500
    job.wallclock_time_seconds = 100

    monkeypatch.setattr(scheduler, '_get_top_level_job', lambda flux_id: job)

    runtime = get_pool_runtime(group)
    runtime['current_flux_id'] = 'old-id'
    group.base.extras.set(
        'aiida_flux_scheduler',
        {
            **group.base.extras.get('aiida_flux_scheduler'),
            'runtime': runtime,
        },
    )

    node = SimpleNamespace(user=group.user, computer=computer)
    flux_id = scheduler._get_or_create_pooled_allocation(
        node,
        parent_pk=42,
        allocation_config={'pool': 'shared'},
    )

    assert flux_id == 'old-id'
    runtime = get_pool_runtime(group)
    assert runtime['current_flux_id'] == 'old-id'
    assert runtime['last_used_by_parent_pk'] == 42
    assert runtime['remaining_seconds'] == 400
    assert releases == [scheduler._get_pool_lock_name(node, 'shared')]


def test_validate_job_fits_gpu_pool():
    scheduler = FluxScheduler()
    node = SimpleNamespace(
        get_metadata_inputs=lambda: {
            'metadata': {
                'options': {
                    'resources': {
                        'flux_pool': 'shared',
                        'num_machines': 1,
                        'num_mpiprocs_per_machine': 4,
                        'num_cores_per_mpiproc': 4,
                        'num_gpus_per_mpiproc': 1,
                    }
                }
            }
        }
    )
    pool = {
        'pool': 'shared',
        'resources': {
            'num_machines': 2,
            'num_mpiprocs_per_machine': 4,
            'num_cores_per_mpiproc': 8,
            'num_gpus_per_mpiproc': 1,
        },
    }

    scheduler._validate_job_fits_pool(node, pool)

    pool['resources']['num_gpus_per_mpiproc'] = 0
    with pytest.raises(Exception, match='GPUs per node=4'):
        scheduler._validate_job_fits_pool(node, pool)


def test_get_or_create_pooled_allocation_drains_expired_runtime(
    aiida_profile_clean,
    aiida_computer_local,
    monkeypatch,
):
    computer = aiida_computer_local(label='flux-replace', configure=True)
    group, _ = create_or_update_pool_group(
        pool_name='shared',
        computer_label=computer.label,
        resources={
            'num_machines': 2,
            'num_mpiprocs_per_machine': 8,
            'max_wallclock_seconds': 600,
        },
    )

    runtime = get_pool_runtime(group)
    runtime['current_flux_id'] = 'old-id'
    group.base.extras.set(
        'aiida_flux_scheduler',
        {
            **group.base.extras.get('aiida_flux_scheduler'),
            'runtime': runtime,
        },
    )

    scheduler = FluxScheduler()
    scheduler.flux_values = {'t': 120}

    monkeypatch.setattr(scheduler, '_acquire_pool_lock', lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, '_release_pool_lock', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        scheduler,
        '_cancel_job_if_exists',
        lambda job_id: pytest.fail(f'unexpected cancellation of shared allocation {job_id}'),
    )
    monkeypatch.setattr(scheduler, '_start_allocation', lambda *args, **kwargs: 'new-id')

    job = JobInfo()
    job.job_id = 'old-id'
    job.requested_wallclock_time_seconds = 150
    job.wallclock_time_seconds = 50

    monkeypatch.setattr(scheduler, '_get_top_level_job', lambda flux_id: job)

    node = SimpleNamespace(user=group.user, computer=computer)
    flux_id = scheduler._get_or_create_pooled_allocation(
        node,
        parent_pk=77,
        allocation_config={'pool': 'shared'},
    )

    assert flux_id == 'new-id'
    runtime = get_pool_runtime(group)
    assert runtime['current_flux_id'] == 'new-id'
    assert runtime['last_used_by_parent_pk'] == 77
    assert runtime['state'] == 'active'
