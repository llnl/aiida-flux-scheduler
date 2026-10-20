import json

from click.testing import CliRunner

from aiida_flux_scheduler import cli as cli_module
from aiida_flux_scheduler.pools import (
    get_current_user,
    get_pool_group,
    get_pool_payload,
)


def test_pool_cli_roundtrip(aiida_profile_clean, aiida_computer_local, monkeypatch):
    monkeypatch.setattr(cli_module, '_load_profile', lambda: None)

    computer = aiida_computer_local(label='flux-localhost', configure=True)
    runner = CliRunner()

    result = runner.invoke(
        cli_module.cli,
        [
            'pool',
            'create',
            'production',
            '--computer',
            computer.label,
            '--num-machines',
            '2',
            '--num-mpiprocs-per-machine',
            '4',
            '--num-cores-per-mpiproc',
            '8',
            '--num-gpus-per-mpiproc',
            '1',
            '--max-wallclock-seconds',
            '600',
            '--queue-name',
            'debug',
            '--account',
            'project-a',
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli_module.cli,
        ['pool', 'show', 'production', '--computer', computer.label],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload['resources']['num_machines'] == 2
    assert payload['resources']['num_mpiprocs_per_machine'] == 4
    assert payload['resources']['num_cores_per_mpiproc'] == 8
    assert payload['resources']['num_gpus_per_mpiproc'] == 1
    assert payload['resources']['queue_name'] == 'debug'

    result = runner.invoke(
        cli_module.cli,
        ['pool', 'disable', 'production', '--computer', computer.label],
    )
    assert result.exit_code == 0

    user = get_current_user()
    group = get_pool_group(user.email, computer.label, 'production')
    assert get_pool_payload(group)['enabled'] is False

    result = runner.invoke(
        cli_module.cli,
        ['pool', 'enable', 'production', '--computer', computer.label],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        cli_module.cli,
        ['pool', 'list', '--computer', computer.label],
    )
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert rows == [
        {
            'computer': computer.label,
            'enabled': True,
            'max_wallclock_seconds': 600,
            'name': 'production',
            'num_machines': 2,
            'num_mpiprocs_per_machine': 4,
            'num_cores_per_mpiproc': 8,
            'num_gpus_per_mpiproc': 1,
        }
    ]

    result = runner.invoke(
        cli_module.cli,
        ['pool', 'delete', 'production', '--computer', computer.label],
    )
    assert result.exit_code == 0
