"""
CLI utilities for managing Flux pool definitions.
"""

import json
from typing import Any

import click
from aiida import load_profile
from aiida.common import exceptions

from aiida_flux_scheduler.pools import (
    create_or_update_pool_group,
    delete_pool_group,
    get_current_user,
    get_pool_group,
    get_pool_payload,
    list_pool_groups,
    set_pool_enabled_state,
)


def _load_profile() -> None:
    """
    Load the active AiiDA profile for ORM access.
    """

    load_profile()


def _get_pool_for_current_user(
    name: str,
    computer_label: str,
):
    """
    Load a pool group for the active user or raise a CLI-friendly error.
    """

    user = get_current_user()

    try:
        return get_pool_group(user.email, computer_label, name)
    except exceptions.NotExistent as exception:
        raise click.ClickException(
            f'No Flux pool named `{name}` exists for computer `{computer_label}`.'
        ) from exception


@click.group()
def cli() -> None:
    """Manage AiiDA Flux scheduler resources."""


@cli.group()
def pool() -> None:
    """Manage shared Flux pool definitions."""


@pool.command('create')
@click.argument('name')
@click.option('--computer', 'computer_label', required=True, help='Computer label.')
@click.option('--queue-name', default=None, help='Flux queue/partition name.')
@click.option('--account', default=None, help='Flux account/bank.')
@click.option('--num-machines', type=int, required=True, help='Number of nodes in the pool allocation.')
@click.option(
    '--num-mpiprocs-per-machine',
    type=int,
    required=True,
    help='MPI processes per machine in the pool allocation.',
)
@click.option(
    '--num-cores-per-mpiproc',
    type=click.IntRange(min=1),
    default=None,
    help='CPU cores reserved for each pool resource slot.',
)
@click.option(
    '--num-gpus-per-mpiproc',
    type=click.IntRange(min=1),
    default=None,
    help='GPUs reserved for each pool resource slot.',
)
@click.option(
    '--max-wallclock-seconds',
    type=int,
    required=True,
    help='Maximum wallclock time for the pool allocation in seconds.',
)
@click.option('--timeout', default='5m', show_default=True, help='Idle watcher timeout passed to Flux.')
@click.option('--disabled', is_flag=True, default=False, help='Create the pool in a disabled state.')
def pool_create(
    name: str,
    computer_label: str,
    queue_name: str | None,
    account: str | None,
    num_machines: int,
    num_mpiprocs_per_machine: int,
    num_cores_per_mpiproc: int | None,
    num_gpus_per_mpiproc: int | None,
    max_wallclock_seconds: int,
    timeout: str,
    disabled: bool,
) -> None:
    """Create or update a named Flux pool definition."""

    _load_profile()

    resources: dict[str, Any] = {
        'num_machines': num_machines,
        'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
        'max_wallclock_seconds': max_wallclock_seconds,
    }
    if queue_name:
        resources['queue_name'] = queue_name
    if account:
        resources['account'] = account
    if num_cores_per_mpiproc is not None:
        resources['num_cores_per_mpiproc'] = num_cores_per_mpiproc
    if num_gpus_per_mpiproc is not None:
        resources['num_gpus_per_mpiproc'] = num_gpus_per_mpiproc

    group, created = create_or_update_pool_group(
        pool_name=name,
        computer_label=computer_label,
        resources=resources,
        timeout=timeout,
        enabled=not disabled,
    )

    action = 'Created' if created else 'Updated'
    click.echo(f'{action} Flux pool `{name}` in group `{group.label}`.')


@pool.command('list')
@click.option('--computer', 'computer_label', default=None, help='Filter by computer label.')
def pool_list(computer_label: str | None) -> None:
    """List configured Flux pools for the current AiiDA user."""

    _load_profile()
    user = get_current_user()
    groups = list_pool_groups(user.email, computer_label=computer_label)

    if not groups:
        click.echo('No Flux pools configured.')
        return

    rows = []
    for group in groups:
        payload = get_pool_payload(group)
        resources = payload.get('resources', {})
        rows.append(
            {
                'name': payload.get('name', ''),
                'computer': payload.get('computer', ''),
                'enabled': payload.get('enabled', True),
                'num_machines': resources.get('num_machines', ''),
                'num_mpiprocs_per_machine': resources.get('num_mpiprocs_per_machine', ''),
                'num_cores_per_mpiproc': resources.get('num_cores_per_mpiproc', ''),
                'num_gpus_per_mpiproc': resources.get('num_gpus_per_mpiproc', ''),
                'max_wallclock_seconds': resources.get('max_wallclock_seconds', ''),
            }
        )

    click.echo(json.dumps(rows, indent=2, sort_keys=True))


@pool.command('show')
@click.argument('name')
@click.option('--computer', 'computer_label', required=True, help='Computer label.')
def pool_show(name: str, computer_label: str) -> None:
    """Show a configured Flux pool definition."""

    _load_profile()
    group = _get_pool_for_current_user(name, computer_label)
    click.echo(json.dumps(get_pool_payload(group), indent=2, sort_keys=True))


@pool.command('enable')
@click.argument('name')
@click.option('--computer', 'computer_label', required=True, help='Computer label.')
def pool_enable(name: str, computer_label: str) -> None:
    """Enable a configured Flux pool definition."""

    _load_profile()
    user = get_current_user()
    group = _get_pool_for_current_user(name, computer_label)
    set_pool_enabled_state(user.email, computer_label, name, enabled=True)
    click.echo(f'Enabled Flux pool `{name}` in group `{group.label}`.')


@pool.command('disable')
@click.argument('name')
@click.option('--computer', 'computer_label', required=True, help='Computer label.')
def pool_disable(name: str, computer_label: str) -> None:
    """Disable a configured Flux pool definition."""

    _load_profile()
    user = get_current_user()
    group = _get_pool_for_current_user(name, computer_label)
    set_pool_enabled_state(user.email, computer_label, name, enabled=False)
    click.echo(f'Disabled Flux pool `{name}` in group `{group.label}`.')


@pool.command('delete')
@click.argument('name')
@click.option('--computer', 'computer_label', required=True, help='Computer label.')
@click.option(
    '--force',
    is_flag=True,
    default=False,
    help='Delete the pool even if its runtime state still references an allocation.',
)
def pool_delete(name: str, computer_label: str, force: bool) -> None:
    """Delete a configured Flux pool definition."""

    _load_profile()
    user = get_current_user()
    group = _get_pool_for_current_user(name, computer_label)
    group_label = group.label

    try:
        current_flux_id = delete_pool_group(
            user.email,
            computer_label,
            name,
            force=force,
        )
    except exceptions.ValidationError as exception:
        raise click.ClickException(str(exception)) from exception

    message = f'Deleted Flux pool `{name}` from group `{group_label}`.'
    if current_flux_id:
        message = (
            f'{message} Removed runtime reference to allocation '
            f'`{current_flux_id}`.'
        )
    click.echo(message)
