import pytest
from aiida.common import exceptions

from aiida_flux_scheduler.pools import (
    POOL_RUNTIME_KEY,
    create_or_update_pool_group,
    delete_pool_group,
    get_current_user,
    get_pool_payload,
    update_pool_runtime,
)


def test_create_or_update_pool_group_clears_runtime_when_resources_change(
    aiida_profile_clean,
    aiida_computer_local,
):
    computer = aiida_computer_local(label='flux-pools', configure=True)

    group, created = create_or_update_pool_group(
        pool_name='production',
        computer_label=computer.label,
        resources={
            'num_machines': 2,
            'num_mpiprocs_per_machine': 4,
            'max_wallclock_seconds': 600,
        },
    )
    assert created is True

    runtime = update_pool_runtime(group, current_flux_id='f123')
    assert runtime['current_flux_id'] == 'f123'

    group, created = create_or_update_pool_group(
        pool_name='production',
        computer_label=computer.label,
        resources={
            'num_machines': 4,
            'num_mpiprocs_per_machine': 8,
            'max_wallclock_seconds': 1200,
        },
        timeout='10m',
    )
    assert created is False

    payload = get_pool_payload(group)
    assert payload['resources']['num_machines'] == 4
    assert payload['timeout'] == '10m'
    assert POOL_RUNTIME_KEY not in payload


def test_create_or_update_pool_group_preserves_runtime_when_unchanged(
    aiida_profile_clean,
    aiida_computer_local,
):
    computer = aiida_computer_local(label='flux-unchanged', configure=True)
    resources = {
        'num_machines': 2,
        'num_mpiprocs_per_machine': 4,
        'num_gpus_per_mpiproc': 1,
        'max_wallclock_seconds': 600,
    }
    group, _ = create_or_update_pool_group(
        pool_name='production',
        computer_label=computer.label,
        resources=resources,
    )
    update_pool_runtime(group, current_flux_id='f123')

    group, created = create_or_update_pool_group(
        pool_name='production',
        computer_label=computer.label,
        resources=resources,
    )

    assert created is False
    assert get_pool_payload(group)[POOL_RUNTIME_KEY]['current_flux_id'] == 'f123'


def test_delete_pool_group_requires_force_when_runtime_is_present(
    aiida_profile_clean,
    aiida_computer_local,
):
    computer = aiida_computer_local(label='flux-delete', configure=True)
    user = get_current_user()

    group, _ = create_or_update_pool_group(
        pool_name='production',
        computer_label=computer.label,
        resources={
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1,
            'max_wallclock_seconds': 60,
        },
    )
    update_pool_runtime(group, current_flux_id='f999')

    with pytest.raises(exceptions.ValidationError):
        delete_pool_group(user.email, computer.label, 'production')

    current_flux_id = delete_pool_group(
        user.email,
        computer.label,
        'production',
        force=True,
    )
    assert current_flux_id == 'f999'
