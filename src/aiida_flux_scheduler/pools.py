"""
Helpers for Flux pool definitions stored in the AiiDA database.
"""

from datetime import datetime, timezone
from typing import Any

from aiida.common import exceptions
from aiida.orm import Computer, Group, User

POOL_GROUP_LABEL_PREFIX = 'aiida_flux_scheduler:pool'
POOL_EXTRA_KEY = 'aiida_flux_scheduler'
POOL_RUNTIME_KEY = 'runtime'


def validate_pool_resources(resources: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the resource envelope for a persistent pool."""

    normalized = dict(resources)
    required = (
        'num_machines',
        'num_mpiprocs_per_machine',
        'max_wallclock_seconds',
    )
    optional_integers = ('num_cores_per_mpiproc', 'num_gpus_per_mpiproc')

    for key in required + optional_integers:
        if key not in normalized:
            if key in required:
                raise exceptions.ValidationError(f'Pool resource `{key}` is required.')
            continue
        try:
            normalized[key] = int(normalized[key])
        except (TypeError, ValueError) as exception:
            raise exceptions.ValidationError(
                f'Pool resource `{key}` must be an integer.'
            ) from exception
        if normalized[key] < 1:
            raise exceptions.ValidationError(
                f'Pool resource `{key}` must be greater than or equal to one.'
            )

    return normalized


def build_pool_group_label(
    user_email: str,
    computer_label: str,
    pool_name: str,
) -> str:
    """
    Build the deterministic group label for a pool definition.
    """

    return f'{POOL_GROUP_LABEL_PREFIX}:{user_email}:{computer_label}:{pool_name}'


def get_current_user() -> User:
    """
    Return the current default AiiDA user.
    """

    user = User.collection.get_default()
    if user is None:
        raise exceptions.NotExistent('No default AiiDA user is configured for the current profile.')

    return user


def get_pool_group(
    user_email: str,
    computer_label: str,
    pool_name: str,
) -> Group:
    """
    Return the stored group for a named pool.
    """

    return Group.collection.get(
        label=build_pool_group_label(user_email, computer_label, pool_name)
    )


def list_pool_groups(
    user_email: str,
    computer_label: str | None = None,
) -> list[Group]:
    """
    List stored pool groups for a user and optional computer.
    """

    label_prefix = f'{POOL_GROUP_LABEL_PREFIX}:{user_email}:'
    if computer_label is not None:
        label_prefix = f'{label_prefix}{computer_label}:'

    groups = Group.collection.find(
        filters={
            'label': {
                'like': f'{label_prefix}%'
            }
        }
    )

    return sorted(groups, key=lambda group: group.label)


def create_or_update_pool_group(
    pool_name: str,
    computer_label: str,
    resources: dict[str, Any],
    timeout: str = '5m',
    enabled: bool = True,
) -> tuple[Group, bool]:
    """
    Create or update a pool definition group for the current user.
    """

    user = get_current_user()
    computer = Computer.collection.get(label=computer_label)
    label = build_pool_group_label(user.email, computer.label, pool_name)

    resources = validate_pool_resources(resources)
    payload = {
        'name': pool_name,
        'computer': computer.label,
        'user': user.email,
        'enabled': enabled,
        'timeout': timeout,
        'resources': resources,
    }

    created = False
    try:
        group = Group.collection.get(label=label)
    except exceptions.NotExistent:
        group = Group(
            label=label,
            user=user,
            description=f'Flux pool definition `{pool_name}` for computer `{computer.label}`.',
        ).store()
        created = True

    existing_payload = group.base.extras.all.get(POOL_EXTRA_KEY, {})
    configuration_unchanged = (
        isinstance(existing_payload, dict)
        and existing_payload.get('resources') == resources
        and existing_payload.get('timeout') == timeout
    )
    if configuration_unchanged and POOL_RUNTIME_KEY in existing_payload:
        payload[POOL_RUNTIME_KEY] = existing_payload[POOL_RUNTIME_KEY]

    group.base.extras.set(POOL_EXTRA_KEY, payload)

    return group, created


def set_pool_enabled_state(
    user_email: str,
    computer_label: str,
    pool_name: str,
    enabled: bool,
) -> Group:
    """
    Update the enabled state for an existing pool definition.
    """

    group = get_pool_group(user_email, computer_label, pool_name)
    payload = get_pool_payload(group)
    payload['enabled'] = enabled
    group.base.extras.set(POOL_EXTRA_KEY, payload)

    return group


def delete_pool_group(
    user_email: str,
    computer_label: str,
    pool_name: str,
    force: bool = False,
) -> str | None:
    """
    Delete a stored pool definition.

    If runtime state still points at an active allocation reference, require
    `force=True` so users do not accidentally orphan a live pool record.
    """

    group = get_pool_group(user_email, computer_label, pool_name)
    runtime = get_pool_runtime(group)
    current_flux_id = runtime.get('current_flux_id')

    if current_flux_id and not force:
        raise exceptions.ValidationError(
            f'Flux pool `{pool_name}` still references allocation `{current_flux_id}`. '
            'Use `force=True` to delete the definition anyway.'
        )

    Group.collection.delete(group.pk)

    return current_flux_id


def get_pool_payload(group: Group) -> dict[str, Any]:
    """
    Return the plugin payload stored on a pool group.
    """

    payload = group.base.extras.all.get(POOL_EXTRA_KEY)
    if not isinstance(payload, dict):
        raise exceptions.NotExistent(
            f'Group `{group.label}` is missing the `{POOL_EXTRA_KEY}` payload.'
        )

    return payload


def get_pool_runtime(group: Group) -> dict[str, Any]:
    """
    Return the runtime payload stored on a pool group.
    """

    payload = get_pool_payload(group)
    runtime = payload.get(POOL_RUNTIME_KEY, {})

    if runtime is None:
        return {}

    if not isinstance(runtime, dict):
        raise exceptions.ValidationError(
            f'Group `{group.label}` has a non-dictionary `{POOL_RUNTIME_KEY}` payload.'
        )

    return runtime


def update_pool_runtime(group: Group, **updates: Any) -> dict[str, Any]:
    """
    Merge updates into the runtime payload of a pool group.
    """

    payload = get_pool_payload(group)
    runtime = dict(get_pool_runtime(group))
    runtime.update(updates)
    runtime['updated_at'] = datetime.now(timezone.utc).isoformat()
    payload[POOL_RUNTIME_KEY] = runtime
    group.base.extras.set(POOL_EXTRA_KEY, payload)

    return runtime


def clear_pool_runtime(group: Group) -> None:
    """
    Clear the runtime payload of a pool group.
    """

    payload = get_pool_payload(group)
    payload[POOL_RUNTIME_KEY] = {}
    group.base.extras.set(POOL_EXTRA_KEY, payload)
