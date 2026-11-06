"""AiiDA-side approval for safe remote journal maintenance."""

from typing import Any

from aiida.common import exceptions
from aiida.orm import CalcJobNode, Computer, load_node

from aiida_flux_scheduler.pools import get_current_user


def cleanup_approval(row: dict[str, Any], computer: Computer) -> str | None:
    """Approve only sealed CalcJobs with matching computer, owner, and scheduler ID."""
    try:
        node = load_node(row["calculation_uuid"])
    except exceptions.NotExistent:
        return "CalcJob not found in the active profile"
    if not isinstance(node, CalcJobNode) or not node.is_sealed:
        return "CalcJob is not sealed"
    if (
        node.computer is None
        or node.computer.uuid != computer.uuid
        or node.user.pk != get_current_user().pk
    ):
        return "CalcJob computer or owner does not match"
    if (
        node.base.attributes.get("job_id", None)
        != f"{row['allocation_id']}:pending-{node.uuid}"
    ):
        return "CalcJob scheduler ID does not match"
    return None
