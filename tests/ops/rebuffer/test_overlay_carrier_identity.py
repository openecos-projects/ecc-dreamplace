"""Physical overlay name checks retain exact base membership and read-only inputs."""

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from dreamplace.ops.rebuffer.payloads import steiner_payload
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_timing_carrier import _base


@pytest.mark.parametrize("kind", ["node", "net"])
@pytest.mark.parametrize("collision", [False, True])
def test_overlay_checks_exact_generated_name_membership_without_mutating_topology(kind, collision):
    base = _base()
    name = "__dreamplace_rebuffer_7" + ("_out" if kind == "net" else "")
    existing = name if collision else name + "_original"
    if kind == "node":
        base.node_names[0] = existing
        base.pin_names[1:3] = [existing + ":A", existing + ":Y"]
    else:
        base.net_names[0] = existing
    owner = RebufferTimingModel(base, use_pin_offset=False)
    steiner = steiner_payload(base, owner)
    state = owner.create_net_overlay("n1")
    rebuffer_cpp.apply_incremental_buffer_split(
        state,
        {
            "action_id": 7,
            "parent_node_id": 2,
            "child_node_id": 3,
            "split_ratio": 0.5,
            "downstream_pin_ids": [3],
            "buffer_master_name": "BUF2",
        },
    )
    actions = [{"session_buffer_id": 7, "buffer_master_name": "BUF2"}]
    before = rebuffer_cpp.incremental_topology_view(state)

    if collision:
        with pytest.raises(ValueError, match="overlay identity collides with the base design"):
            rebuffer_cpp.export_overlay_carrier(base, steiner, [state], actions)
    else:
        result = rebuffer_cpp.export_overlay_carrier(base, steiner, [state], actions)
        assert result == rebuffer_cpp.export_overlay_carrier(base, steiner, [state], actions)
        names = result["physical"][kind + "_names"]
        assert name in names and existing in names
        assert len(names) == len(set(names))
    assert rebuffer_cpp.incremental_topology_view(state) == before
