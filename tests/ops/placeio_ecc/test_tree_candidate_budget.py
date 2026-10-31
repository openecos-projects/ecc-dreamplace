"""The PyDB context must allow complete native tree candidates."""

from dreamplace.ops.rebuffer import rebuffer_cpp as native
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from test_timing_carrier import _base


def test_default_tree_payload_allows_chain_and_explicit_single_move_limit_is_respected():
    base = _base()
    base.node_x[2] = 1_000_000.0
    model = RebufferTimingModel(base, use_pin_offset=False)
    candidates = []
    for override in ({}, {"max_actions": 1}):
        context = RebufferPlanContext(
            base,
            model,
            config={"tree_enforce_electrical": False, **override},
        )
        payload = context.native_current_tree_payload(1)
        result = native.optimize_current_driver_tree(
            payload["design"],
            payload["timing"],
            payload["liberty"],
            payload["characterization"],
            payload["config"],
            payload["current_tree"](),
        )
        assert result["status"] == "ok"
        candidates.append(result["candidates"][0])
    complete, single = candidates
    assert len(complete["actions"]) > 1 == len(single["actions"])
    assert complete["slack"] > single["slack"]
    assert {
        (action["source_parent_node_id"], action["source_child_node_id"])
        for action in complete["actions"]
    } == {(2, 3)}
