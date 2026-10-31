"""OpenROAD-compatible ``repair_design`` planning over ``placedb.pydb``.

The native planner owns violation detection, driver resize selection, buffer
selection, and Steiner split-point choice.  This module only marshals the
PyPlaceDB/Liberty/STA arrays and maps native ids back to ECC names.
"""

from __future__ import annotations

from .characterization import characterize_buffers_from_pydb
from .liberty import build_liberty_view_from_pydb
from .payloads import steiner_payload, timing_payload
from .pydb_adapter import build_rebuffer_design_from_pydb
from .timing_model import RebufferTimingModel

ARTIFACT = "rebuffer_repair_design_plan"
ARTIFACT_VERSION = 1

DEFAULT_CONFIG = {
    "slew_margin": 0.0,
    "cap_margin": 0.0,
    "r_unit": 0.914,
    "c_unit": 1.89e-4,
    "max_fanout": 32,
    "max_actions": 1000,
    "insert_buffers": True,
    "resize_drivers": True,
    "resize_slew_threshold": 0.03,
    "steiner_provider": "ecc",
}


def map_buffer_actions(result, pydb):
    net_names = [str(name) for name in pydb.net_names]
    pin_names = [str(name).replace(":", "/") for name in pydb.pin_names]
    cell_names = [str(name) for name in pydb.flat_libcell_names]
    node_names = [str(name) for name in pydb.node_names]

    actions = []
    for order, action in enumerate(result.get("actions", ())):
        mapped = dict(action)
        mapped["action_kind"] = "buffer_insert"
        mapped["net_name"] = net_names[int(action["net_id"])]
        mapped["driver_pin_name"] = pin_names[int(action["driver_pin"])]
        mapped["downstream_pin_names"] = [
            pin_names[int(pin)] for pin in action["downstream_pins"]
        ]
        mapped["buffer_master_name"] = cell_names[int(action["buffer_cell_index"])]
        mapped["candidate_location_x_dbu"] = int(action["x_dbu"])
        mapped["candidate_location_y_dbu"] = int(action["y_dbu"])
        mapped["commit_order"] = order
        actions.append(mapped)

    resizes = []
    for resize in result.get("resizes", ()):
        mapped = dict(resize)
        mapped["action_kind"] = "cell_resize"
        mapped["node_name"] = node_names[int(resize["node_id"])]
        mapped["target_cell_name"] = cell_names[int(resize["target_cell_index"])]
        resizes.append(mapped)
    return actions, resizes


def plan_repair_design_from_pydb(pydb, timing_model=None, *, config=None):
    """Return a native ``repair_design`` plan without mutating the database."""

    if pydb is None:
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": ["pydb_is_none"],
            "actions": [],
            "resizes": [],
        }

    design = build_rebuffer_design_from_pydb(pydb)
    if design.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(
                design.get("unsupported_reasons", ["design_view_not_ok"])
            ),
            "actions": [],
            "resizes": [],
        }
    liberty = build_liberty_view_from_pydb(pydb)
    if liberty.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(
                liberty.get("unsupported_reasons", ["liberty_view_not_ok"])
            ),
            "actions": [],
            "resizes": [],
        }
    characterization = characterize_buffers_from_pydb(pydb)
    if characterization.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(
                characterization.get(
                    "unsupported_reasons", ["characterization_not_ok"]
                )
            ),
            "actions": [],
            "resizes": [],
        }

    model = timing_model if timing_model is not None else RebufferTimingModel(pydb)
    merged_config = dict(DEFAULT_CONFIG)
    merged_config.update(config or {})
    steiner_values = steiner_payload(
        pydb,
        model,
        provider=str(merged_config.get("steiner_provider", "ecc")),
        r_unit=merged_config["r_unit"],
    )
    if steiner_values.get("status") != "ok":
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "status": "unsupported",
            "unsupported_reasons": list(
                steiner_values.get(
                    "unsupported_reasons", ["steiner_payload_not_ok"]
                )
            ),
            "actions": [],
            "resizes": [],
        }

    from . import rebuffer_cpp

    result = dict(
        rebuffer_cpp.plan_repair_design(
            design,
            timing_payload(pydb, model),
            steiner_values,
            liberty,
            characterization,
            merged_config,
        )
    )
    result.update(
        {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "config": merged_config,
            "baseline": model.baseline.as_dict(),
        }
    )
    if result.get("status") != "ok":
        result.setdefault("actions", [])
        result.setdefault("resizes", [])
        return result

    actions, resizes = map_buffer_actions(result, pydb)
    result["actions"] = actions
    result["resizes"] = resizes
    return result


__all__ = ["DEFAULT_CONFIG", "map_buffer_actions", "plan_repair_design_from_pydb"]
