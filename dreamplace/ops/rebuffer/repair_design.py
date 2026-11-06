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
    "r_unit": 0.653,
    "c_unit": 0.00018889000086286245,
    "max_fanout": 32,
    # Per-net DP safety bound.  repair_design has no global action budget.
    "tree_max_buffers": 1024,
    "insert_buffers": True,
    "resize_drivers": True,
    "resize_slew_threshold": 0.03,
    # OpenROAD repair_design processes nets in reverse topological order and
    # refreshes the affected timing state before deciding the next driver.
    # These three form one contract: local update and load-slew annotation are
    # rejected unless reverse order is on.
    "repair_net_tree": True,
    "repair_design_reverse_topological": True,
    "repair_design_local_update": True,
    "repair_design_load_slew_annotation": True,
    "repair_design_openroad_driver_resize": True,
    "repair_design_fanout_repair": True,
    # OpenROAD placement-parasitics repair design normalizes every logic
    # driver to its target load before checking slew/cap violations.
    "repair_design_resize_to_target_slew": True,
    "repair_design_target_cell_math": True,
    "repair_design_target_cell_selection": True,
    "repair_design_openroad_repeater_resize": True,
    "sizing_area_limit": 0.0,
    "sizing_leakage_limit": 0.0,
    "slew_rc_factor": 1.39,
    "repair_design_max_transition_load": True,
    "repair_design_max_wire_length_um": 0.0,
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
    merged_config = dict(DEFAULT_CONFIG)
    merged_config.update(config or {})
    explicit_target_selection = (
        "repair_design_target_cell_selection" in (config or {})
    )
    has_library_source = getattr(pydb, "repair_library_source", None) is not None
    if not explicit_target_selection and not has_library_source:
        merged_config["repair_design_target_cell_selection"] = False
    if bool(merged_config.get("repair_design_target_cell_selection", False)):
        from dreamplace.ops.placeio_ecc.library_metadata import ensure_repair_library_metadata

        ensure_repair_library_metadata(pydb)
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
    explicit_target_math = bool(
        (config or {}).get("repair_design_target_cell_math", False)
    )
    characterization_options = (
        {"canonical_target_load": True}
        if explicit_target_math
        or (
            bool(merged_config.get("repair_design_target_cell_math", False))
            and has_library_source
        )
        else {}
    )
    characterization = characterize_buffers_from_pydb(pydb, **characterization_options)
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
    if timing_model is None:
        for key in ("r_unit", "c_unit"):
            if key not in (config or {}):
                pydb_val = getattr(pydb, key, None)
                if pydb_val is not None and float(pydb_val) > 0.0 and float(pydb_val) != 1.0:
                    merged_config[key] = float(pydb_val)
        provider_options = (
            {"steiner_provider": merged_config["steiner_provider"]}
            if merged_config["steiner_provider"] != "ecc"
            else {}
        )
        model = RebufferTimingModel(
            pydb,
            r_unit=merged_config["r_unit"],
            c_unit=merged_config["c_unit"],
            **provider_options,
        )
    else:
        model = timing_model
        if "steiner_provider" not in (config or {}):
            merged_config["steiner_provider"] = getattr(model, "steiner_provider", "ecc")
        for key in ("r_unit", "c_unit"):
            value = float(getattr(model, key))
            if key in (config or {}) and float(config[key]) != value:
                raise ValueError(f"repair_design {key} does not match its timing model")
            merged_config[key] = value
    if "repair_design_max_wire_length_um" not in (config or {}) and "max_wire_length" in (
        config or {}
    ):
        merged_config["repair_design_max_wire_length_um"] = float(
            config["max_wire_length"]
        )
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
            timing_payload(
                pydb,
                model,
                max_transition_load=bool(merged_config["repair_design_max_transition_load"]),
                include_pin_rank=bool(
                    merged_config.get("repair_design_reverse_topological", False)
                    or merged_config.get("repair_design_local_update", False)
                    or merged_config.get(
                        "repair_design_load_slew_annotation", False
                    )
                ),
            ),
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
    if (
        merged_config["repair_design_target_cell_selection"]
        and getattr(pydb, "repair_library_metadata", None) is not None
    ):
        result["library_metadata"] = dict(pydb.repair_library_metadata.summary())
    if merged_config["steiner_provider"] == "openroad_stt":
        result["steiner_input_identity"] = dict(steiner_values["input_identity"])
    if result.get("status") != "ok":
        result.setdefault("actions", [])
        result.setdefault("resizes", [])
        return result

    actions, resizes = map_buffer_actions(result, pydb)
    result["actions"] = actions
    result["resizes"] = resizes
    return result


__all__ = ["DEFAULT_CONFIG", "map_buffer_actions", "plan_repair_design_from_pydb"]
