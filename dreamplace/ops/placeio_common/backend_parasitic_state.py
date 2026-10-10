import json
import os


def _load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _workspace_file(workspace, *parts):
    return os.path.join(workspace, *parts)


def _find_files(root, suffixes):
    matches = []
    if not root or not os.path.isdir(root):
        return matches
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            lower_name = filename.lower()
            if any(lower_name.endswith(suffix) for suffix in suffixes):
                matches.append(os.path.join(dirpath, filename))
    matches.sort()
    return matches


def _read_existing_text(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read()


def _openroad_rc_summary(rc_tcl):
    text = _read_existing_text(rc_tcl)
    return {
        "rc_tcl": rc_tcl or "",
        "rc_tcl_exists": bool(rc_tcl and os.path.exists(rc_tcl)),
        "has_set_layer_rc": "set_layer_rc" in text,
        "has_set_wire_rc_signal": "set_wire_rc -signal" in text,
        "requires_estimate_parasitics_placement": bool(rc_tcl and os.path.exists(rc_tcl)),
    }


def _ieda_state_summary(workspace, design_path, discovered_spef):
    spef_path = str(design_path.get("spef_path") or "")
    return {
        "workspace": workspace,
        "workspace_spef_path": spef_path,
        "workspace_spef_exists": bool(spef_path and os.path.exists(spef_path)),
        "discovered_spef_count": len(discovered_spef),
        "discovered_spef_examples": discovered_spef[:5],
        "current_pydb_parasitics_initialization": "unverified",
        "current_pydb_state_status": "diagnostic_unverified",
        "known_candidate_initializers": [
            "iSTA readSpef when a SPEF exists",
            "iTO EstimateParasitics::estimateAllNetParasitics()",
            "iEDA evaluation timing buildRCTree APIs",
        ],
    }


def _recommended_next_step(openroad, ieda):
    if ieda["workspace_spef_exists"]:
        return "run_same_spef_ieda_openroad_endpoint_compare"
    if openroad["requires_estimate_parasitics_placement"]:
        return "calibrate_ieda_placement_parasitic_estimator_against_opensta"
    return "provide_shared_spef_or_shared_rc_initialization"


def audit_backend_parasitic_state(workspace, rc_tcl=""):
    design_path_path = _workspace_file(workspace, "config", "design_path.json")
    workspace_json_path = _workspace_file(workspace, "config", "workspace.json")
    design_path = _load_json(design_path_path)
    workspace_json = _load_json(workspace_json_path)
    discovered_spef = _find_files(workspace, (".spef", ".spef.gz"))
    openroad = _openroad_rc_summary(rc_tcl)
    ieda = _ieda_state_summary(workspace, design_path, discovered_spef)
    return {
        "artifact_version": 1,
        "workspace": workspace,
        "workspace_config": {
            "workspace_json": workspace_json_path if os.path.exists(workspace_json_path) else None,
            "design_path_json": design_path_path if os.path.exists(design_path_path) else None,
            "workspace": workspace_json.get("workspace", {}),
            "design_path": design_path,
        },
        "openroad": openroad,
        "ieda": ieda,
        "comparison_policy": {
            "golden_backend": "openroad_opensta",
            "ieda_role": "diagnostic_until_parasitics_calibrated",
            "same_state_parasitics_required_for_native_sta_parity": True,
        },
        "recommended_next_step": _recommended_next_step(openroad, ieda),
    }


def parasitic_state_passed(report):
    if not isinstance(report, dict):
        return False
    return report.get("recommended_next_step") == "run_same_spef_ieda_openroad_endpoint_compare"
