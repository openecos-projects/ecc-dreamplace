"""Audit saved native route inputs without repeating routing or optimization."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR


def canonical(value):
    return str(value).replace("\\", "")


def def_components(path):
    text = path.read_text()
    section = text.split("COMPONENTS ", 1)[1].split("END COMPONENTS", 1)[0]
    components = {}
    for match in re.finditer(r"-\s+(\S+)\s+(\S+)(.*?);", section, re.S):
        location = re.search(
            r"\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)",
            match[3],
        )
        assert location is not None, match[0]
        components[canonical(match[1])] = (
            match[2], int(location[1]), int(location[2]), location[3]
        )
    return components


def lef_pin_bounds(lefs, master, pin, orientation, units):
    for path in lefs:
        match = re.search(
            rf"^MACRO {re.escape(master)}\s*$(.*?)^END {re.escape(master)}\s*$",
            Path(path).read_text(), re.M | re.S,
        )
        if match is None:
            continue
        geometry = re.search(
            rf"\bPIN {re.escape(pin)}\s*$(.*?)^\s*END {re.escape(pin)}\s*$",
            match[1], re.M | re.S,
        )
        assert geometry is not None, (master, pin)
        rectangles = [
            [float(value) * units for value in rect]
            for rect in re.findall(
                r"\bRECT\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*;",
                geometry[1],
            )
        ]
        assert rectangles, (master, pin)
        bounds = np.array(rectangles)
        lx, ly = bounds[:, :2].min(axis=0)
        hx, hy = bounds[:, 2:].max(axis=0)
        # This clocked standard-cell fixture uses horizontal N/FS rows.
        assert orientation in ("N", "FS"), orientation
        if orientation == "FS":
            size = re.search(r"\bSIZE\s+[\d.]+\s+BY\s+([\d.]+)", match[1])
            height = float(size[1]) * units
            ly, hy = height - hy, height - ly
        return [(lx + hx) / 2, (ly + hy) / 2, hx - lx, hy - ly]
    raise AssertionError(f"missing LEF master: {master}")


def audit(args):
    manifest = json.loads(args.manifest.read_text())
    report = json.loads((args.routes / "report.json").read_text())
    operator = XplaceGPUGR(SimpleNamespace(), SimpleNamespace())
    IOParser, *_ = operator._import_xplace_modules()
    lefs = [manifest["pdk"]["tech_lef"], *manifest["pdk"]["lefs"]]
    orient_names = ("N", "W", "S", "E", "FN", "FW", "FS", "FE")
    stages, parsed_nodes, parsed_pins = [], [], []
    for record in report["routes"]:
        stage = record["stage"]
        route_def, = (args.routes / stage).glob("gpugr_*/gcd_gpugr.def")
        components = def_components(route_def)
        assert components == def_components(args.routes / f"{stage}.def")
        rawdb, gpdb = IOParser().read(
            {"def": str(route_def), "lefs": lefs},
            lite_mode=True, random_place=False, num_threads=2,
        )
        units = gpdb.microns()
        masters = list(gpdb.node_id2celltype_name())
        nodes = {
            canonical(node.name()): (
                masters[node.id()].rsplit("/", 1)[-1],
                node.lx(), node.ly(), orient_names[node.orient()],
            )
            for node in gpdb.nodes()
            if canonical(node.name()) in components
        }
        assert nodes == components, stage
        nets = list(gpdb.net_names())
        pins = {
            canonical(pin.name()): (
                canonical(nets[pin.netId()]),
                pin.rel_lx() + pin.width() / 2,
                pin.rel_ly() + pin.height() / 2,
                pin.width(), pin.height(),
            )
            for pin in gpdb.pins()
        }
        snapshot = record["snapshot"]
        parsed_masters = {node: value[0] for node, value in nodes.items()}
        expected_masters = {
            canonical(node): master for node, master in snapshot["nodes"].items()
        }
        # Timing snapshots exclude physical-only taps/endcaps without a
        # Liberty family. Their geometry/master was still checked against DEF.
        assert set(expected_masters) <= set(parsed_masters)
        assert {node: parsed_masters[node] for node in expected_masters} == expected_masters
        physical_only = set(parsed_masters) - set(expected_masters)
        assert not any(pin.startswith(node + ":") for node in physical_only for pin in pins)
        expected_pins = {canonical(pin): value for pin, value in snapshot["pins"].items()}
        assert set(expected_pins) <= set(pins), sorted(set(expected_pins) - set(pins))
        assert all(pins[pin][0] == value[0] for pin, value in expected_pins.items())
        stages.append({
            "stage": stage, "def": str(route_def),
            "sha256": hashlib.sha256(route_def.read_bytes()).hexdigest(),
            "component_master_coordinate_orientation_checks": len(nodes),
            "signal_pin_connectivity_checks": len(expected_pins),
            "physical_only_components_without_signal_pins": len(physical_only),
        })
        parsed_nodes.append(nodes)
        parsed_pins.append(pins)
        # rawdb owns the storage used by gpdb for this stage.
        del gpdb, rawdb
    changed, = [
        node for node in parsed_nodes[0]
        if parsed_nodes[0][node][0] != parsed_nodes[1][node][0]
    ]
    selected_pins = [pin for pin in parsed_pins[1] if pin.startswith(changed + ":")]
    assert selected_pins
    geometry_changes = [
        pin for pin in selected_pins
        if parsed_pins[0][pin][1:] != parsed_pins[1][pin][1:]
    ]
    assert geometry_changes, "new master must change parsed LEF pin geometry"
    pydb_center_gaps = {}
    for stage_id in (0, 1):
        node = parsed_nodes[stage_id][changed]
        for pin in selected_pins:
            expected = lef_pin_bounds(lefs, node[0], pin.rsplit(":", 1)[-1], node[3], units)
            np.testing.assert_allclose(
                parsed_pins[stage_id][pin][1:], expected, rtol=0, atol=1,
                err_msg="Xplace pin bounding box differs from current-master LEF",
            )
            native_center = report["routes"][stage_id]["snapshot"]["pins"][pin][1:3]
            pydb_center_gaps[f"{stage_id}:{pin}"] = (
                np.asarray(parsed_pins[stage_id][pin][1:3]) - native_center
            ).tolist()
    inserted, = sorted(set(parsed_nodes[2]) - set(parsed_nodes[1]))
    assert parsed_nodes[2][inserted][0] == "BUFX3H7L"
    result = {
        "passed": True, "backend": report["backend"], "stages": stages,
        "changed_master": changed, "changed_pin_geometry": geometry_changes,
        "current_master_lef_pin_geometry_checks": 2 * len(selected_pins),
        "lef_geometry_tolerance_dbu": 1, "inserted_buffer": inserted,
        "pydb_center_gaps_dbu": pydb_center_gaps,
        "pin_center_semantics": "Xplace bbox center; placement PyDB rectangle-center average",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args())
