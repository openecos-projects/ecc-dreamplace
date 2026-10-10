"""Verify excluded GCD standard cells survive a real ECC macro-only handoff."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from native_refresh_route import make_engine
from native_route_geometry_audit import def_components
from dreamplace.ops.placeio_common.physical_mutation import write_native_def
from dreamplace.ops.routability.gpugr_context import (
    build_parser_cache_inputs, get_cached_gpugr_operator, write_back_movable_lpos,
)


def run(args):
    args.backend = "ecc"
    engine, module = make_engine(args, {"macro_only": 1, "with_sta": 0})
    db, params = engine.placedb, engine.params
    # GCD has no movable macros: every proposed standard-cell move must be
    # excluded, including row-power normalization and placement-status changes.
    assert not np.any(db.macro_writeback_candidate[:db.num_movable_nodes])
    before = args.output / "before.def"
    after = args.output / "after.def"
    write_native_def(db, before)
    pos = torch.from_numpy(np.concatenate([db.node_x, db.node_y])).clone()
    pos[:db.num_movable_nodes] += 37 * params.scale_factor
    pos[db.num_nodes:db.num_nodes + db.num_movable_nodes] += 19 * params.scale_factor
    write_back_movable_lpos(pos, params, db)
    assert build_parser_cache_inputs(pos, params, db) == (None, None)
    operator = get_cached_gpugr_operator(params, db)
    assert not operator.parser_cache_would_hit()
    write_native_def(db, after)
    assert def_components(before) == def_components(after)
    def component_section(path):
        return path.read_text().split("COMPONENTS ", 1)[1].split("END COMPONENTS", 1)[0]
    assert component_section(before) == component_section(after)
    result = {
        "passed": True,
        "excluded_movable_nodes": int(db.num_movable_nodes),
        "component_coordinate_orientation_status_checks": len(def_components(before)),
        "selected_macros": 0,
        "scope": "native excluded-cell handoff; no macro-move or OpenROAD qualification",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    module.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rc-tcl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
