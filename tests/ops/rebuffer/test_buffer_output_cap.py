from types import SimpleNamespace

import numpy as np
from dreamplace.ops.rebuffer.liberty import build_liberty_view_from_pydb


def test_output_cap_resolves_the_timing_arc_sink_port():
    db = SimpleNamespace(
        dbu=1000,
        flat_libcell_info=[["BUF", 0, 1, 0]],
        flat_libcell_names=["BUF"],
        cell_id_2_libpin_id_start=[0, 2],
        cell_id_2_arc_id_start=[0, 1],
        flat_lib_pin_names=[b"A", b"Y"],
        flat_lib_pin_cap=[0.1, 0.02],
        flat_lib_pin_rcap=[0.11, 0.02],
        flat_lib_pin_fcap=[0.12, 0.02],
        flat_libarc_info=[["A", "Y", 0, 0, 1, 0]],
        buffer_main_type_candidate_indices=[0],
    )
    for prefix in ("f_delay", "r_delay", "f_trans", "r_trans"):
        setattr(db, prefix + "_flat_luts_values", [[0.01] * 4])
        setattr(db, prefix + "_flat_luts_dim", [[2, 2]])
        setattr(db, prefix + "_flat_luts_trans_table", [[0.01, 0.1]])
        setattr(db, prefix + "_flat_luts_cap_table", [[0.001, 0.1]])

    view = build_liberty_view_from_pydb(db)

    assert view["status"] == "ok"
    np.testing.assert_array_equal(view["cell_output_cap"], [0.02])


