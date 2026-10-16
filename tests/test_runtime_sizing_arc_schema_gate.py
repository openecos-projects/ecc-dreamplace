import numpy as np

from dreamplace.macroPlaceDB import _gate_sizeable_families_by_runtime_arc_schema


def test_runtime_arc_schema_gate_disables_multiplicity_drift_only():
    sizeable, summary = _gate_sizeable_families_by_runtime_arc_schema(
        main_id_2_cell_id_start=np.array([0, 2, 4], dtype=np.int32),
        cell_id_2_arc_id_start=np.array([0, 1, 3, 5, 7], dtype=np.int32),
        flat_libarc_info=np.array(
            [
                [4, 1, 0, 0, -1, 0],
                [4, 1, 1, 0, -1, 0],
                [4, 1, 1, 1, -1, 0],
                [2, 3, 2, 0, 1, 0],
                [0, 3, 2, 1, -1, 0],
                [0, 3, 3, 0, -1, 0],
                [2, 3, 3, 1, 1, 0],
            ],
            dtype=np.int32,
        ),
        main_id_is_sizeable=np.array([True, True]),
        flat_libcell_names=np.array([b"bad_x1", b"bad_x2", b"good_x1", b"good_x2"]),
    )

    assert np.array_equal(sizeable, np.array([False, True]))
    assert summary["status"] == "degraded"
    assert summary["input_sizeable_family_count"] == 2
    assert summary["output_sizeable_family_count"] == 1
    assert summary["disabled_family_count"] == 1
    assert summary["disabled_families"] == [
        {
            "main_id": 0,
            "reference_cell_id": 0,
            "reference_cell_name": "bad_x1",
            "candidate_cell_id": 1,
            "candidate_cell_name": "bad_x2",
            "differences": [
                {
                    "signature": [4, 1, -1, 0],
                    "reference_count": 1,
                    "candidate_count": 2,
                }
            ],
        }
    ]


def test_runtime_arc_schema_gate_keeps_existing_non_sizeable_family_disabled():
    sizeable, summary = _gate_sizeable_families_by_runtime_arc_schema(
        main_id_2_cell_id_start=np.array([0, 2], dtype=np.int32),
        cell_id_2_arc_id_start=np.array([0, 1, 3], dtype=np.int32),
        flat_libarc_info=np.array(
            [
                [4, 1, 0, 0, -1, 0],
                [4, 1, 1, 0, -1, 0],
                [4, 1, 1, 1, -1, 0],
            ],
            dtype=np.int32,
        ),
        main_id_is_sizeable=np.array([False]),
    )

    assert np.array_equal(sizeable, np.array([False]))
    assert summary["status"] == "pass"
    assert summary["disabled_family_count"] == 0
