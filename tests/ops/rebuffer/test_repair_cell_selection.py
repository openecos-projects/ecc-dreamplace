"""Source eligibility, traversal and initial-cell scoring for native repair."""

import gzip
from copy import deepcopy

import pytest
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.rebuffer.test_repair_design import _characterization, _post_resize_liberty
from tests.ops.rebuffer.test_repair_design_target_math import _table, _tables


def _sources(
    tmp_path, names, *, attributes=None, ports=None, classes=None, order=None, compressed=False
):
    attributes = attributes or {}
    ports = ports or {}
    classes = classes or {}
    cells = []
    macros = []
    for name in order or names:
        input_name, output_name, function = ports.get(name, ("A", "Y", "(A)"))
        cells.append(
            f'cell ("{name}") {{ {attributes.get(name, "cell_leakage_power : 1;")}'
            f" pin ({input_name}) {{ direction : input; }}"
            f' pin ({output_name}) {{ direction : output; function : "{function}";'
            ' timing () { related_pin : "A"; values ("a } quoted brace"); } } }'
        )
        if classes.get(name, "CORE") is not None:
            macros.append(
                f"MACRO {name}\n CLASS {classes.get(name, 'CORE')} ; SITE core ;\n"
                " PIN A\n PORT\n LAYER M1 ; RECT 0 0 1 1 ;\n END\n END A\n"
                " OBS\n LAYER M1 ; RECT 0 0 1 1 ;\n END\n"
                f"END {name}\n"
            )
    text = (
        '/* metadata source */ library (test) { comment : ""; leakage_power_unit : "1nW"; '
        + "\n".join(cells)
        + " }"
    )
    liberty = tmp_path / ("cells.lib.gz" if compressed else "cells.lib")
    if compressed:
        with gzip.open(liberty, "wt") as stream:
            stream.write(text)
    else:
        liberty.write_text(text)
    lef = tmp_path / "cells.lef"
    lef.write_text("\n".join(macros))
    return liberty, lef


def _liberty(tmp_path, **kwargs):
    liberty = deepcopy(_post_resize_liberty())
    liberty.update(
        names=["BASE", "MID", "BIG", "OTHER"],
        cell_main_id=[0, 1, 2, 3],
        cell_vt=[0, 1, 2, 3],
        cell_is_buffer=[True] * 4,
        cell_area=[1.0, 2.0, 5.0, 2.0],
        cell_drive_resistance=[40.0, 30.0, 20.0, 10.0],
    )
    delay = [_table(value, 0.0, 0.0) for value in (3.0, 2.0, 1.0, 4.0)]
    transition = _table(0.0, 0.0, 1.0)
    _tables(liberty, [[[table, table, transition, transition]] for table in delay])
    paths = _sources(tmp_path, liberty["names"], **kwargs)
    liberty["_native_cell_metadata"] = rebuffer_cpp.read_repair_library_metadata(
        [str(paths[0])], [str(paths[1])], liberty["names"]
    )
    return liberty


@pytest.mark.parametrize("compressed", [False, True])
def test_metadata_reader_preserves_source_attributes_and_nested_lef(tmp_path, compressed):
    liberty, lef = _sources(
        tmp_path,
        ["B"],
        compressed=compressed,
        attributes={
            "B": 'dont_use : true; cell_footprint : "shape"; '
            'user_function_class : "special"; leakage_power () { value : 2; } '
            "leakage_power () { value : 4; }"
        },
    )

    metadata = rebuffer_cpp.read_repair_library_metadata([str(liberty)], [str(lef)], ["B"])

    assert metadata.records() == [
        dict(
            name="B",
            library="test",
            source_order=0,
            dont_use=True,
            has_master=True,
            master_is_core=True,
            site="core",
            footprint="shape",
            function_class="special",
            buffer_signature="A:input::;Y:output:A:;",
            has_leakage=True,
            leakage=pytest.approx(3.0e-9),
        )
    ]


def test_swappable_domain_uses_current_area_and_crosses_physical_family_and_vt(tmp_path):
    liberty = _liberty(tmp_path)

    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 0) == [0, 1, 3]
    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 1) == [0, 1, 2, 3]


def test_swappable_domain_applies_dont_use_master_function_class_and_port_identity(tmp_path):
    liberty = _liberty(
        tmp_path,
        attributes={"MID": "dont_use : true;", "BASE": 'user_function_class : "private";'},
        classes={"BIG": None},
        ports={"OTHER": ("I", "Z", "I")},
    )

    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 0) == [0]
    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 3) == [3]


def test_source_core_requirement_and_candidate_master_requirement_are_distinct(tmp_path):
    liberty = _liberty(tmp_path, classes={"OTHER": "BLOCK"})

    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 0) == [0, 1, 3]
    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 3) == []


def test_leakage_filter_uses_current_cell_and_includes_exact_four_times(tmp_path):
    liberty = _liberty(
        tmp_path,
        attributes={"MID": "cell_leakage_power : 4;", "OTHER": "cell_leakage_power : 4.01;"},
    )
    liberty["cell_area"] = [1.0] * 4

    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 0) == [0, 1, 2]
    assert rebuffer_cpp.repair_buffer_cell_candidates(liberty, 1) == [0, 1, 2, 3]


def test_target_selection_restarts_from_current_cell_with_new_eligibility(tmp_path):
    liberty = _liberty(tmp_path)
    characterization = _characterization()
    characterization.update(target_slew_rise=1.0, target_slew_fall=1.0)

    assert rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, 0, 0.8) == 1
    assert rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, 1, 0.8) == 2


def test_equal_scores_preserve_initial_cell_even_when_it_is_not_first(tmp_path):
    liberty = _liberty(tmp_path, order=["MID", "BASE", "BIG", "OTHER"])
    liberty["cell_area"] = [1.0] * 4
    delay = _table(2.0, 0.0, 0.0)
    transition = _table(0.0, 0.0, 1.0)
    _tables(liberty, [[[delay, delay, transition, transition]]] * 4)
    characterization = _characterization()
    characterization.update(target_slew_rise=1.0, target_slew_fall=1.0)

    assert rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, 0, 0.8) == 0
    assert rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, 1, 0.8) == 1


def test_target_selection_does_not_add_electrical_prefilters(tmp_path):
    liberty = _liberty(tmp_path)
    liberty["cell_max_cap"] = [0.001] * 4
    liberty["cell_max_slew"] = [0.001] * 4
    characterization = _characterization()
    characterization.update(target_slew_rise=1.0, target_slew_fall=1.0)

    assert rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, 0, 0.8) == 1


def test_repeating_from_final_cell_can_change_choice_without_a_load_change(tmp_path):
    liberty = _liberty(tmp_path, ports={"OTHER": ("I", "Z", "I")})
    liberty["cell_area"] = [1.0] * 4
    liberty["cell_drive_resistance"] = [30.0, 20.0, 40.0, 10.0]
    arcs = []
    for target_load, delay in [(1.0, 10.0), (1.05, 9.5), (1.13, 8.0), (1.0, 20.0)]:
        delay_table = _table(delay, 0.0, 0.0)
        slew_table = _table(0.0, 0.0, 1.0 / target_load)
        arcs.append([[delay_table, delay_table, slew_table, slew_table]])
    _tables(liberty, arcs)
    characterization = _characterization()
    characterization.update(target_slew_rise=1.0, target_slew_fall=1.0)

    # BIG is visited before MID. It lies outside BASE's distance tolerance,
    # then inside MID's tolerance after MID wins. A second pass reaches BIG
    # even though the load is unchanged.
    first = rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, 0, 0.0)
    repeated = rebuffer_cpp.repair_buffer_target_cell(liberty, characterization, first, 0.0)

    assert (first, repeated) == (1, 2)


def test_duplicate_link_name_and_missing_source_cell_are_rejected(tmp_path):
    liberty, lef = _sources(tmp_path, ["B"])
    with pytest.raises(ValueError, match="ambiguous linked"):
        rebuffer_cpp.read_repair_library_metadata([str(liberty)] * 2, [str(lef)], ["B"])
    with pytest.raises(ValueError, match="missing from Liberty"):
        rebuffer_cpp.read_repair_library_metadata([str(liberty)], [str(lef)], ["OTHER"])


def test_buffer_driver_slew_repair_uses_source_eligibility_across_physical_families(tmp_path):
    from tests.ops.rebuffer.test_repair_design import _design, _steiner, _timing

    liberty = _liberty(tmp_path, attributes={"OTHER": "dont_use : true;"})
    liberty["cell_max_slew"] = [0.1] * 4
    arcs = []
    for slew in (0.2, 0.05, 0.01, 0.01):
        delay = _table(1.0, 0.0, 0.0)
        transition = _table(slew, 0.0, 0.0)
        arcs.append([[delay, delay, transition, transition]])
    _tables(liberty, arcs)
    design = _design()
    design["main_id_2_cell_id_start"] = [0, 1, 2, 3, 4]

    result = rebuffer_cpp.plan_repair_design(
        design,
        _timing(),
        _steiner(),
        liberty,
        _characterization(),
        dict(
            insert_buffers=False,
            repair_design_openroad_driver_resize=True,
            repair_design_target_cell_math=True,
            repair_design_target_cell_selection=True,
        ),
    )

    # MID fixes slew across the physical-family boundary. BIG is beyond the
    # source's 4x area limit; OTHER is dont-use despite its better transition.
    assert result["status"] == "ok"
    assert result["resizes"] == [dict(node_id=0, target_cell_index=1)]


def test_junction_resize_uses_selected_initial_cell_as_eligibility_source(tmp_path):
    from tests.ops.rebuffer.test_repair_design_tree_state import _fork

    design, timing, steiner, _ = _fork()
    liberty = _liberty(tmp_path)
    design["main_id_2_cell_id_start"] = [0, 1, 2, 3, 4]
    characterization = _characterization()
    characterization.update(
        target_slew_rise=1.0,
        target_slew_fall=1.0,
        buffer_lowest_drive="BASE",
    )
    for field in ("filtered_buffers", "fast_buffers"):
        characterization[field][0].update(cell_id=0, name="BASE", main_id=0)
    config = dict(
        resize_drivers=False,
        repair_net_tree=True,
        repair_design_openroad_repeater_resize=True,
        repair_design_target_cell_math=True,
        repair_design_target_cell_selection=True,
    )

    result = rebuffer_cpp.plan_repair_design(
        design, timing, steiner, liberty, characterization, config
    )

    assert result["status"] == "ok"
    assert [
        (row["buffer_cell_index"], row["downstream_pins"], row["reason"])
        for row in result["actions"]
    ] == [(2, [1], "junction_cap")]
    assert result["actions"][0]["downstream_load"] == pytest.approx(0.007)
    action = result["actions"][0]
    assert action["initial_buffer_cell_index"] == 1
    assert action["initial_downstream_load"] == pytest.approx(0.012)
    assert (
        rebuffer_cpp.repair_buffer_target_cell(
            liberty,
            characterization,
            action["initial_buffer_cell_index"],
            action["downstream_load"],
        )
        == action["buffer_cell_index"]
    )
