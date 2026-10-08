import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "random_init_snapshot.py"
spec = importlib.util.spec_from_file_location("random_init_snapshot", MODULE_PATH)
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
coordinate_sha256 = snapshot.coordinate_sha256
parse_def_component_placements = snapshot.parse_def_component_placements
xy_coordinate_sha256 = snapshot.xy_coordinate_sha256


def _write_def(path: Path, placed_x: int = 10) -> None:
    path.write_text(
        f"""VERSION 5.8 ;
COMPONENTS 3 ;
- u1 INVx1
  + PLACED ( {placed_x} 20 ) N ;
- macro0 SRAM + PLACED ( 30 40 ) FS ;
- fixed0 TAP
  + FIXED ( 50 60 ) N ;
END COMPONENTS
END DESIGN
""",
        encoding="utf-8",
    )


def test_parse_def_component_placements_handles_multiline_records(tmp_path):
    path = tmp_path / "fixture.def"
    _write_def(path)

    records = parse_def_component_placements(path)

    assert records["u1"]["placement"] == {
        "status": "PLACED",
        "x": 10,
        "y": 20,
        "orient": "N",
    }
    assert records["macro0"]["master"] == "SRAM"
    assert records["fixed0"]["placement"]["status"] == "FIXED"


def test_coordinate_hash_changes_only_for_selected_coordinate_changes(tmp_path):
    first = tmp_path / "first.def"
    second = tmp_path / "second.def"
    _write_def(first, placed_x=10)
    _write_def(second, placed_x=11)
    first_records = parse_def_component_placements(first)
    second_records = parse_def_component_placements(second)

    assert coordinate_sha256(first_records) != coordinate_sha256(second_records)
    assert coordinate_sha256(first_records, {"macro0", "fixed0"}) == (
        coordinate_sha256(second_records, {"macro0", "fixed0"})
    )
    assert xy_coordinate_sha256(first_records, {"u1"}) != xy_coordinate_sha256(
        second_records,
        {"u1"},
    )
