from types import SimpleNamespace

import numpy as np
import torch

from dreamplace import Placer


def test_size_only_write_back_uses_refreshed_placedb_coordinates(tmp_path, monkeypatch):
    written = {}

    def fake_write(_bridge, path, _format, node_x, node_y):
        written["path"] = path
        written["x"] = np.asarray(node_x).copy()
        written["y"] = np.asarray(node_y).copy()

    monkeypatch.setattr(
        Placer.placeio_openroad.PlaceIOFunction,
        "write",
        fake_write,
    )

    engine = Placer.PlacementEngine.__new__(Placer.PlacementEngine)
    engine.params = SimpleNamespace(
        place_io_engine="openroad",
        placement_sizing_mode="size_only",
        scale_factor=1.0,
        shift_factor=[0.0, 0.0],
        result_dir=None,
    )
    engine.placedb = SimpleNamespace(
        num_movable_nodes=2,
        num_nodes=2,
        node_x=np.asarray([11.0, 22.0]),
        node_y=np.asarray([33.0, 44.0]),
        openroad_bridge=object(),
        unscale_pl_positions=lambda node_x, node_y: (node_x, node_y),
    )
    engine.placer = SimpleNamespace(
        pos=[torch.tensor([101.0, 202.0, 303.0, 404.0])]
    )
    engine._rewrite_openroad_def_component_placements = (
        lambda *_args, **_kwargs: {
            "applied": False,
            "reason": "test",
            "rewrite_count": 0,
            "missing_count": 0,
        }
    )
    engine._design_name = lambda: "unit"

    output_def = tmp_path / "sized.def"
    engine.write_back(output_def)

    np.testing.assert_array_equal(written["x"], [11, 22])
    np.testing.assert_array_equal(written["y"], [33, 44])
    assert engine.last_openroad_write_back_summary["source"] == "placedb_node_arrays"
    assert engine.last_openroad_write_back_summary["max_delta_to_placer_pos"] == 0.0
