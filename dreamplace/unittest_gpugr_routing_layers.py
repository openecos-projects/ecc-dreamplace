import os
import sys
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import torch

AUTODMP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if AUTODMP_ROOT not in sys.path:
    sys.path.insert(0, AUTODMP_ROOT)

from dreamplace.ops.gpugr.xplace_backend import XplaceGPUGR  # noqa: E402


_RANGE_LEF = dedent(
    """
    VERSION 5.8 ;
    BUSBITCHARS "[]" ;
    DIVIDERCHAR "/" ;
    UNITS DATABASE MICRONS 1000 ;
    END UNITS
    LAYER MET1
      TYPE ROUTING ; DIRECTION HORIZONTAL ; PITCH 100 ; WIDTH 40 ; SPACING 40 ;
    END MET1
    LAYER VIA1
      TYPE CUT ; SPACING 40 ; WIDTH 40 ;
    END VIA1
    LAYER MET2
      TYPE ROUTING ; DIRECTION VERTICAL ; PITCH 100 ; WIDTH 40 ; SPACING 40 ;
    END MET2
    LAYER VIA2
      TYPE CUT ; SPACING 40 ; WIDTH 40 ;
    END VIA2
    LAYER MET3
      TYPE ROUTING ; DIRECTION HORIZONTAL ; PITCH 100 ; WIDTH 40 ; SPACING 40 ;
    END MET3
    LAYER VIA3
      TYPE CUT ; SPACING 40 ; WIDTH 40 ;
    END VIA3
    LAYER MET4
      TYPE ROUTING ; DIRECTION VERTICAL ; PITCH 100 ; WIDTH 40 ; SPACING 40 ;
    END MET4
    SITE core
      CLASS CORE ; SIZE 0.1 BY 0.1 ;
    END core
    MACRO BUF
      CLASS CORE ; ORIGIN 0 0 ; SIZE 100 BY 100 ; SYMMETRY X Y ; SITE core ;
      PIN A
        DIRECTION INPUT ; USE SIGNAL ;
        PORT
          LAYER MET1 ; RECT 0 0 40 40 ;
        END
      END A
      PIN Y
        DIRECTION OUTPUT ; USE SIGNAL ;
        PORT
          LAYER MET1 ; RECT 60 60 100 100 ;
        END
      END Y
    END BUF
    END LIBRARY
    """
)

_RANGE_DEF = dedent(
    """
    VERSION 5.8 ;
    DIVIDERCHAR "/" ; BUSBITCHARS "[]" ;
    DESIGN RANGE_SMOKE ; UNITS DISTANCE MICRONS 1000 ;
    DIEAREA ( 0 0 ) ( 1000 1000 ) ;
    ROW ROW_0 core 0 0 N DO 10 BY 1 STEP 100 0 ;
    TRACKS X 0 DO 11 STEP 100 LAYER MET1 ;
    TRACKS Y 0 DO 11 STEP 100 LAYER MET1 ;
    TRACKS X 0 DO 11 STEP 100 LAYER MET2 ;
    TRACKS Y 0 DO 11 STEP 100 LAYER MET2 ;
    TRACKS X 0 DO 11 STEP 100 LAYER MET3 ;
    TRACKS Y 0 DO 11 STEP 100 LAYER MET3 ;
    TRACKS X 0 DO 11 STEP 100 LAYER MET4 ;
    TRACKS Y 0 DO 11 STEP 100 LAYER MET4 ;
    COMPONENTS 2 ;
    - U1 BUF + PLACED ( 100 100 ) N ;
    - U2 BUF + PLACED ( 700 700 ) N ;
    END COMPONENTS
    PINS 0 ; END PINS
    SPECIALNETS 0 ; END SPECIALNETS
    NETS 1 ;
    - N1 ( U1 Y ) ( U2 A ) ;
    END NETS
    END DESIGN
    """
)


def _gpugr_extension_ready():
    xplace_root = Path(__file__).resolve().parents[1] / "thirdparty" / "xplace"
    cpybin = xplace_root / "cpp_to_py" / "cpybin"
    return cpybin.is_dir() and any(cpybin.glob("gpugr*.so"))


def _cuda_gpugr_ready():
    """Require both the CUDA runtime and a CUDA-built native GPUGR module."""

    if not _gpugr_extension_ready() or not torch.cuda.is_available():
        return False
    xplace_root = Path(__file__).resolve().parents[1] / "thirdparty" / "xplace"
    xplace_root_str = str(xplace_root)
    if xplace_root_str not in sys.path:
        sys.path.insert(0, xplace_root_str)
    try:
        from cpp_to_py.cpybin import gpugr

        return bool(gpugr.cuda_enabled())
    except Exception:
        return False


def _write_range_design(root: Path):
    lef = root / "range.lef"
    design_def = root / "range.def"
    lef.write_text(_RANGE_LEF, encoding="utf-8")
    design_def.write_text(_RANGE_DEF, encoding="utf-8")
    return lef, design_def


def _params(lef: Path, result_dir: Path):
    return SimpleNamespace(
        lef_input=[str(lef)],
        result_dir=str(result_dir),
        gpugr_bottom_routing_layer="",
        gpugr_top_routing_layer="",
        design_name=lambda: "RANGE_SMOKE",
    )


@unittest.skipUnless(_gpugr_extension_ready(), "Xplace gpugr extension is not built")
class GPUGRRoutingLayerWindowTest(unittest.TestCase):
    def _run(self, backend, bottom, top, include_l_shape_topology=False):
        with tempfile.TemporaryDirectory(prefix="gpugr_layer_") as tmp:
            root = Path(tmp)
            lef, design_def = _write_range_design(root)
            backend_obj = XplaceGPUGR(params=_params(lef, root), placedb=None)
            kwargs = dict(
                input_def=str(design_def),
                out_dir=str(root / "out"),
                design_name="RANGE_SMOKE",
                gpu=0,
                threads=1,
                route_xsize=8,
                route_ysize=8,
                rrr_iters=0,
                skip_m1_route=False,
                backend=backend,
                bottom_routing_layer=bottom,
                top_routing_layer=top,
                include_route_entries=True,
            )
            if include_l_shape_topology:
                kwargs.update(
                    include_l_shape_topology_pack=True,
                    topology_pin_name_to_id={"U1:Y": 0, "U2:A": 1},
                    topology_net_name_to_id={"N1": 0},
                    topology_flat_net2pin_map=np.asarray([0, 1], dtype=np.int64),
                    topology_flat_net2pin_start_map=np.asarray([0, 2], dtype=np.int64),
                    topology_num_pins=2,
                    topology_num_nets=1,
                )
            return backend_obj.run_gpugr(**kwargs)

    def test_cpu_pr_window_reports_enabled_met2_met3(self):
        result = self._run("cpu_pr", "MET2", "MET3")
        metrics = result["metrics"]
        self.assertEqual(metrics["routing_layer_begin"], 1)
        self.assertEqual(metrics["routing_layer_end"], 2)
        self.assertEqual(metrics["enabled_routing_layer_names"], ["MET2", "MET3"])
        demand = result["maps"]["wire_demand_map"]
        self.assertEqual(tuple(demand.shape[:1]), (4,))
        self.assertEqual(float(demand[0].sum()), 0.0)
        self.assertEqual(float(demand[3].sum()), 0.0)
        self.assertGreater(float(demand[1:3].sum()), 0.0)
        layers = {
            int(entry["layer_idx"])
            for net in result["route_entries"]
            for entry in net.get("entries", [])
        }
        self.assertTrue(layers)
        self.assertTrue(layers <= {1, 2})

    def test_cpu_pr_mt_window_and_native_stats_match_cpu_contract(self):
        result = self._run("cpu_pr_mt", "MET2", "MET3")
        metrics = result["metrics"]
        stats = result["native_stats"]
        self.assertEqual(metrics["gpugr_backend"], "cpu_pr_mt")
        self.assertEqual(metrics["parser_threads"], 1)
        self.assertEqual(metrics["native_route_stats"], stats)
        self.assertEqual(stats["terminal_status"], "completed")
        self.assertEqual(stats["requested_workers"], 1)
        self.assertEqual(stats["effective_workers"], 1)
        self.assertFalse(stats["flute_parallel"])
        self.assertEqual(metrics["enabled_routing_layer_names"], ["MET2", "MET3"])
        demand = result["maps"]["wire_demand_map"]
        self.assertEqual(float(demand[0].sum()), 0.0)
        self.assertEqual(float(demand[3].sum()), 0.0)
        self.assertGreater(float(demand[1:3].sum()), 0.0)

    def test_unknown_layer_name_is_rejected(self):
        with self.assertRaises(RuntimeError) as raised:
            self._run("cpu_pr", "MET9", "MET3")
        self.assertIn("Unknown GPUGR bottom routing layer", str(raised.exception))

    @unittest.skipUnless(_cuda_gpugr_ready(), "CUDA GPUGR extension/runtime is unavailable")
    def test_cuda_window_matches_cpu_pr_enabled_layers(self):
        result = self._run("cuda", "MET2", "MET3")
        metrics = result["metrics"]
        self.assertEqual(metrics["routing_layer_begin"], 1)
        self.assertEqual(metrics["routing_layer_end"], 2)
        layers = {
            int(entry["layer_idx"])
            for net in result["route_entries"]
            for entry in net.get("entries", [])
        }
        self.assertTrue(layers <= {1, 2})


if __name__ == "__main__":
    unittest.main()
