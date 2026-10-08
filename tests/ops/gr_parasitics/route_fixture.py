"""Reuse Xplace's native toy design with the supported M2/M3 routing pair."""

import importlib.util


def make_route_fixture(xplace, *, small_pins=False, pin_layer="M2"):
    path = xplace / "cpp_to_py/gpugr/cpu_route_toy_fixture.py"
    spec = importlib.util.spec_from_file_location("cpu_route_toy_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    lef = fixture.TOY_LEF.replace("      LAYER M1 ;", f"      LAYER {pin_layer} ;")
    lef = lef.replace(
        "SITE core\n",
        """LAYER VIA2
  TYPE CUT ;
  SPACING 0.2 ;
END VIA2
LAYER M3
  TYPE ROUTING ;
  DIRECTION HORIZONTAL ;
  PITCH 1 ;
  WIDTH 0.2 ;
  SPACING 0.2 ;
END M3
VIA VIA23 DEFAULT
  LAYER M2 ;
    RECT -0.1 -0.1 0.1 0.1 ;
  LAYER VIA2 ;
    RECT -0.1 -0.1 0.1 0.1 ;
  LAYER M3 ;
    RECT -0.1 -0.1 0.1 0.1 ;
END VIA23
SITE core
""",
    )
    # Explicit synthetic physical RC inputs, independent of the production PDK.
    for layer, resistance, capacitance, edgecap in (
        ("M1", 0.2, 0.005, 0.0005),
        ("M2", 0.4, 0.01, 0.001),
        ("M3", 0.6, 0.02, 0.0015),
    ):
        lef = lef.replace(
            f"END {layer}\n",
            f"""  RESISTANCE RPERSQ {resistance} ;
  CAPACITANCE CPERSQDIST {capacitance} ;
  EDGECAPACITANCE {edgecap} ;
END {layer}
""",
            1,
        )
    lef = lef.replace(" DEFAULT\n", " DEFAULT\n  RESISTANCE 2.5 ;\n")
    if not small_pins:
        lef = lef.replace("RECT 0.1 0.4 0.3 0.6", "RECT 0 0 1 1")
        lef = lef.replace("RECT 0.7 0.4 0.9 0.6", "RECT 0 0 1 1")
    design = fixture.TOY_DEF.replace("NETS 2 ;", "NETS 1 ;")
    design = design.replace("LAYER M1 ;", "LAYER M1 M3 ;")
    design = design.replace("- N1 ( U0 Y ) ( U1 A ) ;", "- N1 ( U0 Y ) ( U1 A ) ( U3 A ) ;")
    design = design.replace("- N2 ( U2 Y ) ( U3 A ) ;\n", "")
    if not small_pins:
        design = design.replace(
            "PINS 0 ;",
            """PINS 1 ;
- IN + NET N1 + DIRECTION INPUT + USE SIGNAL
  + LAYER M2 ( 500 500 ) ( 1250 1250 ) + PLACED ( 1000 1000 ) N ;""",
        )
        design = design.replace("( U0 Y )", "( PIN IN )")
    return lef, design
