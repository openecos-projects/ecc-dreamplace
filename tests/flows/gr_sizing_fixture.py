"""Synthetic clocked native fixture with actual coarse-grid pin access.

Pin shapes cover each cell footprint on M2. This qualifies the strict GR mode
on a real parsed/routed design; it is not an ICS55 physical/QoR benchmark.
"""

import importlib.util
import json
from pathlib import Path


def make_fixture(directory):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "gr_route_fixture", root / "tests/ops/gr_parasitics/route_fixture.py"
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    lef, _ = fixture.make_route_fixture(root / "thirdparty/xplace")
    lef = lef.split("MACRO BUF\n")[0]
    cells = []
    for kind in ("BUF", "DFF"):
        pins = (
            (("A", "INPUT"), ("Y", "OUTPUT"))
            if kind == "BUF"
            else (("D", "INPUT"), ("CK", "INPUT"), ("Q", "OUTPUT"))
        )
        for size in (1, 2):
            for vt in ("R", "L"):
                name = f"{kind}X{size}H7{vt}"
                lef += (
                    f"MACRO {name}\n CLASS CORE ;\n ORIGIN 0 0 ;\n SIZE {size} BY 1 ;\n"
                    " SYMMETRY X Y ;\n SITE core ;\n"
                )
                for pin, direction in pins:
                    lef += (
                        f" PIN {pin}\n DIRECTION {direction} ;\n USE SIGNAL ;\n PORT\n"
                        f" LAYER M2 ;\n RECT 0 0 {size} 1 ;\n END\n END {pin}\n"
                    )
                lef += f"END {name}\n"
                delay = (0.5 if vt == "R" else 0.35) / size
                table = 'index_1("0.01, 1.0"); index_2("0.001, 1.0");'
                values = (
                    f'values("{delay}, {delay + 0.4 / size}", '
                    f'"{delay + 0.05}, {delay + 0.45 / size}");'
                )
                arc = " ".join(
                    f"{metric}(delay) {{{table} {values}}}"
                    for metric in ("cell_rise", "cell_fall", "rise_transition", "fall_transition")
                )
                cell = f"cell({name}) {{area: {size}; cell_leakage_power: {size};"
                if kind == "BUF":
                    cell += f"pin(A) {{direction: input; capacitance: {0.01 * size};}}"
                    cell += (
                        'pin(Y) {direction: output; function: "A"; max_capacitance: 0.05; '
                        'max_transition: 0.2; timing() {related_pin: "A"; '
                        "timing_sense: positive_unate; timing_type: combinational;"
                    )
                else:
                    cell += 'ff(IQ, IQN) {clocked_on: "CK"; next_state: "D";}'
                    cell += (
                        f"pin(CK) {{direction: input; clock: true; capacitance: {0.01 * size};}}"
                    )
                    cell += (
                        f"pin(D) {{direction: input; capacitance: {0.01 * size}; "
                        'timing() {related_pin: "CK"; timing_type: setup_rising; '
                        'rise_constraint(check) {values("0.01, 0.02", "0.02, 0.03");} '
                        'fall_constraint(check) {values("0.01, 0.02", "0.02, 0.03");}}}'
                    )
                    cell += (
                        'pin(Q) {direction: output; function: "IQ"; max_capacitance: 0.05; '
                        'max_transition: 0.2; timing() {related_pin: "CK"; '
                        "timing_type: rising_edge; timing_sense: non_unate;"
                    )
                cells.append(cell + arc + "}}}")
    lef += "END LIBRARY\n"
    liberty = (
        """library(gr_fixture) {
 delay_model: table_lookup;
 time_unit: "1ns"; voltage_unit: "1V"; current_unit: "1mA";
 pulling_resistance_unit: "1kohm"; capacitive_load_unit(1,pf);
 nom_voltage: 1.0; nom_temperature: 25; nom_process: 1;
 default_max_transition: 0.2; default_input_pin_cap: 0.01;
 lu_table_template(delay) {variable_1: input_net_transition;
 variable_2: total_output_net_capacitance;
 index_1("0.01, 1.0"); index_2("0.001, 1.0");}
 lu_table_template(check) {variable_1: related_pin_transition;
 variable_2: constrained_pin_transition;
 index_1("0.01, 1.0"); index_2("0.01, 1.0");}
"""
        + "\n".join(cells)
        + "\n}\n"
    )
    rows = "".join(
        f"ROW R{y} core 0 {y * 1000} {'N' if y % 2 == 0 else 'FS'} DO 10 BY 1 STEP 1000 0 ;\n"
        for y in range(10)
    )
    design = (
        """VERSION 5.8 ;
 DIVIDERCHAR "/" ; BUSBITCHARS "[]" ; DESIGN gr_fixture ;
 UNITS DISTANCE MICRONS 1000 ; DIEAREA ( 0 0 ) ( 10000 10000 ) ;
"""
        + rows
        + """TRACKS Y 500 DO 10 STEP 1000 LAYER M1 M3 ;
 TRACKS X 500 DO 10 STEP 1000 LAYER M2 ;
 COMPONENTS 4 ;
 - launch DFFX1H7R + PLACED ( 1000 1000 ) N ;
 - b1 BUFX1H7R + PLACED ( 5000 1000 ) N ;
 - b2 BUFX1H7R + PLACED ( 8000 5000 ) N ;
 - capture DFFX1H7R + PLACED ( 8000 8000 ) N ;
 END COMPONENTS
 PINS 3 ;
 - clk + NET clk + DIRECTION INPUT + USE CLOCK
   + LAYER M2 ( 0 0 ) ( 1000 1000 ) + PLACED ( 1000 5000 ) N ;
 - in + NET in + DIRECTION INPUT + USE SIGNAL
   + LAYER M2 ( 0 0 ) ( 1000 1000 ) + PLACED ( 1000 8000 ) N ;
 - out + NET out + DIRECTION OUTPUT + USE SIGNAL
   + LAYER M2 ( 0 0 ) ( 1000 1000 ) + PLACED ( 5000 8000 ) N ;
 END PINS
 SPECIALNETS 0 ; END SPECIALNETS
 NETS 6 ;
 - clk ( PIN clk ) ( launch CK ) ( capture CK ) + USE CLOCK ;
 - in ( PIN in ) ( launch D ) ;
 - n1 ( launch Q ) ( b1 A ) ;
 - n2 ( b1 Y ) ( b2 A ) ;
 - n3 ( b2 Y ) ( capture D ) ;
 - out ( capture Q ) ( PIN out ) ;
 END NETS
 END DESIGN
"""
    )
    files = {
        "fixture.lef": lef,
        "fixture.lib": liberty,
        "fixture.def": design,
        "fixture.v": (
            "module gr_fixture(input clk, input in, output out); wire n1,n2,n3;\n"
            "DFFX1H7R launch(.CK(clk),.D(in),.Q(n1)); BUFX1H7R b1(.A(n1),.Y(n2));\n"
            "BUFX1H7R b2(.A(n2),.Y(n3)); DFFX1H7R capture(.CK(clk),.D(n3),.Q(out));\n"
            "endmodule\n"
        ),
        "fixture.sdc": (
            "create_clock -name clk -period 0.8 [get_ports clk]\n"
            "set_input_delay 0.01 -clock clk [get_ports in]\n"
            "set_output_delay 0.01 -clock clk [get_ports out]\n"
            "set_input_transition 0.02 [get_ports in]\nset_load 0.02 [get_ports out]\n"
        ),
    }
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (directory / name).write_text(content)
    (directory / "db.json").write_text(
        json.dumps(
            {
                "INPUT": {
                    "tech_lef_path": str(directory / "fixture.lef"),
                    "lef_paths": [],
                    "def_path": str(directory / "fixture.def"),
                },
                "OUTPUT": {"output_dir_path": str(directory / "native")},
            }
        )
    )
    return directory
