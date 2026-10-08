"""Write OpenROAD Verilog with the compression requested by the filename."""

import gzip
import shutil
import tempfile
from pathlib import Path

from .place_io import PlaceIOFunction


def write_verilog(bridge, filename):
    output = Path(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != ".gz":
        return bridge.eval_tcl_string(
            "write_verilog " + PlaceIOFunction._format_tcl_value(str(output))
        )
    with tempfile.TemporaryDirectory(prefix="openroad-verilog-", dir=output.parent) as directory:
        plain = Path(directory) / "output.v"
        bridge.eval_tcl_string(
            "write_verilog " + PlaceIOFunction._format_tcl_value(str(plain))
        )
        with plain.open("rb") as source, gzip.open(output, "wb") as target:
            shutil.copyfileobj(source, target)
