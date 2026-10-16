import gzip
from pathlib import Path

import pytest
from dreamplace.ops.placeio_openroad.verilog_writeback import write_verilog


@pytest.mark.parametrize("suffix", [".v", ".v.gz"])
def test_verilog_output_content_matches_requested_compression(tmp_path, suffix):
    content = "module top(); endmodule\n"

    class Bridge:
        def eval_tcl_string(self, command):
            filename = command.removeprefix("write_verilog ").strip("{}")
            Path(filename).write_text(content, encoding="utf-8")

    output = tmp_path / ("top" + suffix)
    write_verilog(Bridge(), output)
    if suffix == ".v.gz":
        with gzip.open(output, "rt") as stream:
            assert stream.read() == content
    else:
        assert output.read_text(encoding="utf-8") == content
