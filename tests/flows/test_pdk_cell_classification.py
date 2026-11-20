"""Compile the actual native classifier without requiring the optional OpenROAD bridge."""
import subprocess
from pathlib import Path


def test_native_classifier_uses_only_declared_suffixes(tmp_path):
    include = Path(__file__).parents[2] / "dreamplace/ops/placeio_openroad/src"
    source = tmp_path / "classification.cpp"
    source.write_text('''
#include <cassert>
#include "pdk_cell_classification.h"
using namespace dreamplace::placeio_openroad::impl::sizing_metadata;
int main() {
  auto configured = classifyCell("vendor_buf_fast", {"_slow", "_fast"});
  assert(configured.drive_key == "vendor_buf" && configured.vt_key == "_fast");
  auto unknown = classifyCell("BUFX8_9TLVT", {});
  assert(unknown.drive_key == "BUFX8_9TLVT" && unknown.vt_key == "default");
  auto no_match = classifyCell("vendor_fast_extra", {"_fast"});
  assert(no_match.drive_key == "vendor_fast_extra");
}
''')
    binary = tmp_path / "classify"
    subprocess.run(["c++", "-std=c++17", "-I", str(include), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
