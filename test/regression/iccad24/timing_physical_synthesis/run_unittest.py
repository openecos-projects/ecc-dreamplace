#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("timing_physical_synthesis_run", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_input_domains_default_to_both_protocol_domains():
    assert runner._selected_input_domains(None) == ("R0", "D_post")


def test_input_domains_can_select_only_r0_without_duplicates():
    assert runner._selected_input_domains(["R0", "R0"]) == ("R0",)
