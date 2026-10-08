import argparse
import importlib


HANDOFF_MODES = ("buffer-only", "no-handoff")
PLACE_IO_ENGINES = ("openroad", "ieda")


def buffer_only_repair_command(margin):
    return (
        "repair_timing -setup "
        "-setup_margin %.12g "
        '-sequence "unbuffer,buffer,split" '
        "-skip_last_gasp -skip_pin_swap -skip_gate_cloning -skip_size_down"
    ) % float(margin)


def install_buffer_only_margin_command_patch(margin):
    placeio_openroad = importlib.import_module(
        "dreamplace.ops.placeio_openroad.place_io"
    )

    command = buffer_only_repair_command(margin)
    for alias in ("buffer-only", "buffer_only"):
        placeio_openroad._BUFFER_INSERTION_STRATEGY_ALIASES[alias]["command"] = command
    return command


def optional_gate_float(value):
    if isinstance(value, str) and value.lower() in ("none", "off", "disabled"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "expected a float or one of: none, off, disabled"
        )
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("expected a float in [0, 1]")
    return parsed


def build_openroad_handoff_config(
    output_def,
    rc_tcl,
    handoff_mode="buffer-only",
    trigger_period=None,
    max_handoff_overflow=None,
    pre_repair_tcl=None,
):
    if handoff_mode == "no-handoff":
        return {"enabled": False, "trigger": {"mode": "disabled"}}
    if handoff_mode != "buffer-only":
        raise ValueError("unsupported handoff_mode: %s" % handoff_mode)
    if not output_def:
        raise ValueError("buffer-only handoff requires output_def")
    if not rc_tcl and not pre_repair_tcl:
        raise ValueError("buffer-only handoff requires rc_tcl")
    if pre_repair_tcl is None:
        pre_repair_tcl = [
            "source %s" % rc_tcl,
            "estimate_parasitics -placement",
        ]

    trigger = {"mode": "disabled"}
    if trigger_period is not None:
        trigger = {
            "mode": "interval",
            "interval": int(trigger_period),
            "dedupe_by": "absolute_iteration",
        }
        if max_handoff_overflow is not None:
            trigger["guards"] = [
                {
                    "name": "overflow",
                    "op": "<=",
                    "value": float(max_handoff_overflow),
                }
            ]

    return {
        "enabled": True,
        "trigger": trigger,
        "buffer_insertion": {
            "enabled": True,
            "strategy": "buffer-only",
            "options": {
                "pre_repair_tcl": list(pre_repair_tcl),
            },
            "write_def_after": output_def,
        },
    }
