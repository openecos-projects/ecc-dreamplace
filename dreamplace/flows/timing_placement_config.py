"""Expand an explicit placement timing carrier into mutually exclusive controls."""


def configure_timing_placement_carrier(params):
    carrier = getattr(params, "timing_placement_carrier", "direct_loss")
    selected = getattr(params, "_timing_placement_carrier_explicit", False)
    if not (selected or carrier == "pin2pin"):
        return
    if not getattr(params, "global_place_flag", True):
        return
    if carrier not in {"direct_loss", "pin2pin", "gradient_net_weight"}:
        raise ValueError(
            "timing_placement_carrier must be direct_loss, pin2pin, or gradient_net_weight"
        )
    if carrier == "gradient_net_weight":
        return
    pin2pin = carrier == "pin2pin"
    controls = {
        "with_sta": 1,
        "timing_eval_flag": 1,
        "diff_timing_driven_placement": int(not pin2pin),
        "differentiable_timing_obj": int(not pin2pin),
        "enable_net_weighting": int(pin2pin),
        "pin2pin_net_weighting": int(pin2pin),
        "net_weighting_scheme": "pin2pin" if pin2pin else "lilith",
    }
    for name, value in controls.items():
        setattr(params, name, value)
        setattr(params, f"_{name}_explicit", True)
