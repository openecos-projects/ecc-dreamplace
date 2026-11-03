"""Transport the same placement timing inputs to initial and fresh TP."""


def placement_timing_kwargs(config, *, use_pin_offset=False):
    resolved = dict(config or {})
    options = {"use_pin_offset": bool(use_pin_offset)}
    options.update({key: resolved[key] for key in ("r_unit", "c_unit") if key in resolved})
    provider = str(resolved.get("steiner_provider", "ecc"))
    if provider != "ecc":
        options["steiner_provider"] = provider
    return options
