from .buffer_library import (
    lookup_buffer_delay_for_bsu_with_status,
    lookup_buffer_transition_for_bsu_with_status,
)


def _fallback_lut_status(source):
    return {
        "source": str(source),
        "input_slew_clamped": False,
        "input_slew_clamp": "none",
        "input_slew_axis_min": None,
        "input_slew_axis_max": None,
        "output_cap_clamped": False,
        "output_cap_clamp": "none",
        "output_cap_axis_min": None,
        "output_cap_axis_max": None,
    }


def build_buffer_surrogate(buffer_library, *, metadata=None, contract_artifact=None):
    def surrogate(candidate, *, bsu, input_slew, output_cap):
        entry = dict(buffer_library[int(bsu)])
        delay = float(entry["delay"])
        delay_source = str(entry.get("delay_source", entry.get("source", "buffer_library")))
        output_slew = float(entry.get("output_slew", input_slew))
        transition_source = str(entry.get("transition_source", "buffer_library"))
        delay_lut_status = _fallback_lut_status("buffer_library")
        transition_lut_status = _fallback_lut_status("buffer_library")
        if metadata is not None and contract_artifact is not None:
            try:
                delay, delay_source, delay_lut_status = lookup_buffer_delay_for_bsu_with_status(
                    metadata,
                    contract_artifact,
                    bsu=bsu,
                    input_slew=input_slew,
                    output_cap=output_cap,
                )
                delay_lut_status = dict(delay_lut_status)
                delay_lut_status["source"] = delay_source
            except ValueError:
                pass
            try:
                output_slew, transition_source, transition_lut_status = lookup_buffer_transition_for_bsu_with_status(
                    metadata,
                    contract_artifact,
                    bsu=bsu,
                    input_slew=input_slew,
                    output_cap=output_cap,
                )
                transition_lut_status = dict(transition_lut_status)
                transition_lut_status["source"] = transition_source
            except ValueError:
                pass
        return {
            "buffer_input_cap": float(entry["input_cap"]),
            "buffer_delay": float(delay),
            "buffer_output_slew": float(output_slew),
            "delay_source": delay_source,
            "transition_source": transition_source,
            "delay_lut_status": delay_lut_status,
            "transition_lut_status": transition_lut_status,
            "input_slew": float(input_slew),
            "output_cap": float(output_cap),
        }

    return surrogate
