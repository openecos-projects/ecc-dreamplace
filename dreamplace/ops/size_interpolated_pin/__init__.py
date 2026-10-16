"""Size-interpolated pin property operator.

The compiled operator and the sizing utilities that live next to it are loaded
lazily: importing this package (or ``.sizing_limit_utils``) must stay possible
when the C++ extension is unavailable, because ``sizing_limit_utils`` falls back
to its pure-Python path in that case (see its guarded native import).
"""

__all__ = ["size_interpolated_pin"]


def __getattr__(name):
    if name == "size_interpolated_pin":
        from .size_interpolated_pin_op import size_interpolated_pin

        return size_interpolated_pin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
