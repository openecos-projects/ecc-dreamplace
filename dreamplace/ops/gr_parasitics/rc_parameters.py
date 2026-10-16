"""Read the declared LEF ground-capacitance and wire/via resistance model."""

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RCParameters:
    layer_names: tuple[str, ...]
    resistance_per_um: np.ndarray
    capacitance_per_um: np.ndarray
    via_resistance: np.ndarray
    sources: tuple[tuple[str, str], ...]
    via_cap_model: str = "omitted"

    @classmethod
    def from_lefs(cls, paths, layer_names):
        """Use layer order from the router; ambiguous or absent via R stays invalid.

        Invalid entries are rejected when the native pack actually uses them.
        Unused layers therefore do not make a legal routing window unsupported.
        Values are LEF physical units: ohms, microns and picofarads.
        """
        layers, vias, sources = {}, [], []
        for path in paths:
            path = Path(path)
            raw = path.read_bytes()
            sources.append((str(path.resolve()), hashlib.sha256(raw).hexdigest()))
            text = re.sub(r"#[^\n]*", "", raw.decode())
            for match in re.finditer(
                r"^LAYER\s+(\S+)\s*\n(.*?)^END\s+\1\s*$", text, re.MULTILINE | re.DOTALL
            ):
                name, body = match.groups()
                if not re.search(r"\bTYPE\s+ROUTING\s*;", body):
                    continue
                values = []
                for field in (
                    "WIDTH",
                    "RESISTANCE RPERSQ",
                    "CAPACITANCE CPERSQDIST",
                    "EDGECAPACITANCE",
                ):
                    item = re.search(r"(?:^|;)\s*" + field + r"\s+([^\s;]+)\s*;", body)
                    values.append(float(item[1]) if item else math.nan)
                if name in layers and layers[name] != values:
                    raise ValueError(f"Conflicting LEF RC definitions for layer {name}")
                layers[name] = values
            for match in re.finditer(
                r"^VIA\s+(\S+)\s+DEFAULT\s*\n(.*?)^END\s+\1\s*$", text, re.MULTILINE | re.DOTALL
            ):
                name, body = match.groups()
                resistance = re.search(r"\bRESISTANCE\s+([^\s;]+)\s*;", body)
                via_layers = re.findall(r"\bLAYER\s+(\S+)\s*;", body)
                vias.append((name, via_layers, float(resistance[1]) if resistance else math.nan))
        names = tuple(layer_names)
        r, c = np.full(len(names), np.nan), np.full(len(names), np.nan)
        for index, name in enumerate(names):
            if name not in layers:
                continue
            width, sheet_r, area_c, edge_c = layers[name]
            if width > 0:
                r[index] = sheet_r / width
                c[index] = area_c * width + 2 * edge_c
        via_r = np.full(max(0, len(names) - 1), np.nan)
        for index in range(len(via_r)):
            pair = {names[index], names[index + 1]}
            candidates = [value for _, layers_used, value in vias if pair.issubset(layers_used)]
            if candidates and all(math.isfinite(x) and x == candidates[0] for x in candidates):
                via_r[index] = candidates[0]
        for values in (r, c, via_r):
            values.flags.writeable = False
        return cls(names, r, c, via_r, tuple(sources))
