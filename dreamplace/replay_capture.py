"""Optional placement sampling; independent of plotting and of the ECC Python package."""

import json
import logging
import os
import time
from pathlib import Path

import numpy as np


class PlacementReplayCapture:
    def __init__(self, directory, start_ns):
        self.directory = Path(directory)
        self.start_ns = start_ns
        self.sequence = 0
        self.sizes = {}

    def sample(self, params, placedb, pos, phase, iteration=None, committed=False):
        if self.directory is None:
            return
        try:
            elapsed = time.monotonic_ns() - self.start_ns
            if hasattr(pos, "detach"):
                pos = pos.detach().cpu().numpy()
            pos = np.asarray(pos)
            if not np.isfinite(pos).all():
                raise ValueError("nonfinite placement coordinates")
            # Committed node_x/y exclude virtual fillers; optimizer pos includes them.
            n = len(pos) // 2
            if len(pos) % 2 or n < placedb.num_physical_nodes:
                raise ValueError("invalid placement coordinate vector")
            scale = float(params.scale_factor)
            shift = params.shift_factor
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("invalid placement scale")
            x = np.rint(pos[:n] / scale + shift[0]).astype(np.int64)
            y = np.rint(pos[n:2*n] / scale + shift[1]).astype(np.int64)
            objects = {}
            macro_mask = placedb.node_is_hard_macro
            blockage_count = placedb.num_place_blockages
            blockage_end = placedb.fixed_slice.stop
            for i in range(placedb.num_physical_nodes):
                # Solver obstacles are not design instances.
                if blockage_end - blockage_count <= i < blockage_end:
                    continue
                raw = placedb.node_names[i]
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
                is_macro = bool(macro_mask[i]) if macro_mask is not None and i < len(macro_mask) else False
                if getattr(params, "macro_only", False) and not is_macro:
                    continue
                key = "inst:" + name
                # pydb keeps original DBU master dimensions; optimization arrays
                # may include padding, halos, or congestion inflation.
                self.sizes.setdefault(key, (int(np.rint(placedb.pydb.node_size_x[i])), int(np.rint(placedb.pydb.node_size_y[i]))))
                width, height = self.sizes[key]
                if width <= 0 or height <= 0:
                    continue
                objects[key] = {"kind": "macro" if is_macro else "instance", "name": name,
                                "bbox": [int(x[i]), int(y[i]), int(x[i])+width, int(y[i])+height]}
            # Keep the physical envelope available even when a flow starts at placement.
            objects["die"] = {"kind": "die", "name": "placement region", "bbox": [
                int(np.rint(placedb.xl/scale+shift[0])), int(np.rint(placedb.yl/scale+shift[1])),
                int(np.rint(placedb.xh/scale+shift[0])), int(np.rint(placedb.yh/scale+shift[1]))]}
            macro = bool(getattr(params, "macro_only", False))
            frame = {"sequence": self.sequence, "elapsed_ns": elapsed, "scope": "macros" if macro else "instances",
                     "phase": phase, "iteration": iteration,
                     "subflow": "save data" if committed else ("macro placement" if macro else "run placement"),
                     "state_domain": "native_committed" if committed else "algorithm_state",
                     "objects": objects, "metrics": {"sampled_instances": len(objects)-1}}
            path = self.directory / f"dp-{self.sequence:06d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(frame, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
            self.sequence += 1
        except Exception:
            logging.exception("Replay sampling disabled after capture failure")
            try:
                (self.directory / "error.json").write_text('{"error":"placement capture failed"}')
            except OSError:
                logging.exception("Cannot write Replay failure marker")
            self.directory = None


def initialize(params):
    params._replay_capture = None
    directory = os.environ.get("ECC_REPLAY_CAPTURE_DIR")
    start = os.environ.get("ECC_REPLAY_START_NS")
    if directory and start:
        params._replay_capture = PlacementReplayCapture(directory, int(start))


def capture(params, placedb, pos, phase, iteration=None, committed=False):
    recorder = getattr(params, "_replay_capture", None)
    if recorder is not None:
        recorder.sample(params, placedb, pos, phase, iteration, committed)
