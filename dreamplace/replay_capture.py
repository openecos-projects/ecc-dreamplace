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
        self.interval = int(os.environ.get("ECC_REPLAY_SAMPLE_ITERATIONS", "30"))
        if self.interval <= 0:
            raise ValueError("Replay sampling interval must be positive")

    def due(self, iteration):
        return self.directory is not None and iteration % self.interval == 0

    def sample(self, params, placedb, pos, phase, iteration=None, committed=False, metrics=None):
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
            # Non-interacting terminals are IO ports, even if their LEF shape
            # has nonzero area. They must never become instance lifetimes.
            instance_end = placedb.num_physical_nodes - getattr(placedb, "num_terminal_NIs", 0)
            for i in range(instance_end):
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
                width, height = (int(np.rint(placedb.pydb.node_size_x[i])),
                                 int(np.rint(placedb.pydb.node_size_y[i])))
                if width <= 0 or height <= 0:
                    continue
                objects[key] = {"kind": "macro" if is_macro else "instance", "name": name,
                                "bbox": [int(x[i]), int(y[i]), int(x[i])+width, int(y[i])+height], "placed": True}
            # Keep the physical envelope available even when a flow starts at placement.
            objects["die"] = {"kind": "die", "name": "placement region", "bbox": [
                int(np.rint(placedb.xl/scale+shift[0])), int(np.rint(placedb.yl/scale+shift[1])),
                int(np.rint(placedb.xh/scale+shift[0])), int(np.rint(placedb.yh/scale+shift[1]))]}
            macro = bool(getattr(params, "macro_only", False))
            frame = {"sequence": self.sequence, "elapsed_ns": elapsed, "scope": "macros" if macro else "instances",
                     "phase": phase, "iteration": iteration,
                     "subflow": "save data" if committed else ("macro placement" if macro else "run placement"),
                     "state_domain": "native_committed" if committed else "algorithm_state",
                     "objects": objects, "metrics": {"sampled_instances": len(objects)-1, **(metrics or {})}}
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
        try:
            params._replay_capture = PlacementReplayCapture(directory, int(start))
        except (ValueError, OSError):
            logging.exception("Replay initialization disabled")


def capture(params, placedb, pos, phase, iteration=None, committed=False):
    recorder = getattr(params, "_replay_capture", None)
    if recorder is not None:
        recorder.sample(params, placedb, pos, phase, iteration, committed)


def due(params, iteration):
    recorder = getattr(params, "_replay_capture", None)
    return recorder is not None and recorder.due(iteration)


def capture_native_backend(raw_db, phase, metrics=None):
    """Capture accepted native master/topology transactions without virtual positions."""
    directory = os.environ.get("ECC_REPLAY_CAPTURE_DIR")
    if not directory:
        return
    try:
        stamp = time.monotonic_ns()
        module = raw_db.module
        api = getattr(module, "ecc", module)
        objects = json.loads(api.replay_geometry())
        frame = {"sequence": stamp, "elapsed_ns": stamp - int(os.environ["ECC_REPLAY_START_NS"]),
                 "phase": phase, "subflow": "native commit", "state_domain": "native_committed",
                 "objects": objects, "metrics": metrics or {}}
        path = Path(directory) / f"dp-native-{stamp}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(frame, allow_nan=False))
        temporary.replace(path)
    except Exception:
        logging.exception("Native optimization Replay capture failed")
        try:
            (Path(directory) / "error.json").write_text('{"error":"native optimization capture failed"}')
        except OSError:
            logging.exception("Cannot write Replay failure marker")


def capture_decision(params, phase, summary):
    """Record evaluated virtual decisions without publishing candidate geometry."""
    recorder = getattr(params, "_replay_capture", None)
    if recorder is None or recorder.directory is None:
        return
    try:
        stamp = time.monotonic_ns()
        scalar = {key: value for key, value in summary.items()
                  if isinstance(value, (str, bool, int, float))
                  and (not isinstance(value, float) or np.isfinite(value))}
        rounds = summary.get("rounds")
        if isinstance(rounds, (list, tuple)):
            scalar["evaluated_rounds"] = len(rounds)
        frame = {"sequence": stamp, "elapsed_ns": stamp - recorder.start_ns,
                 "phase": phase, "subflow": "optimization decisions", "scope": "patch",
                 "state_domain": "analysis_state", "objects": {}, "metrics": scalar}
        path = recorder.directory / f"decision-{stamp}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(frame, allow_nan=False))
        temporary.replace(path)
    except Exception:
        logging.exception("Replay decision capture failed")
        try:
            (recorder.directory / "error.json").write_text('{"error":"decision capture failed"}')
        except OSError:
            logging.exception("Cannot write Replay failure marker")
