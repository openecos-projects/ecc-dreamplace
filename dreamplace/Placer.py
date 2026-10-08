# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#

import matplotlib
from matplotlib import pyplot as plt
import copy
import json
import hashlib
# matplotlib.use("Agg")
import os
import re
import sys
import time
from typing import Any
import torch
import random
import numpy as np
import logging
import math

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

os.environ['eda_tool'] = "ecc"
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

top_root_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if top_root_dir not in sys.path:
    sys.path.append(top_root_dir)
import dreamplace.configure as configure
from dreamplace.Params import Params
from dreamplace.macroPlaceDB import MacroPlaceDB as PlaceDB
import dreamplace.NonLinearPlace as NonLinearPlace
from dreamplace.ops.openroad_handoff import launcher as handoff_launcher
import dreamplace.placer_cli as placer_cli
from dreamplace.flows.optimization_flow import run_optimization_flow
from dreamplace.flows.flow_config import (
    FlowKind,
    STAGED_JOINT_SIZING_ITERATIONS,
    STAGED_JOINT_SIZING_STAGE_DEFAULTS,
    STAGED_JOINT_TDP_ITERATIONS,
    STAGED_JOINT_TDP_STAGE_DEFAULTS,
)


def _load_placeio_openroad():
    import dreamplace.ops.placeio_openroad.place_io as placeio_openroad

    return placeio_openroad


class _LazyOpenRoadPlaceIO:
    def __getattr__(self, name):
        return getattr(_load_placeio_openroad(), name)


placeio_openroad = _LazyOpenRoadPlaceIO()

# from data_manager.aimp_dm import AimpDataManager


def _load_placeio_ieda():
    """Load the optional AiEDA/iEDA backend only when it is selected."""
    os.environ["eda_tool"] = "iEDA"
    top_root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    aieda_root_dir = os.path.dirname(top_root_dir)
    for path in (top_root_dir, top_root_dir + "/third_party/aieda", aieda_root_dir):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)

    import dreamplace.ops.placeio_ieda.place_io as placeio_ieda

    return placeio_ieda


build_arg_parser = placer_cli.build_arg_parser
build_effective_params_from_args = placer_cli.build_effective_params_from_args

def seed_all(seed, deterministic=False):
    deterministic = bool(deterministic)
    if deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(deterministic)


class PlacementEngine:
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """

    def __init__(self, params: Params):
        # load parameters
        self.params = Params()
        self.update_params(params)
        # self.params.printWelcome()
        if self.params.evaluate_pl == 1:
            self.params.global_place_flag = 1
            self.params.global_place_stages[0]["iteration"] = 0
            self.params.random_center_init_flag = 0
            self.params.enable_fillers = 0
            self.params.detailed_place_flag = 0
            self.params.legalize_flag = 0
            self.params.routability_opt_flag = 0
            self.params.macro_halo_x = 0
            self.params.macro_halo_y = 0
            self.params.macro_pin_halo = 0
            self.params.plot_flag = 1
            self.params.gp_noise_ratio = 0.0
            self.params.pin_density = -1
            logging.critical("running in evaluation mode")

        # seed for reproducibility
        seed_all(
            self.params.random_seed,
            deterministic=getattr(self.params, "deterministic_flag", False),
        )

        # control multithreading
        os.environ["OMP_NUM_THREADS"] = "%d" % (self.params.num_threads)
        torch.set_num_threads(self.params.num_threads)

        assert (not self.params.gpu) or configure.compile_configurations[
            "CUDA_FOUND"
        ] == "TRUE", "CANNOT enable GPU without CUDA compiled"

        self.placedb = None
        self.placer = None

        self.rsmt = float("inf")
        self.hpwl = float("inf")
        self.congestion = float("inf")
        self.density = float("inf")
        self.metrics = None

    def _prepare_ieda_data_manager_for_sta(self, data_manager):
        if (
            not self.params.with_sta
            or getattr(self.params, "place_io_engine", "ieda") in {"openroad", "ecc"}
        ):
            return data_manager
        if hasattr(data_manager, "read_def") and hasattr(data_manager, "get_dmInst_ptr"):
            return data_manager

        ieda_io = _load_placeio_ieda().PlaceIOFunction.make_io(data_manager.dir_workspace)
        design_inputs = getattr(self.params, "design_inputs", {}) or {}
        input_def = getattr(self.params, "def_input", "") or design_inputs.get("def", "")
        ieda_io.read_def(input_def)
        return ieda_io

    def setup_rawdb(self, data_manager):
        # read cpp database
        tt = time.time()
        if self.placedb is None:
            self.data_manager = self._prepare_ieda_data_manager_for_sta(data_manager)
            self.placedb = PlaceDB(self.data_manager)
            if self.params.with_sta and getattr(self.params, "place_io_engine", "ieda") == "ieda":
                ieda_sta_class = _load_placeio_ieda()._load_ieda_sta_class()
                ieda_sta = ieda_sta_class(self.data_manager.dir_workspace)
                ieda_sta.init_sta()
            self.placedb.setup_rawdb(self.params)

        logging.info("setting up raw database takes %.2f seconds" %
                     (time.time() - tt))

    def setup_placedb(self):
        # setup python placement database
        tt = time.time()
        self.placedb.init_db(self.params)
        if self.params.timing_rc_mode == "gr":
            from dreamplace.flows.gr_sizing import prepare_gr_sizing

            prepare_gr_sizing(self.params, self.placedb)
        else:
            self.placedb.gr_sizing = None
        logging.info(
            "setting up placement database takes %.2f seconds" % (
                time.time() - tt)
        )

    def _decode_def_name(self, value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    def _def_component_orient(self, value):
        orient = self._decode_def_name(value).strip()
        orient_map = {
            "": "N",
            "R0": "N",
            "R180": "S",
            "R90": "W",
            "R270": "E",
            "MY": "FN",
            "MX": "FS",
            "MYR90": "FE",
            "MXR90": "FW",
        }
        return orient_map.get(orient, orient)

    def _rewrite_openroad_def_component_placements(
        self,
        def_file,
        node_x,
        node_y,
        *,
        num_movable,
    ):
        path = str(def_file)
        if not os.path.exists(path):
            return {
                "applied": False,
                "reason": "def_missing",
                "rewrite_count": 0,
                "missing_count": int(num_movable),
            }

        raw_node_names = getattr(self.placedb, "node_names", None)
        raw_node_orient = getattr(self.placedb, "node_orient", None)
        node_names = [] if raw_node_names is None else list(raw_node_names)
        node_orient = [] if raw_node_orient is None else list(raw_node_orient)
        movable: dict[str, tuple[int, int, str]] = {}
        for idx in range(int(num_movable)):
            if idx >= len(node_names) or idx >= len(node_x) or idx >= len(node_y):
                continue
            name = self._decode_def_name(node_names[idx])
            orient = "N"
            if idx < len(node_orient):
                orient = self._def_component_orient(node_orient[idx])
            movable[name] = (
                int(round(float(node_x[idx]))),
                int(round(float(node_y[idx]))),
                orient,
            )

        if not movable:
            return {
                "applied": False,
                "reason": "empty_movable_map",
                "rewrite_count": 0,
                "missing_count": int(num_movable),
            }

        with open(path, "r", encoding="utf-8", errors="ignore") as stream:
            lines = stream.readlines()

        in_components = False
        rewritten: list[str] = []
        rewrite_count = 0
        seen: set[str] = set()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("COMPONENTS "):
                in_components = True
                rewritten.append(line)
                continue
            if in_components and stripped.startswith("END COMPONENTS"):
                in_components = False
                rewritten.append(line)
                continue
            if in_components and stripped.startswith("- "):
                parts = stripped.split()
                if len(parts) >= 3:
                    inst_name = parts[1]
                    master_name = parts[2]
                    loc = movable.get(inst_name)
                    if loc is not None:
                        x, y, orient = loc
                        indent = line[: len(line) - len(line.lstrip())]
                        rewritten.append(
                            f"{indent}- {inst_name} {master_name} "
                            f"+ PLACED ( {x} {y} ) {orient} ;\n"
                        )
                        seen.add(inst_name)
                        rewrite_count += 1
                        continue
            rewritten.append(line)

        if rewrite_count:
            with open(path, "w", encoding="utf-8") as stream:
                stream.writelines(rewritten)

        missing_count = max(0, len(movable) - len(seen))
        return {
            "applied": bool(rewrite_count),
            "reason": "" if rewrite_count else "no_matching_components",
            "rewrite_count": int(rewrite_count),
            "missing_count": int(missing_count),
            "movable_count": int(len(movable)),
        }

    def write_back(self, def_file="output.def"):
        place_io_engine = getattr(self.params, "place_io_engine", "ieda")
        if place_io_engine == "openroad":
            import dreamplace.ops.placeio_openroad.place_io as placeio_openroad
            num_movable = self.placedb.num_movable_nodes
            node_x = np.asarray(self.placedb.node_x[:num_movable]).copy()
            node_y = np.asarray(self.placedb.node_y[:num_movable]).copy()
            source = "placedb_node_arrays"
            max_delta_to_placer_pos = 0.0
            placer_pos = getattr(getattr(self, "placer", None), "pos", None)
            # size_only has already committed masters and coordinates through
            # placedb.apply(), which refreshes the OpenROAD-backed node order.
            # The older placer tensor still uses its pre-refresh order.
            use_runtime_pos = (
                getattr(self.params, "placement_sizing_mode", "place_only")
                != "size_only"
            )
            if use_runtime_pos and placer_pos is not None and len(placer_pos) > 0:
                try:
                    pos = placer_pos[0].detach().cpu().numpy()
                    pos_x = np.asarray(pos[:num_movable])
                    pos_y = np.asarray(pos[self.placedb.num_nodes : self.placedb.num_nodes + num_movable])
                    if pos_x.shape == node_x.shape and pos_y.shape == node_y.shape:
                        max_delta_to_placer_pos = float(
                            max(
                                np.max(np.abs(pos_x - node_x)) if node_x.size else 0.0,
                                np.max(np.abs(pos_y - node_y)) if node_y.size else 0.0,
                            )
                        )
                        if np.isfinite(max_delta_to_placer_pos) and max_delta_to_placer_pos > 1e-3:
                            logging.warning(
                                "OpenROAD write_back uses final placer pos because placedb arrays differ by %.6f",
                                max_delta_to_placer_pos,
                            )
                            node_x = pos_x.copy()
                            node_y = pos_y.copy()
                            source = "placer_final_pos"
                except Exception as exc:
                    logging.warning("OpenROAD write_back final-pos consistency check failed: %s", exc)
            # OpenROAD bridge calls OpenDB setLocation and expects DEF DBU.
            # MacroPlaceDB stores placement coordinates in DREAMPlace's shifted
            # and scaled coordinate system, so convert back before export.
            write_node_x, write_node_y = self.placedb.unscale_pl_positions(node_x, node_y)
            write_node_x = np.asarray(write_node_x).round().astype(np.int64, copy=False)
            write_node_y = np.asarray(write_node_y).round().astype(np.int64, copy=False)
            placeio_openroad.PlaceIOFunction.write(
                self.placedb.openroad_bridge,
                def_file,
                placeio_openroad.SolutionFileFormat.DEF,
                write_node_x,
                write_node_y,
            )
            component_rewrite_summary = self._rewrite_openroad_def_component_placements(
                def_file,
                write_node_x,
                write_node_y,
                num_movable=num_movable,
            )
            raw_shift_factor = getattr(self.params, "shift_factor", [])
            if raw_shift_factor is None:
                shift_factor_summary = []
            else:
                shift_factor_summary = list(raw_shift_factor)
            summary = {
                "artifact": "openroad_write_back_summary",
                "artifact_version": 1,
                "component_placement_rewrite": component_rewrite_summary,
                "coordinate_source": "unscaled_def_dbu",
                "def_file": str(def_file),
                "design_name": self._design_name(),
                "max_delta_to_placer_pos": max_delta_to_placer_pos,
                "num_movable_nodes": int(num_movable),
                "scale_factor": float(getattr(self.params, "scale_factor", 0.0) or 0.0),
                "shift_factor": shift_factor_summary,
                "source": source,
            }
            if os.path.exists(def_file):
                summary["def_sha256"] = self._sha256_file(def_file)
                summary["def_bytes"] = os.path.getsize(def_file)
            result_dir = getattr(self.params, "result_dir", None)
            if result_dir:
                self._write_json_artifact(
                    os.path.join(result_dir, f"{self._design_name()}_openroad_write_back_summary.json"),
                    summary,
                )
            self.last_openroad_write_back_summary = summary
        elif place_io_engine == "ecc":
            import dreamplace.ops.placeio_ecc.place_io as placeio_ecc

            placeio_ecc.PlaceIOFunction.write_def(self.placedb.rawdb, def_file)
        else:
            ieda_io = _load_placeio_ieda().PlaceIOFunction.make_io(
                self.data_manager.dir_workspace
            )
            ieda_io.def_save(def_file)

    def write_verilog(self, verilog_file):
        place_io_engine = getattr(self.params, "place_io_engine", "ieda")
        if place_io_engine == "ecc":
            import dreamplace.ops.placeio_ecc.place_io as placeio_ecc

            os.makedirs(os.path.dirname(str(verilog_file)) or ".", exist_ok=True)
            return placeio_ecc.PlaceIOFunction.write_verilog(
                self.placedb.rawdb,
                str(verilog_file),
            )
        if place_io_engine != "openroad":
            raise RuntimeError("Verilog writeback is only supported by the OpenROAD or ECC backend")
        from dreamplace.ops.placeio_openroad.verilog_writeback import write_verilog

        bridge = getattr(self.placedb, "openroad_bridge", None)
        if bridge is None or not hasattr(bridge, "eval_tcl_string"):
            raise RuntimeError("OpenROAD Verilog writeback requires an initialized bridge")
        return write_verilog(bridge, verilog_file)

    def update_params(self, new_params: Params):
        self.params.fromJson(new_params.__dict__)
        # self.params = new_params.clone()
        logging.info("parameters = %s" % (self.params))

    def place(self, *, reuse_existing=False):
        # solve placement
        import dreamplace.NonLinearPlace as NonLinearPlace  # deferred to avoid compiled-op imports at module level
        tt = time.time()
        if self.params.timing_opt_flag:
            raise RuntimeError(
                "timing_opt_flag is no longer supported because OpenTimer integration has been removed"
            )
        if not reuse_existing or self.placer is None:
            timer = None
            self.placer = NonLinearPlace.NonLinearPlace(
                self.params, self.placedb, timer)
            logging.info(
                "non-linear placement initialization takes %.2f seconds"
                % (time.time() - tt)
            )
        self.rsmt, self.hpwl, self.metrics = self.placer(
            self.params, self.placedb)
        logging.info("non-linear placement takes %.2f seconds" %
                     (time.time() - tt))

    def _post_commit_control_result(self):
        metrics = self.metrics
        if isinstance(metrics, dict):
            control = metrics.get("post_commit_control_result")
            if isinstance(control, dict):
                return control
        control = getattr(self.placer, "last_post_commit_control_result", None)
        return control if isinstance(control, dict) else None

    def _physical_mutation_backend(self):
        from dreamplace.ops.placeio_common.physical_mutation import (
            physical_mutation_backend_for,
        )

        return physical_mutation_backend_for(self.placedb)

    def _query_segment_joint_pre_buffer_opensta(self):
        """Refresh placement parasitics before measuring the buffer-free state."""
        params = getattr(self, "params", None)
        if params is None or not self._is_segment_direct_joint():
            return {
                "status": "disabled",
                "reason": "not_segment_direct_joint",
            }
        started_at = time.perf_counter()
        self._openroad_tcl("estimate_parasitics -placement")
        metrics = dict(
            placeio_openroad.PlaceIOFunction.query_diff_guided_batch_timing_metrics(
                self.placedb.openroad_bridge
            )
            or {}
        )
        timing_refresh = dict(metrics.get("timing_refresh", {}) or {})
        if (
            str(metrics.get("status") or "") != "ok"
            or str(timing_refresh.get("status") or "") != "ok"
            or metrics.get("wns") is None
            or metrics.get("tns") is None
        ):
            raise RuntimeError(
                "segment joint pre-buffer OpenSTA refresh failed: " + str(metrics)
            )
        return {
            "status": "ok",
            "parasitic_refresh": "estimate_parasitics -placement",
            "timing_refresh": timing_refresh,
            "wns": float(metrics["wns"]),
            "tns": float(metrics["tns"]),
            "runtime_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    def _run_segment_joint_buffer_preflight(
        self,
        request,
        *,
        output_dir,
        source_baseline=None,
    ):
        """Measure the final buffer batch on a disposable OpenROAD design."""
        params = getattr(self, "params", None)
        if params is None or not self._is_segment_direct_joint():
            return {
                "status": "disabled",
                "reason": "not_segment_direct_joint",
                "qor_pass": True,
            }
        actions = tuple(getattr(request, "actions", ()) or ())
        if not actions:
            return {
                "status": "skipped",
                "reason": "no_buffer_actions",
                "qor_pass": True,
            }

        result_dir = str(getattr(params, "result_dir", None) or output_dir or "")
        if not result_dir:
            raise RuntimeError("segment joint buffer preflight requires an output directory")
        os.makedirs(result_dir, exist_ok=True)
        pre_buffer_def = os.path.join(
            result_dir,
            f"{self._design_name()}_segment_joint_pre_buffer.def",
        )
        bridge = getattr(self.placedb, "openroad_bridge", None)
        if bridge is None or not hasattr(bridge, "write_def"):
            raise RuntimeError("segment joint buffer preflight requires OpenROAD DEF export")
        bridge.write_def(pre_buffer_def)

        design_inputs = copy.deepcopy(dict(getattr(params, "design_inputs", {}) or {}))
        if not design_inputs:
            raise RuntimeError("segment joint buffer preflight requires resolved design_inputs")
        source_def_path = design_inputs.get("def")
        for key in ("verilog", "netlist", "verilog_input"):
            design_inputs.pop(key, None)
        design_inputs["def"] = pre_buffer_def
        replay_params = copy.copy(params)
        replay_params.design_inputs = design_inputs
        replay_params.def_input = pre_buffer_def
        replay_params.verilog_input = ""

        started_at = time.perf_counter()
        replay_bridge = placeio_openroad.PlaceIOFunction.read(replay_params)
        if hasattr(replay_bridge, "eval_tcl_string"):
            replay_bridge.eval_tcl_string("estimate_parasitics -placement")
        from dreamplace.ops.buffer_insertion.coordinate_backend import (
            commit_coordinate_buffer_actions,
        )

        preflight_output_dir = os.path.join(result_dir, "segment_joint_buffer_preflight")
        trial = dict(
            commit_coordinate_buffer_actions(
                actions,
                openroad_bridge=replay_bridge,
                output_dir=preflight_output_dir,
                realization_config={"disposable_design": True},
            )
            or {}
        )
        accepted_count = int(trial.get("accepted_action_count", 0) or 0)
        all_actions_realized = accepted_count == len(actions)
        qor_pass = bool(trial.get("qor_pass", False))
        backend_pass = (
            str(trial.get("status") or "") == "accepted"
            and int(trial.get("failed_action_count", 0) or 0) == 0
            and all_actions_realized
        )
        baseline_gap = None
        baseline_match = True
        if isinstance(source_baseline, dict) and source_baseline.get("status") == "ok":
            baseline_gap = {
                "wns": float(trial.get("before_wns"))
                - float(source_baseline["wns"]),
                "tns": float(trial.get("before_tns"))
                - float(source_baseline["tns"]),
            }
            baseline_match = (
                abs(baseline_gap["wns"]) <= 1.0e-3
                and abs(baseline_gap["tns"]) <= 1.0e-3
            )
        passed = backend_pass and qor_pass and baseline_match
        if not baseline_match:
            reason = "preflight_source_baseline_mismatch"
        elif not backend_pass:
            reason = "preflight_backend_or_action_failure"
        elif not qor_pass:
            reason = "preflight_opensta_qor_rejected"
        else:
            reason = "preflight_opensta_qor_passed"
        payload = {
            "artifact_version": 1,
            "artifact_scope": "segment_joint_buffer_commit_preflight",
            "status": (
                "pass"
                if passed
                else "failed"
                if not baseline_match
                else "rejected"
            ),
            "reason": reason,
            "qor_pass": passed,
            "source_def_path": source_def_path,
            "pre_buffer_def_path": pre_buffer_def,
            "pre_buffer_def_sha256": self._sha256_file(pre_buffer_def),
            "trial_action_count": len(actions),
            "trial_accepted_action_count": accepted_count,
            "all_actions_realized": all_actions_realized,
            "before_wns": trial.get("before_wns"),
            "before_tns": trial.get("before_tns"),
            "after_wns": trial.get("after_wns"),
            "after_tns": trial.get("after_tns"),
            "delta_wns": trial.get("delta_wns"),
            "delta_tns": trial.get("delta_tns"),
            "trial_qor_status": trial.get("qor_status"),
            "trial_status": trial.get("status"),
            "trial_artifact_path": trial.get("artifact_path"),
            "source_baseline": dict(source_baseline or {}),
            "source_baseline_gap": baseline_gap,
            "source_baseline_match_atol_ps": 1.0e-3,
            "source_baseline_match": baseline_match,
            "runtime_ms": (time.perf_counter() - started_at) * 1000.0,
        }
        artifact_path = os.path.join(
            result_dir,
            f"{self._design_name()}_segment_joint_buffer_preflight.json",
        )
        self._write_json_artifact(artifact_path, payload)
        payload["artifact_path"] = artifact_path
        return payload

    @staticmethod
    def _decode_master_name(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def _verify_segment_joint_final_sizing_masters(self, final_def_path, request):
        actions = tuple(getattr(request, "sizing_actions", ()) or ())
        if not actions:
            return {
                "status": "skipped",
                "reason": "no_sizing_actions",
                "expected_count": 0,
                "matched_count": 0,
                "missing_count": 0,
                "mismatch_count": 0,
            }
        if not final_def_path or not os.path.isfile(final_def_path):
            raise RuntimeError("segment joint final DEF is unavailable for sizing audit")

        raw_node_names = getattr(self.placedb, "node_names", None)
        node_names = [] if raw_node_names is None else list(raw_node_names)
        expected = {}
        for action in actions:
            instance_id = int(action.get("instance_id", -1))
            if instance_id < 0 or instance_id >= len(node_names):
                raise RuntimeError(
                    "segment joint sizing action has invalid instance ID: "
                    f"{instance_id}"
                )
            instance_name = str(action.get("instance_name") or "")
            if not instance_name:
                instance_name = self._decode_def_name(node_names[instance_id])
            master_name = self._decode_master_name(action.get("after_master", ""))
            if not instance_name or not master_name:
                raise RuntimeError(
                    "segment joint sizing action is missing instance/master identity"
                )
            if instance_name in expected:
                raise RuntimeError(
                    "segment joint sizing action repeats instance: " + instance_name
                )
            expected[instance_name] = master_name

        observed = {}
        in_components = False
        with open(final_def_path, "r", encoding="utf-8", errors="ignore") as stream:
            for line in stream:
                stripped = line.strip()
                if stripped.startswith("COMPONENTS "):
                    in_components = True
                    continue
                if in_components and stripped.startswith("END COMPONENTS"):
                    break
                if not in_components or not stripped.startswith("- "):
                    continue
                parts = stripped.split()
                if len(parts) >= 3 and parts[1] in expected:
                    observed[parts[1]] = parts[2]

        missing = sorted(set(expected) - set(observed))
        mismatches = sorted(
            (name, expected[name], observed[name])
            for name in expected.keys() & observed.keys()
            if expected[name] != observed[name]
        )
        if missing or mismatches:
            raise RuntimeError(
                "segment joint final DEF sizing masters do not match request: "
                f"missing={len(missing)}, mismatched={len(mismatches)}"
            )

        digest = hashlib.sha256()
        for instance_name in sorted(expected):
            digest.update(
                f"{instance_name}:{observed[instance_name]};".encode("utf-8")
            )
        return {
            "status": "ok",
            "expected_count": len(expected),
            "matched_count": len(observed),
            "missing_count": 0,
            "mismatch_count": 0,
            "request_action_digest": getattr(request, "sizing_action_digest", None),
            "matched_target_digest": "sha256:" + digest.hexdigest(),
        }

    def _execute_segment_joint_physical_actions(
        self,
        request,
        *,
        mutation_backend,
        output_dir,
    ):
        trace = []
        if request.placement_node_x is None or request.placement_node_y is None:
            raise RuntimeError("segment joint final request has no placement snapshot")
        coordinate_started_at = time.perf_counter()
        node_x = np.asarray(request.placement_node_x, dtype=self.placedb.dtype)
        node_y = np.asarray(request.placement_node_y, dtype=self.placedb.dtype)
        placeio_openroad.PlaceIOFunction.apply(
            self.placedb.openroad_bridge,
            node_x,
            node_y,
        )
        trace.append(
            {
                "stage": "coordinate_sync",
                "status": "ok",
                "count": int(node_x.size),
                "runtime_ms": float(
                    (time.perf_counter() - coordinate_started_at) * 1000.0
                ),
            }
        )

        sizing_result = None
        if request.sizing_actions:
            if request.sizing_cell_ids is None:
                raise RuntimeError("segment joint sizing actions have no final cell IDs")
            sizing_started_at = time.perf_counter()
            cell_ids = np.asarray(request.sizing_cell_ids, dtype=np.int32)
            master_names = [
                self._decode_master_name(name)
                for name in self.placedb.flat_libcell_names
            ]
            sizing_result = placeio_openroad.PlaceIOFunction.apply_sizing(
                self.placedb.openroad_bridge,
                cell_ids,
                master_names,
            )
            sizing_result = dict(sizing_result or {})
            if int(sizing_result.get("applied", -1)) != len(
                request.sizing_actions
            ):
                raise RuntimeError(
                    "segment joint sizing apply count does not match request: "
                    f"expected {len(request.sizing_actions)}, got "
                    f"{sizing_result.get('applied')}"
                )
            if int(sizing_result.get("missing_masters", 0) or 0) != 0 or int(
                sizing_result.get("invalid_cell_ids", 0) or 0
            ) != 0:
                raise RuntimeError(
                    "segment joint sizing apply reported invalid masters or cell IDs"
                )
            trace.append(
                {
                    "stage": "apply_sizing",
                    "status": "ok",
                    "action_count": len(request.sizing_actions),
                    "runtime_ms": float(
                        (time.perf_counter() - sizing_started_at) * 1000.0
                    ),
                }
            )

        if request.actions:
            buffer_started_at = time.perf_counter()
            source_baseline = self._query_segment_joint_pre_buffer_opensta()
            preflight = self._run_segment_joint_buffer_preflight(
                request,
                output_dir=output_dir,
                source_baseline=source_baseline,
            )
            if str(preflight.get("status") or "") == "rejected":
                committed_def_path = str(
                    getattr(request, "committed_def_path", "") or ""
                )
                if committed_def_path:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(committed_def_path)),
                        exist_ok=True,
                    )
                    self.placedb.openroad_bridge.write_def(committed_def_path)
                commit_summary = {
                    "status": "rejected",
                    "reason": "segment_joint_buffer_preflight_rejected",
                    "qor_status": preflight.get("trial_qor_status"),
                    "qor_pass": False,
                    "action_count": len(request.actions),
                    "attempted_action_count": 0,
                    "accepted_action_count": 0,
                    "rejected_action_count": len(request.actions),
                    "failed_action_count": 0,
                    "skipped_action_count": 0,
                    "before_wns": preflight.get("before_wns"),
                    "before_tns": preflight.get("before_tns"),
                    "artifact_path": "",
                    "records": [],
                    "preflight_rejected": True,
                    "preflight": preflight,
                    "committed_def_path": committed_def_path or None,
                    "refresh_required": False,
                    "refresh_mode": request.refresh_mode,
                    "rebuild_mode": request.rebuild_mode,
                    "continuation_policy": "preserve_placement_sizing_state",
                }
            elif str(preflight.get("status") or "") in {"pass", "disabled"}:
                commit_summary = dict(
                    mutation_backend.commit_buffers(
                        request,
                        output_dir=output_dir,
                    )
                    or {}
                )
                commit_summary["preflight"] = preflight
                commit_summary["source_pre_buffer_opensta"] = source_baseline
                if (
                    commit_summary.get("before_wns") is not None
                    and commit_summary.get("before_tns") is not None
                    and source_baseline.get("status") == "ok"
                ):
                    commit_summary["source_before_requery_gap"] = {
                        "wns": float(commit_summary["before_wns"])
                        - float(source_baseline["wns"]),
                        "tns": float(commit_summary["before_tns"])
                        - float(source_baseline["tns"]),
                    }
            else:
                raise RuntimeError(
                    "segment joint buffer preflight did not produce a terminal decision: "
                    + str(preflight)
                )
            trace.append(
                {
                    "stage": "insert_buffers",
                    "status": (
                        "preflight_rejected"
                        if commit_summary.get("preflight_rejected")
                        else str(commit_summary.get("status") or "unknown")
                    ),
                    "action_count": len(request.actions),
                    "accepted_action_count": int(
                        commit_summary.get("accepted_action_count", 0) or 0
                    ),
                    "preflight_status": preflight.get("status"),
                    "preflight_delta_wns": preflight.get("delta_wns"),
                    "preflight_delta_tns": preflight.get("delta_tns"),
                    "source_pre_buffer_wns": source_baseline.get("wns"),
                    "source_pre_buffer_tns": source_baseline.get("tns"),
                    "runtime_ms": float(
                        (time.perf_counter() - buffer_started_at) * 1000.0
                    ),
                }
            )
        else:
            commit_summary = {
                "status": "accepted" if request.sizing_actions else "noop",
                "reason": "sizing_only" if request.sizing_actions else "no_actions",
                "accepted_action_count": 0,
                "records": [],
                "refresh_mode": request.refresh_mode,
                "rebuild_mode": request.rebuild_mode,
            }
        commit_summary.update(
            {
                "physical_action_trace": trace,
                "placement_snapshot_digest": request.placement_snapshot_digest,
                "sizing_action_count": len(request.sizing_actions),
                "sizing_action_digest": request.sizing_action_digest,
                "sizing_area_delta_internal": float(
                    sum(
                        float(action.get("area_delta_internal") or 0.0)
                        for action in request.sizing_actions
                    )
                ),
                "sizing_result": sizing_result,
                "buffer_action_count": len(request.actions),
            }
        )
        return commit_summary

    def _execute_pending_joint_buffer_commit(self, control_result):
        if not isinstance(control_result, dict) or control_result.get("status") != (
            "pending_buffer_commit_request"
        ):
            return None

        request_summary = dict(control_result.get("commit_request", {}) or {})
        request = getattr(self.placer, "last_buffer_commit_request", None)
        coordinator = getattr(self.placer, "joint_coordinator", None)
        buffering_lane = getattr(coordinator, "buffering_lane", None)
        if buffering_lane is None:
            buffering_lane = getattr(self.placer, "last_buffering_lane", None)

        def fail(reason, error=None):
            result = {
                "status": "failed",
                "reason": str(reason),
                "topology_mutated": False,
                "refresh_required": False,
                "continuation_policy": "terminate_outer_flow",
                "commit_request": request_summary,
            }
            if error is not None:
                result["error"] = str(error)
            if isinstance(self.metrics, dict):
                self.metrics["post_commit_control_result"] = dict(result)
            self.last_outer_buffer_commit_result = dict(result)
            return result

        if request is None:
            return fail("missing_buffer_commit_request")

        try:
            mutation_backend = self._physical_mutation_backend()
            output_dir = getattr(
                getattr(buffering_lane, "config", None),
                "output_dir",
                getattr(self.params, "result_dir", None),
            )
            if self._is_segment_direct_joint():
                commit_summary = self._execute_segment_joint_physical_actions(
                    request,
                    mutation_backend=mutation_backend,
                    output_dir=output_dir,
                )
            elif getattr(request, "placement_node_x", None) is not None:
                from dreamplace.flows.joint_physical_actions import commit_joint_physical_actions

                commit_summary = commit_joint_physical_actions(
                    self, request, mutation_backend=mutation_backend, output_dir=output_dir,
                )
            else:
                commit_summary = mutation_backend.commit_buffers(
                    request,
                    output_dir=output_dir,
                )
            self.last_physical_mutation_backend = mutation_backend
        except Exception as exc:
            return fail("buffer_commit_executor_exception", exc)

        request_summary = request.to_summary()
        execution_result = {
            "status": "completed",
            "iteration": request_summary.get("iteration"),
            "projected_buffer_count": request_summary.get("action_count", 0),
            "runtime_refresh": dict(
                request_summary.get("runtime_refresh_summary", {}) or {}
            ),
            "commit_request": request_summary,
            "commit": dict(commit_summary or {}),
        }
        outer_result = None
        record = getattr(self.placer, "record_outer_buffer_commit_result", None)
        if callable(record):
            outer_result = record(execution_result)
        if not isinstance(outer_result, dict):
            accepted_action_count = int(
                commit_summary.get("accepted_action_count", 0) or 0
            )
            topology_mutated = (
                str(commit_summary.get("status") or "") == "accepted"
                and accepted_action_count > 0
            )
            outer_result = {
                "status": (
                    "safe_stopped_after_accepted_buffer_commit"
                    if topology_mutated
                    else "completed"
                ),
                "topology_mutated": topology_mutated,
                "refresh_required": topology_mutated,
                "refresh_mode": commit_summary.get("refresh_mode", "topo"),
                "rebuild_mode": commit_summary.get("rebuild_mode", "topo"),
                "continuation_policy": (
                    "rebuild_placer_outer_loop"
                    if topology_mutated
                    else "continue_same_topology"
                ),
                "commit_request": request_summary,
                "commit": dict(commit_summary or {}),
            }

        commit_status = str(commit_summary.get("status") or "unknown")
        if commit_status in {"blocked", "failed", "unsupported"}:
            outer_result = dict(outer_result)
            outer_result.update(
                {
                    "status": "failed",
                    "reason": str(
                        commit_summary.get("reason") or "buffer_commit_failed"
                    ),
                    "topology_mutated": False,
                    "refresh_required": False,
                    "continuation_policy": "terminate_outer_flow",
                }
            )
        if isinstance(self.metrics, dict):
            self.metrics["post_commit_control_result"] = dict(outer_result)
        self.last_outer_buffer_commit_result = dict(outer_result)
        return outer_result

    def _joint_buffer_outer_iterations(self):
        return max(
            1,
            int(getattr(self.params, "joint_buffer_outer_iterations", 1) or 1),
        )

    def _is_segment_direct_joint(self):
        return (
            str(getattr(self.params, "flow_kind", "") or "") == FlowKind.JOINT.value
            and self._joint_quality_profile() == "segment_count_direct_joint_v1"
        )

    def _run_segment_direct_joint_outer_flow(self):
        """Run virtual placement/buffer windows before one physical commit."""
        outer_windows = []
        outer_count = self._joint_buffer_outer_iterations()
        for window_index in range(outer_count):
            self.place(reuse_existing=window_index > 0)
            placement_debug = dict(
                getattr(self.placer, "last_placement_debug_summary", None) or {}
            )
            request = getattr(self.placer, "last_buffer_commit_request", None)
            request_summary = request.to_summary() if request is not None else {}
            if window_index + 1 < outer_count:
                outer_windows.append(
                    {
                        "window_index": int(window_index),
                        "status": "virtual_completed",
                        "boundary": "virtual",
                        "request_action_digest": request_summary.get("action_digest"),
                        "projected_buffer_count": request_summary.get("action_count", 0),
                        "actual_placement_steps": placement_debug.get(
                            "optimizer_steps"
                        ),
                        "physical_commit": False,
                    }
                )
                continue

            control_result = self._post_commit_control_result()
            if control_result is None and request is not None and request.commit_enabled:
                control_result = {
                    "status": "pending_buffer_commit_request",
                    "topology_mutated": False,
                    "refresh_required": False,
                    "refresh_mode": request.refresh_mode,
                    "rebuild_mode": request.rebuild_mode,
                    "continuation_policy": "execute_in_placement_engine",
                    "accepted_action_count": 0,
                    "commit_request": request_summary,
                }
            outer_commit_result = self._execute_pending_joint_buffer_commit(
                control_result
            )
            if outer_commit_result is not None:
                control_result = outer_commit_result
            if isinstance(control_result, dict) and control_result.get("status") == "failed":
                outer_windows.append(
                    {
                        "window_index": int(window_index),
                        "status": "failed",
                        "boundary": "final",
                        "commit": dict(control_result),
                    }
                )
                self.last_run_result = {
                    "status": "failed",
                    "stop_reason": "buffer_commit_execution_failed",
                    "joint_buffer_commit": dict(control_result),
                    "outer_windows": outer_windows,
                }
                self.last_run_result["outer_lifecycle_artifact_path"] = (
                    self._write_joint_outer_lifecycle_artifact(
                        status="failed",
                        windows=outer_windows,
                        result=self.last_run_result,
                    )
                )
                return self.last_run_result

            post_commit_result = self._continue_after_joint_buffer_commit(control_result)
            outer_windows.append(
                {
                    "window_index": int(window_index),
                        "status": "completed",
                        "boundary": "final",
                        "physical_commit": bool(outer_commit_result is not None),
                        "request_action_digest": request_summary.get("action_digest"),
                        "projected_buffer_count": request_summary.get("action_count", 0),
                        "actual_placement_steps": placement_debug.get(
                            "optimizer_steps"
                        ),
                        "post_commit": dict(post_commit_result or {}),
                }
            )
            self.last_run_result = {
                "status": "ok",
                "rsmt": self.rsmt,
                "hpwl": placement_debug.get("final_hpwl", self.hpwl),
                "congestion": self.congestion,
                "density": placement_debug.get("final_max_density", self.density),
                "iteration": placement_debug.get("optimizer_steps", 0),
                "objective": -1,
                "overflow": placement_debug.get("final_overflow", -1),
                "max_density": placement_debug.get("final_max_density", -1),
                "joint_post_rebuild": post_commit_result,
                "outer_windows": outer_windows,
            }
            self.last_run_result["outer_lifecycle_artifact_path"] = (
                self._write_joint_outer_lifecycle_artifact(
                    status="completed",
                    windows=outer_windows,
                    result=self.last_run_result,
                )
            )
            return self.last_run_result
        raise RuntimeError("segment direct joint outer flow ended without a final window")

    def _run_segment_direct_joint_continuous_outer_flow(self):
        """Run virtual windows inside one placement optimizer lifecycle."""
        window_steps = int(
            getattr(self.params, "joint_segment_virtual_window_steps", 0) or 0
        )
        outer_count = self._joint_buffer_outer_iterations()
        warmup_steps = int(
            getattr(self.params, "joint_segment_virtual_warmup_steps", 0) or 0
        )
        stages = list(getattr(self.params, "global_place_stages", None) or ())
        configured_steps = sum(
            int(stage.get("iteration", 0))
            * int(stage.get("Llambda_density_weight_iteration", 1))
            * int(stage.get("Lsub_iteration", 1))
            for stage in stages
            if isinstance(stage, dict)
        )
        expected_steps = window_steps * outer_count + warmup_steps
        if window_steps <= 0 or configured_steps != expected_steps:
            raise ValueError(
                "continuous segment joint windows require configured optimizer "
                f"steps={expected_steps}, got {configured_steps}"
            )

        self.place()
        coordinator = getattr(self.placer, "joint_coordinator", None)
        coordinator_summary = (
            coordinator.summarize() if coordinator is not None else {}
        )
        outer_windows = copy.deepcopy(
            coordinator_summary.get("segment_virtual_windows", [])
        )
        request = getattr(self.placer, "last_buffer_commit_request", None)
        request_summary = request.to_summary() if request is not None else {}
        control_result = self._post_commit_control_result()
        if control_result is None and request is not None and request.commit_enabled:
            control_result = {
                "status": "pending_buffer_commit_request",
                "topology_mutated": False,
                "refresh_required": False,
                "refresh_mode": request.refresh_mode,
                "rebuild_mode": request.rebuild_mode,
                "continuation_policy": "execute_in_placement_engine",
                "accepted_action_count": 0,
                "commit_request": request_summary,
            }
        outer_commit_result = self._execute_pending_joint_buffer_commit(
            control_result
        )
        if outer_commit_result is not None:
            control_result = outer_commit_result
        if isinstance(control_result, dict) and control_result.get("status") == "failed":
            self.last_run_result = {
                "status": "failed",
                "stop_reason": "buffer_commit_execution_failed",
                "joint_buffer_commit": dict(control_result),
                "outer_windows": outer_windows,
            }
            self.last_run_result["outer_lifecycle_artifact_path"] = (
                self._write_joint_outer_lifecycle_artifact(
                    status="failed",
                    windows=outer_windows,
                    result=self.last_run_result,
                )
            )
            return self.last_run_result

        post_commit_result = self._continue_after_joint_buffer_commit(control_result)
        if not outer_windows:
            outer_windows = [
                {
                    "window_index": outer_count - 1,
                    "status": "missing_virtual_window_trace",
                    "boundary": "final",
                    "actual_placement_steps": configured_steps,
                }
            ]
        outer_windows[-1].update(
            {
                "physical_commit": bool(outer_commit_result is not None),
                "request_action_digest": request_summary.get("action_digest"),
                "projected_buffer_count": request_summary.get("action_count", 0),
                "post_commit": dict(post_commit_result or {}),
            }
        )
        self.last_run_result = {
            "status": "ok",
            "iteration": configured_steps,
            "joint_post_rebuild": post_commit_result,
            "outer_windows": outer_windows,
        }
        self.last_run_result["outer_lifecycle_artifact_path"] = (
            self._write_joint_outer_lifecycle_artifact(
                status="completed",
                windows=outer_windows,
                result=self.last_run_result,
            )
        )
        return self.last_run_result

    def _write_joint_outer_lifecycle_artifact(self, *, status, windows, result=None):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        path = os.path.join(
            result_dir,
            f"{self._design_name()}_joint_outer_lifecycle_summary.json",
        )
        self._write_json_artifact(
            path,
            {
                "artifact_version": 1,
                "artifact_scope": "joint_buffer_outer_lifecycle",
                "status": str(status),
                "configured_outer_iterations": self._joint_buffer_outer_iterations(),
                "windows": [dict(window) for window in windows],
                "result": dict(result or {}),
            },
        )
        return path

    def _design_name(self):
        design_name_getter = getattr(self.params, "design_name", None)
        if callable(design_name_getter):
            return design_name_getter()
        return getattr(self.params, "base_design_name", "unknown_design")

    def _write_json_artifact(self, path, payload):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")

    def _update_json_artifact(self, path, updater):
        if not path or not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        updater(payload)
        self._write_json_artifact(path, payload)
        return True

    def _openroad_tcl(self, cmd):
        bridge = getattr(self.placedb, "openroad_bridge", None)
        if bridge is None:
            raise RuntimeError("OpenROAD ECO requested but placedb has no openroad_bridge")
        return placeio_openroad.PlaceIOFunction.eval_tcl_string(bridge, cmd)

    def _checked_openroad_tcl(self, label, command):
        started_at = time.perf_counter()
        catch_code = self._openroad_tcl(
            "set __autodmp_command_code "
            f"[catch {{{command}}} __autodmp_command_output]; "
            "set __autodmp_command_code"
        )
        output = self._openroad_tcl("set __autodmp_command_output")
        try:
            return_code = int(str(catch_code).strip())
        except (TypeError, ValueError):
            return_code = None
        return {
            "label": label,
            "command": command,
            "status": "ok" if return_code == 0 else "failed",
            "tcl_return_code": return_code,
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
            "output": output,
        }

    @staticmethod
    def _full_core_detailed_placement_command():
        return "; ".join(
            (
                "set __autodmp_dpl_block [ord::get_db_block]",
                "set __autodmp_dpl_core [$__autodmp_dpl_block getCoreArea]",
                "set __autodmp_dpl_dbu_per_micron "
                "[$__autodmp_dpl_block getDbUnitsPerMicron]",
                "set __autodmp_dpl_max_x_um "
                "[expr {max(1, int(ceil(double([$__autodmp_dpl_core xMax] - "
                "[$__autodmp_dpl_core xMin]) / "
                "double($__autodmp_dpl_dbu_per_micron))))}]",
                "set __autodmp_dpl_max_y_um "
                "[expr {max(1, int(ceil(double([$__autodmp_dpl_core yMax] - "
                "[$__autodmp_dpl_core yMin]) / "
                "double($__autodmp_dpl_dbu_per_micron))))}]",
                "detailed_placement -max_displacement "
                "[list $__autodmp_dpl_max_x_um $__autodmp_dpl_max_y_um]",
            )
        )

    def _write_joint_post_commit_openroad_eco_summary(self, payload):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        design_name = self._design_name()
        path = os.path.join(
            result_dir,
            f"{design_name}_joint_post_commit_openroad_eco_summary.json",
        )
        self._write_json_artifact(path, payload)
        return path

    def _write_joint_staged_smoke_summary(self, payload):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        design_name = self._design_name()
        path = os.path.join(
            result_dir,
            f"{design_name}_joint_staged_smoke_summary.json",
        )
        self._write_json_artifact(path, payload)
        return path

    def _sha256_file(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _joint_staged_smoke_final_def_path(self):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        design_name = self._design_name()
        return os.path.join(result_dir, f"{design_name}_staged_smoke_v1_final.def")

    def _run_joint_post_commit_openroad_eco(self, control_result):
        mode = str(getattr(self.params, "joint_post_commit_openroad_eco", "none") or "none")
        if mode in ("none", "off", "disabled", "0", "false"):
            return {"status": "disabled", "mode": mode}
        if mode != "resynth_once":
            raise RuntimeError(f"unsupported joint_post_commit_openroad_eco: {mode}")
        commit = dict((control_result or {}).get("commit", {}) or {})
        commands = [
            (
                "before_update_timing",
                "update_timing",
            ),
            (
                "before_report_checks_json",
                "report_checks -path_delay max -slack_max 0 -sort_by_slack "
                "-group_path_count 20 -endpoint_path_count 1 -format json",
            ),
            (
                "before_report_worst_slack",
                "report_worst_slack -max",
            ),
            (
                "before_report_tns",
                "report_tns -max",
            ),
            (
                "resynth",
                "resynth",
            ),
            (
                "after_update_timing",
                "update_timing",
            ),
            (
                "after_report_checks_json",
                "report_checks -path_delay max -slack_max 0 -sort_by_slack "
                "-group_path_count 20 -endpoint_path_count 1 -format json",
            ),
            (
                "after_report_worst_slack",
                "report_worst_slack -max",
            ),
            (
                "after_report_tns",
                "report_tns -max",
            ),
        ]
        started = time.time()
        results = []
        payload = {
            "artifact_version": 1,
            "artifact_scope": "joint_post_commit_openroad_eco",
            "design_name": self._design_name(),
            "flow_kind": getattr(self.params, "flow_kind", None),
            "mode": mode,
            "commit_status": commit.get("status"),
            "accepted_action_count": commit.get("accepted_action_count"),
            "commands": results,
        }
        failed_error = None
        try:
            for label, command in commands:
                command_start = time.time()
                output = self._openroad_tcl(command)
                results.append(
                    {
                        "label": label,
                        "command": command,
                        "status": "ok",
                        "elapsed_ms": (time.time() - command_start) * 1000.0,
                        "output": output,
                    }
                )
            payload["status"] = "ok"
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = str(exc)
            payload["failed_command_index"] = len(results)
            failed_error = exc
        payload["elapsed_ms"] = (time.time() - started) * 1000.0
        summary_path = self._write_joint_post_commit_openroad_eco_summary(payload)
        if summary_path:
            payload["summary_path"] = summary_path
        if failed_error is not None:
            raise failed_error
        return payload

    def _write_buffer_commit_refresh_summary(
        self,
        control_result,
        refresh_summary,
        post_rebuild_result,
    ):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        design_name = self._design_name()
        commit = dict((control_result or {}).get("commit", {}) or {})
        rebuild = dict((refresh_summary or {}).get("rebuild", {}) or {})
        committed_buffer_names = self._committed_buffer_names(commit)
        accepted_action_count = commit.get("accepted_action_count")
        try:
            accepted_action_count_int = int(accepted_action_count or 0)
        except (TypeError, ValueError):
            accepted_action_count_int = 0
        path = os.path.join(
            result_dir,
            f"{design_name}_buffer_commit_pydb_refresh_summary.json",
        )
        payload = {
            "artifact_version": 1,
            "artifact_scope": "buffer_commit_pydb_refresh",
            "design_name": design_name,
            "flow_kind": getattr(self.params, "flow_kind", None),
            "commit_status": commit.get("status"),
            "accepted_action_count": accepted_action_count,
            "before_counts": rebuild.get("pre_counts"),
            "after_counts": rebuild.get("post_counts"),
            "committed_buffer_names": committed_buffer_names,
            "committed_buffer_names_found": (
                len(committed_buffer_names) >= accepted_action_count_int
                if accepted_action_count_int > 0
                else True
            ),
            "committed_def_path": commit.get("committed_def_path"),
            "opensta_before": {
                "wns": commit.get("before_wns"),
                "tns": commit.get("before_tns"),
            },
            "opensta_after": {
                "wns": commit.get("after_wns"),
                "tns": commit.get("after_tns"),
            },
            "delta_wns": commit.get("delta_wns"),
            "delta_tns": commit.get("delta_tns"),
            "refresh_status": (refresh_summary or {}).get("status"),
            "refresh_generation": rebuild.get("topology_generation"),
            "refresh_summary": dict(refresh_summary or {}),
            "openroad_eco_summary_path": (post_rebuild_result or {}).get(
                "openroad_eco_summary_path"
            ),
            "rebuilt_payloads": [
                "pydb_topology",
                "macroPlaceDB_topology",
                "NonLinearPlace",
                "data_collections",
                "op_collections",
                "plain_pysta_timing",
            ],
            "invalidated_payloads": [
                "pre_commit_NonLinearPlace",
                "pre_commit_optimizer_param_groups",
                "pre_commit_relaxed_buffer_state",
                "pre_commit_dynamic_net_arc_payload",
                "pre_commit_buffer_candidates",
            ],
            "freshness_checks": {
                "fresh_nonlinear_place_constructed": bool(
                    (post_rebuild_result or {}).get(
                        "fresh_nonlinear_place_constructed"
                    )
                ),
                "post_rebuild_timing_status": (post_rebuild_result or {}).get(
                    "post_rebuild_timing_status"
                ),
                "post_rebuild_timing_mode": (post_rebuild_result or {}).get(
                    "post_rebuild_timing_mode"
                ),
            },
            "continuation_policy": (control_result or {}).get("continuation_policy"),
            "post_rebuild_summary_path": (post_rebuild_result or {}).get(
                "summary_path"
            ),
        }
        self._write_json_artifact(path, payload)
        return path

    def _committed_buffer_names(self, commit):
        if "committed_buffer_instance_names" in commit:
            return list(commit["committed_buffer_instance_names"])
        return [
            record["inserted_buffer_name"]
            for record in self._committed_buffer_records(commit)
        ]

    def _committed_buffer_records(self, commit):
        records = []
        artifact_path = commit.get("artifact_path")
        if not artifact_path or not os.path.exists(artifact_path):
            return records
        try:
            with open(artifact_path, "r", encoding="utf-8") as fp:
                artifact = json.load(fp)
        except Exception:
            return records
        for result in artifact.get("results", []) or []:
            if not isinstance(result, dict) or not result.get("accepted"):
                continue
            name = result.get("inserted_buffer_name")
            if not name:
                connectivity = result.get("connectivity")
                if isinstance(connectivity, dict):
                    name = connectivity.get("inserted_buffer_name")
            if name:
                record = dict(result)
                record["inserted_buffer_name"] = str(name)
                records.append(record)
        return records

    def _committed_buffer_coordinate_records(self, commit_records):
        placedb = self.placedb
        raw_names_value = getattr(placedb, "node_names", None)
        raw_names = [] if raw_names_value is None else list(raw_names_value)
        name_to_index = {
            self._decode_def_name(name): index
            for index, name in enumerate(raw_names)
        }
        node_x = np.asarray(getattr(placedb, "node_x", ()))
        node_y = np.asarray(getattr(placedb, "node_y", ()))
        if node_x.shape == node_y.shape and node_x.size:
            final_x, final_y = placedb.unscale_pl_positions(node_x, node_y)
            final_x = np.asarray(final_x)
            final_y = np.asarray(final_y)
        else:
            final_x = np.asarray(())
            final_y = np.asarray(())

        coordinates = []
        for record in commit_records:
            name = str(record["inserted_buffer_name"])
            index = name_to_index.get(name)
            analytical_x = record.get("candidate_location_x_dbu_analytical")
            analytical_y = record.get("candidate_location_y_dbu_analytical")
            rounded_x = record.get("candidate_location_x_dbu")
            rounded_y = record.get("candidate_location_y_dbu")
            item = {
                "inserted_buffer_name": name,
                "action_id": record.get("action_id"),
                "segment_id": record.get("segment_id"),
                "analytical_x_dbu": analytical_x,
                "analytical_y_dbu": analytical_y,
                "rounded_x_dbu": rounded_x,
                "rounded_y_dbu": rounded_y,
                "coordinate_source": record.get("coordinate_source"),
                "placement_snapshot_identity": record.get(
                    "placement_snapshot_identity"
                ),
                "topology_generation": record.get("topology_generation"),
                "frozen_topology_generation": record.get(
                    "frozen_topology_generation"
                ),
            }
            if index is None or index >= final_x.size:
                item.update(
                    {
                        "status": "missing_after_refresh",
                        "legalized_x_dbu": None,
                        "legalized_y_dbu": None,
                        "displacement_x_dbu": None,
                        "displacement_y_dbu": None,
                        "displacement_manhattan_dbu": None,
                        "displacement_euclidean_dbu": None,
                    }
                )
            else:
                legalized_x = float(final_x[index])
                legalized_y = float(final_y[index])
                source_x = analytical_x if analytical_x is not None else rounded_x
                source_y = analytical_y if analytical_y is not None else rounded_y
                dx = None if source_x is None else legalized_x - float(source_x)
                dy = None if source_y is None else legalized_y - float(source_y)
                item.update(
                    {
                        "status": "ok",
                        "legalized_x_dbu": legalized_x,
                        "legalized_y_dbu": legalized_y,
                        "displacement_x_dbu": dx,
                        "displacement_y_dbu": dy,
                        "displacement_manhattan_dbu": (
                            None if dx is None or dy is None else abs(dx) + abs(dy)
                        ),
                        "displacement_euclidean_dbu": (
                            None
                            if dx is None or dy is None
                            else math.hypot(dx, dy)
                        ),
                    }
                )
            coordinates.append(item)
        return coordinates

    def _segment_joint_final_def_path(self):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        return os.path.join(
            result_dir,
            f"{self._design_name()}_segment_joint_final.def",
        )

    def _run_segment_joint_def_only_opensta_replay(
        self,
        final_def_path,
        *,
        in_process_metrics,
    ):
        if not final_def_path or not os.path.isfile(final_def_path):
            raise RuntimeError("segment joint DEF-only replay requires a final DEF")
        design_inputs = copy.deepcopy(
            dict(getattr(self.params, "design_inputs", {}) or {})
        )
        if not design_inputs:
            raise RuntimeError(
                "segment joint DEF-only replay requires resolved design_inputs"
            )

        source_def_path = design_inputs.get("def")
        for key in ("verilog", "netlist", "verilog_input"):
            design_inputs.pop(key, None)
        design_inputs["def"] = final_def_path
        replay_params = copy.copy(self.params)
        replay_params.design_inputs = design_inputs
        replay_params.def_input = final_def_path
        replay_params.verilog_input = ""

        started_at = time.perf_counter()
        replay_bridge = placeio_openroad.PlaceIOFunction.read(replay_params)
        replay_metrics = dict(
            placeio_openroad.PlaceIOFunction.query_diff_guided_batch_timing_metrics(
                replay_bridge
            )
            or {}
        )
        if str(replay_metrics.get("status") or "") != "ok":
            raise RuntimeError(
                "fresh DEF-only OpenSTA replay failed: %s" % replay_metrics
            )
        try:
            replay_wns = float(replay_metrics["wns"])
            replay_tns = float(replay_metrics["tns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "fresh DEF-only OpenSTA replay returned invalid WNS/TNS: %s"
                % replay_metrics
            ) from exc
        if not math.isfinite(replay_wns) or not math.isfinite(replay_tns):
            raise RuntimeError(
                "fresh DEF-only OpenSTA replay returned non-finite WNS/TNS"
            )

        in_process_wns = float(in_process_metrics["wns"])
        in_process_tns = float(in_process_metrics["tns"])

        def input_count(name):
            value = design_inputs.get(name)
            if not value:
                return 0
            return len(value) if isinstance(value, list) else 1

        payload = {
            "artifact_version": 1,
            "artifact_scope": "segment_joint_def_only_opensta_replay",
            "status": "ok",
            "design_name": self._design_name(),
            "load_contract": {
                "design_carrier": "final_def",
                "fresh_openroad_bridge": True,
                "verilog_read": False,
                "source_def_path": source_def_path,
                "final_def_path": final_def_path,
                "final_def_sha256": self._sha256_file(final_def_path),
                "sdc_path": design_inputs.get("sdc"),
                "rc_tcl_path": design_inputs.get("rc_tcl"),
                "tech_lef_count": input_count("tech_lef"),
                "lef_count": input_count("lef"),
                "liberty_count": input_count("lib"),
            },
            "opensta_after": {"wns": replay_wns, "tns": replay_tns},
            "in_process_opensta_after": {
                "wns": in_process_wns,
                "tns": in_process_tns,
            },
            "def_only_vs_in_process_gap": {
                "wns": replay_wns - in_process_wns,
                "tns": replay_tns - in_process_tns,
            },
            "opensta_metrics": replay_metrics,
            "runtime_ms": (time.perf_counter() - started_at) * 1000.0,
        }
        result_dir = getattr(self.params, "result_dir", None)
        if result_dir:
            artifact_path = os.path.join(
                result_dir,
                f"{self._design_name()}_segment_joint_def_only_opensta_replay.json",
            )
            self._write_json_artifact(artifact_path, payload)
            payload["artifact_path"] = artifact_path
        return payload

    def _finalize_segment_joint_committed_state(self, control_result):
        started_at = time.perf_counter()
        commit = dict((control_result or {}).get("commit", {}) or {})
        request = dict((control_result or {}).get("commit_request", {}) or {})
        provenance = dict(request.get("projection_metadata", {}) or {}).get(
            "projection_provenance", {}
        )
        if provenance.get("coordinate_source") != "final_live_geometry":
            raise RuntimeError(
                "segment joint terminal commit is missing final live geometry provenance"
            )

        commands = []
        for label, command in (
            ("legalize", self._full_core_detailed_placement_command()),
            ("check_placement", "check_placement -verbose"),
            ("estimate_parasitics", "estimate_parasitics -placement"),
        ):
            command_result = self._checked_openroad_tcl(label, command)
            commands.append(command_result)
            if command_result["status"] != "ok":
                error = (
                    f"segment joint OpenROAD {label} failed with Tcl return code "
                    f"{command_result['tcl_return_code']}: {command_result['output']}"
                )
                failure_payload = {
                    "artifact_version": 1,
                    "artifact_scope": "segment_joint_committed_state_verification",
                    "status": "failed",
                    "design_name": self._design_name(),
                    "profile": "segment_count_direct_joint_v1",
                    "commands": commands,
                    "detailed_placement_search_window": "full_core",
                    "legalization_status": (
                        "failed"
                        if label in {"legalize", "check_placement"}
                        else "ok"
                    ),
                    "parasitic_refresh_status": (
                        "failed" if label == "estimate_parasitics" else "not_run"
                    ),
                    "error": error,
                    "runtime_ms": (time.perf_counter() - started_at) * 1000.0,
                }
                result_dir = getattr(self.params, "result_dir", None)
                if result_dir:
                    failure_path = os.path.join(
                        result_dir,
                        f"{self._design_name()}_segment_joint_committed_verification.json",
                    )
                    self._write_json_artifact(failure_path, failure_payload)
                    failure_payload["summary_path"] = failure_path
                self.last_segment_joint_committed_verification = failure_payload
                raise RuntimeError(error)

        opensta_metrics = dict(
            placeio_openroad.PlaceIOFunction.query_diff_guided_batch_timing_metrics(
                self.placedb.openroad_bridge
            )
            or {}
        )
        timing_refresh = dict(opensta_metrics.get("timing_refresh", {}) or {})
        timing_refresh_status = str(timing_refresh.get("status") or "missing")
        if timing_refresh_status != "ok":
            raise RuntimeError(
                "fresh OpenSTA timing refresh failed: %s" % timing_refresh
            )
        commands.append(
            {
                "label": "timing_refresh",
                "command": timing_refresh.get(
                    "method", "in_process_opensta_timing_refresh"
                ),
                "status": timing_refresh_status,
                "elapsed_ms": float(timing_refresh.get("elapsed_ms", 0.0) or 0.0),
                "output": timing_refresh,
            }
        )
        if str(opensta_metrics.get("status", "ok")) == "failed":
            raise RuntimeError(
                "fresh OpenSTA committed-state query failed: %s" % opensta_metrics
            )
        if opensta_metrics.get("wns") is None or opensta_metrics.get("tns") is None:
            raise RuntimeError("fresh OpenSTA committed-state query returned no WNS/TNS")

        pre_area = float(getattr(self.placedb, "total_movable_node_area", 0.0) or 0.0)
        commit_records = self._committed_buffer_records(commit)
        mutation_backend = getattr(self, "last_physical_mutation_backend", None)
        if mutation_backend is None:
            mutation_backend = self._physical_mutation_backend()
        if commit_records:
            refresh_summary = mutation_backend.refresh(
                refresh_mode=control_result.get("refresh_mode", "topo"),
                rebuild_mode=control_result.get("rebuild_mode", "topo"),
            )
            if str(refresh_summary.get("status") or "") != "ok":
                raise RuntimeError(
                    "segment joint terminal topology refresh failed: %s"
                    % refresh_summary
                )
        else:
            refresh_summary = {
                "status": "skipped",
                "reason": "no_topology_changing_buffer_actions",
                "refresh_mode": None,
                "rebuild_mode": None,
            }
        post_area = float(
            getattr(self.placedb, "total_movable_node_area", pre_area) or pre_area
        )
        buffer_added_area_internal = post_area - pre_area
        sizing_added_area_internal = float(
            commit.get("sizing_area_delta_internal", 0.0) or 0.0
        )
        added_area_internal = sizing_added_area_internal + buffer_added_area_internal
        scale_factor = float(getattr(self.params, "scale_factor", 1.0) or 1.0)
        dbu = float(getattr(self.placedb, "dbu", 1.0) or 1.0)
        area_scale = (scale_factor * dbu) ** 2
        added_area_um2 = added_area_internal / area_scale
        coordinate_records = self._committed_buffer_coordinate_records(commit_records)
        missing_legalized_coordinates = sum(
            record["status"] != "ok" for record in coordinate_records
        )
        if missing_legalized_coordinates:
            raise RuntimeError(
                "segment joint committed buffers are missing legalized coordinates"
            )

        final_def_path = self._segment_joint_final_def_path()
        if final_def_path:
            os.makedirs(os.path.dirname(final_def_path), exist_ok=True)
            self.placedb.openroad_bridge.write_def(final_def_path)

        request_object = getattr(
            getattr(self, "placer", None),
            "last_buffer_commit_request",
            None,
        )
        if int(request.get("sizing_action_count", 0) or 0) > 0 and request_object is None:
            raise RuntimeError(
                "segment joint final sizing verification has no in-memory request"
            )
        sizing_master_verification = (
            self._verify_segment_joint_final_sizing_masters(
                final_def_path,
                request_object,
            )
            if request_object is not None
            else {
                "status": "skipped",
                "reason": "no_sizing_actions",
                "expected_count": 0,
                "matched_count": 0,
                "missing_count": 0,
                "mismatch_count": 0,
            }
        )

        def_only_replay = self._run_segment_joint_def_only_opensta_replay(
            final_def_path,
            in_process_metrics=opensta_metrics,
        )

        before_wns = commit.get("before_wns")
        before_tns = commit.get("before_tns")
        after_wns = float(opensta_metrics["wns"])
        after_tns = float(opensta_metrics["tns"])
        delta_wns = None if before_wns is None else after_wns - float(before_wns)
        delta_tns = None if before_tns is None else after_tns - float(before_tns)
        runtime_ms = (time.perf_counter() - started_at) * 1000.0
        physical_action_trace = list(commit.get("physical_action_trace", ()) or ())
        expected_action_order = ["coordinate_sync"]
        if int(request.get("sizing_action_count", 0) or 0) > 0:
            expected_action_order.append("apply_sizing")
        if int(request.get("action_count", 0) or 0) > 0:
            expected_action_order.append("insert_buffers")
        actual_action_order = [str(row.get("stage")) for row in physical_action_trace]
        if actual_action_order != expected_action_order:
            raise RuntimeError(
                "segment joint physical action order mismatch: "
                f"expected {expected_action_order}, got {actual_action_order}"
            )
        full_action_order = actual_action_order + [
            str(row.get("label")) for row in commands
        ]
        payload = {
            "artifact_version": 1,
            "artifact_scope": "segment_joint_committed_state_verification",
            "status": "ok",
            "design_name": self._design_name(),
            "profile": "segment_count_direct_joint_v1",
            "coordinate_source": "final_live_geometry",
            "projection_provenance": dict(provenance),
            "physical_action_trace": physical_action_trace,
            "full_action_order": full_action_order,
            "commands": commands,
            "detailed_placement_search_window": "full_core",
            "legalization_status": "ok",
            "parasitic_refresh_status": "ok",
            "opensta_update_status": timing_refresh_status,
            "opensta_update_method": timing_refresh.get("method"),
            "opensta_before": {"wns": before_wns, "tns": before_tns},
            "opensta_after": {"wns": after_wns, "tns": after_tns},
            "def_only_opensta_verification_status": def_only_replay["status"],
            "def_only_opensta_after": dict(def_only_replay["opensta_after"]),
            "def_only_vs_in_process_gap": dict(
                def_only_replay["def_only_vs_in_process_gap"]
            ),
            "def_only_opensta_replay_path": def_only_replay.get("artifact_path"),
            "def_only_opensta_replay": def_only_replay,
            "delta_wns": delta_wns,
            "delta_tns": delta_tns,
            "opensta_metrics": opensta_metrics,
            "buffer_count": len(commit_records),
            "sizing_action_count": int(
                commit.get("sizing_action_count", 0) or 0
            ),
            "sizing_action_digest": commit.get("sizing_action_digest"),
            "sizing_master_verification": sizing_master_verification,
            "accepted_action_count": int(
                commit.get("accepted_action_count", 0) or 0
            ),
            "added_area_internal": added_area_internal,
            "added_area_um2": added_area_um2,
            "sizing_added_area_internal": sizing_added_area_internal,
            "sizing_added_area_um2": sizing_added_area_internal / area_scale,
            "buffer_added_area_internal": buffer_added_area_internal,
            "buffer_added_area_um2": buffer_added_area_internal / area_scale,
            "final_placement_metrics": self._metrics_summary(),
            "buffer_coordinates": coordinate_records,
            "missing_legalized_coordinate_count": missing_legalized_coordinates,
            "refresh_summary": dict(refresh_summary),
            "runtime_ms": runtime_ms,
            "qor_status": (
                "buffer_preflight_rejected_preserved_prebuffer_state"
                if commit.get("preflight_rejected")
                else "absolute_only"
                if delta_tns is None
                else "pass"
                if delta_tns > 0.0 and (delta_wns is None or delta_wns >= 0.0)
                else "fail_tns_or_wns"
            ),
            "final_def_path": final_def_path,
        }
        if final_def_path and os.path.exists(final_def_path):
            payload["final_def_sha256"] = self._sha256_file(final_def_path)

        result_dir = getattr(self.params, "result_dir", None)
        summary_path = None
        if result_dir:
            summary_path = os.path.join(
                result_dir,
                f"{self._design_name()}_segment_joint_committed_verification.json",
            )
            self._write_json_artifact(summary_path, payload)
            payload["summary_path"] = summary_path
        artifact_path = commit.get("artifact_path")
        if artifact_path and summary_path:
            self._update_json_artifact(
                artifact_path,
                lambda artifact: artifact.update(
                    {
                        "segment_joint_committed_verification_path": summary_path,
                        "terminal_opensta_after": payload["opensta_after"],
                        "terminal_def_only_opensta_after": payload[
                            "def_only_opensta_after"
                        ],
                        "terminal_def_only_opensta_replay_path": payload[
                            "def_only_opensta_replay_path"
                        ],
                        "terminal_delta_wns": delta_wns,
                        "terminal_delta_tns": delta_tns,
                    }
                ),
            )

        commit.update(
            {
                "after_wns": after_wns,
                "after_tns": after_tns,
                "delta_wns": delta_wns,
                "delta_tns": delta_tns,
                "qor_status": payload["qor_status"],
                "qor_pass": payload["qor_status"] == "pass",
                "terminal_opensta_after": payload["opensta_after"],
                "terminal_def_only_opensta_after": payload[
                    "def_only_opensta_after"
                ],
                "terminal_def_only_opensta_replay_path": payload[
                    "def_only_opensta_replay_path"
                ],
                "terminal_delta_wns": delta_wns,
                "terminal_delta_tns": delta_tns,
                "segment_joint_committed_verification_path": summary_path,
                "committed_def_path": final_def_path,
            }
        )
        control_result["commit"] = commit
        if isinstance(self.metrics, dict):
            self.metrics["post_commit_control_result"] = dict(control_result)
        placer = getattr(self, "placer", None)
        trace = getattr(placer, "joint_buffer_commit_trace", None)
        if isinstance(trace, list) and trace:
            trace[-1]["commit"] = dict(commit)
        if placer is not None:
            placer.last_post_commit_control_result = dict(control_result)
            update_joint_artifact = getattr(placer, "_update_joint_flow_artifact", None)
            if callable(update_joint_artifact):
                update_joint_artifact(self.params)
        result = {
            "status": "ok",
            "continuation_policy": "terminal_verification_only",
            "terminal_only": True,
            "topology_generation": int(
                getattr(self.placedb, "topology_generation", 0) or 0
            ),
            "wns": after_wns,
            "tns": after_tns,
            "def_only_wns": payload["def_only_opensta_after"]["wns"],
            "def_only_tns": payload["def_only_opensta_after"]["tns"],
            "delta_wns": delta_wns,
            "delta_tns": delta_tns,
            "buffer_count": len(commit_records),
            "added_area_um2": added_area_um2,
            "summary_path": summary_path,
            "final_def_path": final_def_path,
            "refresh_summary": dict(refresh_summary),
        }
        self.last_segment_joint_committed_verification = payload
        self.last_joint_post_rebuild_result = result
        return result

    def _reference_buffer_commit_refresh_summary(
        self,
        control_result,
        path,
        *,
        joint_flow_artifact_path=None,
    ):
        if not path:
            return
        commit = (control_result or {}).get("commit")
        if isinstance(commit, dict):
            commit["buffer_commit_pydb_refresh_summary_path"] = path
        metrics = self.metrics
        if isinstance(metrics, dict):
            control = metrics.get("post_commit_control_result")
            if isinstance(control, dict):
                control["buffer_commit_pydb_refresh_summary_path"] = path
                metric_commit = control.get("commit")
                if isinstance(metric_commit, dict):
                    metric_commit["buffer_commit_pydb_refresh_summary_path"] = path
        joint_path = joint_flow_artifact_path or getattr(
            self.placer,
            "last_joint_flow_artifact_path",
            None,
        )
        if joint_path:
            def update_joint(payload):
                summary = payload.get("summary")
                if not isinstance(summary, dict):
                    return
                summary["buffer_commit_pydb_refresh_summary_path"] = path
                summary_control = summary.get("post_commit_control_result")
                if isinstance(summary_control, dict):
                    summary_control["buffer_commit_pydb_refresh_summary_path"] = path
                trace = summary.get("buffer_commit_trace")
                if isinstance(trace, list) and trace:
                    trace_commit = trace[-1].get("commit")
                    if isinstance(trace_commit, dict):
                        trace_commit[
                            "buffer_commit_pydb_refresh_summary_path"
                        ] = path

            self._update_json_artifact(joint_path, update_joint)
        artifact_path = (
            (commit or {}).get("artifact_path")
            if isinstance(commit, dict)
            else None
        )
        if artifact_path:
            def update_commit(payload):
                payload["buffer_commit_pydb_refresh_summary_path"] = path

            self._update_json_artifact(artifact_path, update_commit)

    def _reference_joint_post_commit_openroad_eco_summary(
        self,
        control_result,
        path,
        *,
        joint_flow_artifact_path=None,
    ):
        if not path:
            return
        commit = (control_result or {}).get("commit")
        if isinstance(commit, dict):
            commit["joint_post_commit_openroad_eco_summary_path"] = path
        metrics = self.metrics
        if isinstance(metrics, dict):
            control = metrics.get("post_commit_control_result")
            if isinstance(control, dict):
                control["joint_post_commit_openroad_eco_summary_path"] = path
                metric_commit = control.get("commit")
                if isinstance(metric_commit, dict):
                    metric_commit["joint_post_commit_openroad_eco_summary_path"] = path
        joint_path = joint_flow_artifact_path or getattr(
            self.placer,
            "last_joint_flow_artifact_path",
            None,
        )
        if joint_path:
            def update_joint(payload):
                summary = payload.get("summary")
                if not isinstance(summary, dict):
                    return
                summary["joint_post_commit_openroad_eco_summary_path"] = path
                summary_control = summary.get("post_commit_control_result")
                if isinstance(summary_control, dict):
                    summary_control["joint_post_commit_openroad_eco_summary_path"] = path
                trace = summary.get("buffer_commit_trace")
                if isinstance(trace, list) and trace:
                    trace_commit = trace[-1].get("commit")
                    if isinstance(trace_commit, dict):
                        trace_commit[
                            "joint_post_commit_openroad_eco_summary_path"
                        ] = path

            self._update_json_artifact(joint_path, update_joint)
        artifact_path = (
            (commit or {}).get("artifact_path")
            if isinstance(commit, dict)
            else None
        )
        if artifact_path:
            def update_commit(payload):
                payload["joint_post_commit_openroad_eco_summary_path"] = path

            self._update_json_artifact(artifact_path, update_commit)

    def _write_joint_post_rebuild_summary(self, control_result, refresh_summary, post_rebuild_result):
        result_dir = getattr(self.params, "result_dir", None)
        if not result_dir:
            return None
        os.makedirs(result_dir, exist_ok=True)
        design_name = self._design_name()
        path = os.path.join(result_dir, f"{design_name}_joint_post_rebuild_summary.json")
        payload = {
            "artifact_version": 1,
            "artifact_scope": "joint_post_commit_rebuild",
            "post_commit_control_result": dict(control_result or {}),
            "refresh_summary": dict(refresh_summary or {}),
            "post_rebuild_result": dict(post_rebuild_result or {}),
        }
        self._write_json_artifact(path, payload)
        return path

    def _continue_after_joint_buffer_commit(self, control_result):
        if not control_result:
            return None
        if self._joint_quality_profile() == "segment_count_direct_joint_v1":
            commit = dict(control_result.get("commit", {}) or {})
            if commit.get("preflight_rejected"):
                # The preflight trial is disposable.  It intentionally leaves
                # the live OpenROAD design at the placement-only snapshot, so
                # do not run legalization or terminal verification against a
                # rejected buffer mutation.  Those commands can fail on input
                # BTerms without shapes and would hide the QoR guard result.
                preflight = dict(commit.get("preflight", {}) or {})
                result = {
                    "status": "ok",
                    "terminal_only": True,
                    "continuation_policy": (
                        "preflight_rejected_preserve_prebuffer_state"
                    ),
                    "qor_status": commit.get("qor_status")
                    or preflight.get("trial_qor_status"),
                    "qor_pass": False,
                    "accepted_action_count": 0,
                    "requested_action_count": int(
                        commit.get("action_count", 0) or 0
                    ),
                    "pre_buffer_def_path": preflight.get("pre_buffer_def_path"),
                    "preflight": preflight,
                }
                self.last_joint_post_rebuild_result = result
                return result
            if control_result.get("status") == "failed":
                return None
            return self._finalize_segment_joint_committed_state(control_result)
        if not control_result.get("topology_mutated"):
            return None
        if control_result.get("continuation_policy") != "rebuild_placer_outer_loop":
            return None
        if not bool(getattr(self.params, "joint_post_commit_continue", True)):
            logging.info("joint post-commit continuation disabled; keeping safe-stop result")
            return None
        from dreamplace.flows.committed_topology import refresh_committed_topology

        return refresh_committed_topology(self, control_result)

    def _joint_quality_profile(self):
        return str(getattr(self.params, "joint_quality_profile", "") or "")

    def _is_joint_staged_smoke_v1(self):
        return (
            str(getattr(self.params, "flow_kind", "") or "") == FlowKind.JOINT.value
            and self._joint_quality_profile() == "staged_smoke_v1"
        )

    def _snapshot_stage_params(self, names):
        return {
            name: (hasattr(self.params, name), getattr(self.params, name, None))
            for name in names
        }

    def _restore_stage_params(self, snapshot):
        for name, (existed, value) in snapshot.items():
            if existed:
                setattr(self.params, name, value)
            elif hasattr(self.params, name):
                delattr(self.params, name)

    def _sync_current_placer_pos_to_placedb(self):
        placer = self.placer
        placedb = self.placedb
        if placer is None or placedb is None or not hasattr(placer, "pos"):
            return {"status": "skipped", "reason": "missing_placer_or_placedb"}
        try:
            pos = placer.pos[0].detach().cpu().numpy()
        except Exception as exc:
            return {"status": "skipped", "reason": str(exc)}
        movable = int(getattr(placedb, "num_movable_nodes", 0) or 0)
        num_nodes = int(getattr(placedb, "num_nodes", 0) or 0)
        if movable <= 0 or num_nodes <= 0:
            return {"status": "skipped", "reason": "empty_placedb"}
        pos_x = pos[:movable]
        pos_y = pos[num_nodes:num_nodes + movable]
        node_x = getattr(placedb, "node_x", None)
        node_y = getattr(placedb, "node_y", None)
        if node_x is not None and node_y is not None:
            try:
                if (
                    np.allclose(np.asarray(node_x[:movable]), pos_x)
                    and np.allclose(np.asarray(node_y[:movable]), pos_y)
                ):
                    return {
                        "status": "already_synced",
                        "movable_nodes": movable,
                        "num_nodes": num_nodes,
                    }
            except Exception:
                pass
        placedb.apply(
            self.params,
            pos_x,
            pos_y,
        )
        return {
            "status": "ok",
            "movable_nodes": movable,
            "num_nodes": num_nodes,
        }

    def _json_safe_scalar(self, value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _metrics_summary(self):
        metrics = self.metrics
        if isinstance(metrics, dict):
            summary = {}
            for key in ("objective", "overflow", "density"):
                values = metrics.get(key)
                if isinstance(values, (list, tuple)) and values:
                    summary[key] = self._json_safe_scalar(values[-1])
            for key in ("wns", "tns"):
                if key in metrics:
                    summary[key] = self._json_safe_scalar(metrics[key])
            return summary
        try:
            niter = len(metrics)
        except Exception:
            niter = None
        return {"metric_count": niter}

    def _run_joint_staged_smoke_window(
        self,
        stage_name,
        overrides,
        stage_role=None,
        outer_iteration=None,
    ):
        snapshot = self._snapshot_stage_params(overrides.keys())
        started = time.time()
        old_placer = self.placer
        old_placer_id = id(old_placer) if old_placer is not None else None
        try:
            for name, value in overrides.items():
                setattr(self.params, name, value)
            self.place()
            sync_summary = self._sync_current_placer_pos_to_placedb()
            result = {
                "stage_name": stage_name,
                "stage_role": stage_role,
                "outer_iteration": outer_iteration,
                "status": "ok",
                "elapsed_ms": (time.time() - started) * 1000.0,
                "overrides": dict(overrides),
                "old_nonlinear_place_id": old_placer_id,
                "new_nonlinear_place_id": id(self.placer),
                "fresh_nonlinear_place_constructed": id(self.placer) != old_placer_id,
                "rsmt": self.rsmt,
                "hpwl": self.hpwl,
                "metrics": self._metrics_summary(),
                "placedb_sync": sync_summary,
            }
        except Exception:
            raise
        finally:
            self._restore_stage_params(snapshot)
        return result

    def _parse_opensta_report_scalar(self, output):
        text = str(output or "").strip()
        if not text:
            return None
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        if not matches:
            return None
        try:
            return float(matches[-1])
        except ValueError:
            return None

    def _joint_staged_smoke_update_timing_summary(self, timing_summary, label, output):
        prefix = None
        if label.startswith("pre_"):
            prefix = "pre_eco"
        elif label.startswith("post_repair_design_"):
            prefix = "post_repair_design"
        elif label.startswith("post_repair_timing_"):
            prefix = "post_repair_timing"
        if prefix is None:
            return
        if "report_worst_slack" in label:
            key = "wns"
        elif "report_tns" in label:
            key = "tns"
        else:
            return
        value = self._parse_opensta_report_scalar(output)
        if value is not None:
            timing_summary.setdefault(prefix, {})[key] = value

    def _joint_staged_smoke_record_direct_metric(self, timing_summary, prefix, key, command):
        output = self._openroad_tcl(command)
        value = self._parse_opensta_report_scalar(output)
        if value is not None:
            timing_summary.setdefault(prefix, {})[key] = value
        return {
            "command": command,
            "output": output,
            "value": value,
        }

    def _write_joint_staged_smoke_openroad_def(self, final_def_path):
        bridge = getattr(self.placedb, "openroad_bridge", None)
        if bridge is None or not hasattr(bridge, "write_def"):
            raise RuntimeError(
                "joint staged smoke final DEF requires OpenROAD bridge.write_def"
            )
        os.makedirs(os.path.dirname(os.path.abspath(final_def_path)), exist_ok=True)
        bridge.write_def(final_def_path)

    def _write_joint_staged_smoke_final_def(self, payload):
        final_def_path = self._joint_staged_smoke_final_def_path()
        if not final_def_path:
            return None
        started = time.time()
        self._write_joint_staged_smoke_openroad_def(final_def_path)
        final_def = {
            "path": final_def_path,
            "sha256": self._sha256_file(final_def_path),
            "elapsed_ms": (time.time() - started) * 1000.0,
            "writer": "openroad_bridge.write_def",
        }
        payload["final_def"] = final_def
        return final_def

    def _joint_staged_smoke_stage_copy(self):
        stages = copy.deepcopy(getattr(self.params, "global_place_stages", None) or [{}])
        if not isinstance(stages, list) or not stages:
            return [{}]
        return stages

    def _joint_staged_smoke_tdp_overrides(self):
        stages = self._joint_staged_smoke_stage_copy()
        for stage in stages:
            if isinstance(stage, dict):
                stage["optimizer"] = "nesterov"
                stage["iteration"] = max(
                    int(stage.get("iteration", 0) or 0),
                    STAGED_JOINT_TDP_ITERATIONS,
                )
        overrides = {
            "flow_kind": FlowKind.PLACEMENT.value,
            "global_place_stages": stages,
            "placement_sizing_mode": "place_only",
            "continuous_size_dynamics_mode": "none",
            "buffering_continuous_relaxed_optimization": False,
            "buffering_segment_count_tns_gradient": False,
            "random_center_init_flag": 0,
        }
        overrides.update(dict(STAGED_JOINT_TDP_STAGE_DEFAULTS))
        return overrides

    def _joint_staged_smoke_sizing_overrides(self):
        stages = self._joint_staged_smoke_stage_copy()
        for stage in stages:
            if isinstance(stage, dict):
                stage["optimizer"] = "adam"
                stage["iteration"] = STAGED_JOINT_SIZING_ITERATIONS
        overrides = {
            "flow_kind": FlowKind.SIZING.value,
            "global_place_stages": stages,
            "buffering_continuous_relaxed_optimization": False,
            "buffering_segment_count_tns_gradient": False,
            "diff_timing_driven_placement": 0,
            "differentiable_timing_obj": 1,
            "with_sta": 1,
            "legalize_flag": 0,
            "detailed_place_flag": 0,
            "random_center_init_flag": 0,
        }
        overrides.update(dict(STAGED_JOINT_SIZING_STAGE_DEFAULTS))
        return overrides

    def _run_joint_staged_smoke_openroad_finalization(self):
        commands = [
            (
                "pre_refresh_timing",
                'if {[llength [info commands update_timing]]} { update_timing } else { worst_slack -max }',
            ),
            ("pre_report_worst_slack", "report_worst_slack -max"),
            ("pre_report_tns", "report_tns -max"),
            ("repair_design", "repair_design"),
            (
                "post_repair_design_refresh_timing",
                'if {[llength [info commands update_timing]]} { update_timing } else { worst_slack -max }',
            ),
            ("post_repair_design_report_worst_slack", "report_worst_slack -max"),
            ("post_repair_design_report_tns", "report_tns -max"),
            ("repair_timing_setup", "repair_timing -setup"),
            (
                "post_repair_timing_refresh_timing",
                'if {[llength [info commands update_timing]]} { update_timing } else { worst_slack -max }',
            ),
            ("post_repair_timing_report_worst_slack", "report_worst_slack -max"),
            ("post_repair_timing_report_tns", "report_tns -max"),
        ]
        results = []
        timing_summary = {}
        started = time.time()
        payload = {
            "artifact_scope": "joint_staged_smoke_openroad_finalization",
            "artifact_version": 1,
            "design_name": self._design_name(),
            "commands": results,
            "timing_summary": timing_summary,
        }
        failed_error = None
        try:
            for label, command in commands:
                command_started = time.time()
                output = self._openroad_tcl(command)
                self._joint_staged_smoke_update_timing_summary(
                    timing_summary,
                    label,
                    output,
                )
                results.append(
                    {
                        "label": label,
                        "command": command,
                        "status": "ok",
                        "elapsed_ms": (time.time() - command_started) * 1000.0,
                        "output": output,
                    }
                )
            direct_metrics = {
                "post_repair_timing_wns": self._joint_staged_smoke_record_direct_metric(
                    timing_summary,
                    "post_repair_timing",
                    "wns",
                    "worst_slack -max",
                ),
                "post_repair_timing_tns": self._joint_staged_smoke_record_direct_metric(
                    timing_summary,
                    "post_repair_timing",
                    "tns",
                    "total_negative_slack -max",
                ),
            }
            payload["direct_metrics"] = direct_metrics
            self._write_joint_staged_smoke_final_def(payload)
            payload["status"] = "ok"
        except Exception as exc:
            payload["status"] = "failed"
            payload["error"] = str(exc)
            payload["failed_command_index"] = len(results)
            failed_error = exc
        payload["elapsed_ms"] = (time.time() - started) * 1000.0
        if failed_error is not None:
            raise failed_error
        return payload

    def _run_joint_staged_smoke_openroad_buffering_stage(
        self,
        stage_name,
        outer_iteration=None,
    ):
        started = time.time()
        old_placer_id = id(self.placer) if self.placer is not None else None
        commands = [
            (
                "pre_refresh_timing",
                'if {[llength [info commands update_timing]]} { update_timing } else { worst_slack -max }',
            ),
            ("pre_report_worst_slack", "report_worst_slack -max"),
            ("pre_report_tns", "report_tns -max"),
            ("repair_timing_setup", "repair_timing -setup"),
            (
                "post_repair_timing_refresh_timing",
                'if {[llength [info commands update_timing]]} { update_timing } else { worst_slack -max }',
            ),
            ("post_repair_timing_report_worst_slack", "report_worst_slack -max"),
            ("post_repair_timing_report_tns", "report_tns -max"),
        ]
        command_results = []
        timing_summary = {}
        for label, command in commands:
            command_started = time.time()
            output = self._openroad_tcl(command)
            self._joint_staged_smoke_update_timing_summary(
                timing_summary,
                label,
                output,
            )
            command_results.append(
                {
                    "label": label,
                    "command": command,
                    "status": "ok",
                    "elapsed_ms": (time.time() - command_started) * 1000.0,
                    "output": output,
                }
            )
        refresh_summary = self.placedb.refresh_from_openroad_bridge(
            refresh_mode="topo",
            rebuild_mode="topo",
        )
        self.placer = None
        return {
            "stage_name": stage_name,
            "stage_role": "openroad_repair_timing_buffering",
            "outer_iteration": outer_iteration,
            "status": "ok",
            "elapsed_ms": (time.time() - started) * 1000.0,
            "old_nonlinear_place_id": old_placer_id,
            "new_nonlinear_place_id": None,
            "fresh_nonlinear_place_constructed": True,
            "topology_refreshed": True,
            "refresh_mode": "topo",
            "rebuild_mode": "topo",
            "openroad_commands": command_results,
            "timing_summary": timing_summary,
            "refresh_summary": refresh_summary,
        }

    def _run_joint_staged_smoke_v1(self):
        started = time.time()
        self.setup_placedb()
        max_outer_iterations = int(
            getattr(self.params, "joint_staged_smoke_outer_iterations", 1) or 1
        )
        stages = []
        stages.append(
            self._run_joint_staged_smoke_window(
                "initial_tdp_place",
                self._joint_staged_smoke_tdp_overrides(),
                stage_role="timing_driven_placement",
                outer_iteration=None,
            )
        )
        for outer_iter in range(max_outer_iterations):
            prefix = f"outer_{outer_iter:03d}"
            stages.append(
                self._run_joint_staged_smoke_window(
                    f"{prefix}_fixed_position_sizing",
                    self._joint_staged_smoke_sizing_overrides(),
                    stage_role="fixed_position_sizing",
                    outer_iteration=outer_iter,
                )
            )
            stages.append(
                self._run_joint_staged_smoke_window(
                    f"{prefix}_after_sizing_tdp_return",
                    self._joint_staged_smoke_tdp_overrides(),
                    stage_role="timing_driven_placement_after_sizing",
                    outer_iteration=outer_iter,
                )
            )
            stages.append(
                self._run_joint_staged_smoke_openroad_buffering_stage(
                    f"{prefix}_buffering",
                    outer_iteration=outer_iter,
                )
            )
            stages.append(
                self._run_joint_staged_smoke_window(
                    f"{prefix}_tdp_return",
                    self._joint_staged_smoke_tdp_overrides(),
                    stage_role="timing_driven_placement_after_buffering",
                    outer_iteration=outer_iter,
                )
            )

        finalization = self._run_joint_staged_smoke_openroad_finalization()
        payload = {
            "artifact_scope": "joint_staged_smoke",
            "artifact_version": 1,
            "design_name": self._design_name(),
            "flow_kind": FlowKind.JOINT.value,
            "joint_quality_profile": self._joint_quality_profile(),
            "max_outer_iterations": max_outer_iterations,
            "stage_count": len(stages),
            "stages": stages,
            "finalization": finalization,
            "elapsed_ms": (time.time() - started) * 1000.0,
            "status": "ok",
        }
        summary_path = self._write_joint_staged_smoke_summary(payload)
        if summary_path:
            payload["summary_path"] = summary_path
            self._write_joint_staged_smoke_summary(payload)
        self.last_joint_staged_smoke_summary = payload
        return {
            "rsmt": self.rsmt,
            "hpwl": self.hpwl,
            "congestion": self.congestion,
            "density": self.density,
            "iteration": 0,
            "objective": -1,
            "overflow": -1,
            "max_density": -1,
            "joint_staged_smoke": payload,
        }

    def run_sta_only(self):
        tt = time.time()
        setup_placedb_start = time.time()
        self.setup_placedb()
        setup_placedb_ms = (time.time() - setup_placedb_start) * 1000.0
        nonlinear_init_start = time.time()
        self.placer = NonLinearPlace.NonLinearPlace(
            self.params, self.placedb, None)
        nonlinear_place_init_ms = (time.time() - nonlinear_init_start) * 1000.0
        sta_eval_start = time.time()
        self.rsmt, self.hpwl, self.metrics = self.placer.run_sta_only(
            self.params, self.placedb)
        sta_eval_ms = (time.time() - sta_eval_start) * 1000.0
        if hasattr(self.placer, "initialization_profile"):
            self.placer.initialization_profile.update(
                {
                    "setup_placedb_ms": setup_placedb_ms,
                    "nonlinear_place_init_ms": nonlinear_place_init_ms,
                    "sta_eval_ms": sta_eval_ms,
                }
            )
        logging.info("STA-only flow takes %.2f seconds" % (time.time() - tt))
        sta_record = self.metrics["timing_stage_summary"]["sta"]
        if not all(
            math.isfinite(float(sta_record[field]))
            for field in ("wns", "tns", "timing_objective")
        ):
            raise RuntimeError("STA-only flow produced non-finite timing metrics")
        return {
            "status": "ok",
            "rsmt": self.rsmt,
            "hpwl": self.hpwl,
            "congestion": self.congestion,
            "density": self.density,
            "iteration": 0,
            "objective": -1,
            "overflow": -1,
            "max_density": -1,
        }

    def external_detailed_placer(self):
        # call external detailed placement
        # TODO: support more external placers, currently only support
        # 1. NTUplace3/NTUplace4h with Bookshelf format
        # 2. NTUplace_4dr with LEF/DEF format
        logging.info(
            "Use external detailed placement engine %s"
            % (self.params.detailed_place_engine)
        )
        if self.params.solution_file_suffix() == "pl" and any(
            dp_engine in self.params.detailed_place_engine
            for dp_engine in ["ntuplace3", "ntuplace4h"]
        ):
            dp_out_file = self.gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if (
                self.params.target_density < 1.0
                and not self.params.routability_opt_flag
            ):
                target_density_cmd = " -util %f" % (self.params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                self.params.detailed_place_engine,
                self.params.aux_input,
                self.gp_out_file,
                target_density_cmd,
                dp_out_file,
                self.params.detailed_place_command,
            )
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info(
                "External detailed placement takes %.2f seconds" % (
                    time.time() - tt)
            )

            if self.params.plot_flag:
                # read solution and evaluate
                self.placedb.read_pl(self.params, dp_out_file + ".ntup.pl")
                iteration = len(self.metrics)
                pos = self.placer.init_pos
                pos[0: self.placedb.num_physical_nodes] = self.placedb.node_x
                pos[
                    self.placedb.num_nodes: self.placedb.num_nodes
                    + self.placedb.num_physical_nodes
                ] = self.placedb.node_y
                hpwl, density_overflow, max_density = self.placer.validate(
                    self.placedb, pos, iteration
                )
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density)
                )
                self.placer.plot(self.params, self.placedb, iteration, pos)
        elif "ntuplace_4dr" in self.params.detailed_place_engine:
            dp_out_file = self.gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (self.params.detailed_place_engine)
            for lef in self.params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (self.gp_out_file)
            if self.params.verilog_input:
                cmd += " -verilog %s" % (self.params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(self.params.verilog_input))
                benchmark_dir
            )
            cmd += " -noglobal %s ; " % (self.params.detailed_place_command)
            # cmd += " %s ; " % (self.params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (
                dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file
            )
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (
                dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info(
                "External detailed placement takes %.2f seconds" % (
                    time.time() - tt)
            )
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )

    def get_congestion(self):
        pos = self.placer.data_collections.pos[0]
        congestion_map = self.placer.op_collections.get_congestion_map_op(
            pos
        )
        congestion, _ = torch.topk(
            congestion_map.flatten(), k=int(0.1 * congestion_map.numel())
        )
        self.congestion = float(congestion.mean())
        logging.info(f"Congestion score {self.congestion}")

    def save_placement(self):
        # write placement solution
        self.path = "%s/%s" % (self.params.result_dir,
                               self.params.design_name())
        if not os.path.exists(self.path):
            os.system("mkdir -p %s" % (self.path))
        self.gp_out_file = os.path.join(
            self.path,
            "%s.gp.def"
            % (self.params.design_name()),
        )
        self.write_back(self.gp_out_file)
        tcl_file = os.path.join(
            self.path,
            "%s.tcl"
            % (self.params.design_name()),
        )
        place_io_engine = getattr(self.params, "place_io_engine", "ieda")
        if place_io_engine == "ecc":
            import dreamplace.ops.placeio_ecc.place_io as placeio_ecc

            placeio_ecc.PlaceIOFunction.write_tcl(self.placedb.rawdb, tcl_file)
        elif place_io_engine != "openroad":
            placeio_ieda = _load_placeio_ieda()
            placeio_ieda.PlaceIOFunction.write_tcl(
                self.data_manager.dir_workspace,
                tcl_file,
            )
        # self.placedb.write(self.params, self.gp_out_file)

    def run(self):
        # run entire placement flow
        print('read verilog done')
        # self.placedb = PlaceDB(engine_data_ieda)
        # self.placedb.init_db(params)
        print('init db done')
        if getattr(self.params, "flow_kind", None) == FlowKind.STA.value:
            return self.run_sta_only()
        if self._is_joint_staged_smoke_v1():
            return self._run_joint_staged_smoke_v1()
        tt = time.time()
        if getattr(self.params, "macro_only", False):
            candidate_count = int(
                np.count_nonzero(self.placedb.pydb.macro_writeback_candidate)
            )
            if candidate_count == 0:
                result = {
                    "executed": False,
                    "candidate_count": 0,
                    "reason": "no_unplaced_hard_macros",
                }
                logging.info("Macro placement skipped: %s", result)
                return result
        self.setup_placedb()
        if self._is_segment_direct_joint() and self._joint_buffer_outer_iterations() > 1:
            if int(
                getattr(self.params, "joint_segment_virtual_window_steps", 0) or 0
            ) > 0:
                return self._run_segment_direct_joint_continuous_outer_flow()
            return self._run_segment_direct_joint_outer_flow()
        outer_windows = []
        outer_window_index = 0
        post_commit_result = None
        while True:
            if outer_window_index == 0:
                self.place()
            else:
                self.place(reuse_existing=True)
            placement_debug = dict(
                getattr(self.placer, "last_placement_debug_summary", None) or {}
            )
            if placement_debug.get("status") == "failed":
                self.last_run_result = placement_debug
                return self.last_run_result
            if bool(getattr(self.params, "timing_opt_enabled", False)):
                from dreamplace.flows.committed_topology import finalize_inflation_s5b1

                return finalize_inflation_s5b1(self)
            control_result = self._post_commit_control_result()
            outer_commit_result = self._execute_pending_joint_buffer_commit(
                control_result
            )
            if outer_commit_result is not None:
                control_result = outer_commit_result
            if isinstance(control_result, dict) and control_result.get("status") == "failed":
                outer_windows.append(
                    {
                        "window_index": outer_window_index,
                        "status": "failed",
                        "commit": dict(control_result),
                    }
                )
                self.last_run_result = {
                    "status": "failed",
                    "stop_reason": "buffer_commit_execution_failed",
                    "joint_buffer_commit": dict(control_result),
                    "outer_windows": outer_windows,
                }
                self.last_run_result["outer_lifecycle_artifact_path"] = (
                    self._write_joint_outer_lifecycle_artifact(
                        status="failed",
                        windows=outer_windows,
                        result=self.last_run_result,
                    )
                )
                return self.last_run_result
            try:
                post_commit_result = self._continue_after_joint_buffer_commit(
                    control_result
                )
            except Exception as exc:
                outer_windows.append(
                    {
                        "window_index": outer_window_index,
                        "status": "failed",
                        "commit": dict(control_result or {}),
                        "error": str(exc),
                    }
                )
                self.last_run_result = {
                    "status": "failed",
                    "stop_reason": "buffer_commit_refresh_or_rebuild_failed",
                    "error": str(exc),
                    "joint_buffer_commit": dict(control_result or {}),
                    "outer_windows": outer_windows,
                }
                self.last_run_result["outer_lifecycle_artifact_path"] = (
                    self._write_joint_outer_lifecycle_artifact(
                        status="failed",
                        windows=outer_windows,
                        result=self.last_run_result,
                    )
                )
                return self.last_run_result

            if post_commit_result is None or post_commit_result.get("status") != "ok":
                break

            request = dict((control_result or {}).get("commit_request", {}) or {})
            is_periodic = request.get("reason") == "joint_buffer_periodic_commit"
            outer_windows.append(
                {
                    "window_index": outer_window_index,
                    "status": (
                        "verified"
                        if self._is_segment_direct_joint()
                        else "rebuilt"
                    ),
                    "boundary": "periodic" if is_periodic else "final",
                    "request_action_digest": request.get("action_digest"),
                    "request_sizing_action_digest": request.get(
                        "sizing_action_digest"
                    ),
                    "topology_generation": post_commit_result.get(
                        "topology_generation"
                    ),
                }
            )
            if (
                is_periodic
                and outer_window_index + 1 < self._joint_buffer_outer_iterations()
            ):
                outer_window_index += 1
                continue
            logging.info("joint post-commit rebuild result: %s", post_commit_result)
            self.last_run_result = {
                "status": "ok",
                "rsmt": self.rsmt,
                "hpwl": placement_debug.get("final_hpwl", self.hpwl),
                "congestion": self.congestion,
                "density": placement_debug.get(
                    "final_max_density",
                    self.density,
                ),
                "iteration": placement_debug.get("optimizer_steps", 0),
                "objective": -1,
                "overflow": placement_debug.get("final_overflow", -1),
                "max_density": placement_debug.get("final_max_density", -1),
                "joint_post_rebuild": post_commit_result,
                "outer_windows": outer_windows,
            }
            self.last_run_result["outer_lifecycle_artifact_path"] = (
                self._write_joint_outer_lifecycle_artifact(
                    status="completed",
                    windows=outer_windows,
                    result=self.last_run_result,
                )
            )
            return self.last_run_result
        # with torch.profiler.profile(
        #     activities=[
        #         torch.profiler.ProfilerActivity.CPU,
        #         torch.profiler.ProfilerActivity.CUDA,
        #     ]
        # ) as p:
        #     self.place()
        # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
        # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

        buffering_summary = getattr(
            getattr(self, "placer", None),
            "last_buffering_lane_summary",
            None,
        )
        if (
            getattr(self.params, "flow_kind", None) == FlowKind.BUFFERING.value
            and isinstance(buffering_summary, dict)
            and buffering_summary.get("status") == "completed"
        ):
            self.last_run_result = {
                "status": "ok",
                "flow_kind": FlowKind.BUFFERING.value,
                "buffering_mode": buffering_summary.get("mode"),
                "buffering_commit_status": dict(
                    buffering_summary.get("commit", {}) or {}
                ).get("status"),
            }
            return self.last_run_result

        placement_status = "ok"
        placement_stop_reason = None
        debug_summary = getattr(getattr(self, "placer", None), "last_placement_debug_summary", None)
        if isinstance(debug_summary, dict):
            placement_status = str(debug_summary.get("status") or placement_status)
            placement_stop_reason = debug_summary.get("stop_reason")
        if self.hpwl != float("inf") and placement_status == "ok":
            # self.save_placement()
            self.density = float(self.params.target_density)

            if self.params.detailed_place_engine and os.path.exists(
                self.params.detailed_place_engine
            ):
                self.external_detailed_placer()
            elif self.params.detailed_place_engine:
                logging.warning(
                    "External detailed placement engine %s or aux file NOT found"
                    % (self.params.detailed_place_engine)
                )

            if self.params.get_congestion_map and hasattr(
                self.placer.op_collections, "get_congestion_map_op"
            ):
                tt2 = time.time()
                self.get_congestion()
                logging.info(
                    "congestion extraction takes %.3f seconds" % (
                        time.time() - tt2)
                )

            logging.info("placement takes %.3f seconds" % (time.time() - tt))
        else:
            logging.warning("placement failed")
            if placement_status == "ok":
                placement_status = "failed"

        final_ppa = {
            "status": placement_status,
            "stop_reason": placement_stop_reason,
            "rsmt": self.rsmt,
            "hpwl": self.hpwl,
            "congestion": self.congestion,
            "density": self.density,
        }
        niter = len(self.metrics["objective"])
        other_metrics = {
            "iteration": niter,
            "objective": -1 if niter == 0 else self.metrics["objective"][-1],
            "overflow": -1 if niter == 0 else self.metrics["overflow"][-1],
            "max_density": -1 if niter == 0 else self.metrics["density"][-1],
        }
        final_ppa.update(other_metrics)
        logging.info(f"Final PPA: {final_ppa}")

        if self.params.timing_rc_mode == "gr":
            from dreamplace.flows.gr_sizing import finalize_gr_sizing

            final_ppa = finalize_gr_sizing(self, final_ppa)
        self.last_run_result = final_ppa
        return final_ppa


def main(argv=None, output_stream=None):
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.params_json:
        parser.print_help(file=output_stream or sys.stdout)
        return 0
    try:
        params, launch = build_effective_params_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run_config:
        stream = output_stream or sys.stdout
        json.dump(params.toJson(), stream, indent=2, sort_keys=True)
        stream.write("\n")
        return 0

    if (
        launch["setup_margin"] is not None
        and placer_cli.uses_buffer_only_handoff(params)
    ):
        handoff_launcher.install_buffer_only_margin_command_patch(
            launch["setup_margin"]
        )

    logging.info(
        "launch workspace=%s output_def=%s place_io_engine=%s",
        launch["workspace"],
        launch["output_def"],
        getattr(params, "place_io_engine", None),
    )
    cuda_metrics_enabled = bool(getattr(params, "gpu", 0)) and torch.cuda.is_available()
    if cuda_metrics_enabled:
        torch.cuda.reset_peak_memory_stats()
    flow_started = time.perf_counter()
    try:
        flow_result = run_optimization_flow(
            params,
            launch,
            engine_cls=PlacementEngine,
        )
    finally:
        if cuda_metrics_enabled:
            torch.cuda.synchronize()
            logging.info(
                "TOOLS_METRIC|gpu_peak_allocated_bytes|%d",
                int(torch.cuda.max_memory_allocated()),
            )
            logging.info(
                "TOOLS_METRIC|gpu_peak_reserved_bytes|%d",
                int(torch.cuda.max_memory_reserved()),
            )
        else:
            logging.info("TOOLS_METRIC|gpu_peak_allocated_bytes|0")
            logging.info("TOOLS_METRIC|gpu_peak_reserved_bytes|0")
        logging.info(
            "TOOLS_METRIC|process_flow_runtime_sec|%.9f",
            time.perf_counter() - flow_started,
        )
    run_result = getattr(flow_result.engine, "last_run_result", None)
    if isinstance(run_result, dict) and str(run_result.get("status") or "ok") != "ok":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
