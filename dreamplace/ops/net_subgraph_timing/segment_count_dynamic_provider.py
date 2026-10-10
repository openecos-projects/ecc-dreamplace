import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from dreamplace.ops.buffer_insertion.buffer_library import (
    lookup_lut_value_with_status_mode,
)
from dreamplace.ops.net_subgraph_timing.segment_count_relaxed_timing import (
    _interpolate_size_tables,
    _split_fractions,
    segment_count_relaxed_timing,
)
from dreamplace.ops.net_subgraph_timing.segment_count_prepared_timing import (
    segment_count_prepared_relaxed_timing,
)
from dreamplace.ops.net_subgraph_timing.segment_count_live_geometry import (
    build_live_edge_geometry,
)
from dreamplace.ops.net_subgraph_timing.segment_count_native import (
    segment_count_driver_cap_native_cuda_autograd,
    segment_count_forward_cuda_segment_transfer_explicit_autograd,
    segment_count_forward_native_explicit_autograd,
    segment_count_forward_native,
    segment_count_forward_native_recompute_autograd,
    segment_count_forward_native_segment_transfer_explicit_autograd,
    segment_count_forward_native_segment_transfer_recompute_autograd,
)
from dreamplace.ops.net_subgraph_timing.segment_count_native_packer import (
    select_packed_segment_count_inputs,
    select_packed_segment_count_inputs_native,
)
from dreamplace.ops.net_subgraph_timing.segment_count_tensor_builder import (
    build_segment_count_timing_inputs,
)
from dreamplace.ops.net_subgraph_timing.segment_transfer import (
    SegmentTransferInput,
    analytic_segment_transfer,
)
from dreamplace.ops.timing_propagation.dynamic_net_provider import DynamicNetProvider


CUDA_SEGMENT_COUNT_BACKENDS = {
    "cpp_cuda_segment_transfer_explicit_autograd",
    "cpp_cuda_explicit_autograd",
}
CUDA_SEGMENT_COUNT_ALIAS_BACKEND = "cpp_cuda_segment_transfer_explicit_autograd"
SEGMENT_TRANSFER_TIMING_BACKENDS = {
    "cpp_cpu_segment_transfer_recompute_autograd",
    "cpp_cpu_segment_transfer_explicit_autograd",
    "cpp_cuda_segment_transfer_explicit_autograd",
}
TENSORIZED_DRIVER_GATHER_TIMING_BACKENDS = {
    "prepared_python",
    "cpp_cpu",
    "cpp_cpu_recompute_autograd",
    "cpp_cpu_explicit_autograd",
    *SEGMENT_TRANSFER_TIMING_BACKENDS,
}


def _net_id(net, fallback):
    return int(net.get("net_id", fallback))


def _as_net_id_set(net_ids):
    if torch.is_tensor(net_ids):
        return {int(value) for value in net_ids.detach().cpu().tolist()}
    return {int(value) for value in list(net_ids or [])}


def _env_float_list(name):
    raw = os.environ.get(name, "")
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return []
    return values


def _driver_pin_by_net(nets):
    result = {}
    for index, net in enumerate(list(nets or [])):
        net_id = _net_id(net, index)
        driver_pin = net.get("driver_pin_id")
        if driver_pin is None:
            driver_pin = (net.get("rc_tree") or {}).get("root_node_id")
        if driver_pin is not None:
            result[int(net_id)] = int(driver_pin)
    return result


def _sink_pin_rows(nets):
    sink_pin_id = []
    sink_net_id = []
    for index, net in enumerate(list(nets or [])):
        net_id = _net_id(net, index)
        for sink in (net.get("rc_tree") or {}).get("sink_nodes", []):
            sink_pin_id.append(int(sink))
            sink_net_id.append(int(net_id))
    return sink_pin_id, sink_net_id


def _tree_preorder(root_node_id, children_by_node):
    order = []
    stack = [int(root_node_id)]
    seen = set()
    while stack:
        node_id = int(stack.pop())
        if node_id in seen:
            continue
        seen.add(node_id)
        order.append(node_id)
        children = [int(child) for child in children_by_node.get(node_id, [])]
        stack.extend(reversed(children))
    return order


def _prepare_net(net, fallback):
    net_id = _net_id(net, fallback)
    rc_tree = dict(net.get("rc_tree", {}) or {})
    root = rc_tree.get("root_node_id", net.get("driver_pin_id"))
    if root is not None:
        root = int(root)
    children_by_node = {
        int(node): [int(child) for child in children]
        for node, children in (rc_tree.get("children_by_node", {}) or {}).items()
    }
    prepared = dict(net)
    prepared["_segment_count_prepared"] = {
        "net_id": int(net_id),
        "root_node_id": root,
        "children_by_node": children_by_node,
        "edge_rc": rc_tree.get("edge_rc", {}) or {},
        "node_cap": {
            int(node): cap
            for node, cap in (rc_tree.get("node_cap", {}) or {}).items()
        },
        "topo": _tree_preorder(root, children_by_node) if root is not None else [],
        "sink_nodes": [
            int(sink)
            for sink in rc_tree.get("sink_nodes", [])
        ],
    }
    return prepared


def _cache_key_for_active_nets(active_net_ids):
    return tuple(sorted(int(net_id) for net_id in active_net_ids))


def _interp_arc_scalar(value_by_size, bsu_index):
    if not value_by_size:
        return None
    size_count = len(value_by_size)
    clipped = max(0.0, min(float(bsu_index), float(size_count - 1)))
    lo = int(math.floor(clipped))
    hi = min(lo + 1, size_count - 1)
    alpha = clipped - float(lo)
    return (1.0 - alpha) * float(value_by_size[lo]) + alpha * float(value_by_size[hi])


def _summarize_values(values):
    finite = [float(value) for value in values if torch.isfinite(torch.tensor(float(value)))]
    if not finite:
        return {"count": int(len(values)), "finite_count": 0}
    return {
        "count": int(len(values)),
        "finite_count": int(len(finite)),
        "min": float(min(finite)),
        "max": float(max(finite)),
        "mean": float(sum(finite) / float(len(finite))),
    }


def _finite_quantiles(tensor, fractions=(0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)):
    tensor = tensor.detach().flatten()
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) == 0:
        return {str(fraction): None for fraction in fractions}
    return {
        str(fraction): float(torch.quantile(finite, float(fraction)).detach().cpu().item())
        for fraction in fractions
    }


def _finite_stats(tensor):
    tensor = tensor.detach().flatten()
    finite = tensor[torch.isfinite(tensor)]
    if int(finite.numel()) == 0:
        return {
            "count": int(tensor.numel()),
            "finite_count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "abs_max": None,
            "quantiles": _finite_quantiles(finite),
        }
    return {
        "count": int(tensor.numel()),
        "finite_count": int(finite.numel()),
        "min": float(torch.min(finite).detach().cpu().item()),
        "max": float(torch.max(finite).detach().cpu().item()),
        "mean": float(torch.mean(finite).detach().cpu().item()),
        "abs_max": float(torch.max(torch.abs(finite)).detach().cpu().item()),
        "quantiles": _finite_quantiles(finite),
    }


def _edge_rc_value(edge_rc, parent, child, key):
    value = (edge_rc or {}).get((int(parent), int(child)))
    if value is None:
        value = (edge_rc or {}).get((str(int(parent)), str(int(child))))
    if isinstance(value, dict):
        return float(value.get(key, 0.0) or 0.0)
    if key == "r" and value is not None:
        return float(value)
    return 0.0


def _sum_tree_edge_cap(children_by_node, edge_rc):
    total = 0.0
    edge_count = 0
    missing_cap_count = 0
    for parent, children in (children_by_node or {}).items():
        for child in children:
            edge_count += 1
            cap = _edge_rc_value(edge_rc, parent, child, "c")
            if cap == 0.0:
                missing_cap_count += 1
            total += cap
    return total, edge_count, missing_cap_count


def _parent_by_child(children_by_node):
    result = {}
    for parent, children in (children_by_node or {}).items():
        for child in children:
            result[int(child)] = int(parent)
    return result


def _root_to_node_path(root_node_id, target_node_id, children_by_node):
    root_node_id = int(root_node_id)
    target_node_id = int(target_node_id)
    if root_node_id == target_node_id:
        return [root_node_id]
    parent_by_child = _parent_by_child(children_by_node)
    reverse_path = [target_node_id]
    seen = {target_node_id}
    current = target_node_id
    while current in parent_by_child:
        current = int(parent_by_child[current])
        if current in seen:
            return []
        reverse_path.append(current)
        seen.add(current)
        if current == root_node_id:
            return list(reversed(reverse_path))
    return []


def _path_edge_rc_summary(path_node_ids, edge_rc):
    resistance = 0.0
    capacitance = 0.0
    missing_r_count = 0
    missing_c_count = 0
    edges = []
    for parent, child in zip(path_node_ids, path_node_ids[1:]):
        r_value = _edge_rc_value(edge_rc, parent, child, "r")
        c_value = _edge_rc_value(edge_rc, parent, child, "c")
        if r_value == 0.0:
            missing_r_count += 1
        if c_value == 0.0:
            missing_c_count += 1
        resistance += r_value
        capacitance += c_value
        edges.append(
            {
                "parent_node_id": int(parent),
                "child_node_id": int(child),
                "r": float(r_value),
                "c": float(c_value),
            }
        )
    return {
        "edge_count": int(len(edges)),
        "r_sum": float(resistance),
        "c_sum": float(capacitance),
        "missing_or_zero_r_count": int(missing_r_count),
        "missing_or_zero_c_count": int(missing_c_count),
        "edges": edges,
    }


class SegmentCountDynamicNetProvider(DynamicNetProvider):
    """TimingPropagation provider for segment-count relaxed buffering."""

    def __init__(
        self,
        *,
        nets,
        segment_state,
        per_size_input_cap,
        per_size_delay,
        per_size_output_slew,
        affected_net_ids=None,
        backend="prepared_python",
        profile_enabled=True,
        buffer_device_lut=None,
        segment_retained_upstream_cap=None,
        segment_transfer_backend="static_size_table",
        prepared_timing_inputs=None,
        driver_cap_mode="residual",
    ):
        if driver_cap_mode not in {"residual", "direct"}:
            raise ValueError(f"unsupported segment driver cap mode: {driver_cap_mode}")
        self.driver_cap_mode = driver_cap_mode
        self.backend = str(backend or "prepared_python")
        self.segment_transfer_backend = str(segment_transfer_backend or "static_size_table")
        self.profile_enabled = bool(profile_enabled)
        if self.backend not in {
            "reference_python",
            "prepared_python",
            "cpp_cpu",
            "cpp_cpu_recompute_autograd",
            "cpp_cpu_explicit_autograd",
            "cpp_cpu_segment_transfer_recompute_autograd",
            "cpp_cpu_segment_transfer_explicit_autograd",
            *CUDA_SEGMENT_COUNT_BACKENDS,
        }:
            raise ValueError(f"unsupported segment-count timing backend: {self.backend}")
        self.backend_requested = self.backend
        self.backend_used = (
            CUDA_SEGMENT_COUNT_ALIAS_BACKEND
            if self.backend in CUDA_SEGMENT_COUNT_BACKENDS
            else self.backend
        )
        self.backend_fallback_reason = None
        self.segment_transfer_backend_requested = self.segment_transfer_backend
        self.segment_transfer_backend_used = (
            "segment_transfer_native"
            if self.backend_used in SEGMENT_TRANSFER_TIMING_BACKENDS
            else self.segment_transfer_backend
        )
        if self.segment_transfer_backend not in {
            "static_size_table",
            "segment_transfer_python",
            "segment_transfer_native",
        }:
            raise ValueError(
                f"unsupported segment transfer backend: {self.segment_transfer_backend}"
            )
        if (
            self.segment_transfer_backend != "static_size_table"
            and self.backend_used
            not in {"prepared_python", *SEGMENT_TRANSFER_TIMING_BACKENDS}
        ):
            raise ValueError(
                "segment transfer backend is currently supported only with prepared_python "
                "or cpp_cpu_segment_transfer_*_autograd"
            )
        self.packed_prepared_inputs = (
            dict(prepared_timing_inputs)
            if isinstance(prepared_timing_inputs, dict)
            else None
        )
        self.nets = (
            []
            if self.packed_prepared_inputs is not None
            else [
                _prepare_net(net, index)
                for index, net in enumerate(list(nets or []))
            ]
        )
        self.segment_state = segment_state
        self.per_size_input_cap = torch.as_tensor(per_size_input_cap)
        self.per_size_delay = torch.as_tensor(
            per_size_delay,
            dtype=self.per_size_input_cap.dtype,
            device=self.per_size_input_cap.device,
        )
        self.per_size_output_slew = torch.as_tensor(
            per_size_output_slew,
            dtype=self.per_size_input_cap.dtype,
            device=self.per_size_input_cap.device,
        )
        self.buffer_device_lut = buffer_device_lut
        self.segment_retained_upstream_cap = (
            None
            if segment_retained_upstream_cap is None
            else torch.as_tensor(
                segment_retained_upstream_cap,
                dtype=self.per_size_input_cap.dtype,
                device=self.per_size_input_cap.device,
            )
        )
        if (
            self.segment_retained_upstream_cap is not None
            and int(self.segment_retained_upstream_cap.numel())
            != int(segment_state.z_param.numel())
        ):
            raise ValueError(
                "segment_retained_upstream_cap length must match segment state"
            )
        self.net_ids = (
            self.packed_prepared_inputs["net_ids"].detach().cpu().tolist()
            if self.packed_prepared_inputs is not None
            else [_net_id(net, index) for index, net in enumerate(self.nets)]
        )
        self.net_by_id = {
            int(net_id): net
            for net_id, net in zip(self.net_ids, self.nets)
        }
        self.affected_net_ids = (
            {int(net_id) for net_id in affected_net_ids}
            if affected_net_ids is not None
            else {
                int(net_id)
                for net_id in (
                    segment_state.net_ids.detach().cpu().tolist()
                    if self.packed_prepared_inputs is not None
                    else self.net_ids
                )
            }
        )
        self.driver_pin_by_net = (
            {}
            if self.packed_prepared_inputs is not None
            else _driver_pin_by_net(self.nets)
        )
        self.net_by_driver_pin = {
            int(pin_id): int(net_id)
            for net_id, pin_id in self.driver_pin_by_net.items()
        }
        if self.packed_prepared_inputs is not None:
            sink_pin_id = self.packed_prepared_inputs["sink_node_id"]
            sink_net_id = self.packed_prepared_inputs["sink_net_id"]
        else:
            sink_pin_id, sink_net_id = _sink_pin_rows(self.nets)
        self.sink_pin_id = sink_pin_id
        self.sink_net_id = sink_net_id
        self.sink_count = len(sink_pin_id)
        self._prepared_inputs_by_active_net = {}
        self._active_view_cache = {}
        self._level_view_cache = {}
        self._active_segment_structure_cache = {}
        self._driver_index_cache = {}
        self._static_device_tensor_cache = {}
        self._buffer_device_lut_cache = {}
        self._cap_overlay_probe_written = False
        self._aat_overlay_probe_written = False
        self._aat_overlay_probe_state = None
        self._device_probe_state = None
        self._last_pin_net_cap_rise_after_overlay = None
        self._last_pin_net_cap_fall_after_overlay = None
        self._critical_path_pin_net_delay_rise = None
        self._critical_path_pin_net_delay_fall = None
        self._virtual_slew_violation = self.segment_state.z_param.new_zeros(())
        self._virtual_cap_violation = self.segment_state.z_param.new_zeros(())
        self._live_geometry_context = None
        self.call_count = 0
        self.affected_call_count = 0
        self.metadata = {
            "timing_integration_mode": "segment_count_dynamic_net_provider",
            "net_subgraph_invocation_timing": "during_timing_propagation_net_aat",
            "uses_current_propagated_driver_slew": True,
            "virtual_drv_limit_coverage": getattr(buffer_device_lut, "drv_limit_coverage", None),
            "dynamic_provider_call_count": 0,
            "dynamic_provider_affected_net_call_count": 0,
            "affected_net_call_count": 0,
            "provider_dispatch_ms": 0.0,
            "driver_timing_gather_ms": 0.0,
            "segment_count_relaxed_timing_ms": 0.0,
            "segment_count_timing_forward_ms": 0.0,
            "net_subgraph_forward_relaxed_ms": 0.0,
            "rise_transfer_ms": 0.0,
            "fall_transfer_ms": 0.0,
            "sink_scatter_ms": 0.0,
            "rise_sink_scatter_ms": 0.0,
            "fall_sink_scatter_ms": 0.0,
            "driver_cap_overlay_ms": 0.0,
            "driver_cap_overlay_forward_ms": 0.0,
            "driver_cap_overlay_scatter_ms": 0.0,
            "driver_cap_overlay_backend": None,
            "driver_cap_overlay_path": None,
            "driver_cap_overlay_fallback_reason": None,
            "driver_cap_overlay_count": 0,
            "native_cap_backward_invocation_count": 0,
            "native_cap_backward_temporary_allocation_bytes": 0,
            "native_cap_backward_temporary_allocation_cpu_ms": 0.0,
            "static_payload_cache_hit_count": 0,
            "level_view_cache_hit_count": 0,
            "level_view_build_count": 0,
            "gpu_static_runtime_build_count": 0,
            "gpu_static_runtime_cache_hit_count": 0,
            "gpu_static_runtime_build_wall_ms": 0.0,
            "gpu_static_runtime_reuse_wall_ms": 0.0,
            "gpu_static_runtime_bytes": 0,
            "gpu_static_runtime_h2d_bytes": 0,
            "gpu_static_runtime_d2h_bytes": 0,
            "gpu_peak_allocated_bytes": 0,
            "gpu_peak_reserved_bytes": 0,
            "tensorized_driver_gather_count": 0,
            "active_view_build_count": 0,
            "active_view_build_ms": 0.0,
            "active_view_cache_hit_count": 0,
            "active_view_cache_hit_ms": 0.0,
            "active_segment_structure_build_count": 0,
            "active_segment_structure_build_ms": 0.0,
            "active_segment_structure_cache_hit_count": 0,
            "active_segment_structure_cache_hit_ms": 0.0,
            "prepared_inputs_build_count": 0,
            "prepared_inputs_build_ms": 0.0,
            "prepared_inputs_cache_hit_count": 0,
            "prepared_inputs_cache_hit_ms": 0.0,
            "prepared_inputs_segment_lookup_cpu_ms": 0.0,
            "prepared_inputs_tree_flatten_cpu_ms": 0.0,
            "prepared_inputs_fraction_table_cpu_ms": 0.0,
            "prepared_inputs_tensor_materialize_cpu_ms": 0.0,
            "prepared_inputs_total_cpu_ms": 0.0,
            "active_view_selector_requested": False,
            "active_view_selector_used": "python_reference",
            "active_view_selector_fallback_reason": None,
            "native_active_view_selector_count": 0,
            "segment_driver_cap_mode": self.driver_cap_mode,
            "segment_count_state_count": int(segment_state.z_param.numel()),
            "segment_state_device": str(segment_state.z_param.device),
            "buffer_device_lut_available": buffer_device_lut is not None,
            "max_repeater_count": int(segment_state.max_repeater_count),
            "segment_shared_bsu": True,
            "segment_count_timing_backend_requested": self.backend_requested,
            "segment_count_timing_backend_used": self.backend_used,
            "segment_count_timing_backend_fallback_reason": self.backend_fallback_reason,
            "segment_count_timing_backend_fallback_is_explicit": (
                self.backend_fallback_reason is not None
            ),
            "segment_transfer_backend_requested": self.segment_transfer_backend_requested,
            "segment_transfer_backend": self.segment_transfer_backend_used,
            "segment_transfer_backend_used": self.segment_transfer_backend_used,
            "segment_transfer_forward_ms": 0.0,
            "profile_enabled": self.profile_enabled,
            "segment_transfer_device_lut_source": (
                getattr(buffer_device_lut, "source", None)
                if buffer_device_lut is not None
                else None
            ),
            "segment_retained_upstream_cap_source": (
                "explicit_segment_tensor"
                if self.segment_retained_upstream_cap is not None
                else None
            ),
            "prepared_input_source": (
                "native_packed"
                if self.packed_prepared_inputs is not None
                else "python_net_records"
            ),
            "live_geometry_bound": False,
            "live_geometry_bind_count": 0,
            "live_geometry_release_count": 0,
            "live_rc_full_build_count": 0,
            "live_rc_local_build_count": 0,
            "live_rc_gather_count": 0,
            "live_rc_full_build_ms": 0.0,
            "live_rc_local_build_ms": 0.0,
            "live_rc_gather_ms": 0.0,
            "live_rc_explicit_source_edge_mapping_count": 0,
            "live_rc_identity_source_edge_mapping_count": 0,
            "live_rc_source_edge_index_count": 0,
            "live_geometry_gradient_hook_count": 0,
            "live_geometry_gradient_stats": {},
            "critical_path_delay_overlay_sink_count": 0,
        }

    def _packed_driver_pins(self, net_ids):
        prepared = self.packed_prepared_inputs
        if prepared is None:
            return [self.driver_pin_by_net.get(int(net_id)) for net_id in net_ids]
        packed_net_ids = prepared["net_ids"].to(device="cpu", dtype=torch.long)
        query = torch.as_tensor(list(net_ids), dtype=torch.long)
        positions = torch.searchsorted(packed_net_ids, query)
        result = []
        for net_id, position in zip(query.tolist(), positions.tolist()):
            if position < int(packed_net_ids.numel()) and int(packed_net_ids[position]) == int(net_id):
                result.append(int(prepared["driver_pin_id"][position]))
            else:
                result.append(None)
        return result

    def _native_active_view_supported(self):
        if self.packed_prepared_inputs is None:
            return False, "python_net_records"
        if self.backend_used not in CUDA_SEGMENT_COUNT_BACKENDS:
            return False, "non_cuda_segment_backend"
        if not getattr(self.segment_state, "is_tensor_backed", False):
            return False, "non_tensor_backed_segment_state"
        probe_variables = (
            "AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON",
            "AIMP_BUFFERING_SEGMENT_AAT_OVERLAY_PROBE_JSON",
        )
        if any(bool(os.environ.get(name)) for name in probe_variables):
            return False, "diagnostic_probe_enabled"
        return True, None

    def _level_view(self, net_ids, *, level_id=None, level_view_epoch=None):
        if level_id is None:
            active_net_ids = _as_net_id_set(net_ids) & self.affected_net_ids
            return {
                "active_net_ids": active_net_ids,
                "has_dynamic_nets": bool(active_net_ids),
            }

        cache_key = (int(level_view_epoch or 0), int(level_id))
        cached = self._level_view_cache.get(cache_key)
        if cached is not None:
            self.metadata["level_view_cache_hit_count"] += 1
            self.metadata["static_payload_cache_hit_count"] += 1
            return cached

        use_native, fallback_reason = self._native_active_view_supported()
        self.metadata["active_view_selector_requested"] = True
        if use_native:
            selected = select_packed_segment_count_inputs_native(
                self.packed_prepared_inputs,
                self.segment_state.packed_segment_geometry,
                active_net_ids=net_ids,
                dtype=self.segment_state.z_param.dtype,
                device=self.segment_state.z_param.device,
            )
            source_row_index = selected["source_segment_row_index"]
            driver_pin_index = selected["driver_pin_id"]
            cached = {
                "active_net_ids": (),
                "has_dynamic_nets": bool(int(source_row_index.numel()) > 0),
                "native_active_view": True,
                "prepared_inputs": selected,
                "source_segment_row_index": source_row_index,
                "driver_pin_index": driver_pin_index,
                "driver_pin_valid": driver_pin_index >= 0,
            }
            self.metadata["active_view_selector_used"] = "native"
            self.metadata["active_view_selector_fallback_reason"] = None
            self.metadata["native_active_view_selector_count"] += 1
            self._level_view_cache[cache_key] = cached
            self.metadata["level_view_build_count"] += 1
            return cached

        active_net_ids = _as_net_id_set(net_ids) & self.affected_net_ids
        row_indices, active_nets = self._active_view(active_net_ids)
        ordered_active_net_ids = [
            int(net) if not isinstance(net, dict) else int(
                net.get("_segment_count_prepared", {}).get("net_id", -1)
            )
            for net in active_nets
        ]
        driver_indices = []
        driver_valid = []
        for driver_pin in self._packed_driver_pins(ordered_active_net_ids):
            driver_indices.append(0 if driver_pin is None else int(driver_pin))
            driver_valid.append(driver_pin is not None and int(driver_pin) >= 0)
        cached = {
            "active_net_ids": active_net_ids,
            "has_dynamic_nets": bool(active_net_ids),
            "row_indices": tuple(int(row) for row in row_indices),
            "active_nets": tuple(active_nets),
            "driver_pin_index": torch.as_tensor(
                driver_indices,
                dtype=torch.long,
                device=self.segment_state.z_param.device,
            ),
            "driver_pin_valid": torch.as_tensor(
                driver_valid,
                dtype=torch.bool,
                device=self.segment_state.z_param.device,
            ),
        }
        self._level_view_cache[cache_key] = cached
        self.metadata["active_view_selector_used"] = "python_reference"
        self.metadata["active_view_selector_fallback_reason"] = fallback_reason
        self.metadata["level_view_build_count"] += 1
        return cached

    def has_dynamic_nets(self, net_ids, *, level_id=None, level_view_epoch=None):
        return bool(
            self._level_view(
                net_ids,
                level_id=level_id,
                level_view_epoch=level_view_epoch,
            )["has_dynamic_nets"]
        )

    def reset_runtime_metadata(self):
        if self._live_geometry_context is not None:
            raise RuntimeError("cannot reset provider metadata while live geometry is bound")
        self.call_count = 0
        self.affected_call_count = 0
        self._aat_overlay_probe_state = None
        self._aat_overlay_probe_written = False
        self._last_pin_net_cap_rise_after_overlay = None
        self._last_pin_net_cap_fall_after_overlay = None
        self._critical_path_pin_net_delay_rise = None
        self._critical_path_pin_net_delay_fall = None
        for key in (
            "dynamic_provider_call_count",
            "dynamic_provider_affected_net_call_count",
            "affected_net_call_count",
            "provider_dispatch_ms",
            "driver_timing_gather_ms",
            "segment_count_relaxed_timing_ms",
            "segment_count_timing_forward_ms",
            "net_subgraph_forward_relaxed_ms",
            "rise_transfer_ms",
            "fall_transfer_ms",
            "segment_transfer_forward_ms",
            "sink_scatter_ms",
            "rise_sink_scatter_ms",
            "fall_sink_scatter_ms",
            "driver_cap_overlay_ms",
            "driver_cap_overlay_forward_ms",
            "driver_cap_overlay_scatter_ms",
            "driver_cap_overlay_count",
            "native_cap_backward_invocation_count",
            "native_cap_backward_temporary_allocation_bytes",
            "native_cap_backward_temporary_allocation_cpu_ms",
            "static_payload_cache_hit_count",
            "level_view_cache_hit_count",
            "level_view_build_count",
            "gpu_static_runtime_build_count",
            "gpu_static_runtime_cache_hit_count",
            "gpu_static_runtime_build_wall_ms",
            "gpu_static_runtime_reuse_wall_ms",
            "gpu_static_runtime_h2d_bytes",
            "gpu_static_runtime_d2h_bytes",
            "gpu_peak_allocated_bytes",
            "gpu_peak_reserved_bytes",
            "tensorized_driver_gather_count",
            "active_view_build_count",
            "active_view_build_ms",
            "active_view_cache_hit_count",
            "active_view_cache_hit_ms",
            "active_segment_structure_build_count",
            "active_segment_structure_build_ms",
            "active_segment_structure_cache_hit_count",
            "active_segment_structure_cache_hit_ms",
            "prepared_inputs_build_count",
            "prepared_inputs_build_ms",
            "prepared_inputs_cache_hit_count",
            "prepared_inputs_cache_hit_ms",
            "prepared_inputs_segment_lookup_cpu_ms",
            "prepared_inputs_tree_flatten_cpu_ms",
            "prepared_inputs_fraction_table_cpu_ms",
            "prepared_inputs_tensor_materialize_cpu_ms",
            "prepared_inputs_total_cpu_ms",
            "live_geometry_bind_count",
            "live_geometry_release_count",
            "live_rc_full_build_count",
            "live_rc_local_build_count",
            "live_rc_gather_count",
            "live_rc_full_build_ms",
            "live_rc_local_build_ms",
            "live_rc_gather_ms",
            "live_rc_explicit_source_edge_mapping_count",
            "live_rc_identity_source_edge_mapping_count",
            "live_rc_source_edge_index_count",
            "live_geometry_gradient_hook_count",
            "critical_path_delay_overlay_sink_count",
        ):
            self.metadata[key] = 0
        self.metadata["driver_cap_overlay_backend"] = None
        self.metadata["driver_cap_overlay_path"] = None
        self.metadata["driver_cap_overlay_fallback_reason"] = None
        self.metadata["active_view_selector_requested"] = False
        self.metadata["active_view_selector_used"] = "python_reference"
        self.metadata["active_view_selector_fallback_reason"] = None
        self.metadata["native_active_view_selector_count"] = 0
        self.metadata["live_geometry_bound"] = False
        self.metadata["live_geometry_gradient_stats"] = {}
        return self

    def update_segment_state(self, segment_state):
        if (
            id(getattr(segment_state, "segment_rows", None))
            != id(getattr(self.segment_state, "segment_rows", None))
            or int(segment_state.z_param.numel()) != int(self.segment_state.z_param.numel())
            or segment_state.z_param.device != self.segment_state.z_param.device
            or segment_state.z_param.dtype != self.segment_state.z_param.dtype
        ):
            self._prepared_inputs_by_active_net.clear()
            self._active_view_cache.clear()
            self._level_view_cache.clear()
            self._active_segment_structure_cache.clear()
            self._driver_index_cache.clear()
            self._static_device_tensor_cache.clear()
            self._buffer_device_lut_cache.clear()
            self._cap_overlay_probe_written = False
            self._device_probe_state = None
        self.segment_state = segment_state
        self.metadata["segment_count_state_count"] = int(segment_state.z_param.numel())
        self.metadata["segment_state_device"] = str(segment_state.z_param.device)
        self.metadata["max_repeater_count"] = int(segment_state.max_repeater_count)
        return self

    def bind_live_geometry(
        self,
        *,
        new_x,
        new_y,
        r_unit,
        c_unit,
        scale_factor,
        dbu,
        topology_epoch,
        forward_id,
        require_axis_aligned=True,
        pin_capacitance_by_sense=None,
    ):
        if self._live_geometry_context is not None:
            raise RuntimeError("live geometry is already bound to this provider")
        topology_epoch = int(topology_epoch)
        expected_epoch = int(self.metadata.get("topology_epoch", topology_epoch))
        if topology_epoch != expected_epoch:
            raise ValueError(
                "live geometry topology epoch does not match the provider epoch"
            )
        if pin_capacitance_by_sense is not None:
            pin_capacitance_by_sense = {
                sense: torch.nn.functional.pad(caps, (0, new_x.numel() - caps.numel()))
                for sense, caps in pin_capacitance_by_sense.items()
            }
        context = {
            "new_x": new_x,
            "new_y": new_y,
            "r_unit": float(r_unit),
            "c_unit": float(c_unit),
            "scale_factor": float(scale_factor),
            "dbu": float(dbu),
            "topology_epoch": topology_epoch,
            "forward_id": forward_id,
            "require_axis_aligned": bool(require_axis_aligned),
            "full_geometry": None,
            "pin_capacitance_by_sense": pin_capacitance_by_sense,
        }
        if self.packed_prepared_inputs is not None:
            started_at = time.perf_counter()
            context["full_geometry"] = build_live_edge_geometry(
                new_x,
                new_y,
                self.packed_prepared_inputs["edge_parent_node_id"],
                self.packed_prepared_inputs["edge_child_node_id"],
                r_unit=r_unit,
                c_unit=c_unit,
                scale_factor=scale_factor,
                dbu=dbu,
                require_axis_aligned=require_axis_aligned,
            )
            self.metadata["live_rc_full_build_count"] += 1
            self.metadata["live_rc_full_build_ms"] += (
                time.perf_counter() - started_at
            ) * 1000.0
            self.metadata["live_geometry_census"] = dict(
                context["full_geometry"]["census"]
            )
            self.metadata["live_geometry_rectilinear_path_policy"] = str(
                context["full_geometry"]["rectilinear_path_policy"]
            )
            self._register_live_geometry_gradient_hooks(
                context["full_geometry"],
                edge_to_segment=self.packed_prepared_inputs.get(
                    "edge_to_segment_id"
                ),
            )
        self._live_geometry_context = context
        self.metadata["live_geometry_bound"] = True
        self.metadata["live_geometry_forward_id"] = str(forward_id)
        self.metadata["live_geometry_topology_epoch"] = topology_epoch
        self.metadata["live_geometry_bind_count"] += 1
        return {
            "forward_id": str(forward_id),
            "topology_epoch": topology_epoch,
            "full_prepared_edge_count": (
                None
                if context["full_geometry"] is None
                else int(context["full_geometry"]["length"].numel())
            ),
            "census": (
                None
                if context["full_geometry"] is None
                else dict(context["full_geometry"]["census"])
            ),
        }

    @staticmethod
    def _gradient_stats(gradient):
        detached = gradient.detach()
        finite = torch.isfinite(detached)
        finite_values = detached[finite]
        return {
            "count": int(detached.numel()),
            "finite_count": int(finite.sum().cpu().item()),
            "nonzero_count": int((finite_values != 0.0).sum().cpu().item()),
            "norm": float(finite_values.float().norm().cpu().item()),
            "max_abs": (
                0.0
                if not int(finite_values.numel())
                else float(finite_values.abs().max().cpu().item())
            ),
        }

    def _register_live_geometry_gradient_hooks(
        self,
        full_geometry,
        *,
        edge_to_segment,
    ):
        length = full_geometry["length"]
        if not length.requires_grad:
            return
        z_snapshot = self.segment_state.z_param.detach().cpu()
        edge_to_segment_cpu = (
            None
            if edge_to_segment is None
            else edge_to_segment.detach().to(device="cpu", dtype=torch.long)
        )

        def capture(name):
            def hook(gradient):
                stats = self._gradient_stats(gradient)
                grouped = {}
                if (
                    edge_to_segment_cpu is not None
                    and int(edge_to_segment_cpu.numel()) == int(gradient.numel())
                ):
                    valid = (edge_to_segment_cpu >= 0) & (
                        edge_to_segment_cpu < int(z_snapshot.numel())
                    )
                    edge_z = torch.full_like(edge_to_segment_cpu, -1)
                    edge_z[valid] = torch.round(
                        z_snapshot.index_select(0, edge_to_segment_cpu[valid])
                    ).to(dtype=torch.long)
                    gradient_cpu = gradient.detach().to(device="cpu")
                    for count in range(int(self.segment_state.max_repeater_count) + 1):
                        mask = valid & (edge_z == count)
                        grouped[str(count)] = self._gradient_stats(gradient_cpu[mask])
                stats["by_integer_z"] = grouped
                self.metadata.setdefault("live_geometry_gradient_stats", {})[
                    name
                ] = stats
                return gradient

            return hook

        for name in ("length", "edge_resistance", "edge_capacitance"):
            tensor = full_geometry[name]
            if tensor.requires_grad:
                tensor.register_hook(capture(name))
                self.metadata["live_geometry_gradient_hook_count"] += 1

    def release_live_geometry(self, *, forward_id):
        context = self._live_geometry_context
        if context is None:
            raise RuntimeError("no live geometry is bound to this provider")
        if context["forward_id"] != forward_id:
            raise ValueError("live geometry forward identity mismatch")
        self._live_geometry_context = None
        self.metadata["live_geometry_bound"] = False
        self.metadata["live_geometry_release_count"] += 1

    def _live_edge_overrides(self, prepared_inputs):
        context = self._live_geometry_context
        if context is None:
            return None
        target = prepared_inputs["edge_resistance"]
        full_geometry = context["full_geometry"]
        if full_geometry is not None:
            started_at = time.perf_counter()
            source_index = prepared_inputs.get("source_prepared_edge_index")
            if source_index is None:
                if int(target.numel()) != int(
                    full_geometry["edge_resistance"].numel()
                ):
                    raise ValueError(
                        "live packed active view is missing source_prepared_edge_index"
                    )
                source_index = torch.arange(
                    int(target.numel()),
                    dtype=torch.long,
                    device=full_geometry["edge_resistance"].device,
                )
                self.metadata["live_rc_identity_source_edge_mapping_count"] += 1
            else:
                self.metadata["live_rc_explicit_source_edge_mapping_count"] += 1
                source_index = source_index.to(
                    dtype=torch.long,
                    device=full_geometry["edge_resistance"].device,
                )
            self.metadata["live_rc_source_edge_index_count"] += int(
                source_index.numel()
            )
            edge_resistance = full_geometry["edge_resistance"].index_select(
                0,
                source_index,
            )
            edge_capacitance = full_geometry["edge_capacitance"].index_select(
                0,
                source_index,
            )
            self.metadata["live_rc_gather_count"] += 1
            self.metadata["live_rc_gather_ms"] += (
                time.perf_counter() - started_at
            ) * 1000.0
        else:
            started_at = time.perf_counter()
            local_geometry = build_live_edge_geometry(
                context["new_x"],
                context["new_y"],
                prepared_inputs["edge_parent_node_id"],
                prepared_inputs["edge_child_node_id"],
                r_unit=context["r_unit"],
                c_unit=context["c_unit"],
                scale_factor=context["scale_factor"],
                dbu=context["dbu"],
                require_axis_aligned=context["require_axis_aligned"],
            )
            edge_resistance = local_geometry["edge_resistance"]
            edge_capacitance = local_geometry["edge_capacitance"]
            self.metadata["live_rc_local_build_count"] += 1
            self.metadata["live_rc_local_build_ms"] += (
                time.perf_counter() - started_at
            ) * 1000.0
            self.metadata["live_geometry_census"] = dict(local_geometry["census"])
            self._register_live_geometry_gradient_hooks(
                local_geometry,
                edge_to_segment=prepared_inputs.get("edge_to_segment_id"),
            )
        return {
            "edge_resistance_override": edge_resistance.to(
                dtype=target.dtype,
                device=target.device,
            ),
            "edge_capacitance_override": edge_capacitance.to(
                dtype=target.dtype,
                device=target.device,
            ),
            "canonical_equal_spacing": True,
        }

    def _static_tensor(self, name, value, *, device, dtype):
        if value is None:
            return None
        key = (str(name), torch.device(device), dtype)
        cached = self._static_device_tensor_cache.get(key)
        if cached is not None:
            started_at = self._profile_clock(device) if self.profile_enabled else None
            self.metadata["gpu_static_runtime_cache_hit_count"] += 1
            if self.profile_enabled:
                self.metadata["gpu_static_runtime_reuse_wall_ms"] += (
                    self._profile_clock(device) - started_at
                ) * 1000.0
            return cached
        source = torch.as_tensor(value)
        started_at = self._profile_clock(device) if self.profile_enabled else None
        cached = source.to(dtype=dtype, device=device).contiguous()
        if self.profile_enabled:
            self.metadata["gpu_static_runtime_build_wall_ms"] += (
                self._profile_clock(device) - started_at
            ) * 1000.0
        self._static_device_tensor_cache[key] = cached
        self.metadata["gpu_static_runtime_build_count"] += 1
        self.metadata["gpu_static_runtime_bytes"] += int(
            cached.numel() * cached.element_size()
        )
        if torch.device(device).type == "cuda":
            source_device = source.device
            if source_device.type != "cuda" or source_device != torch.device(device):
                self.metadata["gpu_static_runtime_h2d_bytes"] += int(
                    cached.numel() * cached.element_size()
                )
        return cached

    def _profile_clock(self, device):
        if self.profile_enabled and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def _update_cuda_memory_profile(self, device):
        if not self.profile_enabled or torch.device(device).type != "cuda":
            return
        self.metadata["gpu_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        self.metadata["gpu_peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device)
        )

    def _device_buffer_lut(self, *, device, dtype, sense="base"):
        if self.buffer_device_lut is None:
            return None
        key = (torch.device(device), dtype, sense)
        cached = self._buffer_device_lut_cache.get(key)
        if cached is not None:
            self.metadata["gpu_static_runtime_cache_hit_count"] += 1
            return cached
        source = self.buffer_device_lut
        phase_luts = getattr(source, "phase_luts", None)
        if phase_luts is not None and sense != "base":
            source = phase_luts[sense]
        if hasattr(source, "tensors"):
            tables = source.tensors(dtype=dtype, device=device)
            source = {
                "buffer_input_cap_by_size": tables["input_cap_by_size"],
                "buffer_slew_axis": tables["input_slew_axis"],
                "buffer_load_axis": tables["output_load_axis"],
                "buffer_delay_lut": tables["delay_lut"],
                "buffer_output_slew_lut": tables["output_slew_lut"],
                **{key: tables[key] for key in ("buffer_slew_limits", "buffer_cap_limits")
                   if key in tables},
            }
        required = (
            "buffer_input_cap_by_size",
            "buffer_slew_axis",
            "buffer_load_axis",
            "buffer_delay_lut",
            "buffer_output_slew_lut",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"buffer_device_lut is missing keys: {missing}")
        cached = {
            name: self._static_tensor(
                f"buffer_device_lut:{sense}:{name}",
                source[name],
                device=device,
                dtype=dtype,
            )
            for name in (*required, *(key for key in ("buffer_slew_limits", "buffer_cap_limits")
                                     if key in source))
        }
        self._buffer_device_lut_cache[key] = cached
        return cached

    def _driver_index(self, active_net_ids, *, device):
        cache_key = (_cache_key_for_active_nets(active_net_ids), device)
        cached = self._driver_index_cache.get(cache_key)
        if cached is not None:
            self.metadata["static_payload_cache_hit_count"] += 1
            return cached
        indices = []
        net_ids = []
        for net_id, pin_id in zip(
            cache_key[0],
            self._packed_driver_pins(cache_key[0]),
        ):
            if pin_id is None:
                continue
            indices.append(int(pin_id))
            net_ids.append(int(net_id))
        cached = (
            net_ids,
            torch.as_tensor(indices, dtype=torch.long, device=device),
        )
        self._driver_index_cache[cache_key] = cached
        return cached

    def _driver_values(self, pin_values, active_net_ids):
        net_ids, index = self._driver_index(active_net_ids, device=pin_values.device)
        if not net_ids:
            return {}
        valid = index < int(pin_values.numel())
        if not bool(torch.all(valid).detach().cpu().item()):
            valid_list = valid.detach().cpu().tolist()
            net_ids = [net_id for net_id, keep in zip(net_ids, valid_list) if keep]
            index = index[valid]
        values = pin_values.index_select(0, index)
        values_by_net = {}
        for offset, net_id in enumerate(net_ids):
            values_by_net[int(net_id)] = values[offset]
        return values_by_net

    def _active_view(self, active_net_ids):
        cache_key = _cache_key_for_active_nets(active_net_ids)
        device = self.segment_state.z_param.device
        started_at = self._profile_clock(device) if self.profile_enabled else None
        cached = self._active_view_cache.get(cache_key)
        if cached is not None:
            self.metadata["static_payload_cache_hit_count"] += 1
            if self.profile_enabled:
                self.metadata["active_view_cache_hit_count"] += 1
                self.metadata["active_view_cache_hit_ms"] += (
                    self._profile_clock(device) - started_at
                ) * 1000.0
            return cached
        row_indices = self._active_segment_indices(cache_key)
        active_nets = self._active_nets(cache_key)
        cached = (row_indices, active_nets)
        self._active_view_cache[cache_key] = cached
        if self.profile_enabled:
            self.metadata["active_view_build_count"] += 1
            self.metadata["active_view_build_ms"] += (
                self._profile_clock(device) - started_at
            ) * 1000.0
        return cached

        return values

    def _active_nets(self, active_net_ids):
        if self.packed_prepared_inputs is not None:
            return [
                int(net_id)
                for net_id in sorted(int(net_id) for net_id in active_net_ids)
                if int(net_id) in self.affected_net_ids
            ]
        return [
            self.net_by_id[int(net_id)]
            for net_id in sorted(int(net_id) for net_id in active_net_ids)
            if int(net_id) in self.net_by_id
        ]

    def _prepared_inputs(self, active_net_ids, row_indices, active_nets, active_segment_state):
        cache_key = (
            _cache_key_for_active_nets(active_net_ids),
            tuple(int(row) for row in row_indices),
            active_segment_state.z_param.device,
            active_segment_state.z_param.dtype,
        )
        device = active_segment_state.z_param.device
        started_at = self._profile_clock(device) if self.profile_enabled else None
        cached = self._prepared_inputs_by_active_net.get(cache_key)
        if cached is not None:
            self.metadata["static_payload_cache_hit_count"] += 1
            if self.profile_enabled:
                self.metadata["prepared_inputs_cache_hit_count"] += 1
                self.metadata["prepared_inputs_cache_hit_ms"] += (
                    self._profile_clock(device) - started_at
                ) * 1000.0
            return cached
        if self.packed_prepared_inputs is not None:
            cached = select_packed_segment_count_inputs(
                self.packed_prepared_inputs,
                self.segment_state.packed_segment_geometry,
                active_net_ids=active_net_ids,
                row_indices=row_indices,
                dtype=active_segment_state.z_param.dtype,
                device=active_segment_state.z_param.device,
            )
        else:
            cached = build_segment_count_timing_inputs(
                active_nets,
                active_segment_state,
                dtype=active_segment_state.z_param.dtype,
                device=active_segment_state.z_param.device,
                runtime_profile=self.metadata if self.profile_enabled else None,
            )
        self._prepared_inputs_by_active_net[cache_key] = cached
        if self.profile_enabled:
            self.metadata["prepared_inputs_build_count"] += 1
            self.metadata["prepared_inputs_build_ms"] += (
                self._profile_clock(device) - started_at
            ) * 1000.0
        return cached

    def _active_segment_indices(self, active_net_ids):
        if hasattr(self.segment_state, "segment_indices_for_nets"):
            return list(self.segment_state.segment_indices_for_nets(active_net_ids))
        row_indices = []
        for net_id in sorted(int(net_id) for net_id in active_net_ids):
            row_indices.extend(self.segment_state.net_to_segment_rows.get(int(net_id), []))
        return row_indices

    def _active_segment_state(self, row_indices):
        device = self.segment_state.z_param.device
        cache_key = (tuple(int(row) for row in row_indices), device)
        started_at = self._profile_clock(device) if self.profile_enabled else None
        structure = self._active_segment_structure_cache.get(cache_key)
        if structure is None:
            index = torch.as_tensor(row_indices, dtype=torch.long, device=device)
            compact_cuda_view = (
                self.backend_used in CUDA_SEGMENT_COUNT_BACKENDS
                and not bool(os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON"))
            )
            if compact_cuda_view:
                # The CUDA path consumes tensor order, not per-segment Python
                # metadata. Keep immutable source rows and let the tensor
                # builder assign local IDs by enumeration.
                local_rows = (
                    ()
                    if self.segment_state.segment_rows is None
                    else tuple(
                        self.segment_state.segment_rows[int(row_index)]
                        for row_index in row_indices
                    )
                )
                local_net_to_segment_rows = {}
                summary = {
                    **dict(self.segment_state.summary or {}),
                    "active_segment_view": True,
                    "compact_cuda_view": True,
                    "eligible_segment_count": int(len(row_indices)),
                    "segment_count_state_count": int(len(row_indices)),
                    "source_row_indices": tuple(int(row) for row in row_indices),
                }
            else:
                local_rows = []
                local_net_to_segment_rows = {}
                segment_by_edge = {}
                for local_id, row_index in enumerate(row_indices):
                    row = dict(self.segment_state.segment_rows[int(row_index)])
                    row["stable_segment_id"] = int(row.get("segment_id", row_index))
                    row["active_row_index"] = int(row_index)
                    row["segment_id"] = int(local_id)
                    local_rows.append(row)
                    local_net_to_segment_rows.setdefault(int(row["net_id"]), []).append(local_id)
                    segment_by_edge[
                        (
                            int(row["net_id"]),
                            int(row["parent_node_id"]),
                            int(row["child_node_id"]),
                        )
                    ] = int(local_id)
                summary = {
                    **dict(self.segment_state.summary or {}),
                    "active_segment_view": True,
                    "eligible_segment_count": int(len(row_indices)),
                    "segment_count_state_count": int(len(row_indices)),
                    "segment_by_edge": segment_by_edge,
                    "source_row_indices": tuple(int(row) for row in row_indices),
                }
            structure = {
                "index": index,
                "segment_ids": torch.arange(
                    len(row_indices),
                    dtype=torch.long,
                    device=device,
                ),
                "segment_net_id": self.segment_state.segment_net_id[index],
                "parent_node_id": self.segment_state.parent_node_id[index],
                "child_node_id": self.segment_state.child_node_id[index],
                "segment_rows": tuple(local_rows),
                "net_to_segment_rows": local_net_to_segment_rows,
                "summary": summary,
            }
            self._active_segment_structure_cache[cache_key] = structure
            if self.profile_enabled:
                self.metadata["active_segment_structure_build_count"] += 1
                self.metadata["active_segment_structure_build_ms"] += (
                    self._profile_clock(device) - started_at
                ) * 1000.0
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
            if self.profile_enabled:
                self.metadata["active_segment_structure_cache_hit_count"] += 1
                self.metadata["active_segment_structure_cache_hit_ms"] += (
                    self._profile_clock(device) - started_at
                ) * 1000.0
        index = structure["index"]
        return replace(
            self.segment_state,
            segment_ids=structure["segment_ids"],
            segment_net_id=structure["segment_net_id"],
            parent_node_id=structure["parent_node_id"],
            child_node_id=structure["child_node_id"],
            z_param=self.segment_state.z_param[index],
            bsu_index_param=self.segment_state.bsu_index_param[index],
            segment_rows=structure["segment_rows"],
            net_to_segment_rows=structure["net_to_segment_rows"],
            summary=structure["summary"],
        )

    def _active_segment_state_from_source_rows(self, source_row_index):
        index = source_row_index.to(
            dtype=torch.long,
            device=self.segment_state.z_param.device,
        )
        segment_count = int(index.numel())
        return replace(
            self.segment_state,
            segment_ids=torch.arange(segment_count, dtype=torch.long, device=index.device),
            segment_net_id=self.segment_state.segment_net_id.index_select(0, index),
            parent_node_id=self.segment_state.parent_node_id.index_select(0, index),
            child_node_id=self.segment_state.child_node_id.index_select(0, index),
            z_param=self.segment_state.z_param.index_select(0, index),
            bsu_index_param=self.segment_state.bsu_index_param.index_select(0, index),
            segment_rows=(),
            net_to_segment_rows={},
            summary={
                **dict(self.segment_state.summary or {}),
                "active_segment_view": True,
                "compact_cuda_view": True,
                "eligible_segment_count": segment_count,
                "segment_count_state_count": segment_count,
            },
        )

    def _active_size_table(
        self,
        table,
        row_indices,
        active_segment_state,
        *,
        source_row_index=None,
    ):
        table = table.to(
            dtype=active_segment_state.z_param.dtype,
            device=active_segment_state.z_param.device,
        )
        if source_row_index is not None:
            if int(table.shape[0]) == 1:
                return table.expand(int(active_segment_state.z_param.numel()), -1)
            return table.index_select(0, source_row_index.to(device=table.device))
        if not row_indices:
            return table[:0]
        if int(table.shape[0]) == 1:
            return table.expand(int(active_segment_state.z_param.numel()), -1)
        index = self._active_segment_structure_cache[
            (tuple(int(row) for row in row_indices), active_segment_state.z_param.device)
        ]["index"]
        return table.index_select(0, index)

    def _active_segment_vector(
        self,
        values,
        row_indices,
        active_segment_state,
        *,
        source_row_index=None,
    ):
        if values is None:
            return None
        values = values.to(
            dtype=active_segment_state.z_param.dtype,
            device=active_segment_state.z_param.device,
        ).view(-1)
        if source_row_index is not None:
            return values.index_select(0, source_row_index.to(device=values.device))
        if not row_indices:
            return values[:0]
        index = self._active_segment_structure_cache[
            (tuple(int(row) for row in row_indices), active_segment_state.z_param.device)
        ]["index"]
        return values.index_select(0, index)

    def _live_prepared_inputs(self, prepared_inputs, sense):
        context = self._live_geometry_context
        if context is None or context["pin_capacitance_by_sense"] is None:
            return prepared_inputs
        reference = prepared_inputs["node_capacitance"]
        caps = context["pin_capacitance_by_sense"][sense].to(reference)
        # Steiner nodes have no Liberty cap. Only the static ID mapping is cached;
        # every solve gathers the current differentiable master capacitances.
        return dict(
            prepared_inputs,
            node_capacitance=caps.index_select(
                0, prepared_inputs["flat_topo_node_id"].to(device=caps.device, dtype=torch.long)
            ),
        )

    def _run_lane(
        self,
        *,
        active_net_ids,
        driver_arrival_by_net=None,
        driver_slew_by_net=None,
        level_view=None,
        driver_arrival=None,
        driver_slew=None,
        z_override=None,
        sense="base",
    ):
        native_level_view = bool(level_view and level_view.get("native_active_view"))
        if native_level_view:
            row_indices = ()
            active_nets = ()
            source_row_index = level_view["source_segment_row_index"]
            active_segment_state = self._active_segment_state_from_source_rows(
                source_row_index
            )
            prepared_inputs = level_view["prepared_inputs"]
        elif level_view is not None and "row_indices" in level_view:
            row_indices = level_view["row_indices"]
            active_nets = level_view["active_nets"]
            source_row_index = None
            active_segment_state = self._active_segment_state(row_indices)
        else:
            row_indices, active_nets = self._active_view(active_net_ids)
            source_row_index = None
            active_segment_state = self._active_segment_state(row_indices)
        if z_override is not None:
            active_segment_state = replace(
                active_segment_state,
                z_param=torch.full_like(
                    active_segment_state.z_param,
                    float(z_override),
                ),
            )
        if self.backend_used in {
            "prepared_python",
            "cpp_cpu",
            "cpp_cpu_recompute_autograd",
            "cpp_cpu_explicit_autograd",
            "cpp_cpu_segment_transfer_recompute_autograd",
            "cpp_cpu_segment_transfer_explicit_autograd",
            "cpp_cuda_segment_transfer_explicit_autograd",
        }:
            ordered_net_ids = (
                prepared_inputs["net_ids"].detach().cpu().tolist()
                if native_level_view
                else [
                    int(net)
                    if not isinstance(net, dict)
                    else int(net.get("_segment_count_prepared", {}).get("net_id", _net_id(net, index)))
                    for index, net in enumerate(active_nets)
                ]
            )
            dtype = active_segment_state.z_param.dtype
            device = active_segment_state.z_param.device
            if driver_arrival is None:
                driver_arrival_by_net = driver_arrival_by_net or {}
                driver_arrival = torch.stack(
                    [
                        torch.as_tensor(
                            driver_arrival_by_net.get(net_id, 0.0),
                            dtype=dtype,
                            device=device,
                        )
                        for net_id in ordered_net_ids
                    ]
                ) if ordered_net_ids else torch.empty(0, dtype=dtype, device=device)
            else:
                driver_arrival = driver_arrival.to(dtype=dtype, device=device)
            if driver_slew is None:
                driver_slew_by_net = driver_slew_by_net or {}
                driver_slew = torch.stack(
                    [
                        torch.as_tensor(
                            driver_slew_by_net.get(net_id, 0.0),
                            dtype=dtype,
                            device=device,
                        )
                        for net_id in ordered_net_ids
                    ]
                ) if ordered_net_ids else torch.empty(0, dtype=dtype, device=device)
            else:
                driver_slew = driver_slew.to(dtype=dtype, device=device)
            if not native_level_view:
                prepared_inputs = self._prepared_inputs(
                    active_net_ids,
                    row_indices,
                    active_nets,
                    active_segment_state,
                )
            live_edge_overrides = self._live_edge_overrides(prepared_inputs)
            prepared_inputs = self._live_prepared_inputs(prepared_inputs, sense)
            native_live_edge_overrides = (
                {}
                if live_edge_overrides is None
                else {
                    "edge_resistance_override": live_edge_overrides[
                        "edge_resistance_override"
                    ],
                    "edge_capacitance_override": live_edge_overrides[
                        "edge_capacitance_override"
                    ],
                    "canonical_equal_spacing": True,
                }
            )
            static_per_size_input_cap = self._static_tensor(
                "per_size_input_cap",
                self.per_size_input_cap,
                device=device,
                dtype=dtype,
            )
            static_per_size_delay = self._static_tensor(
                "per_size_delay",
                self.per_size_delay,
                device=device,
                dtype=dtype,
            )
            static_per_size_output_slew = self._static_tensor(
                "per_size_output_slew",
                self.per_size_output_slew,
                device=device,
                dtype=dtype,
            )
            per_size_input_cap = self._active_size_table(
                static_per_size_input_cap,
                row_indices,
                active_segment_state,
                source_row_index=source_row_index,
            )
            per_size_delay = self._active_size_table(
                static_per_size_delay,
                row_indices,
                active_segment_state,
                source_row_index=source_row_index,
            )
            per_size_output_slew = self._active_size_table(
                static_per_size_output_slew,
                row_indices,
                active_segment_state,
                source_row_index=source_row_index,
            )
            active_retained_cap = self._active_segment_vector(
                self._static_tensor(
                    "segment_retained_upstream_cap",
                    self.segment_retained_upstream_cap,
                    device=device,
                    dtype=dtype,
                ),
                row_indices,
                active_segment_state,
                source_row_index=source_row_index,
            )
            device_buffer_lut = self._device_buffer_lut(
                device=device, dtype=dtype, sense=sense
            )
            if (
                getattr(self.buffer_device_lut, "phase_luts", None) is not None
                and sense != "base"
            ):
                per_size_input_cap = (
                    device_buffer_lut["buffer_input_cap_by_size"].unsqueeze(0).expand(
                        int(active_segment_state.z_param.numel()), -1
                    )
                )
            if int(active_segment_state.z_param.numel()) == 0:
                return segment_count_prepared_relaxed_timing(
                    prepared_inputs,
                    segment_state=active_segment_state,
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                    transfer_backend=self.segment_transfer_backend,
                    buffer_device_lut=device_buffer_lut,
                    segment_retained_upstream_cap=active_retained_cap,
                    **(live_edge_overrides or {}),
                )
            if self.backend_used == "cpp_cpu":
                buffer_tensors = _interpolate_size_tables(
                    active_segment_state.bsu_index().to(
                        dtype=active_segment_state.z_param.dtype,
                        device=active_segment_state.z_param.device,
                    ),
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                )
                return segment_count_forward_native(
                    net_topo_start=prepared_inputs["net_topo_start"],
                    flat_topo_node_id=prepared_inputs["flat_topo_node_id"],
                    edge_start=prepared_inputs["edge_start"],
                    edge_parent_compact_id=prepared_inputs["edge_parent_compact_id"],
                    edge_child_compact_id=prepared_inputs["edge_child_compact_id"],
                    edge_resistance=prepared_inputs["edge_resistance"],
                    edge_capacitance=prepared_inputs["edge_capacitance"],
                    node_capacitance=prepared_inputs["node_capacitance"],
                    edge_to_segment_id=prepared_inputs["edge_to_segment_id"],
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                    z_value=active_segment_state.z_value(),
                    buffer_input_cap=buffer_tensors["buffer_input_cap"],
                    buffer_delay=buffer_tensors["buffer_delay"],
                    buffer_output_slew=buffer_tensors["buffer_output_slew"],
                    parent_cap_fraction=prepared_inputs["parent_cap_fraction"],
                    child_cap_fraction=prepared_inputs["child_cap_fraction"],
                    segment_sub_resistance_fraction=prepared_inputs[
                        "segment_sub_resistance_fraction"
                    ],
                    sink_node_id=prepared_inputs["sink_node_id"],
                    sink_net_index=prepared_inputs["sink_net_id"],
                    sink_node_compact_id=prepared_inputs["sink_node_compact_id"],
                )
            if self.backend_used == "cpp_cpu_recompute_autograd":
                return segment_count_forward_native_recompute_autograd(
                    prepared_inputs,
                    segment_state=active_segment_state,
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                )
            if self.backend_used == "cpp_cpu_explicit_autograd":
                return segment_count_forward_native_explicit_autograd(
                    prepared_inputs,
                    segment_state=active_segment_state,
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                )
            if self.backend_used == "cpp_cpu_segment_transfer_recompute_autograd":
                return segment_count_forward_native_segment_transfer_recompute_autograd(
                    prepared_inputs,
                    segment_state=active_segment_state,
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                    buffer_device_lut=device_buffer_lut,
                    segment_retained_upstream_cap=active_retained_cap,
                )
            if self.backend_used == "cpp_cpu_segment_transfer_explicit_autograd":
                return segment_count_forward_native_segment_transfer_explicit_autograd(
                    prepared_inputs,
                    segment_state=active_segment_state,
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                    buffer_device_lut=device_buffer_lut,
                    segment_retained_upstream_cap=active_retained_cap,
                    **native_live_edge_overrides,
                )
            if self.backend_used == "cpp_cuda_segment_transfer_explicit_autograd":
                return segment_count_forward_cuda_segment_transfer_explicit_autograd(
                    prepared_inputs,
                    segment_state=active_segment_state,
                    per_size_input_cap=per_size_input_cap,
                    per_size_delay=per_size_delay,
                    per_size_output_slew=per_size_output_slew,
                    driver_arrival=driver_arrival,
                    driver_slew=driver_slew,
                    buffer_device_lut=device_buffer_lut,
                    segment_retained_upstream_cap=active_retained_cap,
                    **native_live_edge_overrides,
                )
            transfer_start = time.perf_counter()
            result = segment_count_prepared_relaxed_timing(
                prepared_inputs,
                segment_state=active_segment_state,
                per_size_input_cap=per_size_input_cap,
                per_size_delay=per_size_delay,
                per_size_output_slew=per_size_output_slew,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
                transfer_backend=self.segment_transfer_backend,
                buffer_device_lut=device_buffer_lut,
                segment_retained_upstream_cap=active_retained_cap,
                **(live_edge_overrides or {}),
            )
            if self.profile_enabled:
                elapsed_ms = (time.perf_counter() - transfer_start) * 1000.0
                self.metadata["segment_transfer_forward_ms"] += elapsed_ms
            return result
        return segment_count_relaxed_timing(
            active_nets,
            segment_state=active_segment_state,
            per_size_input_cap=self._active_size_table(
                self.per_size_input_cap,
                row_indices,
                active_segment_state,
            ),
            per_size_delay=self._active_size_table(
                self.per_size_delay,
                row_indices,
                active_segment_state,
            ),
            per_size_output_slew=self._active_size_table(
                self.per_size_output_slew,
                row_indices,
                active_segment_state,
            ),
            driver_arrival_by_net=driver_arrival_by_net,
            driver_slew_by_net=driver_slew_by_net,
        )

    def _run_cap_only(self, *, active_net_ids, z_override=None, sense="base"):
        fallback_reason = None
        if self.backend_used not in CUDA_SEGMENT_COUNT_BACKENDS:
            fallback_reason = "non_cuda_segment_backend"
        elif self.segment_state.z_param.device.type != "cuda":
            fallback_reason = "segment_state_not_cuda"
        elif self.buffer_device_lut is None:
            fallback_reason = "missing_buffer_device_lut"
        if fallback_reason is not None:
            self.metadata["driver_cap_overlay_path"] = "legacy_full_transfer"
            self.metadata["driver_cap_overlay_fallback_reason"] = fallback_reason
            return self._run_lane(
                active_net_ids=active_net_ids,
                driver_arrival_by_net={},
                driver_slew_by_net={},
                z_override=z_override,
                sense=sense,
            )
        self.metadata["driver_cap_overlay_path"] = "cuda_cap_only"
        self.metadata["driver_cap_overlay_fallback_reason"] = None
        row_indices, active_nets = self._active_view(active_net_ids)
        active_segment_state = self._active_segment_state(row_indices)
        if z_override is not None:
            active_segment_state = replace(
                active_segment_state,
                z_param=torch.full_like(
                    active_segment_state.z_param,
                    float(z_override),
                ),
            )
        dtype = active_segment_state.z_param.dtype
        device = active_segment_state.z_param.device
        prepared_inputs = self._prepared_inputs(
            active_net_ids,
            row_indices,
            active_nets,
            active_segment_state,
        )
        per_size_input_cap = self._active_size_table(
            self._static_tensor(
                "per_size_input_cap",
                self.per_size_input_cap,
                device=device,
                dtype=dtype,
            ),
            row_indices,
            active_segment_state,
        )
        device_buffer_lut = self._device_buffer_lut(device=device, dtype=dtype, sense=sense)
        if (
            getattr(self.buffer_device_lut, "phase_luts", None) is not None
            and sense != "base"
        ):
            per_size_input_cap = (
                device_buffer_lut["buffer_input_cap_by_size"].unsqueeze(0).expand(
                    int(active_segment_state.z_param.numel()), -1
                )
            )
        runtime_prepared_inputs = self._live_prepared_inputs(prepared_inputs, sense)
        live_edge_overrides = self._live_edge_overrides(prepared_inputs)
        if live_edge_overrides is not None:
            runtime_prepared_inputs = dict(runtime_prepared_inputs)
            runtime_prepared_inputs["edge_capacitance"] = live_edge_overrides[
                "edge_capacitance_override"
            ]
        return segment_count_driver_cap_native_cuda_autograd(
            runtime_prepared_inputs,
            segment_state=active_segment_state,
            per_size_input_cap=per_size_input_cap,
            buffer_input_cap_by_size=device_buffer_lut["buffer_input_cap_by_size"],
            runtime_profile=self.metadata if self.profile_enabled else None,
        )

    def _gather_level_driver_values(self, pin_values, level_view):
        index = level_view["driver_pin_index"].to(
            device=pin_values.device,
            dtype=torch.long,
        )
        if int(index.numel()) == 0:
            return pin_values[:0]
        if int(pin_values.numel()) == 0:
            return torch.zeros(
                index.shape,
                dtype=pin_values.dtype,
                device=pin_values.device,
            )
        valid = level_view["driver_pin_valid"].to(device=pin_values.device)
        valid = valid & (index < int(pin_values.numel()))
        gathered = pin_values.index_select(
            0,
            index.clamp(0, int(pin_values.numel()) - 1),
        )
        return torch.where(valid, gathered, torch.zeros_like(gathered))

    def _device_probe_target_segment_ids(self):
        raw = os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_SEGMENT_IDS", "")
        return {
            int(part.strip())
            for part in str(raw).split(",")
            if part.strip().isdigit()
        }

    def _selected_probe_segment_ids(self, active_segment_state, result):
        targets = self._device_probe_target_segment_ids()
        if targets:
            return targets
        model = result.get("relaxed_buffer_model", {}) if isinstance(result, dict) else {}
        z_value = model.get("z_value")
        if not torch.is_tensor(z_value) or int(z_value.numel()) == 0:
            return set()
        z_cpu = z_value.detach().float().cpu()
        max_value = float(torch.max(z_cpu).item())
        if max_value <= 0.0:
            return set()
        index = int(torch.argmax(z_cpu).item())
        row = active_segment_state.segment_rows[index]
        return {int(row.get("stable_segment_id", row.get("segment_id", index)))}

    def _analytic_transfer_probe(
        self,
        *,
        sense,
        row,
        local_index,
        fractions_by_count,
        net_id,
        parent,
        child,
        driver_arrival_by_net,
        driver_slew_by_net,
        segment_upstream_cap,
        segment_downstream_load,
        segment_parent_arrival,
        segment_parent_slew,
        segment_retained_upstream_cap,
        z_value,
        bsu_index,
    ):
        if self.buffer_device_lut is None:
            return {}
        edge_resistance = torch.as_tensor(
            _edge_rc_value(row.get("edge_rc", {}) or {}, parent, child, "r"),
            dtype=z_value.dtype,
            device=z_value.device,
        )
        edge_capacitance = torch.as_tensor(
            _edge_rc_value(row.get("edge_rc", {}) or {}, parent, child, "c"),
            dtype=z_value.dtype,
            device=z_value.device,
        )
        retained_cap = (
            torch.zeros((), dtype=z_value.dtype, device=z_value.device)
            if segment_retained_upstream_cap is None
            else torch.as_tensor(
                segment_retained_upstream_cap[local_index],
                dtype=z_value.dtype,
                device=z_value.device,
            )
        )
        available_counts = sorted(int(count) for count in fractions_by_count)
        if not available_counts:
            return {"analytic_transfer_error": "missing_split_fractions"}
        z_scalar = float(z_value[local_index].detach().cpu().item())
        repeater_count = max(
            available_counts[0],
            min(available_counts[-1], int(round(z_scalar))),
        )
        split_fractions = tuple(fractions_by_count[str(repeater_count)])

        def arc_replay_summary(*, input_slew, output_load):
            arc_luts_by_size = getattr(self.buffer_device_lut, "arc_luts_by_size", None)
            if not arc_luts_by_size:
                return None
            bsu_value = float(bsu_index[local_index].detach().cpu().item())
            size_count = len(arc_luts_by_size)
            clipped = max(0.0, min(bsu_value, float(size_count - 1)))
            lo = int(math.floor(clipped))
            hi = min(lo + 1, size_count - 1)
            alpha = clipped - float(lo)

            def rows_for_size(size_index):
                rows = []
                for arc_row in arc_luts_by_size[size_index]:
                    delay, _delay_status = lookup_lut_value_with_status_mode(
                        values=arc_row["delay_lut"].reshape(-1).tolist(),
                        trans_axis=arc_row["input_slew_axis"],
                        cap_axis=arc_row["output_load_axis"],
                        dim=list(arc_row["delay_lut"].shape),
                        input_slew=float(input_slew),
                        output_cap=float(output_load),
                        boundary_mode="extrapolate",
                    )
                    output_slew, _slew_status = lookup_lut_value_with_status_mode(
                        values=arc_row["output_slew_lut"].reshape(-1).tolist(),
                        trans_axis=arc_row["input_slew_axis"],
                        cap_axis=arc_row["output_load_axis"],
                        dim=list(arc_row["output_slew_lut"].shape),
                        input_slew=float(input_slew),
                        output_cap=float(output_load),
                        boundary_mode="extrapolate",
                    )
                    rows.append(
                        {
                            "size_index": int(size_index),
                            "arc_id": int(arc_row["arc_id"]),
                            "prefix": str(arc_row["prefix"]),
                            "delay": float(delay),
                            "output_slew": float(output_slew),
                        }
                    )
                return rows

            lo_rows = rows_for_size(lo)
            hi_rows = rows_for_size(hi)
            lo_delay_stats = _summarize_values([row["delay"] for row in lo_rows])
            hi_delay_stats = _summarize_values([row["delay"] for row in hi_rows])
            lo_slew_stats = _summarize_values([row["output_slew"] for row in lo_rows])
            hi_slew_stats = _summarize_values([row["output_slew"] for row in hi_rows])

            def interp_stat(key, left, right):
                if left.get(key) is None or right.get(key) is None:
                    return None
                return (1.0 - alpha) * float(left[key]) + alpha * float(right[key])

            delay_stats = {
                "count": int(max(lo_delay_stats.get("count", 0), hi_delay_stats.get("count", 0))),
                "finite_count": int(max(lo_delay_stats.get("finite_count", 0), hi_delay_stats.get("finite_count", 0))),
                "min": interp_stat("min", lo_delay_stats, hi_delay_stats),
                "max": interp_stat("max", lo_delay_stats, hi_delay_stats),
                "mean": interp_stat("mean", lo_delay_stats, hi_delay_stats),
            }
            slew_stats = {
                "count": int(max(lo_slew_stats.get("count", 0), hi_slew_stats.get("count", 0))),
                "finite_count": int(max(lo_slew_stats.get("finite_count", 0), hi_slew_stats.get("finite_count", 0))),
                "min": interp_stat("min", lo_slew_stats, hi_slew_stats),
                "max": interp_stat("max", lo_slew_stats, hi_slew_stats),
                "mean": interp_stat("mean", lo_slew_stats, hi_slew_stats),
            }
            return {
                "input_slew_ps": float(input_slew),
                "output_load_pf": float(output_load),
                "bsu_index": float(bsu_value),
                "size_lo": int(lo),
                "size_hi": int(hi),
                "size_alpha": float(alpha),
                "arc_count": int(delay_stats["count"]),
                "delay_stats": delay_stats,
                "output_slew_stats": slew_stats,
                "arcs_by_size": {
                    str(lo): lo_rows,
                    str(hi): hi_rows,
                },
            }

        try:
            analytic = analytic_segment_transfer(
                SegmentTransferInput(
                    input_arrival=torch.as_tensor(
                        segment_parent_arrival[local_index],
                        dtype=z_value.dtype,
                        device=z_value.device,
                    ),
                    input_slew=torch.as_tensor(
                        segment_parent_slew[local_index],
                        dtype=z_value.dtype,
                        device=z_value.device,
                    ),
                    downstream_load=segment_downstream_load[local_index],
                    edge_resistance=edge_resistance,
                    edge_capacitance=edge_capacitance,
                    upstream_retained_capacitance=retained_cap,
                    repeater_count=repeater_count,
                    bsu_index=bsu_index[local_index],
                    split_fractions=split_fractions,
                    buffer_device=self.buffer_device_lut,
                )
            )
        except Exception as exc:
            return {"analytic_transfer_error": str(exc)}
        diagnostics = analytic.diagnostics

        def diagnostic_scalar(name):
            value = diagnostics.get(name)
            return None if value is None else float(value.detach().cpu().item())

        def diagnostic_scalars(name):
            return [float(value.detach().cpu().item()) for value in diagnostics.get(name, [])]

        result = {
            "analytic_transfer_source": getattr(
                self.buffer_device_lut,
                "source",
                "buffer_device_lut",
            ),
            "analytic_transfer_repeater_count": int(diagnostics["repeater_count"]),
            "analytic_transfer_segment_delay": float(
                analytic.segment_delay.detach().cpu().item()
            ),
            "analytic_transfer_output_slew": float(
                analytic.output_slew.detach().cpu().item()
            ),
            "analytic_transfer_upstream_visible_input_cap": float(
                analytic.upstream_visible_load.detach().cpu().item()
            ),
            "analytic_transfer_buffer_input_slew": diagnostic_scalar("buffer_input_slew"),
            "analytic_transfer_buffer_output_load": diagnostic_scalar("buffer_output_load"),
            "analytic_transfer_buffer_delay": diagnostic_scalar("buffer_delay"),
            "analytic_transfer_buffer_output_slew": diagnostic_scalar("buffer_output_slew"),
            "analytic_transfer_buffer_input_slews": diagnostic_scalars("buffer_input_slews"),
            "analytic_transfer_buffer_output_loads": diagnostic_scalars("buffer_output_loads"),
            "analytic_transfer_buffer_delays": diagnostic_scalars("buffer_delays"),
            "analytic_transfer_buffer_output_slews": diagnostic_scalars("buffer_output_slews"),
            "analytic_transfer_upstream_retained_capacitance": diagnostic_scalar(
                "upstream_retained_capacitance"
            ),
        }
        opensta_slew_values = _env_float_list(
            "AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_INPUT_SLEW_PS"
        )
        opensta_load_values = _env_float_list(
            "AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_OUTPUT_LOAD_PF"
        )
        sense_index = {"rise": 0, "fall": 1}.get(str(sense).lower())
        if sense_index is not None and opensta_slew_values and opensta_load_values:
            slew_value = opensta_slew_values[
                min(int(sense_index), len(opensta_slew_values) - 1)
            ]
            load_value = opensta_load_values[
                min(int(sense_index), len(opensta_load_values) - 1)
            ]
            try:
                replay = analytic_segment_transfer(
                    SegmentTransferInput(
                        input_arrival=torch.as_tensor(
                            segment_parent_arrival[local_index],
                            dtype=z_value.dtype,
                            device=z_value.device,
                        ),
                        input_slew=torch.as_tensor(
                            float(slew_value),
                            dtype=z_value.dtype,
                            device=z_value.device,
                        ),
                        downstream_load=torch.as_tensor(
                            float(load_value),
                            dtype=z_value.dtype,
                            device=z_value.device,
                        ),
                        edge_resistance=edge_resistance,
                        edge_capacitance=edge_capacitance,
                        upstream_retained_capacitance=retained_cap,
                        repeater_count=repeater_count,
                        bsu_index=bsu_index[local_index],
                        split_fractions=split_fractions,
                        buffer_device=self.buffer_device_lut,
                    )
                )
                result.update(
                    {
                        "opensta_input_replay_input_slew_ps": float(slew_value),
                        "opensta_input_replay_output_load_pf": float(load_value),
                        "opensta_input_replay_segment_delay": float(
                            replay.segment_delay.detach().cpu().item()
                        ),
                        "opensta_input_replay_output_slew": float(
                            replay.output_slew.detach().cpu().item()
                        ),
                        "opensta_input_replay_buffer_input_slew": float(
                            replay.diagnostics["buffer_input_slew"].detach().cpu().item()
                        ),
                        "opensta_input_replay_buffer_output_load": float(
                            replay.diagnostics["buffer_output_load"].detach().cpu().item()
                        ),
                        "opensta_input_replay_buffer_delay": float(
                            replay.diagnostics["buffer_delay"].detach().cpu().item()
                        ),
                        "opensta_input_replay_buffer_output_slew": float(
                            replay.diagnostics.get("buffer_output_slew", torch.zeros(())).detach().cpu().item()
                        ),
                    }
                )
                arc_summary = arc_replay_summary(
                    input_slew=float(slew_value),
                    output_load=float(load_value),
                )
                if arc_summary is not None:
                    result["opensta_input_arc_replay"] = arc_summary
            except Exception as exc:
                result["opensta_input_replay_error"] = str(exc)
        return result

    def _append_device_probe_rows(
        self,
        *,
        sense,
        active_net_ids,
        driver_arrival_by_net,
        driver_slew_by_net,
        pin_net_cap_rise=None,
        pin_net_cap_fall=None,
        result,
    ):
        probe_path = os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON")
        if not probe_path:
            return
        row_indices, active_nets = self._active_view(active_net_ids)
        if not row_indices:
            return
        active_segment_state = self._active_segment_state(row_indices)
        selected_ids = self._selected_probe_segment_ids(active_segment_state, result)
        if not selected_ids:
            return

        local_to_active_row = [int(row) for row in row_indices]
        global_to_local = {
            int(active_row): int(local_index)
            for local_index, active_row in enumerate(local_to_active_row)
        }
        model = result.get("relaxed_buffer_model", {}) if isinstance(result, dict) else {}
        z_value = model.get("z_value")
        bsu_index = model.get("bsu_index")
        buffer_input_cap = model.get("buffer_input_cap")
        buffer_delay = model.get("buffer_delay")
        buffer_output_slew = model.get("buffer_output_slew")
        segment_delay = result.get("segment_delay")
        segment_output_slew = result.get("segment_output_slew")
        segment_upstream_cap = result.get("segment_upstream_visible_input_cap")
        segment_downstream_load = result.get("segment_downstream_load")
        segment_parent_arrival = result.get("segment_parent_arrival")
        segment_parent_slew = result.get("segment_parent_slew")
        segment_retained_upstream_cap = self._active_segment_vector(
            self.segment_retained_upstream_cap,
            row_indices,
            active_segment_state,
        )
        driver_net_cap = result.get("driver_net_cap")
        driver_pin_id_tensor = result.get("driver_pin_id")
        if not all(torch.is_tensor(value) for value in (
            segment_delay,
            segment_output_slew,
            segment_upstream_cap,
            segment_downstream_load,
            segment_parent_arrival,
            segment_parent_slew,
        )):
            return

        per_size_input_cap = self._active_size_table(
            self.per_size_input_cap,
            row_indices,
            active_segment_state,
        ).detach().cpu()
        per_size_delay = self._active_size_table(
            self.per_size_delay,
            row_indices,
            active_segment_state,
        ).detach().cpu()
        per_size_output_slew = self._active_size_table(
            self.per_size_output_slew,
            row_indices,
            active_segment_state,
        ).detach().cpu()
        zero_result = None
        if any(
            int(row.get("segment_id", local_to_active_row[index])) in selected_ids
            or int(row.get("stable_segment_id", local_to_active_row[index])) in selected_ids
            or int(local_to_active_row[index]) in selected_ids
            for index, row in enumerate(tuple(active_segment_state.segment_rows or ()))
        ):
            try:
                zero_result = self._run_lane_with_z_override(
                    active_net_ids=active_net_ids,
                    z_value=0.0,
                )
            except Exception:
                zero_result = None
        cap_by_driver_pin = self._cap_by_driver_pin(
            result,
            device=z_value.device if torch.is_tensor(z_value) else active_segment_state.z_param.device,
            dtype=z_value.dtype if torch.is_tensor(z_value) else active_segment_state.z_param.dtype,
        ) if torch.is_tensor(driver_net_cap) and torch.is_tensor(driver_pin_id_tensor) else {}
        zero_cap_by_driver_pin = (
            self._cap_by_driver_pin(
                zero_result,
                device=active_segment_state.z_param.device,
                dtype=active_segment_state.z_param.dtype,
            )
            if isinstance(zero_result, dict)
            else {}
        )
        if not all(torch.is_tensor(value) for value in (
            z_value,
            bsu_index,
            buffer_input_cap,
            buffer_delay,
            buffer_output_slew,
        )):
            fallback_buffers = _interpolate_size_tables(
                active_segment_state.bsu_index().to(
                    dtype=active_segment_state.z_param.dtype,
                    device=active_segment_state.z_param.device,
                ),
                per_size_input_cap=self._active_size_table(
                    self.per_size_input_cap,
                    row_indices,
                    active_segment_state,
                ),
                per_size_delay=self._active_size_table(
                    self.per_size_delay,
                    row_indices,
                    active_segment_state,
                ),
                per_size_output_slew=self._active_size_table(
                    self.per_size_output_slew,
                    row_indices,
                    active_segment_state,
                ),
            )
            z_value = active_segment_state.z_value()
            bsu_index = fallback_buffers["bsu_index"]
            buffer_input_cap = fallback_buffers["buffer_input_cap"]
            buffer_delay = fallback_buffers["buffer_delay"]
            buffer_output_slew = fallback_buffers["buffer_output_slew"]

        rows = []
        prepared_inputs = self._prepared_inputs(
            active_net_ids,
            row_indices,
            active_nets,
            active_segment_state,
        )
        flat_topo = prepared_inputs["flat_topo_node_id"].detach().cpu().tolist()
        net_topo_start = prepared_inputs["net_topo_start"].detach().cpu().tolist()
        prepared_net_ids = prepared_inputs["net_ids"].detach().cpu().tolist()
        compact_by_net_node = {}
        for prepared_net_offset, prepared_net_id in enumerate(prepared_net_ids):
            start = int(net_topo_start[prepared_net_offset])
            end = int(net_topo_start[prepared_net_offset + 1])
            compact_by_net_node[int(prepared_net_id)] = {
                int(node_id): int(start + local_id)
                for local_id, node_id in enumerate(flat_topo[start:end])
            }
        for local_index, row in enumerate(tuple(active_segment_state.segment_rows or ())):
            active_row_index = int(local_to_active_row[local_index])
            stable_segment_id = int(
                row.get("stable_segment_id", row.get("segment_id", active_row_index))
            )
            local_segment_id = int(row.get("segment_id", local_index))
            if (
                stable_segment_id not in selected_ids
                and active_row_index not in selected_ids
                and local_segment_id not in selected_ids
            ):
                continue
            net_id = int(row.get("net_id", -1))
            parent = int(row.get("parent_node_id", -1))
            child = int(row.get("child_node_id", -1))
            net = self.net_by_id.get(net_id, {})
            prepared = (net.get("_segment_count_prepared", {}) or {})
            root = prepared.get("root_node_id", net.get("driver_pin_id"))
            root = int(root) if root is not None else -1
            driver_pin = self.driver_pin_by_net.get(net_id)
            children_by_node = {
                int(node): [int(child_id) for child_id in children]
                for node, children in (prepared.get("children_by_node", {}) or {}).items()
            }
            root_to_parent_path = (
                _root_to_node_path(root, parent, children_by_node)
                if root >= 0 and parent >= 0
                else []
            )
            root_to_child_path = (
                _root_to_node_path(root, child, children_by_node)
                if root >= 0 and child >= 0
                else []
            )
            edge_rc = prepared.get("edge_rc", {}) or row.get("edge_rc", {}) or {}
            root_to_parent_rc = _path_edge_rc_summary(root_to_parent_path, edge_rc)
            root_to_child_rc = _path_edge_rc_summary(root_to_child_path, edge_rc)
            compact_by_node = compact_by_net_node.get(net_id, {})
            root_cap_decomposition = {}
            if driver_pin is not None:
                original_cap = None
                if torch.is_tensor(pin_net_cap_rise) and int(driver_pin) < int(pin_net_cap_rise.numel()):
                    original_cap = float(pin_net_cap_rise[int(driver_pin)].detach().cpu().item())
                zero_cap = zero_cap_by_driver_pin.get(int(driver_pin))
                if original_cap is not None and zero_cap is not None:
                    root_cap_decomposition = self._root_load_decomposition(
                        net_id=net_id,
                        driver_pin_id=int(driver_pin),
                        original_cap=original_cap,
                        zero_cap=zero_cap,
                    )
            fractions_by_count = {
                str(count): [float(value) for value in _split_fractions(row, count)]
                for count in range(int(active_segment_state.max_repeater_count) + 1)
            }
            analytic_probe = self._analytic_transfer_probe(
                sense=sense,
                row=row,
                local_index=local_index,
                fractions_by_count=fractions_by_count,
                net_id=net_id,
                parent=parent,
                child=child,
                driver_arrival_by_net=driver_arrival_by_net,
                driver_slew_by_net=driver_slew_by_net,
                segment_upstream_cap=segment_upstream_cap,
                segment_downstream_load=segment_downstream_load,
                segment_parent_arrival=segment_parent_arrival,
                segment_parent_slew=segment_parent_slew,
                segment_retained_upstream_cap=segment_retained_upstream_cap,
                z_value=z_value,
                bsu_index=bsu_index,
            )
            rows.append(
                {
                    "sense": str(sense),
                    "segment_id": int(stable_segment_id),
                    "global_segment_id": int(stable_segment_id),
                    "local_segment_id": int(local_index),
                    "row_segment_id": int(stable_segment_id),
                    "active_row_index": int(active_row_index),
                    "active_local_segment_id": int(local_segment_id),
                    "net_id": int(net_id),
                    "net_name": row.get("net_name"),
                    "driver_pin_id": int(driver_pin) if driver_pin is not None else None,
                    "root_node_id": int(root),
                    "root_matches_driver_pin": bool(
                        driver_pin is not None and int(root) == int(driver_pin)
                    ),
                    "is_parent_root": bool(int(parent) == int(root)),
                    "is_child_root": bool(int(child) == int(root)),
                    "parent_compact_id": compact_by_node.get(int(parent)),
                    "child_compact_id": compact_by_node.get(int(child)),
                    "root_compact_id": compact_by_node.get(int(root)),
                    "root_to_parent_path_node_ids": [
                        int(node_id) for node_id in root_to_parent_path
                    ],
                    "root_to_child_path_node_ids": [
                        int(node_id) for node_id in root_to_child_path
                    ],
                    "root_to_parent_path": root_to_parent_rc,
                    "root_to_child_path": root_to_child_rc,
                    "driver_overlay_net_cap": (
                        float(cap_by_driver_pin[int(driver_pin)])
                        if driver_pin is not None and int(driver_pin) in cap_by_driver_pin
                        else None
                    ),
                    "driver_zero_z_net_cap": (
                        float(zero_cap_by_driver_pin[int(driver_pin)])
                        if driver_pin is not None and int(driver_pin) in zero_cap_by_driver_pin
                        else None
                    ),
                    "driver_pin_net_cap_rise_after_overlay": (
                        float(pin_net_cap_rise[int(driver_pin)].detach().cpu().item())
                        if torch.is_tensor(pin_net_cap_rise)
                        and driver_pin is not None
                        and int(driver_pin) < int(pin_net_cap_rise.numel())
                        else None
                    ),
                    "driver_pin_net_cap_fall_after_overlay": (
                        float(pin_net_cap_fall[int(driver_pin)].detach().cpu().item())
                        if torch.is_tensor(pin_net_cap_fall)
                        and driver_pin is not None
                        and int(driver_pin) < int(pin_net_cap_fall.numel())
                        else None
                    ),
                    "root_load_decomposition": root_cap_decomposition,
                    "parent_node_id": int(parent),
                    "child_node_id": int(child),
                    "parent_x_dbu": row.get("parent_x_dbu"),
                    "parent_y_dbu": row.get("parent_y_dbu"),
                    "child_x_dbu": row.get("child_x_dbu"),
                    "child_y_dbu": row.get("child_y_dbu"),
                    "edge_rc": dict(row.get("edge_rc", {}) or {}),
                    "fractions_by_repeater_count": fractions_by_count,
                    "driver_arrival": float(
                        torch.as_tensor(
                            driver_arrival_by_net.get(net_id, 0.0),
                            dtype=z_value.dtype,
                            device=z_value.device,
                        ).detach().cpu().item()
                    ),
                    "driver_slew": float(
                        torch.as_tensor(
                            driver_slew_by_net.get(net_id, 0.0),
                            dtype=z_value.dtype,
                            device=z_value.device,
                        ).detach().cpu().item()
                    ),
                    "segment_parent_arrival": float(
                        segment_parent_arrival[local_index].detach().cpu().item()
                    ),
                    "segment_parent_slew": float(
                        segment_parent_slew[local_index].detach().cpu().item()
                    ),
                    "z_value": float(z_value[local_index].detach().cpu().item()),
                    "bsu_index": float(bsu_index[local_index].detach().cpu().item()),
                    "static_buffer_input_cap": float(
                        buffer_input_cap[local_index].detach().cpu().item()
                    ),
                    "static_buffer_delay": float(
                        buffer_delay[local_index].detach().cpu().item()
                    ),
                    "static_buffer_output_slew": float(
                        buffer_output_slew[local_index].detach().cpu().item()
                    ),
                    "per_size_input_cap": [
                        float(value)
                        for value in per_size_input_cap[local_index].tolist()
                    ],
                    "per_size_delay": [
                        float(value)
                        for value in per_size_delay[local_index].tolist()
                    ],
                    "per_size_output_slew": [
                        float(value)
                        for value in per_size_output_slew[local_index].tolist()
                    ],
                    "segment_delay": float(
                        segment_delay[local_index].detach().cpu().item()
                    ),
                    "segment_output_slew": float(
                        segment_output_slew[local_index].detach().cpu().item()
                    ),
                        "segment_upstream_visible_input_cap": float(
                            segment_upstream_cap[local_index].detach().cpu().item()
                        ),
                        "segment_downstream_load": float(
                            segment_downstream_load[local_index].detach().cpu().item()
                        ),
                    **analytic_probe,
                }
            )
        if not rows:
            return

        if self._device_probe_state is None:
            self._device_probe_state = {
                "artifact": "segment_count_device_model_probe",
                "artifact_version": 1,
                "semantics": {
                    "static_buffer_delay": "Current PySTA segment-count scalar buffer delay after bsu interpolation, used directly by _edge_delay_slew_state.",
                    "static_buffer_output_slew": "Current PySTA segment-count scalar buffer output slew after bsu interpolation; it overwrites upstream wire slew in z>=1 states.",
                    "driver_slew": "TimingPropagation-propagated driver slew entering this net arc call for the recorded sense.",
                    "analytic_transfer_buffer_delay": "Optional diagnostic-only segment_transfer analytic buffer delay when buffer_device_lut is provided to the provider.",
                    "analytic_transfer_buffer_output_slew": "Optional diagnostic-only segment_transfer analytic output slew when buffer_device_lut is provided to the provider.",
                    "opensta_input_replay_buffer_delay": "Optional primitive-only replay. It uses the same segment edge, split, bsu, and buffer_device_lut, but overrides input slew/output load from AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_* environment values.",
                    "opensta_input_replay_buffer_output_slew": "Optional primitive-only replay output slew under the AIMP_BUFFERING_SEGMENT_TRANSFER_REPLAY_* input state.",
                },
                "backend": self.backend,
                "backend_used": self.backend_used,
                "backend_fallback_reason": self.backend_fallback_reason,
                "rows": [],
            }
        self._device_probe_state["rows"].extend(rows)
        os.makedirs(os.path.dirname(os.path.abspath(probe_path)), exist_ok=True)
        with open(probe_path, "w", encoding="utf-8") as stream:
            json.dump(self._device_probe_state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def _scatter_lane(self, pin_aat, pin_tran, result, active_net_ids):
        if self.sink_count <= 0:
            return pin_aat, pin_tran
        if int(result["sink_node_id"].numel()) == 0:
            return pin_aat, pin_tran
        active_pin_id = result["sink_node_id"].to(device=pin_aat.device, dtype=torch.long)
        valid = active_pin_id < int(pin_aat.numel())
        if not bool(torch.any(valid).detach().cpu().item()):
            return pin_aat, pin_tran
        active_pin_id = active_pin_id[valid]
        sink_arrival = result["sink_arrival"].to(device=pin_aat.device)
        sink_slew = result["sink_slew"].to(device=pin_tran.device)
        pin_aat = pin_aat.clone()
        pin_tran = pin_tran.clone()
        pin_aat[active_pin_id] = sink_arrival[valid].to(
            dtype=pin_aat.dtype,
        )
        pin_tran[active_pin_id] = sink_slew[valid].to(
            dtype=pin_tran.dtype,
        )
        return pin_aat, pin_tran

    def _scatter_driver_cap(self, pin_net_cap, result):
        driver_pin_id = result.get("driver_pin_id")
        driver_net_cap = result.get("driver_net_cap")
        if driver_pin_id is None or driver_net_cap is None:
            return pin_net_cap
        if int(driver_pin_id.numel()) == 0 or int(driver_net_cap.numel()) == 0:
            return pin_net_cap
        active_pin_id = driver_pin_id.to(device=pin_net_cap.device, dtype=torch.long)
        valid = active_pin_id < int(pin_net_cap.numel())
        if not bool(torch.any(valid).detach().cpu().item()):
            return pin_net_cap
        pin_net_cap = pin_net_cap.clone()
        pin_net_cap[active_pin_id[valid]] = driver_net_cap.to(
            device=pin_net_cap.device,
            dtype=pin_net_cap.dtype,
        )[valid]
        return pin_net_cap

    def _run_lane_with_z_override(self, *, active_net_ids, z_value):
        return self._run_lane(
            active_net_ids=active_net_ids,
            driver_arrival_by_net={},
            driver_slew_by_net={},
            z_override=z_value,
        )

    def _residualize_driver_cap(self, pin_net_cap, cap_result, zero_cap_result):
        if self.driver_cap_mode == "direct":
            # Use the load of the current segment topology, without carrying
            # the static-versus-zero-state load offset into inserted states.
            self.metadata["driver_cap_residual_baseline"] = "dynamic_segment_load"
            return cap_result
        current_pin = cap_result.get("driver_pin_id")
        current_cap = cap_result.get("driver_net_cap")
        zero_pin = zero_cap_result.get("driver_pin_id")
        zero_cap = zero_cap_result.get("driver_net_cap")
        if any(value is None for value in (current_pin, current_cap, zero_pin, zero_cap)):
            raise RuntimeError("segment-count cap residual requires driver ids and caps")
        current_pin = current_pin.to(device=pin_net_cap.device, dtype=torch.long)
        zero_pin = zero_pin.to(device=pin_net_cap.device, dtype=torch.long)
        if current_pin.shape != zero_pin.shape or not torch.equal(current_pin, zero_pin):
            raise RuntimeError("segment-count cap residual driver ordering changed")
        valid_pin = (current_pin >= 0) & (current_pin < int(pin_net_cap.numel()))
        base = torch.zeros_like(current_cap, device=pin_net_cap.device)
        base[valid_pin] = pin_net_cap.index_select(0, current_pin[valid_pin]).to(base)
        current_cap = current_cap.to(device=pin_net_cap.device, dtype=base.dtype)
        zero_cap = zero_cap.to(device=pin_net_cap.device, dtype=base.dtype)
        finite = torch.isfinite(current_cap) & torch.isfinite(zero_cap)
        residual_cap = torch.where(finite, base + current_cap - zero_cap, base)
        result = dict(cap_result)
        result["driver_pin_id"] = current_pin
        result["driver_net_cap"] = residual_cap
        self.metadata["driver_cap_residual_baseline"] = "static_plus_dynamic_minus_zero_z"
        self.metadata["driver_cap_residual_nonfinite_fallback_count"] = int(
            torch.count_nonzero(~finite).detach().cpu().item()
        )
        return result

    def _residualize_sink_timing(
        self,
        *,
        static_pin_aat,
        static_pin_slew,
        result,
        zero_result,
    ):
        current_pin = result["sink_node_id"].to(
            device=static_pin_aat.device,
            dtype=torch.long,
        )
        zero_pin = zero_result["sink_node_id"].to(
            device=static_pin_aat.device,
            dtype=torch.long,
        )
        if current_pin.shape != zero_pin.shape or not torch.equal(current_pin, zero_pin):
            raise RuntimeError("segment-count timing residual sink ordering changed")
        valid = (current_pin >= 0) & (current_pin < int(static_pin_aat.numel()))
        base_arrival = torch.zeros_like(
            result["sink_arrival"],
            device=static_pin_aat.device,
        )
        base_slew = torch.zeros_like(
            result["sink_slew"],
            device=static_pin_slew.device,
        )
        base_arrival[valid] = static_pin_aat.index_select(0, current_pin[valid]).to(
            base_arrival
        )
        base_slew[valid] = static_pin_slew.index_select(0, current_pin[valid]).to(
            base_slew
        )
        current_arrival = result["sink_arrival"].to(base_arrival)
        zero_arrival = zero_result["sink_arrival"].to(base_arrival)
        current_slew = result["sink_slew"].to(base_slew)
        zero_slew = zero_result["sink_slew"].to(base_slew)
        arrival_finite = torch.isfinite(current_arrival) & torch.isfinite(zero_arrival)
        slew_finite = torch.isfinite(current_slew) & torch.isfinite(zero_slew)
        residual = dict(result)
        residual["sink_node_id"] = current_pin
        residual["sink_arrival"] = torch.where(
            arrival_finite,
            base_arrival + current_arrival - zero_arrival,
            base_arrival,
        )
        residual["sink_slew"] = torch.where(
            slew_finite,
            # Residual calibration can otherwise produce negative transitions,
            # which are not valid inputs to downstream Liberty timing tables.
            torch.clamp_min(base_slew + current_slew - zero_slew, 0.0),
            base_slew,
        )
        self.metadata["sink_timing_residual_baseline"] = (
            "static_plus_dynamic_minus_zero_z"
        )
        self.metadata["sink_timing_residual_nonfinite_fallback_count"] = int(
            torch.count_nonzero(~arrival_finite | ~slew_finite).detach().cpu().item()
        )
        return residual

    def _cap_by_driver_pin(self, cap_result, *, device, dtype):
        driver_pin_id = cap_result.get("driver_pin_id")
        driver_net_cap = cap_result.get("driver_net_cap")
        if driver_pin_id is None or driver_net_cap is None:
            return {}
        driver_pin_id = driver_pin_id.to(device=device, dtype=torch.long)
        driver_net_cap = driver_net_cap.to(device=device, dtype=dtype)
        result = {}
        for pin_id, cap in zip(
            driver_pin_id.detach().cpu().tolist(),
            driver_net_cap.detach().cpu().tolist(),
        ):
            result[int(pin_id)] = float(cap)
        return result

    def _root_load_decomposition(self, *, net_id, driver_pin_id, original_cap, zero_cap):
        if net_id is None:
            return {"status": "missing_net_id"}
        net = self.net_by_id.get(int(net_id))
        if net is None:
            return {"status": "missing_net"}
        prepared = net.get("_segment_count_prepared", {}) or {}
        root_node_id = int(prepared.get("root_node_id", driver_pin_id))
        children_by_node = {
            int(node): [int(child) for child in children]
            for node, children in (prepared.get("children_by_node", {}) or {}).items()
        }
        topology_nodes = [int(node) for node in prepared.get("topo", [])]
        tree_nodes = set(children_by_node)
        for children in children_by_node.values():
            tree_nodes.update(int(child) for child in children)
        node_cap = {
            int(node): float(cap)
            for node, cap in (prepared.get("node_cap", {}) or {}).items()
        }
        sink_nodes = [int(sink) for sink in prepared.get("sink_nodes", [])]
        edge_rc = prepared.get("edge_rc", {}) or {}
        edge_cap_sum, edge_count, missing_edge_cap_count = _sum_tree_edge_cap(
            children_by_node,
            edge_rc,
        )
        root_node_cap = float(node_cap.get(root_node_id, 0.0))
        sink_node_cap_sum = sum(float(node_cap.get(int(sink), 0.0)) for sink in sink_nodes)
        total_node_cap_sum = sum(float(cap) for cap in node_cap.values())
        manual_root_load = total_node_cap_sum + edge_cap_sum
        return {
            "status": "ok",
            "root_node_id": int(root_node_id),
            "driver_pin_id": int(driver_pin_id),
            "root_matches_driver_pin": bool(int(root_node_id) == int(driver_pin_id)),
            "sink_count": int(len(sink_nodes)),
            "topology_node_count": int(len(topology_nodes)),
            "tree_node_count": int(len(tree_nodes)),
            "node_cap_entry_count": int(len(node_cap)),
            "edge_count": int(edge_count),
            "edge_cap_missing_or_zero_count": int(missing_edge_cap_count),
            "nonzero_node_cap_count": int(
                sum(1 for cap in node_cap.values() if float(cap) != 0.0)
            ),
            "node_cap_without_tree_node_count": int(
                sum(1 for node in node_cap if int(node) not in tree_nodes)
            ),
            "tree_node_without_node_cap_count": int(
                sum(1 for node in tree_nodes if int(node) not in node_cap)
            ),
            "sink_without_tree_node_count": int(
                sum(1 for sink in sink_nodes if int(sink) not in tree_nodes)
            ),
            "root_node_cap": float(root_node_cap),
            "sink_node_cap_sum": float(sink_node_cap_sum),
            "total_node_cap_sum": float(total_node_cap_sum),
            "edge_cap_sum": float(edge_cap_sum),
            "manual_root_load_node_plus_edge": float(manual_root_load),
            "zero_z_relaxed_cap": float(zero_cap),
            "original_pin_net_cap": float(original_cap),
            "manual_minus_zero_z_relaxed_cap": float(manual_root_load - float(zero_cap)),
            "manual_minus_original_pin_net_cap": float(manual_root_load - float(original_cap)),
            "zero_z_minus_original_pin_net_cap": float(float(zero_cap) - float(original_cap)),
        }

    def _driver_cap_delta_summary(
        self,
        *,
        pin_net_cap,
        cap_result,
        zero_cap_result=None,
        top_k=20,
    ):
        driver_pin_id = cap_result.get("driver_pin_id")
        driver_net_cap = cap_result.get("driver_net_cap")
        if driver_pin_id is None or driver_net_cap is None:
            return {"status": "missing_driver_cap_result"}
        if int(driver_pin_id.numel()) == 0 or int(driver_net_cap.numel()) == 0:
            return {"status": "empty_driver_cap_result"}

        active_pin_id = driver_pin_id.to(device=pin_net_cap.device, dtype=torch.long)
        valid = active_pin_id < int(pin_net_cap.numel())
        if not bool(torch.any(valid).detach().cpu().item()):
            return {
                "status": "no_valid_driver_pins",
                "driver_pin_count": int(active_pin_id.numel()),
                "pin_cap_count": int(pin_net_cap.numel()),
            }
        active_pin_id = active_pin_id[valid]
        relaxed_cap = driver_net_cap.to(device=pin_net_cap.device, dtype=pin_net_cap.dtype)[valid]
        original_cap = pin_net_cap.index_select(0, active_pin_id)
        delta = relaxed_cap - original_cap
        ratio = relaxed_cap / torch.clamp(torch.abs(original_cap), min=1e-12)

        zero_relaxed_cap = None
        zero_delta = None
        if zero_cap_result is not None and zero_cap_result.get("driver_net_cap") is not None:
            zero_driver_cap = zero_cap_result["driver_net_cap"].to(
                device=pin_net_cap.device,
                dtype=pin_net_cap.dtype,
            )
            zero_driver_pin_id = zero_cap_result["driver_pin_id"].to(
                device=pin_net_cap.device,
                dtype=torch.long,
            )
            zero_valid = zero_driver_pin_id < int(pin_net_cap.numel())
            if bool(torch.any(zero_valid).detach().cpu().item()):
                zero_cap_by_pin = self._cap_by_driver_pin(
                    zero_cap_result,
                    device=pin_net_cap.device,
                    dtype=pin_net_cap.dtype,
                )
                zero_values = [
                    zero_cap_by_pin.get(int(pin_id))
                    for pin_id in active_pin_id.detach().cpu().tolist()
                ]
                if all(value is not None for value in zero_values):
                    zero_relaxed_cap = torch.as_tensor(
                        zero_values,
                        dtype=pin_net_cap.dtype,
                        device=pin_net_cap.device,
                    )
                    zero_delta = zero_relaxed_cap - original_cap

        z_value = self.segment_state.z_value().detach()
        bsu_value = self.segment_state.bsu_index().detach()
        top_count = min(int(top_k), int(delta.numel()))
        if top_count > 0:
            top_values, top_indices = torch.topk(torch.abs(delta), k=top_count)
            top_rows = []
            for rank, local_index in enumerate(top_indices.detach().cpu().tolist()):
                local_index = int(local_index)
                driver_pin = int(active_pin_id[local_index].detach().cpu().item())
                net_id = self.net_by_driver_pin.get(driver_pin)
                row_indices = (
                    self.segment_state.net_to_segment_rows.get(int(net_id), [])
                    if net_id is not None
                    else []
                )
                row_tensor = torch.as_tensor(
                    row_indices,
                    dtype=torch.long,
                    device=z_value.device,
                )
                row = {
                    "rank": int(rank),
                    "net_id": net_id,
                    "driver_pin_id": driver_pin,
                    "original_cap": float(original_cap[local_index].detach().cpu().item()),
                    "relaxed_cap": float(relaxed_cap[local_index].detach().cpu().item()),
                    "delta": float(delta[local_index].detach().cpu().item()),
                    "abs_delta": float(top_values[rank].detach().cpu().item()),
                    "ratio_abs_original": float(ratio[local_index].detach().cpu().item()),
                    "segment_count": int(len(row_indices)),
                }
                if zero_relaxed_cap is not None and zero_delta is not None:
                    row["zero_z_relaxed_cap"] = float(
                        zero_relaxed_cap[local_index].detach().cpu().item()
                    )
                    row["zero_z_delta"] = float(
                        zero_delta[local_index].detach().cpu().item()
                    )
                    row["current_minus_zero_z_cap"] = float(
                        (relaxed_cap[local_index] - zero_relaxed_cap[local_index])
                        .detach()
                        .cpu()
                        .item()
                    )
                    row["root_load_decomposition"] = self._root_load_decomposition(
                        net_id=net_id,
                        driver_pin_id=int(active_pin_id[local_index].detach().cpu().item()),
                        original_cap=float(original_cap[local_index].detach().cpu().item()),
                        zero_cap=float(zero_relaxed_cap[local_index].detach().cpu().item()),
                    )
                if int(row_tensor.numel()) > 0:
                    row_z = z_value.index_select(0, row_tensor)
                    row_bsu = bsu_value.index_select(0, row_tensor)
                    row["z_min"] = float(torch.min(row_z).detach().cpu().item())
                    row["z_max"] = float(torch.max(row_z).detach().cpu().item())
                    row["z_mean"] = float(torch.mean(row_z).detach().cpu().item())
                    row["bsu_min"] = float(torch.min(row_bsu).detach().cpu().item())
                    row["bsu_max"] = float(torch.max(row_bsu).detach().cpu().item())
                    row["bsu_mean"] = float(torch.mean(row_bsu).detach().cpu().item())
                top_rows.append(row)
        else:
            top_rows = []

        summary = {
            "status": "ok",
            "driver_pin_count": int(active_pin_id.numel()),
            "pin_cap_count": int(pin_net_cap.numel()),
            "z_stats": _finite_stats(z_value),
            "bsu_stats": _finite_stats(bsu_value),
            "original_cap_stats": _finite_stats(original_cap),
            "relaxed_cap_stats": _finite_stats(relaxed_cap),
            "delta_stats": _finite_stats(delta),
            "ratio_abs_original_stats": _finite_stats(ratio),
            "delta_positive_count": int(torch.sum(delta > 0).detach().cpu().item()),
            "delta_negative_count": int(torch.sum(delta < 0).detach().cpu().item()),
            "delta_zero_count": int(torch.sum(delta == 0).detach().cpu().item()),
            "top_abs_delta": top_rows,
        }
        if zero_relaxed_cap is not None and zero_delta is not None:
            summary.update(
                {
                    "zero_z_relaxed_cap_stats": _finite_stats(zero_relaxed_cap),
                    "zero_z_delta_stats": _finite_stats(zero_delta),
                    "zero_z_delta_positive_count": int(
                        torch.sum(zero_delta > 0).detach().cpu().item()
                    ),
                    "zero_z_delta_negative_count": int(
                        torch.sum(zero_delta < 0).detach().cpu().item()
                    ),
                    "zero_z_delta_zero_count": int(
                        torch.sum(zero_delta == 0).detach().cpu().item()
                    ),
                    "current_minus_zero_z_cap_stats": _finite_stats(
                        relaxed_cap - zero_relaxed_cap
                    ),
                }
            )
        return summary

    def _maybe_write_cap_overlay_probe(self, *, pin_net_cap, cap_result):
        probe_path = os.environ.get("AIMP_BUFFERING_SEGMENT_CAP_OVERLAY_PROBE_JSON")
        if not probe_path or self._cap_overlay_probe_written:
            return
        zero_cap_result = self._run_lane_with_z_override(
            active_net_ids=self.affected_net_ids,
            z_value=0.0,
        )
        summary = self._driver_cap_delta_summary(
            pin_net_cap=pin_net_cap,
            cap_result=cap_result,
            zero_cap_result=zero_cap_result,
        )
        summary["artifact"] = "segment_count_driver_cap_overlay_probe"
        summary["artifact_version"] = 2
        summary["load_semantics"] = {
            "original_pin_net_cap": "rc_timing LoadOp output: bottom-up load of pin_caps, where pin_caps includes base pin cap plus half-edge wire caps",
            "zero_z_relaxed_cap": "segment-count prepared timing root node_load with z forced to zero",
            "manual_root_load_node_plus_edge": "direct sum over the segment RC tree of all node_cap entries plus all directed edge wire caps",
        }
        summary["backend"] = self.backend
        summary["backend_used"] = self.backend_used
        summary["backend_fallback_reason"] = self.backend_fallback_reason
        summary["affected_net_count"] = int(len(self.affected_net_ids))
        summary["segment_count_state_count"] = int(self.segment_state.z_param.numel())
        Path(probe_path).parent.mkdir(parents=True, exist_ok=True)
        Path(probe_path).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._cap_overlay_probe_written = True

    def apply_dynamic_net_cap_overlay(
        self,
        *,
        pin_net_cap_rise,
        pin_net_cap_fall,
    ):
        device = self.segment_state.z_param.device
        started_at = self._profile_clock(device) if self.profile_enabled else None
        if not self.affected_net_ids:
            return pin_net_cap_rise, pin_net_cap_fall
        cap_result = self._run_cap_only(active_net_ids=self.affected_net_ids, sense="rise")
        zero_cap_result = self._run_cap_only(
            active_net_ids=self.affected_net_ids,
            z_override=0.0,
            sense="rise",
        )
        phase_inputs = (
            getattr(self.buffer_device_lut, "phase_luts", None) is not None
            or (
                self._live_geometry_context is not None
                and self._live_geometry_context["pin_capacitance_by_sense"] is not None
            )
        )
        fall_cap_result = (
            self._run_cap_only(active_net_ids=self.affected_net_ids, sense="fall")
            if phase_inputs else cap_result
        )
        fall_zero_cap_result = (
            self._run_cap_only(active_net_ids=self.affected_net_ids, z_override=0.0, sense="fall")
            if phase_inputs else zero_cap_result
        )
        if self.profile_enabled:
            cap_finished_at = self._profile_clock(device)
            self.metadata["driver_cap_overlay_forward_ms"] += (
                cap_finished_at - started_at
            ) * 1000.0
            self.metadata["driver_cap_overlay_backend"] = (
                cap_result.get("metadata", {}).get("backend")
                if isinstance(cap_result, dict)
                else None
            )
        self._maybe_write_cap_overlay_probe(
            pin_net_cap=pin_net_cap_rise,
            cap_result=cap_result,
        )
        cap_result_rise = self._residualize_driver_cap(
            pin_net_cap_rise,
            cap_result,
            zero_cap_result,
        )
        cap_result_fall = self._residualize_driver_cap(
            pin_net_cap_fall,
            fall_cap_result,
            fall_zero_cap_result,
        )
        scatter_started_at = self._profile_clock(device) if self.profile_enabled else None
        pin_net_cap_rise = self._scatter_driver_cap(pin_net_cap_rise, cap_result_rise)
        pin_net_cap_fall = self._scatter_driver_cap(pin_net_cap_fall, cap_result_fall)
        self._last_pin_net_cap_rise_after_overlay = pin_net_cap_rise
        self._last_pin_net_cap_fall_after_overlay = pin_net_cap_fall
        self.metadata["driver_cap_overlay_count"] += 1
        if self.profile_enabled:
            scatter_finished_at = self._profile_clock(device)
            self.metadata["driver_cap_overlay_scatter_ms"] += (
                scatter_finished_at - scatter_started_at
            ) * 1000.0
            self.metadata["driver_cap_overlay_ms"] += (
                scatter_finished_at - started_at
            ) * 1000.0
            self._update_cuda_memory_profile(device)
        return pin_net_cap_rise, pin_net_cap_fall

    def begin_critical_path_net_delay_snapshot(
        self,
        *,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
    ):
        self._virtual_slew_violation = base_pin_net_delay_rise.new_zeros(())
        self._virtual_cap_violation = base_pin_net_delay_rise.new_zeros(())
        self._critical_path_pin_net_delay_rise = (
            base_pin_net_delay_rise.clone()
        )
        self._critical_path_pin_net_delay_fall = (
            base_pin_net_delay_fall.clone()
        )
        self.metadata["critical_path_delay_overlay_sink_count"] = 0

    def consume_critical_path_net_delays(
        self,
        *,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
    ):
        rise = self._critical_path_pin_net_delay_rise
        fall = self._critical_path_pin_net_delay_fall
        self._critical_path_pin_net_delay_rise = None
        self._critical_path_pin_net_delay_fall = None
        if rise is None or fall is None:
            return base_pin_net_delay_rise, base_pin_net_delay_fall
        return rise, fall

    def virtual_buffer_drv_tensors(self):
        if self.backend_used != "cpp_cpu_segment_transfer_explicit_autograd":
            return None
        return self._virtual_slew_violation / 1000.0, self._virtual_cap_violation

    @staticmethod
    def _level_net_ids(level_view, *, device):
        if level_view.get("native_active_view"):
            values = level_view["prepared_inputs"]["net_ids"]
        else:
            values = [
                int(net)
                if not isinstance(net, dict)
                else int(
                    net.get("_segment_count_prepared", {}).get(
                        "net_id",
                        _net_id(net, index),
                    )
                )
                for index, net in enumerate(level_view.get("active_nets", ()))
            ]
        return torch.as_tensor(values, dtype=torch.long, device=device)

    def _record_critical_path_net_delays(
        self,
        *,
        sense,
        pin_aat,
        result,
        level_view,
    ):
        overlay = (
            self._critical_path_pin_net_delay_rise
            if sense == "rise"
            else self._critical_path_pin_net_delay_fall
        )
        if overlay is None:
            return
        sink_pin = result["sink_node_id"].to(
            device=pin_aat.device,
            dtype=torch.long,
        )
        sink_net = result["sink_net_index"].to(
            device=pin_aat.device,
            dtype=torch.long,
        )
        sink_arrival = result["sink_arrival"].to(device=pin_aat.device)
        net_ids = self._level_net_ids(level_view, device=pin_aat.device)
        driver_pin = level_view["driver_pin_index"].to(
            device=pin_aat.device,
            dtype=torch.long,
        )
        if not int(sink_pin.numel()) or not int(net_ids.numel()):
            return
        sorted_net_ids, order = torch.sort(net_ids)
        sorted_driver_pin = driver_pin.index_select(0, order)
        positions = torch.searchsorted(sorted_net_ids, sink_net)
        valid = positions < int(sorted_net_ids.numel())
        if bool(torch.any(valid).detach().cpu().item()):
            valid_index = torch.nonzero(valid, as_tuple=False).flatten()
            valid_positions = positions.index_select(0, valid_index)
            matched = torch.zeros_like(valid)
            matched[valid_index] = (
                sorted_net_ids.index_select(0, valid_positions)
                == sink_net.index_select(0, valid_index)
            )
            valid &= matched
        valid &= (sink_pin >= 0) & (sink_pin < int(overlay.numel()))
        if not bool(torch.any(valid).detach().cpu().item()):
            return
        selected_positions = positions[valid]
        selected_driver_pin = sorted_driver_pin.index_select(
            0,
            selected_positions,
        )
        driver_valid = (
            (selected_driver_pin >= 0)
            & (selected_driver_pin < int(pin_aat.numel()))
        )
        if not bool(torch.any(driver_valid).detach().cpu().item()):
            return
        selected_sink_pin = sink_pin[valid][driver_valid]
        selected_sink_arrival = sink_arrival[valid][driver_valid]
        selected_driver_pin = selected_driver_pin[driver_valid]
        effective_delay = (
            selected_sink_arrival - pin_aat.index_select(0, selected_driver_pin)
        )
        overlay[selected_sink_pin] = effective_delay.to(
            device=overlay.device,
            dtype=overlay.dtype,
        )
        self.metadata["critical_path_delay_overlay_sink_count"] += int(
            selected_sink_pin.numel()
        )

    def propagate_net_aat_level(
        self,
        *,
        net_ids,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
        base_pin_net_impulse_rise,
        base_pin_net_impulse_fall,
        static_calculate_net_aat_level,
        level_id=None,
        level_view_epoch=None,
    ):
        profile_device = pin_rAAT.device
        started_at = self._profile_clock(profile_device) if self.profile_enabled else None
        self.call_count += 1
        level_view = self._level_view(
            net_ids,
            level_id=level_id,
            level_view_epoch=level_view_epoch,
        )
        active_net_ids = level_view["active_net_ids"]
        self.metadata["dynamic_provider_call_count"] = int(self.call_count)
        if not level_view["has_dynamic_nets"]:
            result = static_calculate_net_aat_level(
                net_ids,
                pin_rAAT,
                pin_fAAT,
                pin_rtran,
                pin_ftran,
                base_pin_net_delay_rise,
                base_pin_net_delay_fall,
                base_pin_net_impulse_rise,
                base_pin_net_impulse_fall,
            )
            if self.profile_enabled:
                finished_at = self._profile_clock(profile_device)
                self.metadata["provider_dispatch_ms"] += (
                    finished_at - started_at
                ) * 1000.0
            return result

        self.affected_call_count += 1
        self.metadata["dynamic_provider_affected_net_call_count"] = int(
            self.affected_call_count
        )
        self.metadata["affected_net_call_count"] = int(self.affected_call_count)
        pin_rAAT, pin_fAAT, pin_rtran, pin_ftran = static_calculate_net_aat_level(
            net_ids,
            pin_rAAT,
            pin_fAAT,
            pin_rtran,
            pin_ftran,
            base_pin_net_delay_rise,
            base_pin_net_delay_fall,
            base_pin_net_impulse_rise,
            base_pin_net_impulse_fall,
        )
        static_pin_rAAT = pin_rAAT
        static_pin_fAAT = pin_fAAT
        static_pin_rtran = pin_rtran
        static_pin_ftran = pin_ftran

        gather_started_at = (
            self._profile_clock(profile_device) if self.profile_enabled else None
        )
        probe_enabled = bool(os.environ.get("AIMP_BUFFERING_SEGMENT_DEVICE_PROBE_JSON"))
        use_tensorized_driver_gather = (
            level_id is not None
            and not probe_enabled
            and self.backend_used in TENSORIZED_DRIVER_GATHER_TIMING_BACKENDS
        )
        if not use_tensorized_driver_gather:
            rise_driver_arrival = self._driver_values(pin_rAAT, active_net_ids)
            rise_driver_slew = self._driver_values(pin_rtran, active_net_ids)
            fall_driver_arrival = self._driver_values(pin_fAAT, active_net_ids)
            fall_driver_slew = self._driver_values(pin_ftran, active_net_ids)
            rise_driver_arrival_tensor = None
            rise_driver_slew_tensor = None
            fall_driver_arrival_tensor = None
            fall_driver_slew_tensor = None
        else:
            rise_driver_arrival = None
            rise_driver_slew = None
            fall_driver_arrival = None
            fall_driver_slew = None
            rise_driver_arrival_tensor = self._gather_level_driver_values(
                pin_rAAT,
                level_view,
            )
            rise_driver_slew_tensor = self._gather_level_driver_values(
                pin_rtran,
                level_view,
            )
            fall_driver_arrival_tensor = self._gather_level_driver_values(
                pin_fAAT,
                level_view,
            )
            fall_driver_slew_tensor = self._gather_level_driver_values(
                pin_ftran,
                level_view,
            )
            self.metadata["tensorized_driver_gather_count"] += 4
        if self.profile_enabled:
            gather_finished_at = self._profile_clock(profile_device)
            self.metadata["driver_timing_gather_ms"] += (
                gather_finished_at - gather_started_at
            ) * 1000.0

        forward_started_at = (
            self._profile_clock(profile_device) if self.profile_enabled else None
        )
        rise_result = self._run_lane(
            sense="rise",
            active_net_ids=active_net_ids,
            driver_arrival_by_net=rise_driver_arrival,
            driver_slew_by_net=rise_driver_slew,
            level_view=level_view,
            driver_arrival=rise_driver_arrival_tensor,
            driver_slew=rise_driver_slew_tensor,
        )
        rise_zero_result = self._run_lane(
            sense="rise",
            active_net_ids=active_net_ids,
            driver_arrival_by_net=rise_driver_arrival,
            driver_slew_by_net=rise_driver_slew,
            level_view=level_view,
            driver_arrival=rise_driver_arrival_tensor,
            driver_slew=rise_driver_slew_tensor,
            z_override=0.0,
        )
        rise_finished_at = (
            self._profile_clock(profile_device) if self.profile_enabled else None
        )
        fall_started_at = (
            self._profile_clock(profile_device) if self.profile_enabled else None
        )
        fall_result = self._run_lane(
            sense="fall",
            active_net_ids=active_net_ids,
            driver_arrival_by_net=fall_driver_arrival,
            driver_slew_by_net=fall_driver_slew,
            level_view=level_view,
            driver_arrival=fall_driver_arrival_tensor,
            driver_slew=fall_driver_slew_tensor,
        )
        fall_zero_result = self._run_lane(
            sense="fall",
            active_net_ids=active_net_ids,
            driver_arrival_by_net=fall_driver_arrival,
            driver_slew_by_net=fall_driver_slew,
            level_view=level_view,
            driver_arrival=fall_driver_arrival_tensor,
            driver_slew=fall_driver_slew_tensor,
            z_override=0.0,
        )
        if "buffer_slew_violation" in rise_result:
            # Match real-pin reporting: worst rise/fall per physical repeater,
            # then sum, rather than taking the worse phase of an entire chain.
            self._virtual_slew_violation = self._virtual_slew_violation + torch.maximum(
                rise_result["buffer_slew_violation"], fall_result["buffer_slew_violation"]
            ).sum()
            self._virtual_cap_violation = self._virtual_cap_violation + torch.maximum(
                rise_result["buffer_cap_violation"], fall_result["buffer_cap_violation"]
            ).sum()
        rise_result = self._residualize_sink_timing(
            static_pin_aat=static_pin_rAAT,
            static_pin_slew=static_pin_rtran,
            result=rise_result,
            zero_result=rise_zero_result,
        )
        fall_result = self._residualize_sink_timing(
            static_pin_aat=static_pin_fAAT,
            static_pin_slew=static_pin_ftran,
            result=fall_result,
            zero_result=fall_zero_result,
        )
        self._record_critical_path_net_delays(
            sense="rise",
            pin_aat=static_pin_rAAT,
            result=rise_result,
            level_view=level_view,
        )
        self._record_critical_path_net_delays(
            sense="fall",
            pin_aat=static_pin_fAAT,
            result=fall_result,
            level_view=level_view,
        )
        if probe_enabled:
            self._append_device_probe_rows(
                sense="rise",
                active_net_ids=active_net_ids,
                driver_arrival_by_net=rise_driver_arrival,
                driver_slew_by_net=rise_driver_slew,
                pin_net_cap_rise=self._last_pin_net_cap_rise_after_overlay,
                pin_net_cap_fall=self._last_pin_net_cap_fall_after_overlay,
                result=rise_result,
            )
            self._append_device_probe_rows(
                sense="fall",
                active_net_ids=active_net_ids,
                driver_arrival_by_net=fall_driver_arrival,
                driver_slew_by_net=fall_driver_slew,
                pin_net_cap_rise=self._last_pin_net_cap_rise_after_overlay,
                pin_net_cap_fall=self._last_pin_net_cap_fall_after_overlay,
                result=fall_result,
            )
        if self.profile_enabled:
            fall_finished_at = self._profile_clock(profile_device)
            self.metadata["rise_transfer_ms"] += (
                rise_finished_at - forward_started_at
            ) * 1000.0
            self.metadata["fall_transfer_ms"] += (
                fall_finished_at - fall_started_at
            ) * 1000.0
            self.metadata["segment_count_relaxed_timing_ms"] += (
                fall_finished_at - forward_started_at
            ) * 1000.0
            self.metadata["segment_count_timing_forward_ms"] = float(
                self.metadata["segment_count_relaxed_timing_ms"]
            )
            self.metadata["net_subgraph_forward_relaxed_ms"] = float(
                self.metadata["segment_count_relaxed_timing_ms"]
            )

        scatter_started_at = (
            self._profile_clock(profile_device) if self.profile_enabled else None
        )
        pin_rAAT, pin_rtran = self._scatter_lane(
            pin_rAAT,
            pin_rtran,
            rise_result,
            active_net_ids,
        )
        rise_scatter_finished_at = (
            self._profile_clock(profile_device) if self.profile_enabled else None
        )
        pin_fAAT, pin_ftran = self._scatter_lane(
            pin_fAAT,
            pin_ftran,
            fall_result,
            active_net_ids,
        )
        self._maybe_write_aat_overlay_probe(
            active_net_ids=active_net_ids,
            static_pin_rAAT=static_pin_rAAT,
            static_pin_fAAT=static_pin_fAAT,
            static_pin_rtran=static_pin_rtran,
            static_pin_ftran=static_pin_ftran,
            dynamic_pin_rAAT=pin_rAAT,
            dynamic_pin_fAAT=pin_fAAT,
            dynamic_pin_rtran=pin_rtran,
            dynamic_pin_ftran=pin_ftran,
        )
        if self.profile_enabled:
            scatter_finished_at = self._profile_clock(profile_device)
            self.metadata["rise_sink_scatter_ms"] += (
                rise_scatter_finished_at - scatter_started_at
            ) * 1000.0
            self.metadata["fall_sink_scatter_ms"] += (
                scatter_finished_at - rise_scatter_finished_at
            ) * 1000.0
            self.metadata["sink_scatter_ms"] += (
                scatter_finished_at - scatter_started_at
            ) * 1000.0
            self.metadata["provider_dispatch_ms"] += (
                scatter_finished_at - started_at
            ) * 1000.0
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran

    def _maybe_write_aat_overlay_probe(
        self,
        *,
        active_net_ids,
        static_pin_rAAT,
        static_pin_fAAT,
        static_pin_rtran,
        static_pin_ftran,
        dynamic_pin_rAAT,
        dynamic_pin_fAAT,
        dynamic_pin_rtran,
        dynamic_pin_ftran,
    ):
        probe_path = os.environ.get("AIMP_BUFFERING_SEGMENT_AAT_OVERLAY_PROBE_JSON")
        if not probe_path:
            return
        target_raw = os.environ.get("AIMP_BUFFERING_SEGMENT_AAT_OVERLAY_NET_IDS", "")
        target_net_ids = {
            int(part.strip())
            for part in str(target_raw).split(",")
            if part.strip().isdigit()
        }
        if target_net_ids:
            active_net_ids = set(int(net_id) for net_id in active_net_ids) & target_net_ids
            if not active_net_ids:
                return
        if self._aat_overlay_probe_state is None:
            self._aat_overlay_probe_state = {
                "call_count": 0,
                "active_net_ids": set(),
                "row_count": 0,
                "top_abs_delta": [],
                "stats": {
                    key: {
                        "count": 0,
                        "min": None,
                        "max": None,
                        "sum": 0.0,
                        "abs_max": None,
                    }
                    for key in ("delta_rAAT", "delta_fAAT", "delta_rtran", "delta_ftran")
                },
            }
        state = self._aat_overlay_probe_state
        state["call_count"] += 1
        state["active_net_ids"].update(int(net_id) for net_id in active_net_ids)
        static_r = static_pin_rAAT.detach().float().cpu()
        static_f = static_pin_fAAT.detach().float().cpu()
        static_rt = static_pin_rtran.detach().float().cpu()
        static_ft = static_pin_ftran.detach().float().cpu()
        dynamic_r = dynamic_pin_rAAT.detach().float().cpu()
        dynamic_f = dynamic_pin_fAAT.detach().float().cpu()
        dynamic_rt = dynamic_pin_rtran.detach().float().cpu()
        dynamic_ft = dynamic_pin_ftran.detach().float().cpu()

        rows = []
        for net_id in sorted(int(net_id) for net_id in active_net_ids):
            net = self.net_by_id.get(int(net_id), {})
            sink_ids = []
            rc_tree = dict(net.get("rc_tree", {}) or {})
            for key in ("sink_pin_ids", "sink_nodes"):
                if rc_tree.get(key):
                    sink_ids = [int(value) for value in rc_tree.get(key, [])]
                    break
            if not sink_ids:
                sink_ids = [int(value) for value in net.get("sink_pin_ids", net.get("sink_nodes", [])) or []]
            for pin_id in sink_ids:
                if pin_id >= static_r.numel():
                    continue
                row = {
                    "net_id": int(net_id),
                    "net_name": net.get("net_name"),
                    "pin_id": int(pin_id),
                    "static_rAAT": float(static_r[pin_id].item()),
                    "dynamic_rAAT": float(dynamic_r[pin_id].item()),
                    "delta_rAAT": float(dynamic_r[pin_id].item() - static_r[pin_id].item()),
                    "static_fAAT": float(static_f[pin_id].item()),
                    "dynamic_fAAT": float(dynamic_f[pin_id].item()),
                    "delta_fAAT": float(dynamic_f[pin_id].item() - static_f[pin_id].item()),
                    "static_rtran": float(static_rt[pin_id].item()),
                    "dynamic_rtran": float(dynamic_rt[pin_id].item()),
                    "delta_rtran": float(dynamic_rt[pin_id].item() - static_rt[pin_id].item()),
                    "static_ftran": float(static_ft[pin_id].item()),
                    "dynamic_ftran": float(dynamic_ft[pin_id].item()),
                    "delta_ftran": float(dynamic_ft[pin_id].item() - static_ft[pin_id].item()),
                }
                rows.append(row)
        state["row_count"] += len(rows)
        for row in rows:
            for key, stats in state["stats"].items():
                value = float(row[key])
                stats["count"] += 1
                stats["sum"] += value
                stats["min"] = value if stats["min"] is None else min(stats["min"], value)
                stats["max"] = value if stats["max"] is None else max(stats["max"], value)
                abs_value = abs(value)
                stats["abs_max"] = (
                    abs_value
                    if stats["abs_max"] is None
                    else max(stats["abs_max"], abs_value)
                )
        state["top_abs_delta"].extend(rows)
        rows.sort(
            key=lambda row: max(
                abs(float(row["delta_rAAT"])),
                abs(float(row["delta_fAAT"])),
                abs(float(row["delta_rtran"])),
                abs(float(row["delta_ftran"])),
            ),
            reverse=True,
        )
        state["top_abs_delta"].sort(
            key=lambda row: max(
                abs(float(row["delta_rAAT"])),
                abs(float(row["delta_fAAT"])),
                abs(float(row["delta_rtran"])),
                abs(float(row["delta_ftran"])),
            ),
            reverse=True,
        )
        state["top_abs_delta"] = state["top_abs_delta"][:100]
        stats_payload = {}
        for key, stats in state["stats"].items():
            count = int(stats["count"])
            stats_payload[key] = {
                "count": count,
                "min": stats["min"],
                "max": stats["max"],
                "mean": (float(stats["sum"]) / count) if count else None,
                "abs_max": stats["abs_max"],
            }
        payload = {
            "artifact": "segment_count_aat_overlay_probe",
            "artifact_version": 2,
            "call_count": int(state["call_count"]),
            "active_net_count": int(len(state["active_net_ids"])),
            "row_count": int(state["row_count"]),
            "stats": stats_payload,
            "top_abs_delta": state["top_abs_delta"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(probe_path)), exist_ok=True)
        with open(probe_path, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        self._aat_overlay_probe_written = True
