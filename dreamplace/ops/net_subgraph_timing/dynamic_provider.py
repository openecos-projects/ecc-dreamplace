import time

import torch

from dreamplace.ops.net_subgraph_timing.candidate_native_packer import (
    select_packed_candidate_inputs_native,
)
from dreamplace.ops.net_subgraph_timing.net_subgraph_timing import (
    net_subgraph_forward_relaxed,
)
from dreamplace.ops.timing_propagation.dynamic_net_provider import DynamicNetProvider


_NET_SUBGRAPH_FORWARD_KEYS = {
    "net_flat_topo_sort",
    "net_flat_topo_sort_start",
    "pin_fa",
    "flat_pin_to_start",
    "flat_pin_to",
    "edge_resistance",
    "node_capacitance",
    "edge_capacitance",
    "sink_node_id",
    "sink_net_index",
}


def _payload_value_on_device(value, *, device, dtype=None):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if dtype is None:
        return value.to(device=device)
    return value.to(device=device, dtype=dtype)


def _net_ids_from_payload(static_payload):
    metadata = dict(static_payload.get("metadata", {}) or {})
    net_ids = metadata.get("net_ids")
    if net_ids is not None:
        return [int(net_id) for net_id in net_ids]
    topo_start = static_payload["net_flat_topo_sort_start"]
    return list(range(max(0, int(topo_start.numel()) - 1)))


def _driver_pin_ids_from_payload(static_payload, net_ids):
    metadata = dict(static_payload.get("metadata", {}) or {})
    compact_node_to_original = list(metadata.get("compact_node_to_original", []))
    topo = static_payload["net_flat_topo_sort"].detach().cpu().tolist()
    topo_start = static_payload["net_flat_topo_sort_start"].detach().cpu().tolist()
    driver_pin_by_net = {}
    for net_index, net_id in enumerate(net_ids):
        if net_index + 1 >= len(topo_start):
            continue
        begin = int(topo_start[net_index])
        end = int(topo_start[net_index + 1])
        if begin == end:
            continue
        root_compact = int(topo[begin])
        if 0 <= root_compact < len(compact_node_to_original):
            driver_pin_by_net[int(net_id)] = int(compact_node_to_original[root_compact])
    return driver_pin_by_net


def _driver_pin_index_from_payload(static_payload, net_ids):
    driver_pin_by_net = _driver_pin_ids_from_payload(static_payload, net_ids)
    fallback = static_payload["driver_arrival"]
    fallback_indices = set()
    indices = []
    for net_index, net_id in enumerate(net_ids):
        driver_pin = driver_pin_by_net.get(int(net_id))
        if driver_pin is None:
            fallback_indices.add(int(net_index))
            indices.append(0)
        else:
            indices.append(int(driver_pin))
    return (
        torch.as_tensor(indices, dtype=torch.long),
        fallback_indices,
        driver_pin_by_net,
    )


def _as_net_id_set(net_ids):
    if torch.is_tensor(net_ids):
        return {int(value) for value in net_ids.detach().cpu().tolist()}
    return {int(value) for value in list(net_ids or [])}


def _as_cpu_long_tensor(values):
    if torch.is_tensor(values):
        return values.detach().cpu().to(dtype=torch.long)
    return torch.as_tensor(list(values or []), dtype=torch.long)


def _index_select_cpu(tensor, index):
    index = _as_cpu_long_tensor(index)
    if int(index.numel()) == 0:
        return tensor.detach().cpu()[:0]
    return tensor.detach().cpu().index_select(0, index)


class RelaxedBufferDynamicNetProvider(DynamicNetProvider):
    """TimingPropagation net AAT provider for relaxed buffer candidates."""

    def __init__(
        self,
        *,
        static_payload,
        buffer_state,
        affected_net_ids=None,
        coordinate_source=None,
        forward_backend="native_explicit_autograd",
        buffer_device_lut=None,
    ):
        self.static_payload = dict(static_payload)
        self.buffer_state = buffer_state
        self.net_ids = _net_ids_from_payload(self.static_payload)
        self._net_index_by_id = {
            int(net_id): int(index)
            for index, net_id in enumerate(self.net_ids)
        }
        self._candidate_net_ids = {
            int(value)
            for value in self.buffer_state.candidate_net_id.detach().cpu().tolist()
        }
        self.affected_net_ids = (
            {int(net_id) for net_id in affected_net_ids}
            if affected_net_ids is not None
            else set(self._candidate_net_ids)
        )
        (
            self._driver_pin_index_cpu,
            self._driver_fallback_indices,
            self.driver_pin_by_net,
        ) = _driver_pin_index_from_payload(
            self.static_payload,
            self.net_ids,
        )
        self._driver_pin_index_cache = {}
        self._driver_arrival_fallback_cache = {}
        self.coordinate_source = str(
            coordinate_source
            if coordinate_source is not None
            else self.static_payload.get("coordinate_source", "current_relaxed")
        )
        self.forward_backend = str(forward_backend or "python")
        self.buffer_device_lut = buffer_device_lut
        self._forward_kwargs_cache = {}
        self._active_forward_kwargs_cache = {}
        self._active_view_cache = {}
        self._per_size_cache = {}
        self._buffer_device_lut_cache = {}
        self._candidate_node_id_cache = None
        self._sink_pin_id_cache = None
        self.call_count = 0
        self.affected_call_count = 0
        (
            self._fixed_bsu_fused_supported,
            self._fixed_bsu_fused_fallback_reason,
        ) = self._fixed_bsu_fused_support()
        self.metadata = {
            "net_subgraph_invocation_timing": "during_timing_propagation_net_aat",
            "uses_current_propagated_driver_slew": True,
            "dynamic_provider_call_count": 0,
            "dynamic_provider_affected_net_call_count": 0,
            "provider_dispatch_ms": 0.0,
            "driver_timing_gather_ms": 0.0,
            "net_subgraph_forward_relaxed_ms": 0.0,
            "sink_scatter_ms": 0.0,
            "static_payload_cache_hit_count": 0,
            "net_subgraph_forward_backend": self.forward_backend,
            "net_subgraph_execution_device": (
                "cuda"
                if self.forward_backend == "cuda_explicit_autograd"
                else "cpu"
                if self.forward_backend.startswith("native")
                else "state_device"
            ),
            "affected_net_count": int(len(self.affected_net_ids)),
            "candidate_net_count": int(len(self._candidate_net_ids)),
            "buffer_device_lut_status": (
                "ok" if self.buffer_device_lut is not None else "unavailable"
            ),
            "buffer_device_lut_source": (
                str(getattr(self.buffer_device_lut, "source", ""))
                if self.buffer_device_lut is not None
                else ""
            ),
            "driver_cap_overlay_count": 0,
            "driver_cap_overlay_path": None,
            "active_view_cache_hit_count": 0,
            "active_view_build_count": 0,
            "active_view_selector_requested": False,
            "active_view_selector_used": "python_reference",
            "active_view_selector_fallback_reason": None,
            "native_active_view_selector_count": 0,
            "native_active_view_selector_ms": 0.0,
            "fixed_bsu_fused_forward_supported": self._fixed_bsu_fused_supported,
            "fixed_bsu_fused_forward_fallback_reason": (
                self._fixed_bsu_fused_fallback_reason
            ),
            "fixed_bsu_fused_forward_count": 0,
        }

    def has_dynamic_nets(self, net_ids, *, level_id=None, level_view_epoch=None):
        return bool(_as_net_id_set(net_ids) & self.affected_net_ids)

    def reset_runtime_metadata(self):
        self.call_count = 0
        self.affected_call_count = 0
        for key in (
            "dynamic_provider_call_count",
            "dynamic_provider_affected_net_call_count",
            "provider_dispatch_ms",
            "driver_timing_gather_ms",
            "net_subgraph_forward_relaxed_ms",
            "sink_scatter_ms",
            "static_payload_cache_hit_count",
            "active_view_cache_hit_count",
            "active_view_build_count",
            "native_active_view_selector_count",
            "native_active_view_selector_ms",
            "fixed_bsu_fused_forward_count",
            "driver_cap_overlay_count",
        ):
            self.metadata[key] = 0
        return self

    def update_buffer_state(self, buffer_state):
        self.buffer_state = buffer_state
        return self

    def _buffer_coordinate_callback(self, **kwargs):
        if self.buffer_device_lut is None:
            return None
        from dreamplace.ops.net_subgraph_timing.segment_transfer import (
            lookup_buffer_device_fixed_size,
            lookup_buffer_device_per_size,
        )

        fixed_bsu_index = getattr(self.buffer_state, "fixed_bsu_index", None)
        if fixed_bsu_index is None:
            values = lookup_buffer_device_per_size(
                self.buffer_device_lut,
                input_slew=kwargs["candidate_input_slew"],
                output_load=kwargs["candidate_output_cap"],
            )
            values["coordinate_source"] = "liberty_lut_candidate_slew_load"
        else:
            values = lookup_buffer_device_fixed_size(
                self.buffer_device_lut,
                size_index=fixed_bsu_index,
                input_slew=kwargs["candidate_input_slew"],
                output_load=kwargs["candidate_output_cap"],
            )
            values["coordinate_source"] = (
                "liberty_lut_candidate_slew_load_fixed_bsu"
            )
        return values

    def _fixed_bsu_fused_support(self):
        if self.forward_backend != "cuda_explicit_autograd":
            return False, "non_cuda_candidate_backend"
        fixed_bsu_index = getattr(self.buffer_state, "fixed_bsu_index", None)
        if fixed_bsu_index is None:
            return False, "non_discrete_candidate_state"
        if self.buffer_device_lut is None:
            return False, "missing_buffer_device_lut"
        per_size_input_cap = self.static_payload.get("per_size_input_cap")
        if per_size_input_cap is None or per_size_input_cap.ndim != 2:
            return False, "missing_per_size_input_cap"
        fixed_bsu_index = int(fixed_bsu_index)
        if not 0 <= fixed_bsu_index < int(per_size_input_cap.shape[1]):
            return False, "fixed_bsu_index_out_of_range"
        table = per_size_input_cap.detach().to(device="cpu")
        lut = self.buffer_device_lut.tensors(dtype=table.dtype, device="cpu")
        if fixed_bsu_index >= int(lut["input_cap_by_size"].numel()):
            return False, "fixed_bsu_lut_index_out_of_range"
        expected = lut["input_cap_by_size"][fixed_bsu_index]
        if not torch.allclose(
            table[:, fixed_bsu_index],
            expected.expand(int(table.shape[0])),
        ):
            return False, "probe_and_liberty_input_cap_mismatch"
        return True, None

    def _buffer_device_lut_tensors(self, *, dtype, device):
        key = (dtype, torch.device(device))
        cached = self._buffer_device_lut_cache.get(key)
        if cached is None:
            cached = self.buffer_device_lut.tensors(dtype=dtype, device=device)
            self._buffer_device_lut_cache[key] = cached
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        return cached

    def _active_key(self, active_net_ids):
        active = {int(net_id) for net_id in active_net_ids}
        return tuple(int(net_id) for net_id in self.net_ids if int(net_id) in active)

    def _native_active_view_supported(self):
        metadata = dict(self.static_payload.get("metadata", {}) or {})
        if metadata.get("topology_source") != "native_candidate_packer":
            return False, "non_native_candidate_payload"
        sink_filter = dict(metadata.get("sink_filter", {}) or {})
        if sink_filter.get("status") != "all_net_sinks":
            return False, "filtered_sink_view"
        required = (
            "net_ids",
            "net_topo_start",
            "net_edge_start",
            "net_sink_start",
            "edge_start",
            "candidate_net_id",
            "candidate_node_id",
        )
        missing = [name for name in required if name not in self.static_payload]
        if missing:
            return False, "missing_native_fields:" + ",".join(missing)
        return True, None

    def _active_view(self, active_net_ids):
        cache_key = self._active_key(active_net_ids)
        cached = self._active_view_cache.get(cache_key)
        if cached is not None:
            self.metadata["static_payload_cache_hit_count"] += 1
            self.metadata["active_view_cache_hit_count"] += 1
            return cached

        use_native, fallback_reason = self._native_active_view_supported()
        self.metadata["active_view_selector_requested"] = True
        if use_native:
            started_at = time.perf_counter()
            view = select_packed_candidate_inputs_native(
                self.static_payload,
                active_net_ids=cache_key,
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            view["cache_key"] = cache_key
            self._active_view_cache[cache_key] = view
            self.metadata["active_view_build_count"] += 1
            self.metadata["active_view_selector_used"] = "native"
            self.metadata["active_view_selector_fallback_reason"] = None
            self.metadata["native_active_view_selector_count"] += 1
            self.metadata["native_active_view_selector_ms"] += elapsed_ms
            return view

        topo = self.static_payload["net_flat_topo_sort"].detach().cpu().tolist()
        topo_start = self.static_payload["net_flat_topo_sort_start"].detach().cpu().tolist()
        pin_fa = self.static_payload["pin_fa"].detach().cpu().tolist()
        flat_pin_to_start = self.static_payload["flat_pin_to_start"].detach().cpu().tolist()
        flat_pin_to = self.static_payload["flat_pin_to"].detach().cpu().tolist()
        candidate_node_id = self.static_payload["candidate_node_id"].detach().cpu().tolist()
        sink_node_id = self.static_payload["sink_node_id"].detach().cpu().tolist()
        sink_net_index = self.static_payload["sink_net_index"].detach().cpu().tolist()

        old_to_local = {}
        local_to_old = []
        local_topo = []
        local_topo_start = [0]
        net_indices = []
        for net_id in cache_key:
            net_index = self._net_index_by_id.get(int(net_id))
            if net_index is None or net_index + 1 >= len(topo_start):
                continue
            net_indices.append(int(net_index))
            for old_node in topo[int(topo_start[net_index]): int(topo_start[net_index + 1])]:
                old_node = int(old_node)
                if old_node not in old_to_local:
                    old_to_local[old_node] = len(local_to_old)
                    local_to_old.append(old_node)
                local_topo.append(old_to_local[old_node])
            local_topo_start.append(len(local_topo))

        local_pin_fa = [-1 for _ in local_to_old]
        local_flat_pin_to = []
        local_flat_pin_to_start = [0]
        for old_node in local_to_old:
            old_node = int(old_node)
            local_node = old_to_local[old_node]
            old_parent = int(pin_fa[old_node]) if 0 <= old_node < len(pin_fa) else -1
            local_pin_fa[local_node] = (
                old_to_local[old_parent] if old_parent in old_to_local else -1
            )
            begin = int(flat_pin_to_start[old_node]) if 0 <= old_node < len(flat_pin_to_start) else 0
            end = int(flat_pin_to_start[old_node + 1]) if old_node + 1 < len(flat_pin_to_start) else begin
            for old_child in flat_pin_to[begin:end]:
                old_child = int(old_child)
                if old_child in old_to_local:
                    local_flat_pin_to.append(old_to_local[old_child])
            local_flat_pin_to_start.append(len(local_flat_pin_to))

        active_set = set(cache_key)
        candidate_net_ids = self.buffer_state.candidate_net_id.detach().cpu().tolist()
        candidate_indices = [
            int(index)
            for index, net_id in enumerate(candidate_net_ids)
            if int(net_id) in active_set and int(candidate_node_id[index]) in old_to_local
        ]
        local_candidate_node_id = [
            old_to_local[int(candidate_node_id[index])]
            for index in candidate_indices
        ]

        sink_indices = [
            int(index)
            for index, net_id in enumerate(sink_net_index)
            if int(net_id) in active_set and int(sink_node_id[index]) in old_to_local
        ]
        local_sink_node_id = [
            old_to_local[int(sink_node_id[index])]
            for index in sink_indices
        ]
        local_net_index_by_id = {
            int(net_id): int(index) for index, net_id in enumerate(cache_key)
        }
        local_sink_net_index = [
            local_net_index_by_id[int(sink_net_index[index])]
            for index in sink_indices
        ]

        node_index = torch.as_tensor(local_to_old, dtype=torch.long)
        view = {
            "cache_key": cache_key,
            "net_indices_cpu": torch.as_tensor(net_indices, dtype=torch.long),
            "candidate_indices_cpu": torch.as_tensor(candidate_indices, dtype=torch.long),
            "sink_indices_cpu": torch.as_tensor(sink_indices, dtype=torch.long),
            "static": {
                "net_flat_topo_sort": torch.as_tensor(
                    local_topo,
                    dtype=self.static_payload["net_flat_topo_sort"].dtype,
                ),
                "net_flat_topo_sort_start": torch.as_tensor(
                    local_topo_start,
                    dtype=self.static_payload["net_flat_topo_sort_start"].dtype,
                ),
                "pin_fa": torch.as_tensor(
                    local_pin_fa,
                    dtype=self.static_payload["pin_fa"].dtype,
                ),
                "flat_pin_to_start": torch.as_tensor(
                    local_flat_pin_to_start,
                    dtype=self.static_payload["flat_pin_to_start"].dtype,
                ),
                "flat_pin_to": torch.as_tensor(
                    local_flat_pin_to,
                    dtype=self.static_payload["flat_pin_to"].dtype,
                ),
                "edge_resistance": _index_select_cpu(
                    self.static_payload["edge_resistance"],
                    node_index,
                ),
                "node_capacitance": _index_select_cpu(
                    self.static_payload["node_capacitance"],
                    node_index,
                ),
                "edge_capacitance": _index_select_cpu(
                    self.static_payload["edge_capacitance"],
                    node_index,
                ),
                "sink_node_id": torch.as_tensor(
                    local_sink_node_id,
                    dtype=self.static_payload["sink_node_id"].dtype,
                ),
                "sink_net_index": torch.as_tensor(
                    local_sink_net_index,
                    dtype=self.static_payload["sink_net_index"].dtype,
                ),
            },
            "candidate_node_id_cpu": torch.as_tensor(
                local_candidate_node_id,
                dtype=self.static_payload["candidate_node_id"].dtype,
            ),
            "sink_pin_id_cpu": _index_select_cpu(
                self.static_payload["sink_pin_id"],
                sink_indices,
            ),
            "sink_net_id_cpu": _index_select_cpu(
                self.static_payload["sink_net_index"],
                sink_indices,
            ),
        }
        self._active_view_cache[cache_key] = view
        self.metadata["active_view_build_count"] += 1
        self.metadata["active_view_selector_used"] = "python_reference"
        self.metadata["active_view_selector_fallback_reason"] = fallback_reason
        return view

    def _driver_values(self, pin_values, active_net_ids=None):
        if not self.net_ids:
            return torch.empty(0, dtype=pin_values.dtype, device=pin_values.device)
        active_view = self._active_view(active_net_ids) if active_net_ids is not None else None
        if active_view is None:
            net_indices_cpu = None
            cache_key = ("all", pin_values.device)
            driver_index_cpu = self._driver_pin_index_cpu
        else:
            net_indices_cpu = active_view["net_indices_cpu"]
            cache_key = (active_view["cache_key"], pin_values.device)
            driver_index_cpu = _index_select_cpu(self._driver_pin_index_cpu, net_indices_cpu)
        driver_index = self._driver_pin_index_cache.get(cache_key)
        if driver_index is None:
            driver_index = driver_index_cpu.to(device=pin_values.device)
            self._driver_pin_index_cache[cache_key] = driver_index
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        values = pin_values.index_select(0, driver_index).to(dtype=pin_values.dtype)
        if self._driver_fallback_indices and active_view is None:
            fallback_key = (pin_values.device, pin_values.dtype)
            fallback = self._driver_arrival_fallback_cache.get(fallback_key)
            if fallback is None:
                fallback = self.static_payload["driver_arrival"].to(
                    device=pin_values.device,
                    dtype=pin_values.dtype,
                )
                self._driver_arrival_fallback_cache[fallback_key] = fallback
            else:
                self.metadata["static_payload_cache_hit_count"] += 1
            for net_index in self._driver_fallback_indices:
                values[net_index] = fallback[net_index]
        elif self._driver_fallback_indices and active_view is not None:
            fallback_key = (pin_values.device, pin_values.dtype)
            fallback = self._driver_arrival_fallback_cache.get(fallback_key)
            if fallback is None:
                fallback = self.static_payload["driver_arrival"].to(
                    device=pin_values.device,
                    dtype=pin_values.dtype,
                )
                self._driver_arrival_fallback_cache[fallback_key] = fallback
            else:
                self.metadata["static_payload_cache_hit_count"] += 1
            for local_index, net_index in enumerate(net_indices_cpu.tolist()):
                if int(net_index) in self._driver_fallback_indices:
                    values[local_index] = fallback[int(net_index)]
        return values

    def _forward_device(self, driver_arrival):
        return (
            torch.device("cpu")
            if self.forward_backend.startswith("native")
            else driver_arrival.device
        )

    def _forward_kwargs(self, *, driver_arrival, driver_slew):
        forward_device = self._forward_device(driver_arrival)
        cache_key = (forward_device, driver_arrival.dtype)
        kwargs = self._forward_kwargs_cache.get(cache_key)
        if kwargs is None:
            kwargs = {}
            for key in _NET_SUBGRAPH_FORWARD_KEYS:
                if key not in self.static_payload:
                    continue
                if key in {
                    "edge_resistance",
                    "node_capacitance",
                    "edge_capacitance",
                }:
                    dtype = driver_arrival.dtype
                elif self.forward_backend == "cuda_explicit_autograd":
                    dtype = torch.long
                else:
                    dtype = None
                kwargs[key] = _payload_value_on_device(
                    self.static_payload[key],
                    device=forward_device,
                    dtype=dtype,
                )
            self._forward_kwargs_cache[cache_key] = kwargs
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        kwargs["driver_arrival"] = driver_arrival.to(device=forward_device)
        kwargs["driver_slew"] = driver_slew.to(device=forward_device)
        return kwargs

    def _active_forward_kwargs(self, *, active_view, driver_arrival, driver_slew):
        forward_device = self._forward_device(driver_arrival)
        cache_key = (active_view["cache_key"], forward_device, driver_arrival.dtype)
        kwargs = self._active_forward_kwargs_cache.get(cache_key)
        if kwargs is None:
            kwargs = {}
            for key, value in active_view["static"].items():
                if key in {
                    "edge_resistance",
                    "node_capacitance",
                    "edge_capacitance",
                }:
                    dtype = driver_arrival.dtype
                elif self.forward_backend == "cuda_explicit_autograd":
                    dtype = torch.long
                else:
                    dtype = None
                kwargs[key] = _payload_value_on_device(
                    value,
                    device=forward_device,
                    dtype=dtype,
                )
            self._active_forward_kwargs_cache[cache_key] = kwargs
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        kwargs["driver_arrival"] = driver_arrival.to(device=forward_device)
        kwargs["driver_slew"] = driver_slew.to(device=forward_device)
        return kwargs

    def _per_size_tensors(self, *, driver_arrival, active_view=None):
        forward_device = self._forward_device(driver_arrival)
        if active_view is not None:
            candidate_indices = active_view["candidate_indices_cpu"]
            cache_key = (
                active_view["cache_key"],
                forward_device,
                driver_arrival.dtype,
            )
            tensors = self._per_size_cache.get(cache_key)
            if tensors is None:
                table_index = candidate_indices.to(
                    device=self.static_payload["per_size_input_cap"].device
                )
                tensors = {
                    "per_size_input_cap": self.static_payload["per_size_input_cap"]
                    .index_select(0, table_index)
                    .to(device=forward_device, dtype=driver_arrival.dtype),
                    "per_size_delay": self.static_payload["per_size_delay"]
                    .index_select(0, table_index.to(device=self.static_payload["per_size_delay"].device))
                    .to(device=forward_device, dtype=driver_arrival.dtype),
                    "per_size_output_slew": self.static_payload["per_size_output_slew"]
                    .index_select(
                        0,
                        table_index.to(device=self.static_payload["per_size_output_slew"].device),
                    )
                    .to(device=forward_device, dtype=driver_arrival.dtype),
                }
                self._per_size_cache[cache_key] = tensors
            else:
                self.metadata["static_payload_cache_hit_count"] += 1
            return tensors
        cache_key = (forward_device, driver_arrival.dtype)
        tensors = self._per_size_cache.get(cache_key)
        if tensors is None:
            tensors = {
                "per_size_input_cap": self.static_payload["per_size_input_cap"].to(
                    device=forward_device,
                    dtype=driver_arrival.dtype,
                ),
                "per_size_delay": self.static_payload["per_size_delay"].to(
                    device=forward_device,
                    dtype=driver_arrival.dtype,
                ),
                "per_size_output_slew": self.static_payload["per_size_output_slew"].to(
                    device=forward_device,
                    dtype=driver_arrival.dtype,
                ),
            }
            self._per_size_cache[cache_key] = tensors
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        return tensors

    def _candidate_node_id(self, active_view=None):
        if active_view is not None:
            device = (
                torch.device("cpu")
                if self.forward_backend.startswith("native")
                else self.buffer_state.candidate_node_id.device
            )
            return active_view["candidate_node_id_cpu"].to(
                device=device,
                dtype=self.buffer_state.candidate_node_id.dtype,
            )
        if self._candidate_node_id_cache is None:
            device = (
                torch.device("cpu")
                if self.forward_backend.startswith("native")
                else self.buffer_state.candidate_node_id.device
            )
            self._candidate_node_id_cache = self.static_payload["candidate_node_id"].to(
                device=device,
                dtype=self.buffer_state.candidate_node_id.dtype,
            )
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        return self._candidate_node_id_cache

    def _run_lane(
        self,
        *,
        driver_arrival,
        driver_slew,
        active_net_ids=None,
        use_buffer_device=True,
    ):
        state_device = (
            torch.device("cpu")
            if self.forward_backend.startswith("native")
            else self.buffer_state.candidate_node_id.device
        )
        active_view = (
            self._active_view(active_net_ids)
            if active_net_ids is not None
            else None
        )
        if active_view is None:
            candidate_indices = torch.arange(
                int(self.buffer_state.candidate_ids.numel()),
                dtype=torch.long,
            )
            candidate_node_id = self._candidate_node_id()
        else:
            candidate_indices = active_view["candidate_indices_cpu"]
            candidate_node_id = self._candidate_node_id(active_view)
        state = self.buffer_state.slice_candidates(
            candidate_indices,
            candidate_node_id=candidate_node_id,
            device=state_device,
        )
        per_size = self._per_size_tensors(
            driver_arrival=driver_arrival,
            active_view=active_view,
        )
        forward_kwargs = (
            self._active_forward_kwargs(
                active_view=active_view,
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
            )
            if active_view is not None
            else self._forward_kwargs(
                driver_arrival=driver_arrival,
                driver_slew=driver_slew,
            )
        )
        result = net_subgraph_forward_relaxed(
            buffer_state=state,
            per_size_input_cap=per_size["per_size_input_cap"],
            per_size_delay=per_size["per_size_delay"],
            per_size_output_slew=per_size["per_size_output_slew"],
            coordinate_source=self.coordinate_source,
            forward_backend=self.forward_backend,
            fixed_bsu_device_lut=(
                self._buffer_device_lut_tensors(
                    dtype=driver_arrival.dtype,
                    device=driver_arrival.device,
                )
                if self._fixed_bsu_fused_supported and use_buffer_device
                else None
            ),
            fixed_bsu_index=(
                int(self.buffer_state.fixed_bsu_index)
                if self._fixed_bsu_fused_supported and use_buffer_device
                else None
            ),
            buffer_coordinate_callback=(
                self._buffer_coordinate_callback
                if use_buffer_device and self.buffer_device_lut is not None
                else None
            ),
            **forward_kwargs,
        )
        if self._fixed_bsu_fused_supported and use_buffer_device:
            self.metadata["fixed_bsu_fused_forward_count"] += 1
        if active_view is not None:
            result["sink_net_index"] = active_view["sink_net_id_cpu"].to(
                device=result["sink_arrival"].device,
                dtype=torch.long,
            )
        return result

    def _run_cap_only(self, active_net_ids):
        active_view = self._active_view(active_net_ids)
        active_net_count = len(active_view["cache_key"])
        candidate_bu = self.buffer_state.candidate_bu()
        forward_device = (
            torch.device("cpu")
            if self.forward_backend.startswith("native")
            else candidate_bu.device
        )
        zeros = torch.zeros(
            active_net_count,
            dtype=candidate_bu.dtype,
            device=forward_device,
        )
        result = self._run_lane(
            driver_arrival=zeros,
            driver_slew=zeros,
            active_net_ids=active_net_ids,
            use_buffer_device=False,
        )
        topo = active_view["static"]["net_flat_topo_sort"].to(dtype=torch.long)
        topo_start = active_view["static"]["net_flat_topo_sort_start"].to(
            dtype=torch.long
        )
        if active_net_count == 0:
            root_index = torch.empty(0, dtype=torch.long)
        else:
            root_index = topo.index_select(0, topo_start[:-1])
        driver_pin_id = torch.as_tensor(
            [
                int(self.driver_pin_by_net[net_id])
                for net_id in active_view["cache_key"]
            ],
            dtype=torch.long,
        )
        return {
            "driver_pin_id": driver_pin_id,
            "driver_net_cap": result["lin"].index_select(
                0,
                root_index.to(device=result["lin"].device),
            ),
        }

    @staticmethod
    def _scatter_driver_cap(pin_net_cap, cap_result):
        driver_pin_id = cap_result["driver_pin_id"].to(
            device=pin_net_cap.device,
            dtype=torch.long,
        )
        valid = driver_pin_id < int(pin_net_cap.numel())
        if not bool(torch.any(valid).detach().cpu().item()):
            return pin_net_cap
        pin_net_cap = pin_net_cap.clone()
        pin_net_cap[driver_pin_id[valid]] = cap_result["driver_net_cap"].to(
            device=pin_net_cap.device,
            dtype=pin_net_cap.dtype,
        )[valid]
        return pin_net_cap

    def apply_dynamic_net_cap_overlay(
        self,
        *,
        pin_net_cap_rise,
        pin_net_cap_fall,
    ):
        if not self.affected_net_ids:
            return pin_net_cap_rise, pin_net_cap_fall
        cap_result = self._run_cap_only(self.affected_net_ids)
        pin_net_cap_rise = self._scatter_driver_cap(pin_net_cap_rise, cap_result)
        pin_net_cap_fall = self._scatter_driver_cap(pin_net_cap_fall, cap_result)
        self.metadata["driver_cap_overlay_count"] = int(
            self.metadata.get("driver_cap_overlay_count", 0)
        ) + 1
        self.metadata["driver_cap_overlay_path"] = "candidate_native_load_only"
        return pin_net_cap_rise, pin_net_cap_fall

    def _sink_pin_id(self, *, pin_aat, active_net_ids=None):
        if active_net_ids is not None:
            view = self._active_view(active_net_ids)
            return view["sink_pin_id_cpu"].to(
                device=pin_aat.device,
                dtype=torch.long,
            )
        cache_key = (pin_aat.device, torch.long)
        if self._sink_pin_id_cache is None:
            self._sink_pin_id_cache = {}
        cached = self._sink_pin_id_cache.get(cache_key)
        if cached is None:
            cached = self.static_payload["sink_pin_id"].to(
                device=pin_aat.device,
                dtype=torch.long,
            )
            self._sink_pin_id_cache[cache_key] = cached
        else:
            self.metadata["static_payload_cache_hit_count"] += 1
        return cached

    def _scatter_lane(self, pin_aat, pin_tran, result, active_net_ids):
        sink_pin_id = self._sink_pin_id(pin_aat=pin_aat, active_net_ids=active_net_ids)
        if int(sink_pin_id.numel()) == int(result["sink_arrival"].numel()):
            active_pins = sink_pin_id.to(device=pin_aat.device, dtype=torch.long)
            pin_aat = pin_aat.clone()
            pin_tran = pin_tran.clone()
            pin_aat[active_pins] = result["sink_arrival"].to(
                device=pin_aat.device,
                dtype=pin_aat.dtype,
            )
            pin_tran[active_pins] = result["sink_slew"].to(
                device=pin_tran.device,
                dtype=pin_tran.dtype,
            )
            return pin_aat, pin_tran
        result_device = result["sink_arrival"].device
        sink_net_index = result["sink_net_index"].to(device=result_device)
        active_ids = torch.as_tensor(
            sorted(int(net_id) for net_id in active_net_ids),
            dtype=sink_net_index.dtype,
            device=sink_net_index.device,
        )
        active_mask = torch.isin(sink_net_index, active_ids)
        if not torch.any(active_mask):
            return pin_aat, pin_tran
        active_pins = sink_pin_id.to(device=result_device)[active_mask].to(
            device=pin_aat.device,
            dtype=torch.long,
        )
        pin_aat = pin_aat.clone()
        pin_tran = pin_tran.clone()
        pin_aat[active_pins] = result["sink_arrival"][active_mask].to(
            device=pin_aat.device,
            dtype=pin_aat.dtype,
        )
        pin_tran[active_pins] = result["sink_slew"][active_mask].to(
            device=pin_tran.device,
            dtype=pin_tran.dtype,
        )
        return pin_aat, pin_tran

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
        provider_started_at = time.perf_counter()
        self.call_count += 1
        active_net_ids = _as_net_id_set(net_ids) & self.affected_net_ids
        self.metadata["dynamic_provider_call_count"] = int(self.call_count)
        if not active_net_ids:
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
            self.metadata["provider_dispatch_ms"] += (
                time.perf_counter() - provider_started_at
            ) * 1000.0
            return result
        self.affected_call_count += 1
        self.metadata["dynamic_provider_affected_net_call_count"] = int(
            self.affected_call_count
        )
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
        gather_started_at = time.perf_counter()
        rise_driver_arrival = self._driver_values(pin_rAAT, active_net_ids)
        rise_driver_slew = self._driver_values(pin_rtran, active_net_ids)
        fall_driver_arrival = self._driver_values(pin_fAAT, active_net_ids)
        fall_driver_slew = self._driver_values(pin_ftran, active_net_ids)
        self.metadata["driver_timing_gather_ms"] += (
            time.perf_counter() - gather_started_at
        ) * 1000.0

        forward_started_at = time.perf_counter()
        rise_result = self._run_lane(
            driver_arrival=rise_driver_arrival,
            driver_slew=rise_driver_slew,
            active_net_ids=active_net_ids,
        )
        fall_result = self._run_lane(
            driver_arrival=fall_driver_arrival,
            driver_slew=fall_driver_slew,
            active_net_ids=active_net_ids,
        )
        self.metadata["net_subgraph_forward_relaxed_ms"] += (
            time.perf_counter() - forward_started_at
        ) * 1000.0

        scatter_started_at = time.perf_counter()
        pin_rAAT, pin_rtran = self._scatter_lane(
            pin_rAAT,
            pin_rtran,
            rise_result,
            active_net_ids,
        )
        pin_fAAT, pin_ftran = self._scatter_lane(
            pin_fAAT,
            pin_ftran,
            fall_result,
            active_net_ids,
        )
        self.metadata["sink_scatter_ms"] += (
            time.perf_counter() - scatter_started_at
        ) * 1000.0
        self.metadata["provider_dispatch_ms"] += (
            time.perf_counter() - provider_started_at
        ) * 1000.0
        return pin_rAAT, pin_fAAT, pin_rtran, pin_ftran
