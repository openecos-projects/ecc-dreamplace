"""LCB -> FF clock arrival using datapath operators.

Root clock nets stay ideal (AT=0). Leaf clock nets that drive FF.CK are
evaluated with the same depth-one star Elmore formula as driver-sink-star
datapath RC, plus combinational Liberty LUTs for the LCB cell. Clock nets
are never added to data ``net_flat_arcs`` / star covering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch


class LcbFfTimingError(RuntimeError):
    """Fail-closed error for LCB -> FF clock arrival."""


def _smooth_extremum_enabled(params_or_op) -> bool:
    raw = 1 if params_or_op is None else getattr(params_or_op, "timing_smooth_extremum", 1)
    try:
        return bool(int(raw))
    except (TypeError, ValueError) as exc:
        raise LcbFfTimingError("timing_smooth_extremum must be 0 or 1") from exc


def _pairwise_lse_max(lhs: torch.Tensor, rhs: torch.Tensor, tau: float) -> torch.Tensor:
    tau_value = float(tau)
    if tau_value <= 0.0:
        raise LcbFfTimingError("timing_aggregation_tau_ps must be positive")
    return tau_value * torch.logsumexp(
        torch.stack([lhs, rhs], dim=0) / tau_value, dim=0
    )


def lcb_ff_propagation_enabled(params) -> bool:
    raw = 1 if params is None else getattr(params, "timing_lcb_ff_propagation", 1)
    try:
        return bool(int(raw))
    except (TypeError, ValueError) as exc:
        raise LcbFfTimingError(
            "timing_lcb_ff_propagation must be 0 or 1"
        ) from exc


def _as_1d_long(value, name: str, *, allow_negative: bool = False) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    tensor = value.reshape(-1).long().contiguous()
    if not allow_negative and tensor.numel() and bool((tensor < 0).any().item()):
        raise LcbFfTimingError("%s contains a negative index" % name)
    return tensor


def _decode_name(value) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="strict")
    return str(value)


def pin_is_output(value) -> bool:
    token = _decode_name(value).strip().upper()
    return token in {"OUTPUT", "O", "OUT", "OUTPUT_TRISTATE"}


def pin_is_input(value) -> bool:
    token = _decode_name(value).strip().upper()
    return token in {"INPUT", "I", "IN"}


def _ignore_net_degree(params) -> int:
    raw = 100 if params is None else getattr(params, "ignore_net_degree", 100)
    try:
        cutoff = int(raw)
    except (TypeError, ValueError) as exc:
        raise LcbFfTimingError("ignore_net_degree must be a positive integer") from exc
    if cutoff <= 0:
        raise LcbFfTimingError("ignore_net_degree must be a positive integer")
    return cutoff


def _physical_node_count(placedb) -> int:
    num_movable = getattr(placedb, "num_movable_nodes", None)
    num_terminals = getattr(placedb, "num_terminals", None)
    num_terminal_nis = getattr(placedb, "num_terminal_NIs", None)
    if (
        num_movable is not None
        and num_terminals is not None
        and num_terminal_nis is not None
        and int(num_terminal_nis) > 0
    ):
        return int(num_movable) + int(num_terminals)
    raw = getattr(placedb, "num_physical_nodes", None)
    if raw is not None:
        return int(raw)
    num_nodes = int(getattr(placedb, "num_nodes"))
    fillers = int(getattr(placedb, "num_filler_nodes", 0) or 0)
    return num_nodes - fillers


def _net_pin_ids(flat_net2pin, netpin_start, net_id: int) -> torch.Tensor:
    begin = int(netpin_start[net_id].item())
    end = int(netpin_start[net_id + 1].item())
    if end <= begin:
        raise LcbFfTimingError("clock net %d has no pins" % net_id)
    pins = flat_net2pin[begin:end]
    if int(torch.unique(pins).numel()) != int(pins.numel()):
        raise LcbFfTimingError("clock net %d repeats a physical pin" % net_id)
    return pins


def _net_driver_pin(pin_ids, pin_direct, pin2node, physical_node_count: int, placedb=None):
    outputs = []
    for pin_id in pin_ids.tolist():
        if pin_is_output(pin_direct[int(pin_id)]):
            outputs.append(int(pin_id))
    if len(outputs) != 1:
        return None, True
    driver = outputs[0]
    node_id = int(pin2node[driver].item())
    is_io = node_id < 0 or node_id >= physical_node_count
    if not is_io and placedb is not None:
        inst_main_id = getattr(placedb, "inst_main_id", None)
        if inst_main_id is not None and (
            node_id >= len(inst_main_id) or int(inst_main_id[node_id]) < 0
        ):
            is_io = True
    return driver, is_io


@dataclass
class LcbFfPlan:
    clock_pins_native: torch.Tensor
    sink_is_root: torch.Tensor
    leaf_driver_pin_ids: torch.Tensor
    leaf_sink_pin_ids: torch.Tensor
    leaf_sink_rows: torch.Tensor
    root_net_ids: tuple = ()
    leaf_net_ids: tuple = ()
    ignore_net_degree: int = 100
    # Clock-star plans use compact RC pin IDs while retaining native IDs for
    # Liberty lookup and sidecar provenance.  Legacy callers leave these
    # fields unset and therefore use the primary IDs for both purposes.
    leaf_driver_native_pin_ids: Optional[torch.Tensor] = None
    leaf_sink_native_pin_ids: Optional[torch.Tensor] = None
    clock_compact_to_native_pin_ids: Optional[torch.Tensor] = None
    clock_row_to_compact_pin_ids: Optional[torch.Tensor] = None
    clock_net_names: tuple = ()

    def leaf_pair_count(self) -> int:
        return int(self.leaf_sink_pin_ids.numel())


@dataclass(frozen=True)
class LcbFfClockStarDomain:
    """Compact clock-star namespace and its native sidecar crosswalk."""

    plan: LcbFfPlan
    compact_to_native_pin_ids: torch.Tensor
    clock_row_to_compact_pin_ids: torch.Tensor
    clock_net_names: tuple
    native_clock_net_ids: tuple
    timing_net_flat_arcs: torch.Tensor
    timing_net_flat_arcs_start: torch.Tensor
    net2driver_pin_map: torch.Tensor


def _name_at(values, index: int) -> str:
    value = values[index]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="strict")
    return str(value)


def build_lcb_ff_clock_star_domain(
    *,
    clock_pins_native,
    pin2net_map,
    pin2node_map,
    pin_direct,
    flat_net2pin_map,
    flat_net2pin_start_map,
    placedb,
    params=None,
) -> LcbFfClockStarDomain:
    """Build a compact, clock-only Star namespace.

    ``build_lcb_ff_plan`` remains the authoritative classifier for root versus
    leaf clock nets.  This function then remaps only leaf drivers and FF.CK
    sinks into a dense local namespace suitable for ``StarPinPair``.  Native
    pin IDs are retained separately for Liberty lookup and geometry gathers.
    """
    native_plan = build_lcb_ff_plan(
        clock_pins_native=clock_pins_native,
        pin2net_map=pin2net_map,
        pin2node_map=pin2node_map,
        pin_direct=pin_direct,
        flat_net2pin_map=flat_net2pin_map,
        flat_net2pin_start_map=flat_net2pin_start_map,
        placedb=placedb,
        params=params,
    )
    native_clock_pins = native_plan.clock_pins_native.reshape(-1).long().cpu()
    row_by_native = {
        int(pin_id): row for row, pin_id in enumerate(native_clock_pins.tolist())
    }
    flat = torch.as_tensor(flat_net2pin_map).reshape(-1).long().cpu()
    starts = torch.as_tensor(flat_net2pin_start_map).reshape(-1).long().cpu()
    pin2node = torch.as_tensor(pin2node_map).reshape(-1).long().cpu()
    clock_set = set(row_by_native)

    compact_by_native = {}
    compact_to_native = []
    compact_drivers = []
    compact_sinks = []
    native_drivers = []
    native_sinks = []
    sink_rows = []
    net_names = []
    native_net_ids = []
    net_driver_compact = []
    net_arcs = []
    net_starts = [0]
    raw_net_names = getattr(placedb, "net_names", ())

    def compact_id(native_pin):
        native_pin = int(native_pin)
        result = compact_by_native.get(native_pin)
        if result is None:
            result = len(compact_to_native)
            compact_by_native[native_pin] = result
            compact_to_native.append(native_pin)
        return result

    for native_net_id in sorted(native_plan.leaf_net_ids):
        native_net_id = int(native_net_id)
        if native_net_id < 0 or native_net_id + 1 >= int(starts.numel()):
            raise LcbFfTimingError("clock leaf net ID is outside net intervals")
        pin_ids = _net_pin_ids(flat, starts, native_net_id).tolist()
        driver, is_io = _net_driver_pin(
            torch.as_tensor(pin_ids, dtype=torch.long),
            pin_direct,
            pin2node,
            _physical_node_count(placedb),
            placedb=placedb,
        )
        if driver is None or is_io:
            raise LcbFfTimingError(
                "clock leaf net %d has no movable LCB driver" % native_net_id
            )
        sink_native = [pin_id for pin_id in pin_ids if int(pin_id) in clock_set]
        sink_native = [pin_id for pin_id in sink_native if int(pin_id) != int(driver)]
        if not sink_native:
            raise LcbFfTimingError(
                "clock leaf net %d has no FF.CK sink" % native_net_id
            )
        driver_compact = compact_id(driver)
        net_driver_compact.append(driver_compact)
        for sink in sink_native:
            sink_compact = compact_id(sink)
            compact_drivers.append(driver_compact)
            compact_sinks.append(sink_compact)
            native_drivers.append(int(driver))
            native_sinks.append(int(sink))
            sink_rows.append(row_by_native[int(sink)])
            net_arcs.append((driver_compact, sink_compact))
        net_starts.append(len(net_arcs))
        native_net_ids.append(native_net_id)
        net_names.append(
            _name_at(raw_net_names, native_net_id)
            if native_net_id < len(raw_net_names)
            else "clock_net_%d" % native_net_id
        )

    compact_to_native_tensor = torch.as_tensor(
        compact_to_native, dtype=torch.long
    )
    row_to_compact = torch.full(
        (native_clock_pins.numel(),), -1, dtype=torch.long
    )
    for native_pin, compact_pin in compact_by_native.items():
        row = row_by_native.get(native_pin)
        if row is not None:
            row_to_compact[row] = int(compact_pin)
    plan = LcbFfPlan(
        clock_pins_native=native_clock_pins,
        sink_is_root=native_plan.sink_is_root.cpu(),
        leaf_driver_pin_ids=torch.as_tensor(compact_drivers, dtype=torch.long),
        leaf_sink_pin_ids=torch.as_tensor(compact_sinks, dtype=torch.long),
        leaf_sink_rows=torch.as_tensor(sink_rows, dtype=torch.long),
        root_net_ids=tuple(
            index
            for index, is_root in enumerate(native_plan.sink_is_root.tolist())
            if bool(is_root)
        ),
        leaf_net_ids=tuple(range(len(net_names))),
        ignore_net_degree=native_plan.ignore_net_degree,
        leaf_driver_native_pin_ids=torch.as_tensor(
            native_drivers, dtype=torch.long
        ),
        leaf_sink_native_pin_ids=torch.as_tensor(native_sinks, dtype=torch.long),
        clock_compact_to_native_pin_ids=compact_to_native_tensor,
        clock_row_to_compact_pin_ids=row_to_compact,
        clock_net_names=tuple(net_names),
    )
    # ``StarPinPair`` needs at least one compact pin.  An all-root clock tree
    # is represented by an empty domain and skipped by the caller.
    return LcbFfClockStarDomain(
        plan=plan,
        compact_to_native_pin_ids=compact_to_native_tensor,
        clock_row_to_compact_pin_ids=row_to_compact,
        clock_net_names=tuple(net_names),
        native_clock_net_ids=tuple(native_net_ids),
        timing_net_flat_arcs=torch.as_tensor(net_arcs, dtype=torch.long).reshape(-1, 2),
        timing_net_flat_arcs_start=torch.as_tensor(net_starts, dtype=torch.long),
        net2driver_pin_map=torch.as_tensor(net_driver_compact, dtype=torch.long),
    )


def build_lcb_ff_plan(
    *,
    clock_pins_native,
    pin2net_map,
    pin2node_map,
    pin_direct,
    flat_net2pin_map,
    flat_net2pin_start_map,
    placedb,
    params=None,
) -> LcbFfPlan:
    """Discover root-ideal vs LCB->FF leaf nets from placement connectivity."""
    ck_pins = _as_1d_long(clock_pins_native, "clock_pins_native")
    pin2net = _as_1d_long(pin2net_map, "pin2net_map", allow_negative=True)
    pin2node = _as_1d_long(pin2node_map, "pin2node_map", allow_negative=True)
    flat = _as_1d_long(flat_net2pin_map, "flat_net2pin_map")
    starts = _as_1d_long(flat_net2pin_start_map, "flat_net2pin_start_map")
    if starts.numel() == 0 or int(starts[0].item()) != 0:
        raise LcbFfTimingError("flat_net2pin_start_map must start at 0")
    if ck_pins.numel() == 0:
        empty = ck_pins.new_empty((0,), dtype=torch.long)
        return LcbFfPlan(
            clock_pins_native=ck_pins,
            sink_is_root=ck_pins.new_zeros((0,), dtype=torch.bool),
            leaf_driver_pin_ids=empty,
            leaf_sink_pin_ids=empty,
            leaf_sink_rows=empty,
            ignore_net_degree=_ignore_net_degree(params),
        )
    if bool((ck_pins >= int(pin2net.numel())).any().item()):
        raise LcbFfTimingError("clock_pins_native is outside pin2net_map")
    pin_domain = int(pin_direct.shape[0]) if hasattr(pin_direct, "shape") else len(pin_direct)
    if int(pin2node.numel()) != pin_domain:
        raise LcbFfTimingError(
            "pin2node_map and pin_direct must describe the same native pin domain"
        )
    if flat.numel() and bool((flat >= pin_domain).any().item()):
        raise LcbFfTimingError(
            "flat_net2pin_map contains a native pin outside the pin domain "
            "(max=%d size=%d)" % (int(flat.max().item()), pin_domain)
        )
    cutoff = _ignore_net_degree(params)
    physical_nodes = _physical_node_count(placedb)
    sink_is_root = torch.zeros(ck_pins.numel(), dtype=torch.bool)
    leaf_drivers = []
    leaf_sinks = []
    leaf_rows = []
    root_nets = set()
    leaf_nets = set()
    # A clock net is classified once, not once per FF.CK pin on it (SB1: 144K pins, 7K leaf nets).  The first row
    # that reaches a net runs its checks, so errors are raised at the same row as before.
    net_class = {}
    for row, ck_pin in enumerate(ck_pins.tolist()):
        net_id = int(pin2net[int(ck_pin)].item())
        if net_id < 0 or net_id + 1 >= int(starts.numel()):
            raise LcbFfTimingError("clock pin %d maps to an invalid net" % ck_pin)
        cached = net_class.get(net_id)
        if cached is None:
            pin_ids = _net_pin_ids(flat, starts, net_id)
            cached = net_class[net_id] = (
                int(pin_ids.numel()),
                *_net_driver_pin(
                    pin_ids, pin_direct, pin2node, physical_nodes, placedb=placedb
                ),
            )
        degree, driver, is_io = cached
        is_root = degree > cutoff or bool(is_io) or driver is None
        if is_root:
            sink_is_root[row] = True
            root_nets.add(net_id)
            continue
        leaf_nets.add(net_id)
        leaf_drivers.append(int(driver))
        leaf_sinks.append(int(ck_pin))
        leaf_rows.append(int(row))
    return LcbFfPlan(
        clock_pins_native=ck_pins,
        sink_is_root=sink_is_root,
        leaf_driver_pin_ids=torch.as_tensor(leaf_drivers, dtype=torch.long),
        leaf_sink_pin_ids=torch.as_tensor(leaf_sinks, dtype=torch.long),
        leaf_sink_rows=torch.as_tensor(leaf_rows, dtype=torch.long),
        root_net_ids=tuple(sorted(root_nets)),
        leaf_net_ids=tuple(sorted(leaf_nets)),
        ignore_net_degree=cutoff,
    )


def split_pin_pos(pin_pos: torch.Tensor, num_pins: int):
    if not torch.is_tensor(pin_pos):
        raise TypeError("pin_pos must be a tensor")
    if pin_pos.dim() == 1:
        if int(pin_pos.numel()) != 2 * num_pins:
            raise LcbFfTimingError("flat pin_pos must contain exactly 2 * num_pins values")
        return pin_pos[:num_pins], pin_pos[num_pins:]
    if pin_pos.dim() == 2 and tuple(pin_pos.shape) == (2, num_pins):
        return pin_pos[0], pin_pos[1]
    if pin_pos.dim() == 2 and tuple(pin_pos.shape) == (num_pins, 2):
        return pin_pos[:, 0], pin_pos[:, 1]
    raise LcbFfTimingError("pin_pos must be flat [x..., y...], [2, P], or [P, 2]")


def star_elmore_sink_delay(
    length_um: torch.Tensor,
    sink_base_cap: torch.Tensor,
    r_unit: float,
    c_unit: float,
) -> torch.Tensor:
    """Match StarPinPair.rc_forward: d = r L (C_sink + 0.5 c L)."""
    if int(length_um.numel()) != int(sink_base_cap.numel()):
        raise LcbFfTimingError("star Elmore length and sink cap length differ")
    wire_cap = length_um * float(c_unit)
    resistance = length_um * float(r_unit)
    return resistance * (sink_base_cap + 0.5 * wire_cap)


def star_elmore_driver_load(
    length_um: torch.Tensor,
    driver_ids: torch.Tensor,
    sink_base_cap: torch.Tensor,
    driver_base_cap_by_pin: torch.Tensor,
    r_unit: float,
    c_unit: float,
) -> torch.Tensor:
    """Full star net cap at each unique driver, matching StarPinPair load."""
    del r_unit
    wire_cap = length_um * float(c_unit)
    sink_cap = sink_base_cap + 0.5 * wire_cap
    load = driver_base_cap_by_pin.clone()
    load = load.scatter_add(0, driver_ids, 0.5 * wire_cap)
    load = load.scatter_add(0, driver_ids, sink_cap)
    return load


def _lookup_libcell_ids(plan: LcbFfPlan, data_collections) -> torch.Tensor:
    pin2node = getattr(data_collections, "pin2node_map", None)
    if pin2node is None:
        raise LcbFfTimingError("LCB Liberty lookup requires pin2node_map")
    pin2node = torch.as_tensor(pin2node).reshape(-1).long()
    driver_ids = getattr(plan, "leaf_driver_native_pin_ids", None)
    if driver_ids is None:
        driver_ids = plan.leaf_driver_pin_ids
    driver_ids = driver_ids.reshape(-1).long()
    if driver_ids.numel() and bool((driver_ids >= pin2node.numel()).any().item()):
        raise LcbFfTimingError(
            "LCB driver pin IDs are outside pin2node_map "
            "(max=%d size=%d)"
            % (int(driver_ids.max().item()), int(pin2node.numel()))
        )
    node_ids = pin2node[driver_ids]
    inst_cell_id = getattr(data_collections, "inst_cell_id", None)
    if inst_cell_id is not None:
        cells_all = torch.as_tensor(inst_cell_id).reshape(-1).long()
        if node_ids.numel() and bool((node_ids >= cells_all.numel()).any().item()):
            raise LcbFfTimingError(
                "LCB driver node IDs are outside inst_cell_id "
                "(max=%d size=%d)"
                % (int(node_ids.max().item()), int(cells_all.numel()))
            )
        libcell_ids = cells_all[node_ids.to(device=cells_all.device)].cpu()
        if bool((libcell_ids < 0).any().item()):
            raise LcbFfTimingError("LCB instance has a negative inst_cell_id")
        return libcell_ids
    inst_main_id = getattr(data_collections, "inst_main_id", None)
    inst_libcell_offset = getattr(data_collections, "inst_libcell_offset", None)
    main_id_2_cell_id_start = getattr(data_collections, "main_id_2_cell_id_start", None)
    if (
        inst_main_id is None
        or inst_libcell_offset is None
        or main_id_2_cell_id_start is None
    ):
        raise LcbFfTimingError("LCB Liberty lookup requires instance-to-libcell maps")
    inst_main_id = torch.as_tensor(inst_main_id).reshape(-1).long()
    if node_ids.numel() and bool((node_ids >= inst_main_id.numel()).any().item()):
        raise LcbFfTimingError(
            "LCB driver node IDs are outside inst_main_id "
            "(max=%d size=%d)"
            % (int(node_ids.max().item()), int(inst_main_id.numel()))
        )
    main_ids = inst_main_id[node_ids.to(device=inst_main_id.device)].cpu()
    if bool((main_ids < 0).any().item()):
        raise LcbFfTimingError("LCB instance has no liberty main_id")
    starts = torch.as_tensor(main_id_2_cell_id_start).reshape(-1).long().cpu()
    main_count = max(0, int(starts.numel()) - 1)
    if main_count <= 0 or bool((main_ids >= main_count).any().item()):
        raise LcbFfTimingError("LCB main_id is outside main_id_2_cell_id_start")
    if bool((starts[1:] < starts[:-1]).any().item()):
        raise LcbFfTimingError("main_id_2_cell_id_start is not monotonic")
    offsets_all = torch.as_tensor(inst_libcell_offset).reshape(-1).long()
    if node_ids.numel() and bool((node_ids >= offsets_all.numel()).any().item()):
        raise LcbFfTimingError(
            "LCB driver node IDs are outside inst_libcell_offset "
            "(max=%d size=%d)"
            % (int(node_ids.max().item()), int(offsets_all.numel()))
        )
    offsets = offsets_all[node_ids.to(device=offsets_all.device)].cpu()
    if bool((offsets < 0).any().item()):
        raise LcbFfTimingError("LCB instance has a negative Liberty cell offset")
    libcell_ids = starts[main_ids] + offsets
    if bool((libcell_ids < starts[main_ids]).any().item()):
        raise LcbFfTimingError("LCB Liberty cell ID overflow")
    return libcell_ids


def _lib_arc_ids_for_cells(cell_id_2_arc_id_start, libcell_ids: torch.Tensor) -> torch.Tensor:
    starts = torch.as_tensor(cell_id_2_arc_id_start).reshape(-1).long()
    cell_count = max(0, int(starts.numel()) - 1)
    if cell_count <= 0 or bool((libcell_ids < 0).any().item()) or bool(
        (libcell_ids >= cell_count).any().item()
    ):
        raise LcbFfTimingError("LCB libcell_id is outside cell_id_2_arc_id_start")
    if bool((starts[1:] < starts[:-1]).any().item()):
        raise LcbFfTimingError("cell_id_2_arc_id_start is not monotonic")
    arc_ids = starts[libcell_ids]
    if bool((arc_ids >= starts[libcell_ids + 1]).any().item()):
        raise LcbFfTimingError("LCB libcell has no Liberty delay arc")
    return arc_ids


def evaluate_lcb_cell_delay_slew(
    *,
    timing_op,
    plan: LcbFfPlan,
    data_collections,
    driver_load: torch.Tensor,
    dtype,
    device,
    driver_pin_ids=None,
):
    zeros = torch.zeros(plan.leaf_pair_count(), dtype=dtype, device=device)
    if plan.leaf_pair_count() == 0 or timing_op is None:
        return zeros, zeros.clone(), 0
    if data_collections is None:
        raise LcbFfTimingError("LCB Liberty lookup requires data_collections")
    lut = getattr(timing_op, "lut_entry_vectorized", None)
    arcs_info = getattr(timing_op, "arcs_info", None)
    starts = getattr(data_collections, "cell_id_2_arc_id_start", None)
    if lut is None or arcs_info is None or starts is None:
        raise LcbFfTimingError("LCB Liberty lookup requires cell_id_2_arc_id_start and LUT banks")
    libcell_ids = _lookup_libcell_ids(plan, data_collections)
    lut_ref = arcs_info.r_delay_luts.flat_luts_values
    lut_device = lut_ref.device
    lut_dtype = lut_ref.dtype
    input_slew = torch.zeros(plan.leaf_pair_count(), dtype=lut_dtype, device=lut_device)
    if driver_pin_ids is None:
        driver_pin_ids = plan.leaf_driver_pin_ids
    driver_ids = torch.as_tensor(driver_pin_ids).reshape(-1).long()
    driver_load = torch.as_tensor(driver_load).reshape(-1)
    if driver_ids.numel() and bool((driver_ids >= driver_load.numel()).any().item()):
        raise LcbFfTimingError(
            "LCB driver load vector does not cover driver pins "
            "(max=%d size=%d)"
            % (int(driver_ids.max().item()), int(driver_load.numel()))
        )
    load = driver_load[driver_ids.to(device=driver_load.device)].to(
        device=lut_device, dtype=lut_dtype
    )
    cells = libcell_ids.to(device=lut_device)
    arcs = _lib_arc_ids_for_cells(starts, libcell_ids).to(device=lut_device)
    arc_count = int(lut_ref.shape[0])
    for name, lut_info in (
        ("r_delay", arcs_info.r_delay_luts),
        ("f_delay", arcs_info.f_delay_luts),
        ("r_trans", arcs_info.r_trans_luts),
        ("f_trans", arcs_info.f_trans_luts),
    ):
        lut_count = int(lut_info.flat_luts_values.shape[0])
        if lut_count != arc_count:
            raise LcbFfTimingError(
                "LCB Liberty LUT domains disagree: %s=%d reference=%d"
                % (name, lut_count, arc_count)
            )
    if arcs.numel() and bool((arcs >= arc_count).any().item()):
        raise LcbFfTimingError(
            "LCB Liberty arc IDs are outside the LUT domain "
            "(max=%d size=%d)"
            % (int(arcs.max().item()), arc_count)
        )
    delay_r = lut(cells, input_slew, load, arcs, arcs_info.r_delay_luts)
    delay_f = lut(cells, input_slew, load, arcs, arcs_info.f_delay_luts)
    slew_r = lut(cells, input_slew, load, arcs, arcs_info.r_trans_luts)
    slew_f = lut(cells, input_slew, load, arcs, arcs_info.f_trans_luts)
    if _smooth_extremum_enabled(timing_op):
        tau = float(getattr(timing_op, "timing_aggregation_tau_ps", 1.0))
        delay = _pairwise_lse_max(delay_r, delay_f, tau).to(device=device, dtype=dtype)
        slew = _pairwise_lse_max(slew_r, slew_f, tau).clamp_min(0).to(
            device=device, dtype=dtype
        )
    else:
        delay = torch.maximum(delay_r, delay_f).to(device=device, dtype=dtype)
        slew = torch.maximum(slew_r, slew_f).clamp_min(0).to(
            device=device, dtype=dtype
        )
    if bool((~torch.isfinite(delay)).any().item()) or bool((delay < 0).any().item()):
        raise LcbFfTimingError("LCB Liberty delay is non-finite or negative")
    return delay, slew, int(delay.numel())


def compute_lcb_ff_clock_state(
    *,
    plan: LcbFfPlan,
    pin_pos: torch.Tensor,
    base_caps: Mapping[str, torch.Tensor],
    r_unit: float,
    c_unit: float,
    coordinate_to_um: float,
    timing_op=None,
    data_collections=None,
    rc_inputs=None,
) -> dict:
    """Return compact CK AT/slew using shared RC + LCB Liberty.

    ``rc_inputs`` is an optional compact clock-pin RC state produced by
    ``StarPinPair.rc_forward`` and its cached teacher calibration.  Keeping
    the clock-specific LCB/FF timing wrapper separate preserves launch/capture
    semantics while sharing the data path's validated RC implementation.
    """
    if coordinate_to_um <= 0:
        raise LcbFfTimingError("coordinate_to_um must be positive")
    generic = torch.as_tensor(base_caps["generic"]).reshape(-1)
    rise = torch.as_tensor(base_caps.get("rise", generic)).reshape(-1)
    fall = torch.as_tensor(base_caps.get("fall", generic)).reshape(-1)
    num_pins = int(generic.numel())
    x, y = split_pin_pos(pin_pos, num_pins)
    dtype = generic.dtype
    device = pin_pos.device
    arrival_r = torch.zeros(plan.clock_pins_native.numel(), dtype=dtype, device=device)
    arrival_f = torch.zeros_like(arrival_r)
    slew_r = torch.zeros_like(arrival_r)
    slew_f = torch.zeros_like(arrival_r)
    if plan.leaf_pair_count() == 0:
        return {
            "late_rise_arrival_ps": arrival_r,
            "late_fall_arrival_ps": arrival_f,
            "late_rise_slew_ps": slew_r,
            "late_fall_slew_ps": slew_f,
            "leaf_pair_count": 0,
            "lcb_cell_arc_count": 0,
            "root_sink_count": int(plan.sink_is_root.sum().item())
            if plan.sink_is_root.numel()
            else 0,
        }
    drivers = plan.leaf_driver_pin_ids.to(device=device)
    sinks = plan.leaf_sink_pin_ids.to(device=device)
    rows = plan.leaf_sink_rows.to(device=device)
    for name, indices in (("driver", drivers), ("sink", sinks)):
        if indices.numel() and bool((indices >= num_pins).any().item()):
            raise LcbFfTimingError(
                "LCB->FF %s pin IDs are outside the pin-position domain "
                "(max=%d size=%d)"
                % (name, int(indices.max().item()), num_pins)
            )
    if rows.numel() and bool(
        (rows >= int(plan.clock_pins_native.numel())).any().item()
    ):
        raise LcbFfTimingError(
            "LCB->FF sink rows are outside the compact clock-pin domain "
            "(max=%d size=%d)"
            % (int(rows.max().item()), int(plan.clock_pins_native.numel()))
        )
    length_um = (
        torch.abs(x[drivers] - x[sinks]) + torch.abs(y[drivers] - y[sinks])
    ) / float(coordinate_to_um)
    if rc_inputs is None:
        sink_cap_r = rise.to(device=device)[sinks]
        sink_cap_f = fall.to(device=device)[sinks]
        sink_cap = generic.to(device=device)[sinks]
        net_delay_r = star_elmore_sink_delay(length_um, sink_cap_r, r_unit, c_unit)
        net_delay_f = star_elmore_sink_delay(length_um, sink_cap_f, r_unit, c_unit)
        driver_load = star_elmore_driver_load(
            length_um,
            drivers,
            sink_cap,
            generic.to(device=device),
            r_unit,
            c_unit,
        )
        net_impulse_r = torch.zeros_like(net_delay_r)
        net_impulse_f = torch.zeros_like(net_delay_f)
        rc_model = "star_depth_one"
    else:
        required = ("delays", "impulses")
        if any(key not in rc_inputs for key in required):
            raise LcbFfTimingError(
                "LCB->FF shared RC state requires delays and impulses"
            )
        try:
            # Shared timing snapshots expose per-transition root loads as
            # ``caps``; older callers may provide ``loads.generic``.  Clock
            # cell evaluation needs one generic load, so use the rise view
            # (the two views share the same capacitive load in the RC model).
            load_groups = rc_inputs.get("loads") or rc_inputs.get("caps")
            if load_groups is None:
                raise KeyError("loads/caps")
            driver_load_source = load_groups.get("generic", load_groups.get("rise"))
            if driver_load_source is None and rc_inputs.get("caps") is not load_groups:
                driver_load_source = rc_inputs["caps"].get("rise")
            if driver_load_source is None:
                raise KeyError("loads.generic/loads.rise")
            driver_load_source = torch.as_tensor(driver_load_source).reshape(-1)
            delays_r_source = torch.as_tensor(rc_inputs["delays"]["rise"]).reshape(-1)
            delays_f_source = torch.as_tensor(rc_inputs["delays"]["fall"]).reshape(-1)
            impulses_r_source = torch.as_tensor(rc_inputs["impulses"]["rise"]).reshape(-1)
            impulses_f_source = torch.as_tensor(rc_inputs["impulses"]["fall"]).reshape(-1)
            for label, source, indices in (
                ("driver load", driver_load_source, drivers),
                ("rise delay", delays_r_source, sinks),
                ("fall delay", delays_f_source, sinks),
                ("rise impulse", impulses_r_source, sinks),
                ("fall impulse", impulses_f_source, sinks),
            ):
                if indices.numel() and bool((indices >= source.numel()).any().item()):
                    raise LcbFfTimingError(
                        "LCB->FF shared RC %s does not cover pin IDs "
                        "(max=%d size=%d)"
                        % (label, int(indices.max().item()), int(source.numel()))
                    )
            # Keep the shared load in the full native-pin domain.  The
            # Liberty evaluator performs the single driver-pin gather after
            # validating that domain; gathering here would index it twice.
            driver_load = driver_load_source.to(device=device, dtype=dtype)
            net_delay_r = delays_r_source.to(device=device, dtype=dtype)[sinks]
            net_delay_f = delays_f_source.to(device=device, dtype=dtype)[sinks]
            net_impulse_r = impulses_r_source.to(device=device, dtype=dtype)[sinks]
            net_impulse_f = impulses_f_source.to(device=device, dtype=dtype)[sinks]
        except (KeyError, IndexError, RuntimeError) as error:
            raise LcbFfTimingError("LCB->FF shared RC state does not cover clock pins") from error
        rc_model = str(rc_inputs.get("model", "clock_star_raw"))
    cell_delay, cell_slew, cell_count = evaluate_lcb_cell_delay_slew(
        timing_op=timing_op,
        plan=plan,
        data_collections=data_collections,
        driver_load=driver_load,
        dtype=dtype,
        device=device,
        driver_pin_ids=(
            getattr(plan, "leaf_driver_pin_ids", None)
            if rc_inputs is not None
            else None
        ),
    )
    for name, values in (("rise delay", net_delay_r), ("fall delay", net_delay_f)):
        if bool((~torch.isfinite(values)).any().item()) or bool(
            (values < 0).any().item()
        ):
            raise LcbFfTimingError("LCB->FF clock %s is invalid" % name)
    for name, values in (("rise impulse", net_impulse_r), ("fall impulse", net_impulse_f)):
        if bool((~torch.isfinite(values)).any().item()):
            raise LcbFfTimingError("LCB->FF clock %s is invalid" % name)
    arrival_r[rows] = net_delay_r.to(dtype=dtype) + cell_delay.to(dtype=dtype)
    arrival_f[rows] = net_delay_f.to(dtype=dtype) + cell_delay.to(dtype=dtype)
    slew_r[rows] = torch.sqrt(
        cell_slew.to(dtype=dtype).square() + net_impulse_r.clamp_min(0)
    )
    slew_f[rows] = torch.sqrt(
        cell_slew.to(dtype=dtype).square() + net_impulse_f.clamp_min(0)
    )
    return {
        "late_rise_arrival_ps": arrival_r,
        "late_fall_arrival_ps": arrival_f,
        "late_rise_slew_ps": slew_r,
        "late_fall_slew_ps": slew_f,
        "leaf_pair_count": plan.leaf_pair_count(),
        "lcb_cell_arc_count": cell_count,
        "root_sink_count": int(plan.sink_is_root.sum().item()),
        "length_um": length_um.detach(),
        "net_delay_ps": net_delay_r.detach(),
        "rc_model": rc_model,
    }


def build_clock_star_steiner_net2pin(star) -> tuple[torch.Tensor, torch.Tensor]:
    """Build compact (flat_net2pin_map, flat_net2pin_start_map) for FLUTE teacher RC.

    Each leaf clock net has its compact driver pin first (local pin 0 = FLUTE
    root), followed by its compact sink pins in deterministic pair order.
    """
    num_nets = int(star.num_nets)
    pair_net_ids = star.pair_net_ids.detach().cpu().long()
    net_drivers = star.net_driver_pin_ids.detach().cpu().long()
    sink_pins = star.sink_pin_ids.detach().cpu().long()
    flat_pins: list[int] = []
    starts: list[int] = [0]
    for net_id in range(num_nets):
        mask = pair_net_ids == net_id
        net_sinks = sink_pins[mask].tolist()
        if net_sinks:
            driver = int(net_drivers[net_id].item())
            flat_pins.append(driver)
            flat_pins.extend(int(s) for s in net_sinks)
        starts.append(len(flat_pins))
    return (
        torch.tensor(flat_pins, dtype=torch.int32),
        torch.tensor(starts, dtype=torch.int32),
    )


def apply_lcb_ff_to_compact_clk_pins(timing_op, state: Mapping[str, torch.Tensor]) -> dict:
    arrival_r = state["late_rise_arrival_ps"]
    expected_rows = int(
        getattr(timing_op, "clock_pins", timing_op.clk_pin_rtran).numel()
    )
    if int(arrival_r.numel()) != expected_rows:
        raise LcbFfTimingError("LCB->FF compact CK AT does not match clk_pin domain")
    for name in (
        "late_rise_arrival_ps",
        "late_fall_arrival_ps",
        "late_rise_slew_ps",
        "late_fall_slew_ps",
    ):
        values = state[name]
        if not bool(torch.all(torch.isfinite(values)).item()) or bool(
            torch.any(values < 0).item()
        ):
            raise LcbFfTimingError("LCB->FF compact %s is non-finite or negative" % name)
    timing_op.clk_pin_r_aat = state["late_rise_arrival_ps"].detach().clone()
    timing_op.clk_pin_f_aat = state["late_fall_arrival_ps"].detach().clone()
    timing_op.clk_pin_rtran = state["late_rise_slew_ps"].detach().clone()
    timing_op.clk_pin_ftran = state["late_fall_slew_ps"].detach().clone()
    return {
        "status": "native",
        "owner": "lcb_ff_datapath",
        "compact_row_count": expected_rows,
        "leaf_pair_count": int(state.get("leaf_pair_count", 0)),
        "lcb_cell_arc_count": int(state.get("lcb_cell_arc_count", 0)),
        "root_sink_count": int(state.get("root_sink_count", 0)),
    }
