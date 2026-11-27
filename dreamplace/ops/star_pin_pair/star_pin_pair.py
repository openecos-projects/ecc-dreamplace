"""GPU-vectorized driver-to-sink star topology for timing placement.

The legacy RCTiming implementation evaluates an arbitrary Steiner tree with
CPU custom operators.  A driver-to-sink star has depth one, so its RC
recurrences are closed-form gather/scatter expressions and can remain on the
same PyTorch device as placement.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import nn


_TIMING_MODES = ("generic", "rise", "fall")
_DISTANCE_MODELS = ("l1", "wa", "lse")
_LOAD_GEOMETRIES = ("star", "wa_net")

class StarPinPair(nn.Module):
    """Stable physical driver-to-sink pairs and their depth-one RC model.

    Connectivity is built exactly once from the compact timing-net arc CSR
    plus its authoritative driver map.  ``length_um`` is the physical Manhattan
    length consumed by RC timing.  The default placement surrogate is
    ``structured_delay_objective``: frozen net-delay adjoints times the
    same depth-one Elmore delay used by ``rc_forward``.  The optional
    ``quadratic_pair_objective`` uses Efficient-TDP's placement-unit
    ``||Δp||²`` geometry on the same frozen pair weights.
    """

    def __init__(
        self,
        *,
        net2driver_pin_map: torch.Tensor,
        timing_net_flat_arcs: torch.Tensor,
        timing_net_flat_arcs_start: torch.Tensor,
        num_pins: int,
        r_unit: float = 1.0,
        c_unit: float = 1.0,
        scale_factor: float = 1.0,
        dbu: float = 1.0,
        identity_namespace: str = "data",
        load_geometry: str = "star",
    ) -> None:
        super().__init__()
        self.num_pins = self._validated_positive_int(num_pins, "num_pins")
        self.r_unit = self._validated_nonnegative_float(r_unit, "r_unit")
        self.c_unit = self._validated_nonnegative_float(c_unit, "c_unit")
        self.scale_factor = self._validated_positive_float(
            scale_factor, "scale_factor"
        )
        self.dbu = self._validated_positive_float(dbu, "dbu")
        self.coordinate_to_um = self.scale_factor * self.dbu
        self.identity_namespace = str(identity_namespace or "data")
        self.load_geometry = self._normalize_load_geometry(load_geometry)

        (
            pair_net_ids,
            driver_pin_ids,
            sink_pin_ids,
        ) = self._build_pairs_from_timing_arcs(
            timing_net_flat_arcs,
            timing_net_flat_arcs_start,
            net2driver_pin_map,
            self.num_pins,
        )
        net_driver_pin_ids = (
            torch.as_tensor(net2driver_pin_map)
            .reshape(-1)
            .to(device=pair_net_ids.device, dtype=torch.long)
            .contiguous()
        )
        self.num_nets = int(net_driver_pin_ids.numel())
        self.register_buffer("pair_net_ids", pair_net_ids)
        self.register_buffer("driver_pin_ids", driver_pin_ids)
        self.register_buffer("sink_pin_ids", sink_pin_ids)
        self.register_buffer("net_driver_pin_ids", net_driver_pin_ids)
        if int(pair_net_ids.numel()) > 0:
            (
                wa_active_net_ids,
                wa_pair_compact_net_ids,
                wa_active_net_degrees,
            ) = torch.unique(
                pair_net_ids,
                sorted=True,
                return_inverse=True,
                return_counts=True,
            )
            wa_pair_net_degree = wa_active_net_degrees.index_select(
                0, wa_pair_compact_net_ids
            )
            active_net_count = int(wa_active_net_ids.numel())
            inc_nets = torch.cat(
                (
                    torch.arange(
                        active_net_count,
                        device=pair_net_ids.device,
                        dtype=torch.long,
                    ),
                    wa_pair_compact_net_ids,
                ),
                dim=0,
            )
            inc_pins = torch.cat(
                (
                    net_driver_pin_ids.index_select(0, wa_active_net_ids),
                    sink_pin_ids,
                ),
                dim=0,
            )
            inc_order = torch.argsort(inc_nets, stable=True)
            wa_inc_compact_net_ids = inc_nets.index_select(0, inc_order).contiguous()
            wa_inc_pin_ids = inc_pins.index_select(0, inc_order).contiguous()
        else:
            empty_long = torch.empty(
                0, dtype=torch.long, device=pair_net_ids.device
            )
            wa_active_net_ids = empty_long
            wa_pair_compact_net_ids = empty_long.clone()
            wa_pair_net_degree = empty_long.clone()
            wa_inc_compact_net_ids = empty_long.clone()
            wa_inc_pin_ids = empty_long.clone()
        self.register_buffer(
            "wa_active_net_ids", wa_active_net_ids, persistent=False
        )
        self.register_buffer(
            "wa_pair_compact_net_ids", wa_pair_compact_net_ids, persistent=False
        )
        self.register_buffer(
            "wa_pair_net_degree", wa_pair_net_degree, persistent=False
        )
        self.register_buffer(
            "wa_inc_compact_net_ids", wa_inc_compact_net_ids, persistent=False
        )
        self.register_buffer("wa_inc_pin_ids", wa_inc_pin_ids, persistent=False)
        self.register_buffer(
            "frozen_weight",
            torch.empty(0, dtype=torch.float32, device=pair_net_ids.device),
        )
        self.register_buffer(
            "frozen_load_length_weight",
            torch.empty(0, dtype=torch.float32, device=pair_net_ids.device),
        )
        self.register_buffer(
            "frozen_curvature_load_length_weight",
            torch.empty(0, dtype=torch.float32, device=pair_net_ids.device),
        )
        self.register_buffer(
            "frozen_impulse_length_weight",
            torch.empty(0, dtype=torch.float32, device=pair_net_ids.device),
        )
        self.register_buffer(
            "frozen_impulse_anchor",
            torch.empty(0, dtype=torch.float32, device=pair_net_ids.device),
        )
        # Weight value zero is a valid state for the w0=0 ablation.  Keep
        # membership separately so a seen zero-weight pair can accumulate on
        # a later refresh instead of being mistaken for a new pair.
        self.register_buffer(
            "frozen_weight_seen",
            torch.empty(0, dtype=torch.bool, device=pair_net_ids.device),
        )
        self.register_buffer(
            "frozen_length_um",
            torch.empty(0, dtype=torch.float32, device=pair_net_ids.device),
        )
        self.pair_topology_generation = 1
        self.pair_weight_generation = 0
        self.snapshot_generation = 0
        self.refresh_count = 0
        self.pair_identity_hash = self._pair_identity_hash(
            pair_net_ids,
            driver_pin_ids,
            sink_pin_ids,
            namespace=self.identity_namespace,
        )
        self._frozen_snapshot_metadata: Optional[Dict[str, object]] = None
        self.live_endpoint_hinge_snapshot: Optional[object] = None
        # Lazily derived timing-aware diagonal; publication invalidates it.
        self._timing_precondition_cache_key = None
        self._timing_precondition_cache_x = None
        self._timing_precondition_cache_y = None
        # ``pin2node_map`` is int32 in the production pydb.  Keep one
        # device-local int64 view so diagonal construction can actually reuse
        # the snapshot cache instead of converting millions of entries per
        # optimizer iteration.
        self._timing_precondition_pin_map_source_key = None
        self._timing_precondition_pin_map_long = None

    @property
    def pair_count(self) -> int:
        return int(self.sink_pin_ids.numel())

    @staticmethod
    def _validated_positive_int(value, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError("%s must be a positive integer" % name)
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be a positive integer" % name) from exc
        if result <= 0:
            raise ValueError("%s must be a positive integer" % name)
        return result

    @staticmethod
    def _validated_positive_float(value, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be finite and positive" % name) from exc
        if not math.isfinite(result) or result <= 0.0:
            raise ValueError("%s must be finite and positive" % name)
        return result

    @staticmethod
    def _validated_nonnegative_float(value, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("%s must be finite and non-negative" % name) from exc
        if not math.isfinite(result) or result < 0.0:
            raise ValueError("%s must be finite and non-negative" % name)
        return result

    @staticmethod
    def _first_indices(mask: torch.Tensor, limit: int = 8):
        if not torch.is_tensor(mask) or mask.numel() == 0:
            return []
        return torch.nonzero(mask, as_tuple=False).reshape(-1)[:limit].cpu().tolist()


    @classmethod
    def _build_pairs_from_timing_arcs(
        cls,
        timing_net_flat_arcs: torch.Tensor,
        timing_net_flat_arcs_start: torch.Tensor,
        net2driver_pin_map: torch.Tensor,
        num_pins: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build pairs from the compact timing-net CSR used by the sidecar.

        ``timing_sidecar`` deliberately keeps net-local timing arrays in its
        compact sidecar-net order, while the placement ``flat_net2pin_*``
        arrays remain in native-net order.  The timing arc CSR is therefore
        the authoritative production source: its rows are already mapped to
        native physical pin IDs and enumerate exactly the driver-to-sink arcs
        consumed by timing propagation.  This avoids assuming that two valid
        net namespaces happen to share an ordering.
        """
        arcs = torch.as_tensor(timing_net_flat_arcs)
        if arcs.dim() != 2 or int(arcs.shape[1]) < 2:
            raise ValueError("timing_net_flat_arcs must have at least source and sink columns")
        arcs = arcs[:, :2].to(dtype=torch.long).contiguous()
        starts = (
            torch.as_tensor(timing_net_flat_arcs_start)
            .reshape(-1)
            .to(device=arcs.device, dtype=torch.long)
            .contiguous()
        )
        drivers_by_net = (
            torch.as_tensor(net2driver_pin_map)
            .reshape(-1)
            .to(device=arcs.device, dtype=torch.long)
            .contiguous()
        )
        if starts.numel() < 2:
            raise ValueError("timing_net_flat_arcs_start must contain at least one timing net")
        num_nets = int(starts.numel() - 1)
        if int(drivers_by_net.numel()) != num_nets:
            raise ValueError(
                "net2driver_pin_map length must match timing_net_flat_arcs_start"
            )
        if int(starts[0].cpu().item()) != 0:
            raise ValueError("timing_net_flat_arcs_start must start at zero")
        if int(starts[-1].cpu().item()) != int(arcs.shape[0]):
            raise ValueError(
                "timing_net_flat_arcs_start must end at timing_net_flat_arcs length"
            )
        degrees = starts[1:] - starts[:-1]
        if bool(torch.any(degrees < 0).cpu().item()):
            raise ValueError("timing_net_flat_arcs_start must be monotonic")
        invalid_driver = (degrees > 0) & (
            (drivers_by_net < 0) | (drivers_by_net >= int(num_pins))
        )
        if bool(torch.any(invalid_driver).cpu().item()):
            raise ValueError(
                "net2driver_pin_map contains out-of-range driver IDs for timing nets %s"
                % cls._first_indices(invalid_driver)
            )
        net_ids = torch.repeat_interleave(
            torch.arange(num_nets, device=arcs.device, dtype=torch.long), degrees
        )
        if int(net_ids.numel()) != int(arcs.shape[0]):
            raise RuntimeError("star timing-arc net expansion has inconsistent size")
        sources = arcs[:, 0]
        sinks = arcs[:, 1]
        invalid_pin = (
            (sources < 0)
            | (sources >= int(num_pins))
            | (sinks < 0)
            | (sinks >= int(num_pins))
        )
        if bool(torch.any(invalid_pin).cpu().item()):
            raise ValueError(
                "timing_net_flat_arcs contains out-of-range pin IDs at rows %s"
                % cls._first_indices(invalid_pin)
            )
        expected_drivers = drivers_by_net[net_ids]
        source_mismatch = sources != expected_drivers
        if bool(torch.any(source_mismatch).cpu().item()):
            raise ValueError(
                "timing_net_flat_arcs source must equal the authoritative driver; bad rows %s"
                % cls._first_indices(source_mismatch)
            )
        self_loop = sinks == expected_drivers
        if bool(torch.any(self_loop).cpu().item()):
            raise ValueError(
                "timing_net_flat_arcs must not contain driver self-loop rows %s"
                % cls._first_indices(self_loop)
            )
        pair_keys = (
            net_ids.to(dtype=torch.int64) * int(num_pins)
            + sinks.to(dtype=torch.int64)
        )
        if int(torch.unique(pair_keys).numel()) != int(pair_keys.numel()):
            raise ValueError(
                "timing_net_flat_arcs repeats a physical sink within a timing net"
            )
        return (
            net_ids.contiguous(),
            expected_drivers.contiguous(),
            sinks.contiguous(),
        )

    @staticmethod
    def _pair_identity_hash(
        pair_net_ids: torch.Tensor,
        driver_pin_ids: torch.Tensor,
        sink_pin_ids: torch.Tensor,
        namespace: str = "data",
    ) -> str:
        payload = torch.stack(
            (pair_net_ids, driver_pin_ids, sink_pin_ids), dim=1
        ).detach().to(device="cpu", dtype=torch.int64).contiguous()
        digest = hashlib.sha256()
        if str(namespace) != "data":
            digest.update(str(namespace).encode("utf-8"))
            digest.update(b"\0")
        digest.update(payload.numpy().tobytes())
        return digest.hexdigest()

    def _split_pin_pos(self, pin_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(pin_pos):
            raise TypeError("pin_pos must be a tensor")
        if pin_pos.dim() == 1:
            if int(pin_pos.numel()) != 2 * self.num_pins:
                raise ValueError("flat pin_pos must contain exactly 2 * num_pins values")
            return pin_pos[: self.num_pins], pin_pos[self.num_pins :]
        if pin_pos.dim() == 2 and tuple(pin_pos.shape) == (2, self.num_pins):
            return pin_pos[0], pin_pos[1]
        if pin_pos.dim() == 2 and tuple(pin_pos.shape) == (self.num_pins, 2):
            return pin_pos[:, 0], pin_pos[:, 1]
        raise ValueError("pin_pos must be flat [x..., y...], [2, P], or [P, 2]")

    def geometry(self, pin_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the canonical branch geometry without any host transfer."""
        x, y = self._split_pin_pos(pin_pos)
        driver_ids = self.driver_pin_ids
        sink_ids = self.sink_pin_ids
        raw_dx = x[driver_ids] - x[sink_ids]
        raw_dy = y[driver_ids] - y[sink_ids]
        dx_um = torch.abs(raw_dx) / self.coordinate_to_um
        dy_um = torch.abs(raw_dy) / self.coordinate_to_um
        return {
            "raw_dx": raw_dx,
            "raw_dy": raw_dy,
            "dx_um": dx_um,
            "dy_um": dy_um,
            "length_um": dx_um + dy_um,
        }

    @staticmethod
    def _normalize_distance_model(distance_model) -> str:
        model = str(
            distance_model if distance_model is not None else "l1"
        ).strip().lower()
        if model not in _DISTANCE_MODELS:
            raise ValueError(
                "star distance_model must be one of: %s"
                % ", ".join(_DISTANCE_MODELS)
            )
        return model

    @staticmethod
    def _normalize_load_geometry(load_geometry) -> str:
        mode = str(
            load_geometry if load_geometry is not None else "star"
        ).strip().lower()
        if mode not in _LOAD_GEOMETRIES:
            raise ValueError(
                "star load_geometry must be one of: %s"
                % ", ".join(_LOAD_GEOMETRIES)
            )
        return mode

    def set_load_geometry(self, load_geometry: str) -> str:
        mode = self._normalize_load_geometry(load_geometry)
        if mode != self.load_geometry:
            self.load_geometry = mode
            self._timing_precondition_cache_key = None
            self._timing_precondition_cache_x = None
            self._timing_precondition_cache_y = None
        return self.load_geometry

    def _aggregate_pair_load_to_active_nets(
        self, pair_load: torch.Tensor
    ) -> torch.Tensor:
        active_count = int(self.wa_active_net_ids.numel())
        flat_load = torch.as_tensor(pair_load).reshape(-1)
        if active_count == 0 or int(flat_load.numel()) == 0:
            return flat_load.new_zeros((active_count,))
        if int(flat_load.numel()) != int(self.sink_pin_ids.numel()):
            raise ValueError("pair_load length must match star pair count")
        compact_ids = self.wa_pair_compact_net_ids
        if compact_ids.device != flat_load.device:
            compact_ids = compact_ids.to(device=flat_load.device)
        net_load = flat_load.new_zeros((active_count,))
        return net_load.scatter_add(0, compact_ids, flat_load)

    def net_wa_wirelength_um(
        self,
        pin_pos: torch.Tensor,
        smoothing_gamma,
    ) -> torch.Tensor:
        """Return overflow/cancellation-free net-level WA wirelength (in um) per active net."""
        x, y = self._split_pin_pos(pin_pos)
        active_count = int(self.wa_active_net_ids.numel())
        if active_count == 0:
            return x.new_zeros((0,))
        gamma = self._validated_smoothing_gamma(
            smoothing_gamma, device=x.device, dtype=x.dtype
        )
        inc_nets = self.wa_inc_compact_net_ids
        inc_pins = self.wa_inc_pin_ids
        active_nets = self.wa_active_net_ids
        if inc_nets.device != x.device:
            inc_nets = inc_nets.to(device=x.device)
            inc_pins = inc_pins.to(device=x.device)
            active_nets = active_nets.to(device=x.device)
        drv_pins = self.net_driver_pin_ids.to(device=x.device).index_select(
            0, active_nets
        )

        def _axis_wa(u: torch.Tensor) -> torch.Tensor:
            u_inc = u.index_select(0, inc_pins)
            with torch.no_grad():
                u_det = u.detach()
                u_inc_det = u_det.index_select(0, inc_pins)
                c_max = u_det.index_select(0, drv_pins).clone()
                c_min = c_max.clone()
                c_max.scatter_reduce_(
                    0, inc_nets, u_inc_det, reduce="amax", include_self=True
                )
                c_min.scatter_reduce_(
                    0, inc_nets, u_inc_det, reduce="amin", include_self=True
                )
            delta_plus = u_inc - c_max.index_select(0, inc_nets)
            w_plus = torch.exp(delta_plus / gamma)
            z_plus = u.new_zeros((active_count,)).scatter_add(
                0, inc_nets, w_plus
            )
            num_plus = u.new_zeros((active_count,)).scatter_add(
                0, inc_nets, w_plus * delta_plus
            )
            delta_minus = c_min.index_select(0, inc_nets) - u_inc
            w_minus = torch.exp(delta_minus / gamma)
            z_minus = u.new_zeros((active_count,)).scatter_add(
                0, inc_nets, w_minus
            )
            num_minus = u.new_zeros((active_count,)).scatter_add(
                0, inc_nets, w_minus * delta_minus
            )
            return (c_max - c_min) + (num_plus / z_plus) + (num_minus / z_minus)

        return (_axis_wa(x) + _axis_wa(y)) / self.coordinate_to_um

    @staticmethod
    def _validated_smoothing_gamma(smoothing_gamma, *, device, dtype) -> torch.Tensor:
        gamma = torch.as_tensor(
            smoothing_gamma, device=device, dtype=dtype
        ).reshape(-1)
        if int(gamma.numel()) != 1:
            raise ValueError("star smoothing_gamma must be scalar")
        gamma = gamma[0]
        if not bool(torch.isfinite(gamma).item()) or not bool((gamma > 0).item()):
            raise ValueError("star smoothing_gamma must be finite and positive")
        return gamma

    @classmethod
    def _axis_distance(cls, delta, distance_model, smoothing_gamma=None):
        model = cls._normalize_distance_model(distance_model)
        if model == "l1":
            return delta.abs()
        gamma = cls._validated_smoothing_gamma(
            smoothing_gamma, device=delta.device, dtype=delta.dtype
        )
        z = delta / (2.0 * gamma)
        if model == "wa":
            # Two-pin ePlace weighted-average wirelength.
            return delta * torch.tanh(z)
        # Zero-normalized two-pin LSE: 2*gamma*log(cosh(z)). The offset
        # matters because Elmore uses L**2. logsumexp keeps second derivatives
        # finite at large separations and correct at coincident coordinates.
        return 2.0 * gamma * (
            torch.logsumexp(torch.stack((z, -z), dim=0), dim=0) - math.log(2.0)
        )

    def surrogate_geometry(
        self,
        pin_pos: torch.Tensor,
        *,
        distance_model="l1",
        smoothing_gamma=None,
    ) -> Dict[str, torch.Tensor]:
        """Return geometry for the differentiable timing surrogate.

        The public ``geometry`` and ``length_um`` methods remain physical
        Manhattan measurements for RC/timing refreshes. Smooth WA/LSE models
        are used only here, with gamma in raw placement-coordinate units.
        """
        x, y = self._split_pin_pos(pin_pos)
        raw_dx = x[self.driver_pin_ids] - x[self.sink_pin_ids]
        raw_dy = y[self.driver_pin_ids] - y[self.sink_pin_ids]
        model = self._normalize_distance_model(distance_model)
        surrogate_dx = self._axis_distance(raw_dx, model, smoothing_gamma)
        surrogate_dy = self._axis_distance(raw_dy, model, smoothing_gamma)
        surrogate_dx_um = surrogate_dx / self.coordinate_to_um
        surrogate_dy_um = surrogate_dy / self.coordinate_to_um
        return {
            "raw_dx": raw_dx,
            "raw_dy": raw_dy,
            "dx_um": surrogate_dx_um,
            "dy_um": surrogate_dy_um,
            "length_um": surrogate_dx_um + surrogate_dy_um,
        }

    def length_um(self, pin_pos: torch.Tensor) -> torch.Tensor:
        return self.geometry(pin_pos)["length_um"]

    def elmore_length_slope(
        self,
        length_um: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return d(star Elmore delay)/d(length) for the generic mode."""
        caps = self._normalized_base_caps(base_caps)
        if length_um.dim() != 1 or length_um.numel() != self.sink_pin_ids.numel():
            raise ValueError("length_um must contain one value per star pair")
        return self.r_unit * (
            caps["generic"][self.sink_pin_ids] + self.c_unit * length_um
        )

    def compute_timing_channel_length_gradients(
        self,
        *,
        length_um: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
        input_gradients: Mapping[str, Mapping[str, torch.Tensor]],
        calibration=None,
        position: Optional[torch.Tensor] = None,
        cap_digest: Optional[str] = None,
        timing_mode_identity: str = "generic,rise,fall",
        clock_policy_digest: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Compute the three timing channel gradients with respect to length.

        ``length_um`` is in micrometres. ``base_caps`` and ``input_gradients``
        must come from the same timing refresh; the latter contains
        ``d(loss)/d(delay)``, ``d(loss)/d(impulse)``, and ``d(loss)/d(cap/load)``
        for rise and fall. The returned tensors are the corresponding
        contributions to ``d(loss)/d(length_um)`` in loss units per
        micrometre, plus their sum.

        A cached calibration is applied before differentiation. Constant
        teacher replacements therefore have zero length derivative, while
        multiplicative load corrections remain in the chain. The result is a
        diagnostic; the published Star delay weight keeps its existing
        ``d(loss)/d(delay)`` contract.
        """

        # The timing forward was run under no_grad.  Recreate only the small
        # RC graph needed for this refresh, while keeping the original pair
        # length tensor untouched when it already belongs to another graph.
        # Refresh callers commonly run under ``torch.no_grad()`` because the
        # timing forward itself is detached.  The small RC replay is the one
        # deliberate exception: it needs a local graph for this VJP.
        with torch.enable_grad():
            rc_length = length_um.detach().clone().requires_grad_(True)
            _, raw_loads, raw_delays, _, _, raw_impulses = self.rc_forward(
                rc_length, base_caps
            )
            loads = raw_loads
            delays = raw_delays
            impulses = raw_impulses
            if calibration is not None:
                if position is None or cap_digest is None:
                    raise ValueError(
                        "calibrated channel pullback requires position and cap_digest"
                    )
                loads, delays, impulses = calibration.apply(
                    star=self,
                    position=position,
                    cap_digest=cap_digest,
                    loads=loads,
                    delays=delays,
                    impulses=impulses,
                    timing_mode_identity=timing_mode_identity,
                    clock_policy_digest=clock_policy_digest,
                )

            outputs = {
                "delay": delays,
                "impulse": impulses,
                "cap": loads,
            }
            result: Dict[str, torch.Tensor] = {}
            for group in ("delay", "impulse", "cap"):
                output_map = outputs[group]
                output_values = (output_map["rise"], output_map["fall"])
                output_grads = tuple(
                    input_gradients[group][edge].to(
                        device=output_values[index].device,
                        dtype=output_values[index].dtype,
                    )
                    for index, edge in enumerate(("rise", "fall"))
                )
                result[group] = torch.autograd.grad(
                    outputs=output_values,
                    inputs=rc_length,
                    grad_outputs=output_grads,
                    retain_graph=True,
                )[0].detach()

        result["full"] = result["delay"] + result["impulse"] + result["cap"]
        return result

    @torch.no_grad()
    def normalize_load_length_weight(
        self,
        load_length_weight: torch.Tensor,
        delay_weight: torch.Tensor,
        length_um: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
        *,
        mode: str = "none",
        alpha: float = 1.0,
        eps: float = 1.0e-30,
    ):
        """Scale B against the accumulated delay length slope at one refresh.

        ``delay_weight * elmore_length_slope`` and ``load_length_weight``
        both have units ps/um. Their RMS values use the same pair domain;
        common zero entries cancel in the ratio. This matches length-space
        slopes, before geometry, pin-to-cell aggregation and preconditioning.
        All coefficients and this scale are frozen until the next refresh.
        """
        mode = str(mode or "none").strip().lower()
        if mode not in ("none", "delay_slope_rms"):
            raise ValueError("unsupported star load normalization: %s" % mode)
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise ValueError("star load alpha must be finite and non-negative")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("star load normalization epsilon must be finite and positive")
        if load_length_weight.shape != delay_weight.shape or load_length_weight.shape != length_um.shape:
            raise ValueError("star load normalization tensors must have equal shapes")
        if load_length_weight.device != delay_weight.device or load_length_weight.device != length_um.device:
            raise ValueError("star load normalization tensors must share a device")
        count = max(1, load_length_weight.numel())
        # Accumulate norms in double to avoid float32 square overflow/underflow.
        load_rms = torch.linalg.vector_norm(load_length_weight, dtype=torch.float64) / math.sqrt(count)
        delay_force = delay_weight.detach() * self.elmore_length_slope(length_um, base_caps)
        delay_rms = torch.linalg.vector_norm(delay_force, dtype=torch.float64) / math.sqrt(count)
        if not bool(torch.isfinite(torch.stack((load_rms, delay_rms))).all().item()):
            raise RuntimeError("star load normalization received non-finite slopes")
        if mode == "none":
            scale = load_rms.new_tensor(1.0)
        elif bool((load_rms > eps).item()):
            scale = delay_rms / load_rms
        else:
            scale = load_rms.new_tensor(0.0)
        used = (load_length_weight.double() * scale * alpha).to(load_length_weight.dtype)
        if not bool(torch.isfinite(used).all().item()):
            raise RuntimeError("star load normalization produced non-finite coefficients")
        return used, {
            "mode": mode,
            "alpha": alpha,
            "delay_force_rms": float(delay_rms.item()),
            "raw_load_length_rms": float(load_rms.item()),
            "applied_load_scale": float((scale * alpha).item()),
            "used_load_length_rms": float((torch.linalg.vector_norm(used, dtype=torch.float64) / math.sqrt(count)).item()),
        }

    def structured_delay_objective(
        self,
        pin_pos: torch.Tensor,
        frozen_weight: Optional[torch.Tensor] = None,
        *,
        base_caps: Mapping[str, torch.Tensor],
        distance_model="l1",
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Return the live Elmore surrogate plus the optional frozen load term.

        ``frozen_weight[e]`` is ``K_e = d(-TNS)/d d_e`` at the last refresh.
        ``d_e(L)`` is the generic-mode depth-one Elmore delay already used
        by ``rc_forward``: ``r L (C_sink + 0.5 c L)``.  The quadratic term
        therefore lives in the placement loss, not only in the refresh
        slope.  ``K`` is detached; current distance ``L`` is not.  The
        selected distance model affects only this live surrogate; physical
        RC refreshes continue to use ``geometry``'s Manhattan length.
        When published, ``B = J_load,L^T dF/dload`` adds ``sum(B * L)``;
        in the production FULL8 flow, ``B`` is projected from accumulated
        pin-cap adjoints ``H`` and optionally clamped to ``B^+ = max(B, 0)``.
        """
        geometry = self.surrogate_geometry(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        weight = self.frozen_weight if frozen_weight is None else frozen_weight
        pair_count = int(geometry["length_um"].numel())
        if int(weight.numel()) != pair_count:
            raise RuntimeError(
                "star delay objective requires a published adjoint per pair"
            )
        if weight.device != geometry["length_um"].device:
            raise RuntimeError(
                "star frozen delay adjoints must reside on the geometry device"
            )
        if pair_count == 0:
            return geometry["length_um"].new_zeros(())
        pair_delay = self._generic_pair_delay_from_length(
            geometry["length_um"], base_caps
        )
        return self._weighted_pair_delay_objective(
            pair_delay,
            geometry["length_um"],
            weight,
            pin_pos=pin_pos,
            smoothing_gamma=smoothing_gamma,
        )

    def get_or_set_rcwl_anchor_length(
        self,
        pin_pos: torch.Tensor,
        *,
        distance_model="l1",
        smoothing_gamma=None,
        surrogate_length_um: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return frozen surrogate anchor length for the active snapshot generation."""
        model = self._normalize_distance_model(distance_model)
        gamma_val = (
            float(smoothing_gamma.detach().reshape(-1)[0].item())
            if torch.is_tensor(smoothing_gamma)
            else (None if smoothing_gamma is None else float(smoothing_gamma))
        )
        cache_key = (int(self.snapshot_generation), str(model), gamma_val)
        anchor_len = getattr(self, "frozen_surrogate_length_um", None)
        pair_count = int(self.sink_pin_ids.numel())
        if (
            getattr(self, "_rcwl_anchor_cache_key", None) != cache_key
            or not torch.is_tensor(anchor_len)
            or int(anchor_len.numel()) != pair_count
        ):
            if surrogate_length_um is None:
                surrogate_length_um = self.surrogate_geometry(
                    pin_pos,
                    distance_model=model,
                    smoothing_gamma=smoothing_gamma,
                )["length_um"]
            anchor_len = surrogate_length_um.detach().clone()
            self.frozen_surrogate_length_um = anchor_len
            self._rcwl_anchor_cache_key = cache_key
        return anchor_len

    def rcwl_linear_objective(
        self,
        pin_pos: torch.Tensor,
        frozen_weight: Optional[torch.Tensor] = None,
        *,
        base_caps: Mapping[str, torch.Tensor],
        distance_model="l1",
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Anchor-linearized pure first-order Star RCWL objective."""
        geometry = self.surrogate_geometry(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        weight = self.frozen_weight if frozen_weight is None else frozen_weight
        pair_count = int(geometry["length_um"].numel())
        if int(weight.numel()) != pair_count:
            raise RuntimeError(
                "star rcwl objective requires a published adjoint per pair"
            )
        if weight.device != geometry["length_um"].device:
            raise RuntimeError(
                "star frozen delay adjoints must reside on the geometry device"
            )
        if pair_count == 0:
            return geometry["length_um"].new_zeros(())
        anchor_len = self.get_or_set_rcwl_anchor_length(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
            surrogate_length_um=geometry["length_um"],
        )
        anchor_slope = self.elmore_length_slope(anchor_len.detach(), base_caps)
        pair_delay_linear = (
            anchor_slope.detach().to(dtype=geometry["length_um"].dtype)
            * geometry["length_um"]
        )
        return self._weighted_pair_delay_objective(
            pair_delay_linear,
            geometry["length_um"],
            weight,
            pin_pos=pin_pos,
            smoothing_gamma=smoothing_gamma,
        )

    def _weighted_pair_delay_objective(
        self,
        pair_delay: torch.Tensor,
        length_um: torch.Tensor,
        weight: torch.Tensor,
        *,
        pin_pos: Optional[torch.Tensor] = None,
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Apply one published delay/load snapshot to live geometry."""
        pair_count = int(pair_delay.numel())
        if int(weight.numel()) != pair_count:
            raise RuntimeError(
                "star delay objective requires a published adjoint per pair"
            )
        if weight.device != pair_delay.device or length_um.device != pair_delay.device:
            raise RuntimeError("star delay objective tensors must share one device")
        if pair_count == 0:
            return length_um.new_zeros(())
        objective = torch.sum(pair_delay * weight.detach().to(dtype=pair_delay.dtype))
        load_weight = self.frozen_load_length_weight
        if load_weight.numel():
            if load_weight.shape != weight.shape or load_weight.device != weight.device:
                raise RuntimeError("star load coefficients must match the delay snapshot")
            # B has units ps/um. In the production FULL8 configuration it is
            # projected from accumulated pin-cap adjoints and optionally clamped
            # to B^+ = max(B, 0) at publication. Additive anchor constants do
            # not affect placement, so use B*L directly.
            if getattr(self, "load_geometry", "star") == "wa_net":
                if pin_pos is None or smoothing_gamma is None:
                    raise ValueError(
                        "wa_net load_geometry requires pin_pos and smoothing_gamma"
                    )
                net_load = self._aggregate_pair_load_to_active_nets(
                    load_weight.detach()
                )
                net_wa_um = self.net_wa_wirelength_um(pin_pos, smoothing_gamma)
                objective = objective + torch.sum(
                    net_wa_um * net_load.to(dtype=net_wa_um.dtype)
                )
            else:
                objective = objective + torch.sum(
                    length_um * load_weight.detach()
                )
        return objective

    def structured_impulse_objective(
        self,
        pin_pos: torch.Tensor,
        *,
        base_caps: Mapping[str, torch.Tensor],
        distance_model="l1",
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Add the tangent-neutral live sink-impulse channel."""
        weight = self.frozen_impulse_length_weight
        anchor = self.frozen_impulse_anchor
        if weight.numel() == 0 and anchor.numel() == 0:
            return pin_pos.new_zeros(())
        pair_count = int(self.sink_pin_ids.numel())
        if weight.shape != (pair_count,) or anchor.shape != (pair_count,):
            raise RuntimeError("star impulse snapshot shape is invalid")
        geometry = self.surrogate_geometry(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        _, _, _, _, _, impulses = self.rc_forward(
            geometry["length_um"], base_caps
        )
        current = impulses["generic"][self.sink_pin_ids]
        return torch.sum(
            weight.detach()
            * (current - anchor.detach()).to(dtype=current.dtype)
        )

    def _generic_pair_delay_from_length(
        self,
        length_um: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        _caps, _loads, delays, _ldelays, _betas, _impulses = self.rc_forward(
            length_um, base_caps
        )
        return delays["generic"][self.sink_pin_ids]

    def current_generic_pair_delay(
        self,
        pin_pos: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
        *,
        distance_model="l1",
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Return the live generic Star delay in the placement geometry."""
        geometry = self.surrogate_geometry(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        return self._generic_pair_delay_from_length(
            geometry["length_um"], base_caps
        )

    def live_endpoint_hinge_objective(
        self,
        pin_pos: torch.Tensor,
        *,
        base_caps: Mapping[str, torch.Tensor],
        tau_ps: float,
        distance_model="l1",
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Evaluate the fixed-within-block endpoint hinge snapshot."""
        snapshot = self.live_endpoint_hinge_snapshot
        if snapshot is None:
            raise RuntimeError("live endpoint hinge snapshot is unavailable")
        snapshot.validate_for_pair_state(
            pair_identity_hash=self.pair_identity_hash,
            snapshot_generation=self.snapshot_generation,
            pair_count=int(self.sink_pin_ids.numel()),
        )
        pair_delay = self.current_generic_pair_delay(
            pin_pos,
            base_caps,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        return snapshot.objective(pair_delay, tau_ps=tau_ps)

    def live_endpoint_hinge_correction_objective(
        self,
        pin_pos: torch.Tensor,
        *,
        base_caps: Mapping[str, torch.Tensor],
        tau_ps: float,
        correction_scale: float = 1.0,
        distance_model="l1",
        smoothing_gamma=None,
    ) -> torch.Tensor:
        """Add a tangent-neutral live endpoint correction to memory weights."""
        scale = float(correction_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("live endpoint correction scale must be finite and non-negative")
        snapshot = self.live_endpoint_hinge_snapshot
        if snapshot is None:
            raise RuntimeError("live endpoint hinge snapshot is unavailable")
        snapshot.validate_for_pair_state(
            pair_identity_hash=self.pair_identity_hash,
            snapshot_generation=self.snapshot_generation,
            pair_count=int(self.sink_pin_ids.numel()),
        )
        geometry = self.surrogate_geometry(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        pair_delay = self._generic_pair_delay_from_length(
            geometry["length_um"], base_caps
        )
        memory_objective = self._weighted_pair_delay_objective(
            pair_delay,
            geometry["length_um"],
            self.frozen_weight,
            pin_pos=pin_pos,
            smoothing_gamma=smoothing_gamma,
        )
        correction = snapshot.tangent_neutral_correction(
            pair_delay, tau_ps=tau_ps
        )
        return memory_objective + pair_delay.new_tensor(scale) * correction

    def load_length_adjoint(
        self,
        cap_gradients,
        calibration=None,
        *,
        load_geometry: Optional[str] = None,
    ):
        """Return J_load,L^T H (ps/um), including frozen root calibration.

        In ``star`` mode, transposes the depth-one star load construction in
        ``rc_forward``.  In ``wa_net`` mode, decouples driver wire-load onto
        each net's smooth WA bounding-box length with unit Hanan boundary
        slope, distributing ``B_drv,e / m_e`` evenly across the ``m_e`` pair
        slots of net ``e`` without fanout-multiplicity inflation.
        """
        geom_mode = (
            self.load_geometry
            if load_geometry is None
            else self._normalize_load_geometry(load_geometry)
        )
        result = None
        for mode in ("rise", "fall"):
            seed = cap_gradients[mode].detach().clone()
            if seed.ndim != 1 or seed.numel() != self.num_pins:
                raise ValueError("load adjoints must use the star pin domain")
            if seed.device != self.sink_pin_ids.device:
                raise RuntimeError("load adjoints must share the star device")
            if geom_mode == "wa_net":
                if self.pair_net_ids.numel() == 0:
                    projected = seed.new_zeros((0,))
                else:
                    # By Hanan's RSMT strip theorem, moving an extreme boundary
                    # pin of any k-pin net changes RSMT wirelength with unit
                    # slope (d L_RSMT / d x_max = 1.0 = d HPWL / d x_max),
                    # independent of internal Steiner branches among interior pins.
                    drv_seed = seed.index_select(0, self.driver_pin_ids)
                    drv_term = drv_seed / self.wa_pair_net_degree.to(
                        device=seed.device, dtype=seed.dtype
                    )
                    snk_term = 0.5 * seed.index_select(0, self.sink_pin_ids)
                    projected = self.c_unit * (drv_term + snk_term)
            else:
                if calibration is not None and self.pair_net_ids.numel():
                    nets = torch.unique(self.pair_net_ids, sorted=True)
                    drivers = self.net_driver_pin_ids[nets]
                    scale = calibration.root_load_scale[mode].to(seed)
                    seed[drivers] *= scale[nets]
                cap_seed = seed.scatter_add(
                    0, self.sink_pin_ids, seed[self.driver_pin_ids]
                )
                projected = (0.5 * self.c_unit) * (
                    cap_seed[self.driver_pin_ids] + cap_seed[self.sink_pin_ids]
                )
            result = projected if result is None else result + projected
        return result

    def quadratic_pair_objective(
        self,
        pin_pos: torch.Tensor,
        frozen_weight: Optional[torch.Tensor] = None,
        *,
        include_load_length: bool = False,
        distance_model: str = "l1",
        smoothing_gamma: Optional[float] = None,
    ) -> torch.Tensor:
        """Return frozen pair weights times Efficient-TDP ``||Δp||²``.

        ``Δp`` is the live driver-to-sink displacement in placement
        coordinates, matching Efficient-TDP pin2pin ``w(dx²+dy²)``.  ``K``
        is detached; current coordinates are not.  ``include_load_length``
        preserves the separately derived ``B L`` load channel so a geometry
        ablation can keep both published coefficient channels fixed.
        """
        geometry = self.surrogate_geometry(
            pin_pos,
            distance_model=distance_model,
            smoothing_gamma=smoothing_gamma,
        )
        weight = self.frozen_weight if frozen_weight is None else frozen_weight
        pair_count = int(geometry["raw_dx"].numel())
        if int(weight.numel()) != pair_count:
            raise RuntimeError(
                "star quadratic objective requires a published adjoint per pair"
            )
        if weight.device != geometry["raw_dx"].device:
            raise RuntimeError(
                "star frozen delay adjoints must reside on the geometry device"
            )
        if pair_count == 0:
            return geometry["raw_dx"].new_zeros(())
        square_length = geometry["raw_dx"].square() + geometry["raw_dy"].square()
        objective = torch.sum(
            square_length * weight.detach().to(dtype=square_length.dtype)
        )
        if include_load_length:
            load_weight = self.frozen_load_length_weight
            if load_weight.numel():
                if (
                    load_weight.shape != weight.shape
                    or load_weight.device != weight.device
                ):
                    raise RuntimeError(
                        "star load coefficients must match the quadratic snapshot"
                    )
                if getattr(self, "load_geometry", "star") == "wa_net":
                    if smoothing_gamma is None:
                        raise ValueError(
                            "wa_net load_geometry requires smoothing_gamma"
                        )
                    net_load = self._aggregate_pair_load_to_active_nets(
                        load_weight.detach()
                    )
                    net_wa_um = self.net_wa_wirelength_um(
                        pin_pos, smoothing_gamma
                    )
                    objective = objective + torch.sum(
                        net_wa_um * net_load.to(dtype=net_wa_um.dtype)
                    )
                else:
                    objective = objective + torch.sum(
                        geometry["length_um"]
                        * load_weight.detach().to(
                            dtype=geometry["length_um"].dtype
                        )
                    )
        return objective

    @torch.no_grad()
    def match_quadratic_channel_weight(
        self,
        pin_pos: torch.Tensor,
        weight: torch.Tensor,
        *,
        base_caps: Mapping[str, torch.Tensor],
        pin2node_map: torch.Tensor,
        num_movable_nodes: int,
        num_nodes: int,
        distance_model: str = "l1",
        smoothing_gamma: Optional[float] = None,
        eps: float = 1.0e-30,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Scale ``weight`` so ``K*(dx²+dy²)`` matches ``K*d(L)`` movable L1 force.

        ``d(L)`` has units of picoseconds while ``dx²+dy²`` has units of DBU².
        Without channel conversion, adding ``K*(dx²+dy²)`` to ``B*L`` (in ps)
        under a single global ``lambda`` dilutes the load and clock channels by
        ~1e-5.  Matching the movable-node L1 gradient norm of ``K*(dx²+dy²)`` to
        that of the Elmore delay channel ``K*d(L)`` at the publication snapshot
        converts the quadratic geometry into ps while keeping ``B*L`` and clock
        at their exact physical scale.
        """
        pair_count = int(self.sink_pin_ids.numel())
        if int(weight.numel()) != pair_count:
            raise ValueError("weight length must match star pair count")
        movable_count = int(num_movable_nodes)
        node_count = int(num_nodes)
        if not 0 <= movable_count <= node_count:
            raise ValueError("invalid movable/total node counts")
        if pair_count == 0 or movable_count == 0:
            return weight.detach().clone(), {
                "enabled": True,
                "elmore_delay_movable_l1": 0.0,
                "raw_quadratic_movable_l1": 0.0,
                "applied_quadratic_scale": 0.0,
                "matched_quadratic_movable_l1": 0.0,
            }
        node_map = (
            torch.as_tensor(pin2node_map, device=pin_pos.device, dtype=torch.long)
            .reshape(-1)
            .expand(2, -1)
        )
        with torch.enable_grad():
            local_pin_pos = pin_pos.detach().clone().requires_grad_(True)
            geometry = self.surrogate_geometry(
                local_pin_pos,
                distance_model=distance_model,
                smoothing_gamma=smoothing_gamma,
            )
            pair_delay = self._generic_pair_delay_from_length(
                geometry["length_um"], base_caps
            )
            w_detached = weight.detach().to(
                device=local_pin_pos.device, dtype=pair_delay.dtype
            )
            elmore_obj = torch.sum(pair_delay * w_detached)
            square_length = (
                geometry["raw_dx"].square() + geometry["raw_dy"].square()
            )
            quad_obj = torch.sum(square_length * w_detached)
            elmore_pin_grad = torch.autograd.grad(
                elmore_obj, local_pin_pos, retain_graph=True
            )[0].detach()
            quad_pin_grad = torch.autograd.grad(
                quad_obj, local_pin_pos
            )[0].detach()

        elmore_node = local_pin_pos.new_zeros((2, node_count))
        quad_node = local_pin_pos.new_zeros((2, node_count))
        elmore_node.scatter_add_(1, node_map, elmore_pin_grad.reshape(2, -1))
        quad_node.scatter_add_(1, node_map, quad_pin_grad.reshape(2, -1))
        elmore_l1 = float(
            elmore_node[:, :movable_count].double().abs().sum().item()
        )
        quad_l1 = float(
            quad_node[:, :movable_count].double().abs().sum().item()
        )
        scale = elmore_l1 / quad_l1 if quad_l1 > float(eps) else 0.0
        scaled = (weight.detach().double() * scale).to(dtype=weight.dtype)
        return scaled, {
            "enabled": True,
            "elmore_delay_movable_l1": elmore_l1,
            "raw_quadratic_movable_l1": quad_l1,
            "applied_quadratic_scale": float(scale),
            "matched_quadratic_movable_l1": float(quad_l1 * scale),
        }

    def timing_precondition_diagonal(
        self,
        *,
        pin2node_map: torch.Tensor,
        num_movable_nodes: int,
        num_nodes: int,
        output_template: torch.Tensor,
        trial_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the positive star Elmore quadratic-term diagonal.

        The depth-one Elmore delay contains ``0.5 * r * c * L^2``.  This
        preconditioner uses ``abs(K) * r * c / coordinate_to_um^2`` as a
        shared x/y curvature proxy at each movable endpoint.  Fixed endpoints
        and same-cell pairs are omitted.  The diagonal is piecewise static:
        it is tied to the currently published star snapshot.

        ``trial_weight`` measures an *unpublished* candidate (the refresh probe
        runs before the first snapshot exists).  It skips the published
        metadata guard and never reads or writes the published cache, so a
        candidate can neither be rejected for lacking a snapshot nor poison
        the published diagonal.
        """
        trial = trial_weight is not None
        if not trial:
            self.read_snapshot_metadata()
        if not torch.is_tensor(output_template):
            raise TypeError("output_template must be a tensor")
        template = output_template.reshape(-1)
        try:
            movable_count = int(num_movable_nodes)
            node_count = int(num_nodes)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "num_movable_nodes and num_nodes must be integers"
            ) from exc
        if movable_count < 0 or node_count < 0 or movable_count > node_count:
            raise ValueError("invalid movable/total node counts")
        if int(template.numel()) != node_count:
            raise ValueError("output_template length must equal num_nodes")

        source_pin_map = torch.as_tensor(pin2node_map).reshape(-1)
        source_key = (
            int(source_pin_map.data_ptr()),
            int(getattr(source_pin_map, "_version", 0)),
            str(source_pin_map.device),
            str(source_pin_map.dtype),
            int(source_pin_map.numel()),
            str(template.device),
        )
        if (
            source_pin_map.device == template.device
            and source_pin_map.dtype == torch.long
            and source_pin_map.is_contiguous()
        ):
            pin_map = source_pin_map
        elif source_key == self._timing_precondition_pin_map_source_key:
            pin_map = self._timing_precondition_pin_map_long
        else:
            pin_map = source_pin_map.to(
                device=template.device, dtype=torch.long
            ).contiguous()
            self._timing_precondition_pin_map_source_key = source_key
            self._timing_precondition_pin_map_long = pin_map
        if pin_map is None:
            raise RuntimeError("failed to normalize pin2node_map")
        pin_map = pin_map.reshape(-1)
        pin_map_version = int(getattr(pin_map, "_version", 0))
        weight_version = int(getattr(self.frozen_weight, "_version", 0))
        cache_key = (
            int(self.snapshot_generation),
            int(self.pair_weight_generation),
            int(self.frozen_weight.data_ptr()),
            weight_version,
            int(pin_map.data_ptr()),
            pin_map_version,
            movable_count,
            node_count,
            str(template.device),
            str(template.dtype),
            tuple(template.shape),
        )
        if (
            not trial
            and cache_key == self._timing_precondition_cache_key
            and self._timing_precondition_cache_x is not None
            and self._timing_precondition_cache_y is not None
        ):
            return self._timing_precondition_cache_x, self._timing_precondition_cache_y

        weight = (
            torch.as_tensor(trial_weight).reshape(-1)
            if trial
            else self.frozen_weight.reshape(-1)
        )
        if int(weight.numel()) != int(self.sink_pin_ids.numel()):
            raise RuntimeError("star timing precondition weight shape is stale")
        if weight.device != template.device:
            weight = weight.to(device=template.device)
        weight = weight.to(dtype=template.dtype)
        if not bool(torch.isfinite(weight).all().item()):
            raise RuntimeError(
                "star timing precondition weight contains non-finite value"
            )
        if int(pin_map.numel()) != self.num_pins:
            raise ValueError("pin2node_map length must equal star num_pins")
        if pin_map.numel() and bool(
            ((pin_map < 0) | (pin_map >= node_count)).any().item()
        ):
            raise ValueError("pin2node_map contains out-of-range node IDs")

        driver_pin_ids = self.driver_pin_ids.to(device=template.device)
        sink_pin_ids = self.sink_pin_ids.to(device=template.device)
        driver_nodes = pin_map[driver_pin_ids]
        sink_nodes = pin_map[sink_pin_ids]
        distinct = driver_nodes != sink_nodes
        curvature = weight.abs() * (self.r_unit * self.c_unit) / (
            self.coordinate_to_um * self.coordinate_to_um
        )
        if not bool(torch.isfinite(curvature).all().item()):
            raise RuntimeError("star timing precondition curvature is non-finite")

        diagonal_x = torch.zeros_like(template)
        diagonal_y = torch.zeros_like(template)
        endpoint_weight = curvature * distinct
        for nodes in (driver_nodes, sink_nodes):
            movable = distinct & (nodes < movable_count)
            if bool(movable.any().item()):
                diagonal_x.scatter_add_(0, nodes[movable], endpoint_weight[movable])
                diagonal_y.scatter_add_(0, nodes[movable], endpoint_weight[movable])
        if not bool(torch.isfinite(diagonal_x).all().item()) or not bool(
            torch.isfinite(diagonal_y).all().item()
        ):
            raise RuntimeError("star timing precondition diagonal is non-finite")
        if trial:
            # A candidate measurement must not touch the published cache.
            return diagonal_x.detach(), diagonal_y.detach()
        self._timing_precondition_cache_key = cache_key
        self._timing_precondition_cache_x = diagonal_x.detach()
        self._timing_precondition_cache_y = diagonal_y.detach()
        return self._timing_precondition_cache_x, self._timing_precondition_cache_y

    @torch.no_grad()
    def timing_wa_curvature_diagonal(
        self,
        pin_pos: torch.Tensor,
        *,
        base_caps: Mapping[str, torch.Tensor],
        smoothing_gamma,
        pin2node_map: torch.Tensor,
        num_movable_nodes: int,
        num_nodes: int,
        output_template: torch.Tensor,
        trial_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bound the local WA Elmore/load Hessian on movable coordinates.

        For one pair, ``L = (psi(dx) + psi(dy)) / a`` and
        ``q = K*r*(C_sink+c*L) + B``. Its relative-coordinate Hessian is
        ``K*r*c/a^2 * v*v.T + q/a * diag(psi_second)``, where
        ``v = psi_first``. Absolute row sums, multiplied by the number of
        movable endpoints, give a positive diagonal that dominates the
        assembled local Hessian. Same-cell pairs are constant and omitted.

        This is evaluated at the supplied position, with the published K/B
        held fixed. It is not a finite-step majorizer or a bound on timing
        propagation curvature. No position-dependent result is cached.
        An unpublished trial uses the caller's K and the current trial B,
        without reading metadata or modifying the published RC cache.
        """
        if trial_weight is None:
            self.read_snapshot_metadata()
        template = output_template.reshape(-1)
        movable_count, node_count = int(num_movable_nodes), int(num_nodes)
        if not 0 <= movable_count <= node_count:
            raise ValueError("invalid movable/total node counts")
        if int(template.numel()) != node_count:
            raise ValueError("output_template length must equal num_nodes")
        if pin_pos.device != template.device:
            raise ValueError("pin_pos and output_template must share a device")
        geometry = self.surrogate_geometry(
            pin_pos, distance_model="wa", smoothing_gamma=smoothing_gamma
        )
        relative = torch.stack((geometry["raw_dx"], geometry["raw_dy"]))
        gamma = self._validated_smoothing_gamma(
            smoothing_gamma, device=relative.device, dtype=relative.dtype
        )
        z = relative / (2.0 * gamma)
        tangent = torch.tanh(z)
        sech_square = 1.0 - tangent.square()
        first = tangent + z * sech_square
        second = sech_square * (1.0 - z * tangent) / gamma
        length = geometry["length_um"]
        weight = self.frozen_weight if trial_weight is None else trial_weight
        weight = weight.detach().reshape(-1).to(length)
        if weight.numel() != length.numel():
            raise RuntimeError("star timing curvature weight shape is stale")
        curvature_load = getattr(
            self, "frozen_curvature_load_length_weight", None
        )
        if curvature_load is None:
            curvature_load = getattr(self, "frozen_load_length_weight", None)
        if curvature_load is None:
            curvature_load = self.frozen_weight.new_empty(0)
        load = curvature_load.detach().reshape(-1).to(length)
        if load.numel() == 0 and self._frozen_snapshot_metadata is None:
            # Older offline probes set only ``frozen_load_length_weight`` and
            # predate the separate curvature snapshot.  Preserve their
            # load-in-H behavior until a new snapshot publishes the split.
            load = self.frozen_load_length_weight.detach().reshape(-1).to(length)
        if load.numel() not in (0, length.numel()):
            raise RuntimeError("star timing curvature load shape is stale")
        slope = weight * self.elmore_length_slope(length, base_caps)
        if load.numel() and self.load_geometry != "wa_net":
            slope = slope + load
        rc_rows = (
            weight.abs() * self.r_unit * self.c_unit / self.coordinate_to_um**2
        ).unsqueeze(0) * first.abs() * first.abs().sum(dim=0, keepdim=True)
        geometry_rows = slope.abs().unsqueeze(0) * second.abs() / self.coordinate_to_um
        rows = rc_rows + geometry_rows
        if not bool(torch.isfinite(rows).all().item()):
            raise RuntimeError("star WA timing curvature is non-finite")

        pin_map = torch.as_tensor(
            pin2node_map, device=template.device, dtype=torch.long
        ).reshape(-1)
        if pin_map.numel() != self.num_pins:
            raise ValueError("pin2node_map length must equal star num_pins")
        if bool(((pin_map < 0) | (pin_map >= node_count)).any().item()):
            raise ValueError("pin2node_map contains out-of-range node IDs")
        driver_nodes, sink_nodes = (
            pin_map[self.driver_pin_ids], pin_map[self.sink_pin_ids]
        )
        distinct = driver_nodes != sink_nodes
        driver_movable = distinct & (driver_nodes < movable_count)
        sink_movable = distinct & (sink_nodes < movable_count)
        free_count = driver_movable.to(rows) + sink_movable.to(rows)
        rows = (rows * free_count.unsqueeze(0)).to(template)
        diagonal = template.new_zeros((2, node_count))
        for nodes, movable in (
            (driver_nodes, driver_movable), (sink_nodes, sink_movable)
        ):
            diagonal.scatter_add_(
                1, nodes[movable].unsqueeze(0).expand(2, -1), rows[:, movable]
            )
        if load.numel() and self.load_geometry == "wa_net":
            wa_dx, wa_dy = self.net_wa_load_curvature_diagonal(
                pin_pos,
                load=load,
                smoothing_gamma=smoothing_gamma,
                pin2node_map=pin_map,
                num_movable_nodes=movable_count,
                num_nodes=node_count,
                output_template=template,
            )
            diagonal[0].add_(wa_dx)
            diagonal[1].add_(wa_dy)
        if not bool(torch.isfinite(diagonal).all().item()):
            raise RuntimeError("star WA timing curvature diagonal is non-finite")
        return diagonal[0], diagonal[1]

    @torch.no_grad()
    def net_wa_load_curvature_diagonal(
        self,
        pin_pos: torch.Tensor,
        *,
        load: torch.Tensor,
        smoothing_gamma,
        pin2node_map: torch.Tensor,
        num_movable_nodes: int,
        num_nodes: int,
        output_template: torch.Tensor,
        mm_secant: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact movable-pin Gershgorin diagonal bound for ``sum_e |B_net,e| WA_um,e(x)``.

        When ``mm_secant=True`` (used by the finite-step ``mm_rowsum`` MM
        preconditioner), augments the local Hessian bound with the bounding-box
        secant majorizer ``free_factor * max(|g^+| + |g^-|, p^+ + p^-) / max(span, gamma)``,
        matching ``linear_length_diagonal`` identically on all 2-pin nets while
        assigning zero parasitic curvature to interior non-boundary pins on
        high-fanout nets.
        """
        template = output_template.reshape(-1)
        movable_count, node_count = int(num_movable_nodes), int(num_nodes)
        if not 0 <= movable_count <= node_count:
            raise ValueError("invalid movable/total node counts")
        if int(template.numel()) != node_count:
            raise ValueError("output_template length must equal num_nodes")
        if pin_pos.device != template.device:
            raise ValueError("pin_pos and output_template must share a device")
        pin_map = torch.as_tensor(
            pin2node_map, device=template.device, dtype=torch.long
        ).reshape(-1)
        if int(pin_map.numel()) != self.num_pins:
            raise ValueError("pin2node_map length must equal star num_pins")
        if pin_map.numel() and bool(
            ((pin_map < 0) | (pin_map >= node_count)).any().item()
        ):
            raise ValueError("pin2node_map contains out-of-range node IDs")
        active_count = int(self.wa_active_net_ids.numel())
        load_vec = torch.as_tensor(load, device=template.device).detach().reshape(-1)
        if active_count == 0 or int(load_vec.numel()) == 0 or movable_count == 0:
            zeros = torch.zeros_like(template)
            return zeros, zeros.clone()
        if int(load_vec.numel()) != int(self.sink_pin_ids.numel()):
            raise ValueError("load length must match star pair count")

        x, y = self._split_pin_pos(pin_pos)
        gamma = self._validated_smoothing_gamma(
            smoothing_gamma, device=x.device, dtype=x.dtype
        )
        inc_nets = self.wa_inc_compact_net_ids
        inc_pins = self.wa_inc_pin_ids
        active_nets = self.wa_active_net_ids
        if inc_nets.device != x.device:
            inc_nets = inc_nets.to(device=x.device)
            inc_pins = inc_pins.to(device=x.device)
            active_nets = active_nets.to(device=x.device)
        drv_pins = self.net_driver_pin_ids.to(device=x.device).index_select(
            0, active_nets
        )

        inc_nodes = pin_map.index_select(0, inc_pins)
        drv_nodes_active = pin_map.index_select(0, drv_pins)
        max_node = drv_nodes_active.clone().scatter_reduce_(
            0, inc_nets, inc_nodes, reduce="amax", include_self=True
        )
        min_node = drv_nodes_active.clone().scatter_reduce_(
            0, inc_nets, inc_nodes, reduce="amin", include_self=True
        )
        mov_inc = inc_nodes < movable_count
        if not bool(mov_inc.any().item()):
            zeros = torch.zeros_like(template)
            return zeros, zeros.clone()

        net_load_abs = self._aggregate_pair_load_to_active_nets(
            load_vec.to(dtype=x.dtype)
        ).abs()
        multi_node = (max_node != min_node).to(dtype=x.dtype)
        net_scale = (net_load_abs * multi_node) / self.coordinate_to_um
        mov_nets = inc_nets[mov_inc]
        mov_nodes = inc_nodes[mov_inc]
        mov_scale = net_scale.index_select(0, mov_nets)
        free_factor_mov = None
        if mm_secant:
            mov_pin_count = x.new_zeros((active_count,)).scatter_add(
                0, mov_nets, x.new_ones((int(mov_nets.numel()),))
            )
            free_factor_mov = mov_pin_count.clamp(1.0, 2.0).index_select(
                0, mov_nets
            )

        def _axis_diag(u: torch.Tensor) -> torch.Tensor:
            u_inc = u.index_select(0, inc_pins)
            u_drv = u.index_select(0, drv_pins)
            c_max = u_drv.clone().scatter_reduce_(
                0, inc_nets, u_inc, reduce="amax", include_self=True
            )
            c_min = u_drv.clone().scatter_reduce_(
                0, inc_nets, u_inc, reduce="amin", include_self=True
            )
            pin_curv = u.new_zeros((int(mov_nets.numel()),))
            ga_sum = u.new_zeros((int(mov_nets.numel()),)) if mm_secant else None
            p_sum = u.new_zeros((int(mov_nets.numel()),)) if mm_secant else None
            for delta in (
                u_inc - c_max.index_select(0, inc_nets),
                c_min.index_select(0, inc_nets) - u_inc,
            ):
                w = torch.exp(delta / gamma)
                z = u.new_zeros((active_count,)).scatter_add(0, inc_nets, w)
                p = w / z.index_select(0, inc_nets)
                mu_tilde = u.new_zeros((active_count,)).scatter_add(
                    0, inc_nets, p * delta
                )
                g = p * (
                    1.0 + (delta - mu_tilde.index_select(0, inc_nets)) / gamma
                )
                g_abs = g.abs()
                p_m = p[mov_inc]
                g_m = g[mov_inc]
                ga_m = g_abs[mov_inc]
                if mm_secant:
                    ga_sum = ga_sum + ga_m
                    p_sum = p_sum + p_m
                p_mov_sum = u.new_zeros((active_count,)).scatter_add(
                    0, mov_nets, p_m
                )
                g_mov_sum = u.new_zeros((active_count,)).scatter_add(
                    0, mov_nets, ga_m
                )
                diag_abs = (g_m * (1.0 - p_m) + p_m * (1.0 - g_m)).abs()
                off_abs = ga_m * (
                    p_mov_sum.index_select(0, mov_nets) - p_m
                ).clamp(min=0.0) + p_m * (
                    g_mov_sum.index_select(0, mov_nets) - ga_m
                ).clamp(min=0.0)
                pin_curv = pin_curv + (diag_abs + off_abs) / gamma
            if mm_secant:
                span_mov = (c_max - c_min).clamp(min=gamma).index_select(
                    0, mov_nets
                )
                pin_secant = (
                    free_factor_mov * torch.maximum(ga_sum, p_sum) / span_mov
                )
                pin_curv = torch.maximum(pin_curv, pin_secant)
            out = torch.zeros_like(template)
            out.scatter_add_(
                0, mov_nodes, (pin_curv * mov_scale).to(dtype=template.dtype)
            )
            return out

        diag_x = _axis_diag(x)
        diag_y = _axis_diag(y)
        if not bool(torch.isfinite(diag_x).all().item()) or not bool(
            torch.isfinite(diag_y).all().item()
        ):
            raise RuntimeError("star net WA load curvature diagonal is non-finite")
        return diag_x, diag_y

    def _normalized_base_caps(
        self, base_caps: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(base_caps, Mapping):
            raise TypeError("base_caps must map generic/rise/fall to tensors")
        result: Dict[str, torch.Tensor] = {}
        for mode in _TIMING_MODES:
            cap = base_caps.get(mode)
            if not torch.is_tensor(cap):
                raise ValueError("base_caps requires a %s tensor" % mode)
            cap = cap.reshape(-1)
            if int(cap.numel()) != self.num_pins:
                raise ValueError(
                    "base %s cap length must equal num_pins" % mode
                )
            result[mode] = cap
        return result

    @staticmethod
    def _validate_rc_tensor(name: str, value: torch.Tensor) -> None:
        if not bool(torch.all(torch.isfinite(value)).item()):
            raise RuntimeError("Non-finite star RC %s" % name)

    def rc_forward(
        self,
        length_um: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
        *,
        validate: bool = False,
    ):
        """Evaluate depth-one RC on the device of ``length_um``.

        The return tuple intentionally matches ``RCTiming.forward``:
        ``pin_caps, loads, delays, ldelays, betas, impulses``.
        """
        if not torch.is_tensor(length_um) or length_um.dim() != 1:
            raise TypeError("length_um must be a one-dimensional tensor")
        if int(length_um.numel()) != int(self.sink_pin_ids.numel()):
            raise ValueError("length_um must contain one value per star pair")
        caps_by_mode = self._normalized_base_caps(base_caps)
        first_cap = caps_by_mode["generic"]
        if first_cap.device != length_um.device:
            raise RuntimeError("star RC length and base caps must share a device")
        for cap in caps_by_mode.values():
            if cap.device != length_um.device:
                raise RuntimeError("all star RC base caps must share one device")
        driver_ids = self.driver_pin_ids
        sink_ids = self.sink_pin_ids
        if driver_ids.device != length_um.device or sink_ids.device != length_um.device:
            raise RuntimeError("star pair buffers must share the length device")

        # ``base_cap`` is pF and this recurrence produces ps, so c_unit must
        # be normalized to pF/um before StarPinPair construction.
        wire_cap = length_um * self.c_unit
        resistance = length_um * self.r_unit
        half_wire_cap = wire_cap * 0.5
        pin_caps: Dict[str, torch.Tensor] = {}
        loads: Dict[str, torch.Tensor] = {}
        delays: Dict[str, torch.Tensor] = {}
        ldelays: Dict[str, torch.Tensor] = {}
        betas: Dict[str, torch.Tensor] = {}
        impulses: Dict[str, torch.Tensor] = {}
        for mode in _TIMING_MODES:
            base_cap = caps_by_mode[mode]
            net_cap = torch.zeros_like(base_cap)
            net_cap = net_cap.scatter_add(0, driver_ids, half_wire_cap)
            net_cap = net_cap.scatter_add(0, sink_ids, half_wire_cap)
            cap = base_cap + net_cap
            sink_cap = cap[sink_ids]
            load = cap.scatter_add(0, driver_ids, sink_cap)
            delay = torch.zeros_like(cap).scatter(0, sink_ids, resistance * sink_cap)
            leaf_ldelay = cap * delay
            ldelay = leaf_ldelay.scatter_add(0, driver_ids, leaf_ldelay[sink_ids])
            beta = torch.zeros_like(cap).scatter(
                0, sink_ids, resistance * ldelay[sink_ids]
            )
            impulse = 2.0 * beta - delay.square()
            pin_caps[mode] = cap
            loads[mode] = load
            delays[mode] = delay
            ldelays[mode] = ldelay
            betas[mode] = beta
            impulses[mode] = impulse
            if validate:
                for name, value in (
                    ("%s_cap" % mode, cap),
                    ("%s_load" % mode, load),
                    ("%s_delay" % mode, delay),
                    ("%s_ldelay" % mode, ldelay),
                    ("%s_beta" % mode, beta),
                    ("%s_impulse" % mode, impulse),
                ):
                    self._validate_rc_tensor(name, value)
                if not bool(torch.all(impulse >= 0).item()):
                    raise RuntimeError("Negative star RC impulse in %s mode" % mode)
        return pin_caps, loads, delays, ldelays, betas, impulses

    def forward(
        self,
        pin_pos: torch.Tensor,
        base_caps: Mapping[str, torch.Tensor],
        *,
        validate: bool = False,
    ):
        geometry = self.geometry(pin_pos)
        result = self.rc_forward(
            geometry["length_um"], base_caps, validate=validate
        )
        return (*result, geometry)

    def prepare_snapshot(
        self,
        *,
        length_um: torch.Tensor,
        raw_weight: torch.Tensor,
        load_length_weight: Optional[torch.Tensor] = None,
        curvature_load_length_weight: Optional[torch.Tensor] = None,
        impulse_length_weight: Optional[torch.Tensor] = None,
        impulse_anchor: Optional[torch.Tensor] = None,
        live_endpoint_hinge_snapshot: Optional[object] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> Dict[str, object]:
        """Validate a candidate without mutating the currently frozen state."""
        if not torch.is_tensor(length_um) or not torch.is_tensor(raw_weight):
            raise TypeError("star snapshot requires tensor length and raw weight")
        if length_um.dim() != 1 or raw_weight.dim() != 1:
            raise ValueError("star snapshot tensors must be one-dimensional")
        pair_count = int(self.sink_pin_ids.numel())
        if int(length_um.numel()) != pair_count or int(raw_weight.numel()) != pair_count:
            raise ValueError("star snapshot tensor shape does not match pair count")
        if length_um.device != raw_weight.device:
            raise RuntimeError("star snapshot length and weight must share a device")
        if length_um.device != self.sink_pin_ids.device:
            raise RuntimeError("star snapshot tensors must share the pair-buffer device")
        if not bool(torch.all(torch.isfinite(length_um)).item()):
            raise RuntimeError("star snapshot length contains non-finite values")
        if not bool(torch.all(torch.isfinite(raw_weight)).item()):
            raise RuntimeError("star snapshot raw weight contains non-finite values")
        if bool(torch.any(length_um < 0).item()):
            raise RuntimeError("star snapshot length contains negative values")
        load_weight = self._validated_load_snapshot(load_length_weight, raw_weight)
        curvature_load_weight = (
            load_weight
            if curvature_load_length_weight is None
            else self._validated_load_snapshot(
                curvature_load_length_weight, raw_weight
            )
        )
        impulse_weight = self._validated_load_snapshot(
            impulse_length_weight, raw_weight
        )
        impulse_anchor_value = self._validated_load_snapshot(
            impulse_anchor, raw_weight
        )
        if impulse_weight.numel() != impulse_anchor_value.numel():
            raise ValueError(
                "star impulse weight and anchor must have matching lengths"
            )
        if live_endpoint_hinge_snapshot is not None:
            validator = getattr(
                live_endpoint_hinge_snapshot, "validate_for_pair_state", None
            )
            if not callable(validator):
                raise TypeError(
                    "live_endpoint_hinge_snapshot must implement validate_for_pair_state"
                )
            validator(
                pair_identity_hash=self.pair_identity_hash,
                snapshot_generation=self.snapshot_generation + 1,
                pair_count=pair_count,
            )
            if live_endpoint_hinge_snapshot.anchor_pair_delay.device != length_um.device:
                raise RuntimeError(
                    "live endpoint snapshot must share the Star snapshot device"
                )
        extra = dict(metadata or {})
        return {
            "length_um": length_um.detach().clone(),
            "raw_signed_weight": raw_weight.detach().clone(),
            "load_length_weight": load_weight,
            "curvature_load_length_weight": curvature_load_weight,
            "impulse_length_weight": impulse_weight,
            "impulse_anchor": impulse_anchor_value,
            "live_endpoint_hinge_snapshot": live_endpoint_hinge_snapshot,
            "pair_count": pair_count,
            "pair_identity_hash": self.pair_identity_hash,
            "topology_mode": "driver_sink_star",
            "length_unit": "um",
            "metadata": extra,
        }

    @staticmethod
    def _validated_load_snapshot(value, weight):
        if value is None:
            return weight.new_empty(0)
        if (not torch.is_tensor(value) or value.ndim != 1
                or value.numel() not in (0, weight.numel())):
            raise ValueError("star load snapshot must contain one coefficient per pair")
        if value.device != weight.device or not bool(torch.isfinite(value).all().item()):
            raise RuntimeError("star load snapshot has an invalid device or non-finite values")
        return value.detach().to(dtype=weight.dtype).clone()

    def publish_snapshot(self, candidate: Mapping[str, object]) -> Dict[str, object]:
        """Atomically commit a previously validated candidate in Python state."""
        if not isinstance(candidate, Mapping):
            raise TypeError("star snapshot candidate must be a mapping")
        if candidate.get("topology_mode") != "driver_sink_star":
            raise RuntimeError("star snapshot topology mode mismatch")
        if candidate.get("pair_identity_hash") != self.pair_identity_hash:
            raise RuntimeError("star snapshot pair identity mismatch")
        if int(candidate.get("pair_count", -1)) != int(self.sink_pin_ids.numel()):
            raise RuntimeError("star snapshot pair count mismatch")
        length_um = candidate.get("length_um")
        raw_weight = candidate.get("raw_signed_weight")
        if not torch.is_tensor(length_um) or not torch.is_tensor(raw_weight):
            raise RuntimeError("star snapshot candidate lacks tensors")
        pair_count = int(self.sink_pin_ids.numel())
        if (
            length_um.dim() != 1
            or raw_weight.dim() != 1
            or int(length_um.numel()) != pair_count
            or int(raw_weight.numel()) != pair_count
        ):
            raise RuntimeError("star snapshot candidate tensor shape is invalid")
        if (
            length_um.device != raw_weight.device
            or length_um.device != self.sink_pin_ids.device
        ):
            raise RuntimeError("star snapshot candidate tensor device is invalid")
        if not bool(torch.all(torch.isfinite(length_um)).item()):
            raise RuntimeError("star snapshot candidate length contains non-finite values")
        if not bool(torch.all(torch.isfinite(raw_weight)).item()):
            raise RuntimeError("star snapshot candidate weight contains non-finite values")
        if bool(torch.any(length_um < 0).item()):
            raise RuntimeError("star snapshot candidate length contains negative values")
        frozen_load_weight = self._validated_load_snapshot(
            candidate.get("load_length_weight"), raw_weight
        )
        curvature_candidate = candidate.get(
            "curvature_load_length_weight", frozen_load_weight
        )
        frozen_curvature_load_weight = self._validated_load_snapshot(
            curvature_candidate, raw_weight
        )
        frozen_impulse_weight = self._validated_load_snapshot(
            candidate.get("impulse_length_weight"), raw_weight
        )
        frozen_impulse_anchor = self._validated_load_snapshot(
            candidate.get("impulse_anchor"), raw_weight
        )
        if frozen_impulse_weight.numel() != frozen_impulse_anchor.numel():
            raise RuntimeError(
                "star impulse weight and anchor must have matching lengths"
            )
        # The checks above make the public commit boundary robust even for a
        # caller that did not retain the mapping returned by prepare_snapshot.
        # The following assignments are the one commit boundary observed by
        # placement.
        next_weight_generation = self.pair_weight_generation + 1
        next_snapshot_generation = self.snapshot_generation + 1
        live_snapshot = candidate.get("live_endpoint_hinge_snapshot")
        if live_snapshot is not None:
            validator = getattr(live_snapshot, "validate_for_pair_state", None)
            if not callable(validator):
                raise RuntimeError("star snapshot candidate has an invalid live snapshot")
            validator(
                pair_identity_hash=self.pair_identity_hash,
                snapshot_generation=next_snapshot_generation,
                pair_count=pair_count,
            )
            if live_snapshot.anchor_pair_delay.device != raw_weight.device:
                raise RuntimeError(
                    "star snapshot candidate live snapshot device is invalid"
                )
        frozen_length = length_um.detach().clone()
        frozen_weight = raw_weight.detach().clone()
        metadata = dict(candidate.get("metadata") or {})
        metadata.update(
            {
                "topology_mode": "driver_sink_star",
                "load_geometry": self.load_geometry,
                "pair_topology_generation": int(self.pair_topology_generation),
                "pair_weight_generation": int(next_weight_generation),
                "snapshot_generation": int(next_snapshot_generation),
                "pair_count": int(self.sink_pin_ids.numel()),
                "pair_identity_hash": self.pair_identity_hash,
                "identity_namespace": self.identity_namespace,
                "length_unit": "um",
                "weight_device": str(frozen_weight.device),
                "weight_dtype": str(frozen_weight.dtype),
                "length_device": str(frozen_length.device),
                "length_dtype": str(frozen_length.dtype),
                "live_endpoint_hinge_available": live_snapshot is not None,
                "live_endpoint_hinge_endpoint_count": (
                    int(live_snapshot.endpoint_count) if live_snapshot is not None else 0
                ),
                "live_endpoint_hinge_nnz": (
                    int(live_snapshot.nnz) if live_snapshot is not None else 0
                ),
                "load_gradient_available": bool(frozen_load_weight.numel()),
                "load_gradient_l1": (
                    float(frozen_load_weight.abs().sum().item())
                    if frozen_load_weight.numel()
                    else 0.0
                ),
                "load_curvature_available": bool(
                    frozen_curvature_load_weight.numel()
                ),
                "load_curvature_l1": (
                    float(frozen_curvature_load_weight.abs().sum().item())
                    if frozen_curvature_load_weight.numel()
                    else 0.0
                ),
                "impulse_channel_available": bool(
                    frozen_impulse_weight.numel()
                ),
                "impulse_weight_l1": (
                    float(frozen_impulse_weight.abs().sum().item())
                    if frozen_impulse_weight.numel()
                    else 0.0
                ),
            }
        )
        self.frozen_length_um = frozen_length
        self.frozen_weight = frozen_weight
        self.frozen_load_length_weight = frozen_load_weight
        self.frozen_curvature_load_length_weight = frozen_curvature_load_weight
        self.frozen_impulse_length_weight = frozen_impulse_weight
        self.frozen_impulse_anchor = frozen_impulse_anchor
        self.live_endpoint_hinge_snapshot = live_snapshot
        self.pair_weight_generation = next_weight_generation
        self.snapshot_generation = next_snapshot_generation
        self.refresh_count += 1
        self._frozen_snapshot_metadata = metadata
        self._timing_precondition_cache_key = None
        self._timing_precondition_cache_x = None
        self._timing_precondition_cache_y = None
        self._rcwl_anchor_cache_key = None
        self.frozen_surrogate_length_um = None
        return dict(metadata)

    def read_snapshot_metadata(self) -> Dict[str, object]:
        if self._frozen_snapshot_metadata is None:
            raise RuntimeError("star timing snapshot is unavailable")
        if int(self.frozen_weight.numel()) != int(self.sink_pin_ids.numel()):
            raise RuntimeError("star timing snapshot weight shape is stale")
        if int(self.frozen_length_um.numel()) != int(self.sink_pin_ids.numel()):
            raise RuntimeError("star timing snapshot length shape is stale")
        curvature_load = getattr(
            self, "frozen_curvature_load_length_weight", None
        )
        if curvature_load is None:
            curvature_load = getattr(self, "frozen_load_length_weight", None)
        if curvature_load is None:
            curvature_load = self.frozen_weight.new_empty(0)
        if curvature_load.numel() not in (
            0,
            int(self.sink_pin_ids.numel()),
        ):
            raise RuntimeError("star timing curvature load shape is stale")
        metadata = dict(self._frozen_snapshot_metadata)
        if metadata.get("pair_identity_hash") != self.pair_identity_hash:
            raise RuntimeError("star timing snapshot pair identity is stale")
        if int(metadata.get("pair_weight_generation", -1)) != int(
            self.pair_weight_generation
        ):
            raise RuntimeError("star timing snapshot weight generation is stale")
        if int(metadata.get("snapshot_generation", -1)) != int(
            self.snapshot_generation
        ):
            raise RuntimeError("star timing snapshot generation is stale")
        if bool(metadata.get("live_endpoint_hinge_available", False)):
            if self.live_endpoint_hinge_snapshot is None:
                raise RuntimeError("star live endpoint snapshot is stale")
            self.live_endpoint_hinge_snapshot.validate_for_pair_state(
                pair_identity_hash=self.pair_identity_hash,
                snapshot_generation=self.snapshot_generation,
                pair_count=int(self.sink_pin_ids.numel()),
            )
        return metadata
