import copy

import torch


_PROFILE_NAME = "proximal_alternating_v1"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def proximal_profile_enabled(params):
    profile = str(getattr(params, "joint_quality_profile", "") or "")
    explicit = str(getattr(params, "joint_proximal_enabled", "")).strip().lower()
    return profile == _PROFILE_NAME or explicit in _TRUE_VALUES


def _float_param(params, name, default=0.0):
    try:
        return float(getattr(params, name, default) or 0.0)
    except (TypeError, ValueError):
        return float(default)


def _tensor_summary(tensor):
    value = tensor.detach()
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }
    if value.numel() == 0:
        result.update({"numel": 0, "mean": None, "min": None, "max": None})
        return result
    scalar = value.float()
    result.update(
        {
            "numel": int(value.numel()),
            "mean": float(scalar.mean().detach().cpu().item()),
            "min": float(scalar.min().detach().cpu().item()),
            "max": float(scalar.max().detach().cpu().item()),
        }
    )
    return result


def _scalar_or_none(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return float(value.detach().float().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class JointProximalObjective:
    """Native-scale proximal/resource terms for the joint flow objective."""

    def __init__(self, params):
        self.params = params
        self.enabled = proximal_profile_enabled(params)
        self.anchors = {}
        self._anchorable_names_observed = set()
        self.anchor_metadata = {}
        self.last_summary = {"enabled": bool(self.enabled)}

    def lambdas(self):
        return {
            "lambda_x": _float_param(self.params, "joint_proximal_lambda_x", 0.0),
            "lambda_s": _float_param(self.params, "joint_proximal_lambda_s", 0.0),
            "lambda_z": _float_param(self.params, "joint_proximal_lambda_z", 0.0),
            "lambda_b": _float_param(self.params, "joint_proximal_lambda_b", 0.0),
            "mu": _float_param(self.params, "joint_proximal_buffer_mu", 0.0),
        }

    def _buffer_state(self, model):
        data_collections = getattr(model, "data_collections", None)
        state = getattr(model, "buffer_optimization_state", None)
        if state is None and data_collections is not None:
            state = getattr(data_collections, "buffer_optimization_state", None)
        if state is None and data_collections is not None:
            state = getattr(data_collections, "buffer_segment_count_state", None)
        return state

    def _state_tensors(self, model, pos):
        data_collections = getattr(model, "data_collections", None)
        buffer_state = self._buffer_state(model)
        tensors = {
            "x": pos,
            "size_logits": getattr(data_collections, "size_logits", None),
            "vt_logits": getattr(data_collections, "vt_logits", None),
            "z_param": getattr(buffer_state, "z_param", None),
            "bu_logits": getattr(buffer_state, "bu_logits", None),
            "bsu_index_param": getattr(buffer_state, "bsu_index_param", None),
        }
        return {name: value for name, value in tensors.items() if torch.is_tensor(value)}

    def _metadata(self, model):
        data_collections = getattr(model, "data_collections", None)
        payload = None
        if data_collections is not None:
            payload = getattr(data_collections, "buffer_relaxed_timing_payload", None)
        payload_metadata = {}
        if isinstance(payload, dict):
            payload_metadata = dict(payload.get("metadata", {}) or {})
        return {
            "topology_generation": getattr(data_collections, "topology_generation", None),
            "openroad_design_generation": getattr(
                data_collections,
                "openroad_design_generation",
                None,
            ),
            "buffer_payload_metadata": payload_metadata,
        }

    def capture_anchors(self, model, pos, *, force=False):
        if not self.enabled:
            self.last_summary = {"enabled": False, "reason": "profile_disabled"}
            return self.last_summary
        if self.anchors and not force:
            tensors = self._state_tensors(model, pos)
            missing_anchors = {
                name: tensor.detach().clone()
                for name, tensor in tensors.items()
                if name not in self.anchors
                and name not in self._anchorable_names_observed
                and (tensor.requires_grad or name == "x")
            }
            if missing_anchors:
                self.anchors.update(missing_anchors)
                self._anchorable_names_observed.update(missing_anchors)
                self.anchor_metadata = self._metadata(model)
                return self.anchor_summary(status="extended")
            return self.anchor_summary(status="reused")
        tensors = self._state_tensors(model, pos)
        self.anchors = {
            name: tensor.detach().clone()
            for name, tensor in tensors.items()
            if tensor.requires_grad or name == "x"
        }
        self._anchorable_names_observed = set(self.anchors)
        self.anchor_metadata = self._metadata(model)
        return self.anchor_summary(status="created")

    def anchor_summary(self, *, status):
        return {
            "status": status,
            "metadata": copy.deepcopy(self.anchor_metadata),
            "tensors": {
                name: _tensor_summary(tensor) for name, tensor in self.anchors.items()
            },
        }

    def _mean_square(self, tensor, anchor, name):
        if tensor.shape != anchor.shape:
            raise ValueError(
                f"joint proximal anchor shape mismatch for {name}: "
                f"tensor={tuple(tensor.shape)} anchor={tuple(anchor.shape)}"
            )
        if tensor.numel() == 0:
            return tensor.sum() * 0.0
        delta = tensor - anchor.to(device=tensor.device, dtype=tensor.dtype)
        return 0.5 * torch.mean(delta * delta)

    def adjust_discrete_count_gradient(self, z_tensor, raw_grad):
        """Replace the differential P_z term with its integer +1 increment."""
        if not torch.is_tensor(z_tensor) or not torch.is_tensor(raw_grad):
            raise TypeError("z_tensor and raw_grad must be tensors")
        if z_tensor.shape != raw_grad.shape:
            raise ValueError("z_tensor and raw_grad must have equal shape")
        action_grad = raw_grad.detach().clone()
        anchor = self.anchors.get("z_param")
        lambda_z = self.lambdas()["lambda_z"]
        if anchor is None or lambda_z == 0.0 or z_tensor.numel() == 0:
            return action_grad, {
                "enabled": False,
                "reason": "missing_anchor_or_zero_lambda",
                "finite_increment_applied": False,
            }

        current = z_tensor.detach()
        anchor = anchor.to(device=current.device, dtype=current.dtype)
        normalizer = float(current.numel())
        differential = lambda_z * (current - anchor) / normalizer
        finite_increment = (
            lambda_z
            * ((current + 1.0 - anchor).square() - (current - anchor).square())
            / (2.0 * normalizer)
        )
        correction = finite_increment - differential
        action_grad.add_(correction.to(device=action_grad.device, dtype=action_grad.dtype))
        return action_grad, {
            "enabled": True,
            "finite_increment_applied": True,
            "lambda_z": float(lambda_z),
            "normalizer": int(current.numel()),
            "correction_min": float(correction.min().cpu().item()),
            "correction_max": float(correction.max().cpu().item()),
            "correction_norm": float(correction.float().norm().cpu().item()),
        }

    def _term_or_missing(self, tensors, names, lambda_value, label, missing):
        active_terms = []
        for name in names:
            tensor = tensors.get(name)
            anchor = self.anchors.get(name)
            if tensor is None:
                continue
            if anchor is None:
                if lambda_value != 0.0:
                    missing.append(name)
                continue
            active_terms.append(self._mean_square(tensor, anchor, name))
        if active_terms:
            return sum(active_terms) / float(len(active_terms))
        if lambda_value != 0.0:
            missing.append(label)
        return None

    def add_to_objective(self, model, obj, pos):
        if not self.enabled:
            self.last_summary = {"enabled": False, "reason": "profile_disabled"}
            return obj, self.last_summary
        anchor_summary = self.capture_anchors(model, pos)
        tensors = self._state_tensors(model, pos)
        lambdas = self.lambdas()
        missing = []
        z_tensor = tensors.get("z_param")
        if z_tensor is None:
            z_tensor = tensors.get("bu_logits")
        # The segment lane is deliberately created after its overflow gate.
        # During initial learning-rate estimation there is no buffer state yet;
        # defer P_z until the lane is activated instead of treating it as a
        # malformed active state.
        lambda_z = lambdas["lambda_z"] if z_tensor is not None else 0.0
        terms = {
            "P_x": self._term_or_missing(
                tensors,
                ("x",),
                lambdas["lambda_x"],
                "placement",
                missing,
            ),
            "P_s": self._term_or_missing(
                tensors,
                ("size_logits", "vt_logits"),
                lambdas["lambda_s"],
                "sizing",
                missing,
            ),
            "P_z": self._term_or_missing(
                tensors,
                ("z_param", "bu_logits"),
                lambda_z,
                "buffer_count",
                missing,
            ),
            "P_b": self._term_or_missing(
                tensors,
                ("bsu_index_param",),
                lambdas["lambda_b"],
                "buffer_size",
                missing,
            ),
        }
        if missing:
            raise ValueError(
                "joint proximal profile has nonzero lambda but missing anchors/state: "
                + ", ".join(sorted(set(missing)))
            )
        if z_tensor is not None:
            resource_values = torch.relu(z_tensor)
            if z_tensor.numel():
                terms["P_buf"] = torch.mean(resource_values)
            else:
                terms["P_buf"] = z_tensor.sum() * 0.0
        elif lambdas["mu"] != 0.0:
            raise ValueError("joint proximal buffer resource term requires z_param or bu_logits")
        else:
            terms["P_buf"] = None

        weighted_terms = {
            "lambda_x_P_x": None if terms["P_x"] is None else lambdas["lambda_x"] * terms["P_x"],
            "lambda_s_P_s": None if terms["P_s"] is None else lambdas["lambda_s"] * terms["P_s"],
            "lambda_z_P_z": None if terms["P_z"] is None else lambdas["lambda_z"] * terms["P_z"],
            "lambda_b_P_b": None if terms["P_b"] is None else lambdas["lambda_b"] * terms["P_b"],
            "mu_P_buf": None if terms["P_buf"] is None else lambdas["mu"] * terms["P_buf"],
        }
        total = None
        for value in weighted_terms.values():
            if value is None:
                continue
            total = value if total is None else total + value
        if total is None:
            total = obj.new_zeros(())
        augmented_obj = obj + total
        obj_abs = abs(_scalar_or_none(obj) or 0.0)
        denom = max(obj_abs, 1.0e-12)
        weighted_scalars = {
            name: _scalar_or_none(value) for name, value in weighted_terms.items()
        }
        ratios = {
            name: (None if value is None else abs(float(value)) / denom)
            for name, value in weighted_scalars.items()
        }
        self.last_summary = {
            "enabled": True,
            "profile": _PROFILE_NAME,
            "obj_fn": _scalar_or_none(obj),
            "total_obj": _scalar_or_none(augmented_obj),
            "raw_terms": {name: _scalar_or_none(value) for name, value in terms.items()},
            "weighted_terms": weighted_scalars,
            "total_weighted": _scalar_or_none(total),
            "lambdas": lambdas,
            "lambda_scale_probe": {
                "denominator_abs_obj_fn": float(denom),
                "weighted_ratio_to_abs_obj_fn": ratios,
                "target_band": [0.01, 0.05],
                "band_is_calibration_warning": True,
            },
            "anchors": anchor_summary,
            "active_blocks": {
                "placement": terms["P_x"] is not None,
                "sizing": terms["P_s"] is not None,
                "buffer_count": terms["P_z"] is not None,
                "buffer_size": terms["P_b"] is not None,
                "buffer_resource": terms["P_buf"] is not None,
            },
            "gates": {
                "missing_anchor": False,
                "missing_state": [],
            },
        }
        return augmented_obj, self.last_summary
