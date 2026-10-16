from types import SimpleNamespace

import pytest
import torch

from dreamplace.flows.joint_proximal import JointProximalObjective
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.PlaceObj import PlaceObj


class _FakeModel:
    def __init__(self, data_collections):
        self.data_collections = data_collections


class _FakePrecondition:
    def __call__(self, grad, density_weight, update_mask, fix_nodes_mask):
        return grad


class _MinimalPlaceObj(PlaceObj):
    def __init__(self, params, data_collections):
        torch.nn.Module.__init__(self)
        self.params = params
        self.data_collections = data_collections
        self.joint_proximal_objective = JointProximalObjective(params)
        self.last_joint_proximal_summary = {
            "enabled": bool(self.joint_proximal_objective.enabled)
        }
        self.op_collections = SimpleNamespace(precondition_op=_FakePrecondition())
        self.density_weight = torch.ones(1)
        self.update_mask = None
        self.fix_nodes_mask = None
        self.use_timing_obj = False
        self.use_l_shape_routability = False
        self.l_shape_routability_op = None
        self.size_density_area_penalty = None
        self.init_density = torch.ones(())
        self.targeted_backward = False
        self.size_only_mode = False

    def obj_fn(self, pos):
        return (3.0 * pos).sum()

    def _is_size_only_mode(self):
        return bool(self.size_only_mode)

    def _use_targeted_sizing_backward(self):
        return bool(self.targeted_backward)

    def _size_debug_component_grad_stats_enabled(self):
        return False

    def _timing_backward_component_profile_enabled(self):
        return False


def _params(**overrides):
    values = {
        "joint_quality_profile": "proximal_alternating_v1",
        "joint_proximal_lambda_x": 0.5,
        "joint_proximal_lambda_s": 0.25,
        "joint_proximal_lambda_z": 0.125,
        "joint_proximal_lambda_b": 0.0625,
        "joint_proximal_buffer_mu": 0.03125,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _state():
    pos = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    size_logits = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    vt_logits = torch.nn.Parameter(torch.tensor([0.3, 0.4]))
    z_param = torch.nn.Parameter(torch.tensor([0.2, 0.6]))
    bsu_index_param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    buffer_state = SimpleNamespace(
        z_param=z_param,
        bsu_index_param=bsu_index_param,
    )
    data_collections = SimpleNamespace(
        size_logits=size_logits,
        vt_logits=vt_logits,
        buffer_optimization_state=buffer_state,
        buffer_relaxed_timing_payload={"metadata": {"state_kind": "segment_count"}},
    )
    return pos, size_logits, vt_logits, z_param, bsu_index_param, _FakeModel(data_collections)


def _assert_close(actual, expected):
    assert torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_joint_proximal_zero_base_objective_matches_native_scale_gradients():
    pos, size_logits, vt_logits, z_param, bsu_index_param, model = _state()
    proximal = JointProximalObjective(_params())
    proximal.capture_anchors(model, pos)

    with torch.no_grad():
        pos.add_(torch.tensor([0.3, -0.6, 0.9]))
        size_logits.add_(torch.tensor([0.2, -0.4]))
        vt_logits.add_(torch.tensor([-0.1, 0.5]))
        z_param.add_(torch.tensor([0.4, -0.3]))
        bsu_index_param.add_(torch.tensor([0.25, -0.75]))

    obj = pos.new_zeros(())
    loss, summary = proximal.add_to_objective(model, obj, pos)
    loss.backward()

    expected_pos = 0.5 * (pos.detach() - proximal.anchors["x"]) / pos.numel()
    expected_size = (
        0.25
        * 0.5
        * (size_logits.detach() - proximal.anchors["size_logits"])
        / size_logits.numel()
    )
    expected_vt = (
        0.25
        * 0.5
        * (vt_logits.detach() - proximal.anchors["vt_logits"])
        / vt_logits.numel()
    )
    expected_z = (
        0.125 * (z_param.detach() - proximal.anchors["z_param"]) / z_param.numel()
        + 0.03125 * torch.ones_like(z_param) / z_param.numel()
    )
    expected_bsu = (
        0.0625
        * (bsu_index_param.detach() - proximal.anchors["bsu_index_param"])
        / bsu_index_param.numel()
    )

    _assert_close(pos.grad, expected_pos)
    _assert_close(size_logits.grad, expected_size)
    _assert_close(vt_logits.grad, expected_vt)
    _assert_close(z_param.grad, expected_z)
    _assert_close(bsu_index_param.grad, expected_bsu)
    assert summary["enabled"]
    assert summary["lambda_scale_probe"]["band_is_calibration_warning"]
    assert summary["active_blocks"]["placement"]
    assert summary["active_blocks"]["buffer_resource"]


def test_discrete_count_proximal_uses_finite_forward_increment():
    proximal = JointProximalObjective(
        _params(
            joint_proximal_lambda_x=0.0,
            joint_proximal_lambda_s=0.0,
            joint_proximal_lambda_z=2.0,
            joint_proximal_lambda_b=0.0,
            joint_proximal_buffer_mu=0.0,
        )
    )
    proximal.anchors["z_param"] = torch.tensor([0.0, 1.0])
    z_param = torch.tensor([0.0, 2.0])
    raw_grad = torch.tensor([-3.0, -4.0])

    action_grad, summary = proximal.adjust_discrete_count_gradient(
        z_param,
        raw_grad,
    )

    _assert_close(action_grad, torch.tensor([-2.5, -3.5]))
    assert summary["finite_increment_applied"]
    assert summary["correction_min"] == 0.5
    assert summary["correction_max"] == 0.5


def test_joint_proximal_real_objective_gradient_delta_matches_proximal_part():
    pos, size_logits, vt_logits, z_param, bsu_index_param, model = _state()
    proximal = JointProximalObjective(_params())
    proximal.capture_anchors(model, pos)

    with torch.no_grad():
        pos.add_(torch.tensor([0.3, -0.6, 0.9]))
        size_logits.add_(torch.tensor([0.2, -0.4]))
        vt_logits.add_(torch.tensor([-0.1, 0.5]))
        z_param.add_(torch.tensor([0.4, -0.3]))
        bsu_index_param.add_(torch.tensor([0.25, -0.75]))

    base_obj = (
        (3.0 * pos).sum()
        + (5.0 * size_logits).sum()
        + (7.0 * vt_logits).sum()
        + (11.0 * z_param).sum()
        + (13.0 * bsu_index_param).sum()
    )
    base_obj.backward(retain_graph=True)
    base_grads = {
        "pos": pos.grad.detach().clone(),
        "size": size_logits.grad.detach().clone(),
        "vt": vt_logits.grad.detach().clone(),
        "z": z_param.grad.detach().clone(),
        "bsu": bsu_index_param.grad.detach().clone(),
    }
    for tensor in (pos, size_logits, vt_logits, z_param, bsu_index_param):
        tensor.grad.zero_()

    loss, _ = proximal.add_to_objective(model, base_obj, pos)
    loss.backward()

    _assert_close(
        pos.grad - base_grads["pos"],
        0.5 * (pos.detach() - proximal.anchors["x"]) / pos.numel(),
    )
    _assert_close(
        size_logits.grad - base_grads["size"],
        0.25
        * 0.5
        * (size_logits.detach() - proximal.anchors["size_logits"])
        / size_logits.numel(),
    )
    _assert_close(
        vt_logits.grad - base_grads["vt"],
        0.25
        * 0.5
        * (vt_logits.detach() - proximal.anchors["vt_logits"])
        / vt_logits.numel(),
    )
    _assert_close(
        z_param.grad - base_grads["z"],
        0.125 * (z_param.detach() - proximal.anchors["z_param"]) / z_param.numel()
        + 0.03125 * torch.ones_like(z_param) / z_param.numel(),
    )
    _assert_close(
        bsu_index_param.grad - base_grads["bsu"],
        0.0625
        * (bsu_index_param.detach() - proximal.anchors["bsu_index_param"])
        / bsu_index_param.numel(),
    )


def test_joint_proximal_missing_anchor_fails_for_nonzero_lambda():
    pos, _size_logits, _vt_logits, _z_param, _bsu_index_param, model = _state()
    proximal = JointProximalObjective(_params(joint_proximal_lambda_s=1.0))
    proximal.capture_anchors(model, pos)
    del proximal.anchors["size_logits"]

    with pytest.raises(ValueError, match="missing anchors"):
        proximal.add_to_objective(model, pos.new_zeros(()), pos)


def test_joint_proximal_zero_lambda_keeps_raw_diagnostics_without_grad_delta():
    pos, size_logits, vt_logits, z_param, bsu_index_param, model = _state()
    proximal = JointProximalObjective(
        _params(
            joint_proximal_lambda_x=0.0,
            joint_proximal_lambda_s=0.0,
            joint_proximal_lambda_z=0.0,
            joint_proximal_lambda_b=0.0,
            joint_proximal_buffer_mu=0.0,
        )
    )
    proximal.capture_anchors(model, pos)

    with torch.no_grad():
        pos.add_(1.0)
        size_logits.add_(1.0)
        vt_logits.add_(1.0)
        z_param.add_(1.0)
        bsu_index_param.add_(1.0)

    base_obj = (
        (3.0 * pos).sum()
        + (5.0 * size_logits).sum()
        + (7.0 * vt_logits).sum()
        + (11.0 * z_param).sum()
        + (13.0 * bsu_index_param).sum()
    )
    loss, summary = proximal.add_to_objective(model, base_obj, pos)
    loss.backward()

    _assert_close(pos.grad, torch.full_like(pos, 3.0))
    _assert_close(size_logits.grad, torch.full_like(size_logits, 5.0))
    _assert_close(vt_logits.grad, torch.full_like(vt_logits, 7.0))
    _assert_close(z_param.grad, torch.full_like(z_param, 11.0))
    _assert_close(bsu_index_param.grad, torch.full_like(bsu_index_param, 13.0))
    assert summary["raw_terms"]["P_x"] > 0.0
    assert summary["weighted_terms"]["lambda_x_P_x"] == 0.0
    assert summary["weighted_terms"]["mu_P_buf"] == 0.0


def test_placeobj_obj_and_grad_fn_augments_objective_before_backward():
    params = _params(
        joint_proximal_lambda_x=0.5,
        joint_proximal_lambda_s=0.0,
        joint_proximal_lambda_z=0.0,
        joint_proximal_lambda_b=0.0,
        joint_proximal_buffer_mu=0.0,
    )
    pos = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    data_collections = SimpleNamespace()
    model = _MinimalPlaceObj(params, data_collections)
    model.joint_proximal_objective.capture_anchors(model, pos)

    with torch.no_grad():
        pos.add_(torch.tensor([0.3, -0.6, 0.9]))

    obj, grad = model.obj_and_grad_fn(pos)

    expected_delta = 0.5 * (pos.detach() - model.joint_proximal_objective.anchors["x"]) / pos.numel()
    _assert_close(grad, torch.full_like(pos, 3.0) + expected_delta)
    assert obj.detach().item() > (3.0 * pos.detach()).sum().item()
    assert model.debug_obj_and_grad_profile["joint_proximal"]["enabled"]


def test_placeobj_targeted_sizing_backward_uses_augmented_objective():
    params = _params(
        joint_proximal_lambda_x=0.0,
        joint_proximal_lambda_s=0.5,
        joint_proximal_lambda_z=0.0,
        joint_proximal_lambda_b=0.0,
        joint_proximal_buffer_mu=0.0,
    )
    pos = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    size_logits = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    data_collections = SimpleNamespace(size_logits=size_logits)
    model = _MinimalPlaceObj(params, data_collections)
    model.targeted_backward = True
    model.size_only_mode = True
    model.joint_proximal_objective.capture_anchors(model, pos)

    with torch.no_grad():
        size_logits.add_(torch.tensor([0.25, -0.5]))

    obj, grad = model.obj_and_grad_fn(pos)

    expected_size_grad = (
        0.5
        * (size_logits.detach() - model.joint_proximal_objective.anchors["size_logits"])
        / size_logits.numel()
    )
    _assert_close(size_logits.grad, expected_size_grad)
    assert grad.shape == pos.shape
    assert torch.count_nonzero(grad).item() == 0
    assert model.debug_obj_and_grad_profile["objective_backward_mode"] == (
        "targeted_sizing_autograd_grad"
    )
    assert obj.detach().item() > (3.0 * pos.detach()).sum().item()


def test_joint_flow_artifact_records_proximal_schema(tmp_path):
    params = SimpleNamespace(
        flow_kind="joint",
        joint_quality_profile="proximal_alternating_v1",
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        buffering_commit_enabled=0,
    )
    placer = NonLinearPlace.__new__(NonLinearPlace)

    result = placer._write_joint_flow_artifact(
        params,
        {
            "anchors": {"status": "created"},
            "proximal": {"enabled": True},
            "lambda_scale_probe": {"band_is_calibration_warning": True},
            "gates": {"missing_anchor": False},
            "transactions": {"default_no_hard_buffer_commit": True},
        },
    )

    assert result["summary_path"].endswith("unit_design_joint_flow_summary.json")
    payload = __import__("json").loads(
        (tmp_path / "unit_design_joint_flow_summary.json").read_text(encoding="utf-8")
    )
    assert payload["metadata"]["method"] == "proximal_alternating_joint"
    assert payload["summary"]["proximal"]["enabled"]
    assert payload["summary"]["transactions"]["default_no_hard_buffer_commit"]
