import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class ProjectionContext:
    candidate_terms: dict[str, torch.Tensor] = field(default_factory=dict)
    term_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionCandidates:
    inst_ids: torch.Tensor
    main_ids: torch.Tensor
    current_cell_ids: torch.Tensor
    current_offsets: torch.Tensor
    candidate_cell_ids: torch.Tensor
    candidate_libcell_offsets: torch.Tensor
    candidate_name_ids: torch.Tensor
    candidate_sizes: torch.Tensor
    candidate_vts: torch.Tensor
    candidate_leakages: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_legal_mask: torch.Tensor


@dataclass
class ProjectionRequest:
    inst_ids: torch.Tensor
    main_ids: torch.Tensor
    current_cell_ids: torch.Tensor
    current_offsets: torch.Tensor
    current_leakage: torch.Tensor
    continuous_sizes: torch.Tensor
    vt_distributions: torch.Tensor
    vt_expectations: torch.Tensor


@dataclass
class ProjectionScore:
    total_score: torch.Tensor
    terms: dict[str, torch.Tensor]


@dataclass
class ProjectionResolution:
    selected_candidate_index: torch.Tensor
    has_legal_candidate: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectionValidation:
    is_valid: bool
    issues: list[str]
    metrics: dict[str, Any]


@dataclass
class ProjectionResult:
    inst_ids: torch.Tensor
    main_ids: torch.Tensor
    current_cell_id: torch.Tensor
    current_libcell_offset: torch.Tensor
    current_name_id: torch.Tensor
    current_size: torch.Tensor
    current_vt: torch.Tensor
    current_leakage: torch.Tensor
    continuous_size: torch.Tensor
    continuous_vt_expectation: torch.Tensor
    projected_cell_id: torch.Tensor
    projected_libcell_offset: torch.Tensor
    projected_name_id: torch.Tensor
    projected_size: torch.Tensor
    projected_vt: torch.Tensor
    projected_leakage: torch.Tensor
    leakage_delta: torch.Tensor
    has_legal_candidate: torch.Tensor
    selected_candidate_index: torch.Tensor
    projection_cost: torch.Tensor
    score_terms: dict[str, torch.Tensor]
    resolution_metadata: dict[str, Any]
    validation: ProjectionValidation


@dataclass
class ProjectionPinOffsetArtifacts:
    records: list[dict[str, Any]]
    fieldnames: list[str]
    summary: dict[str, Any]


@dataclass(frozen=True)
class ProjectionTensorMetadata:
    name: str
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    index_space: str


@dataclass(frozen=True)
class ProjectionFrame:
    """Immutable candidate/global state produced by one legal projection.

    Candidate-row tensors retain the exact projection result.  Global tensors
    are the views consumed by runtime state owners, so callers never have to
    infer an index space from a field name.
    """

    projection_result: ProjectionResult
    changed_candidate_mask: torch.Tensor
    changed_inst_ids: torch.Tensor
    changed_node_mask: torch.Tensor
    changed_pin_ids: torch.Tensor
    projected_pin_offset_x: torch.Tensor
    projected_pin_offset_y: torch.Tensor
    projected_node_size_x: torch.Tensor
    projected_node_size_y: torch.Tensor
    projected_node_area: torch.Tensor
    projected_size_global: torch.Tensor
    projected_vt_global: torch.Tensor | None
    projected_cell_id_global: torch.Tensor
    projected_libcell_offset_global: torch.Tensor
    projected_inst_size_init: torch.Tensor | None
    projected_original_pin_offset_x: torch.Tensor | None
    projected_original_pin_offset_y: torch.Tensor | None
    state_digest: str
    topology_generation_before: int
    topology_generation_after: int | None
    timing_model_generation_before: int
    timing_model_generation_after: int | None
    tensor_contract: tuple[ProjectionTensorMetadata, ...] = ()


class ProjectionScorer:
    def term_weights(self) -> dict[str, float]:
        return {}

    def _apply_context_terms(
        self,
        total_score: torch.Tensor,
        terms: dict[str, torch.Tensor],
        context: ProjectionContext | None = None,
        default_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if context is None:
            return total_score, terms

        default_weights = {} if default_weights is None else default_weights
        for name, tensor in context.candidate_terms.items():
            if tensor.shape != total_score.shape:
                raise ValueError(
                    f"candidate term {name} shape {tensor.shape} does not match "
                    f"projection score shape {total_score.shape}"
                )
            weight = float(context.term_weights.get(name, default_weights.get(name, 1.0)))
            total_score = total_score + weight * tensor
            terms[name] = tensor
        return total_score, terms

    def score(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        context: ProjectionContext | None = None,
    ) -> ProjectionScore:
        raise NotImplementedError


class ProjectionValidator:
    def validate(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        result: ProjectionResult,
        context: ProjectionContext | None = None,
    ) -> ProjectionValidation:
        raise NotImplementedError


class ProjectionResolver:
    def resolve(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        score: ProjectionScore,
        context: ProjectionContext | None = None,
    ) -> ProjectionResolution:
        raise NotImplementedError


class DistanceProjectionScorer(ProjectionScorer):
    def __init__(
        self,
        size_weight: float = 1.0,
        vt_weight: float = 1.0,
        offset_weight: float = 1e-3,
    ):
        self.size_weight = float(size_weight)
        self.vt_weight = float(vt_weight)
        self.offset_weight = float(offset_weight)

    def term_weights(self) -> dict[str, float]:
        return {
            "size_distance": self.size_weight,
            "vt_mismatch": self.vt_weight,
            "offset_distance": self.offset_weight,
        }

    def score(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        context: ProjectionContext | None = None,
    ) -> ProjectionScore:
        if context is None:
            context = ProjectionContext()

        vt_indices = candidates.candidate_vts.clamp(min=0)
        if request.vt_distributions.size(1) == 0:
            raise ValueError("vt_distributions must have at least one column")
        if torch.any(
            candidates.candidate_legal_mask
            & (vt_indices >= request.vt_distributions.size(1))
        ):
            raise ValueError("candidate vt index exceeds vt_distributions width")

        size_distance = torch.abs(
            request.continuous_sizes.unsqueeze(1) - candidates.candidate_sizes
        )
        vt_probability = request.vt_distributions.gather(1, vt_indices)
        vt_mismatch = 1.0 - vt_probability
        offset_distance = torch.abs(
            request.current_offsets.unsqueeze(1).float()
            - candidates.candidate_libcell_offsets.float()
        )

        terms = {
            "size_distance": size_distance,
            "vt_mismatch": vt_mismatch,
            "offset_distance": offset_distance,
        }
        total_score = (
            self.size_weight * size_distance
            + self.vt_weight * vt_mismatch
            + self.offset_weight * offset_distance
        )
        total_score, terms = self._apply_context_terms(total_score, terms, context=context)

        total_score = total_score.masked_fill(~candidates.candidate_legal_mask, float("inf"))
        return ProjectionScore(total_score=total_score, terms=terms)


class TimingAwareProjectionScorer(DistanceProjectionScorer):
    def __init__(
        self,
        size_weight: float = 1.0,
        vt_weight: float = 1.0,
        offset_weight: float = 1e-3,
        timing_weight: float = 1.0,
        area_weight: float = 0.0,
        density_weight: float = 0.0,
        leakage_weight: float = 0.0,
        sensitivity_weight: float = 0.0,
    ):
        super(TimingAwareProjectionScorer, self).__init__(
            size_weight=size_weight,
            vt_weight=vt_weight,
            offset_weight=offset_weight,
        )
        self.timing_weight = float(timing_weight)
        self.area_weight = float(area_weight)
        self.density_weight = float(density_weight)
        self.leakage_weight = float(leakage_weight)
        self.sensitivity_weight = float(sensitivity_weight)

    def term_weights(self) -> dict[str, float]:
        return {
            **super(TimingAwareProjectionScorer, self).term_weights(),
            "timing_penalty": self.timing_weight,
            "area_penalty": self.area_weight,
            "density_penalty": self.density_weight,
            "leakage_penalty": self.leakage_weight,
            "sensitivity_penalty": self.sensitivity_weight,
        }

    def score(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        context: ProjectionContext | None = None,
    ) -> ProjectionScore:
        size_distance = torch.abs(
            request.continuous_sizes.unsqueeze(1) - candidates.candidate_sizes
        )
        vt_indices = candidates.candidate_vts.clamp(min=0)
        vt_probability = request.vt_distributions.gather(1, vt_indices)
        vt_mismatch = 1.0 - vt_probability
        offset_distance = torch.abs(
            request.current_offsets.unsqueeze(1).float()
            - candidates.candidate_libcell_offsets.float()
        )

        terms = {
            "size_distance": size_distance,
            "vt_mismatch": vt_mismatch,
            "offset_distance": offset_distance,
        }
        total_score = (
            self.size_weight * size_distance
            + self.vt_weight * vt_mismatch
            + self.offset_weight * offset_distance
        )
        default_weights = {
            "timing_penalty": self.timing_weight,
            "area_penalty": self.area_weight,
            "density_penalty": self.density_weight,
            "leakage_penalty": self.leakage_weight,
            "sensitivity_penalty": self.sensitivity_weight,
        }
        total_score, terms = self._apply_context_terms(
            total_score,
            terms,
            context=context,
            default_weights=default_weights,
        )
        total_score = total_score.masked_fill(~candidates.candidate_legal_mask, float("inf"))
        return ProjectionScore(total_score=total_score, terms=terms)


class BasicProjectionValidator(ProjectionValidator):
    def __init__(self, max_density_proxy_ratio: float = 2.5):
        self.max_density_proxy_ratio = float(max_density_proxy_ratio)

    def validate(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        result: ProjectionResult,
        context: ProjectionContext | None = None,
    ) -> ProjectionValidation:
        issues: list[str] = []
        has_any_candidate = candidates.candidate_mask.any(dim=1)
        has_any_legal_candidate = candidates.candidate_legal_mask.any(dim=1)
        selected_legal = result.projected_cell_id >= 0

        missing_candidate_count = int((~has_any_candidate).sum().item())
        missing_legal_count = int((~has_any_legal_candidate).sum().item())
        illegal_selected_count = int((~selected_legal).sum().item())

        if missing_candidate_count:
            issues.append(f"{missing_candidate_count} instances have no candidates")
        if missing_legal_count:
            issues.append(f"{missing_legal_count} instances have no legal candidates")
        if illegal_selected_count:
            issues.append(f"{illegal_selected_count} instances projected to an illegal fallback")

        legal_mask = selected_legal
        current_total_size = float(result.current_size.clamp(min=0).sum().item())
        continuous_total_size = float(request.continuous_sizes.clamp(min=0).sum().item())
        projected_total_size = float(result.projected_size[legal_mask].clamp(min=0).sum().item())
        density_proxy_ratio = None
        if continuous_total_size > 0:
            density_proxy_ratio = projected_total_size / continuous_total_size
            if density_proxy_ratio > self.max_density_proxy_ratio:
                issues.append(
                    "projection density proxy ratio %.4f exceeds %.4f"
                    % (density_proxy_ratio, self.max_density_proxy_ratio)
                )

        timing_metrics: dict[str, Any] = {}
        if context is not None and isinstance(getattr(context, "metadata", None), dict):
            local_timing_scores = context.metadata.get("local_timing_scores")
            if torch.is_tensor(local_timing_scores):
                local_timing_scores = local_timing_scores.detach().float()
                timing_metrics["mean_local_timing_score"] = float(local_timing_scores.mean().item())
                timing_metrics["max_local_timing_score"] = float(local_timing_scores.max().item())
            stage_timing_summary = context.metadata.get("stage_timing_summary")
            if isinstance(stage_timing_summary, dict):
                timing_metrics["stage_timing_summary"] = stage_timing_summary
        if "timing_penalty" in result.score_terms:
            timing_penalty = result.score_terms["timing_penalty"][legal_mask]
            if timing_penalty.numel() > 0:
                timing_metrics["mean_timing_penalty"] = float(timing_penalty.mean().item())
                timing_metrics["max_timing_penalty"] = float(timing_penalty.max().item())
        leakage_metrics = {
            "current_total_leakage": float(result.current_leakage.clamp(min=0).sum().item()),
            "projected_total_leakage": float(result.projected_leakage[legal_mask].clamp(min=0).sum().item()),
        }
        leakage_metrics["total_leakage_delta"] = (
            leakage_metrics["projected_total_leakage"] - leakage_metrics["current_total_leakage"]
        )
        if legal_mask.any():
            leakage_metrics["num_leakage_improved_cells"] = int(
                (result.leakage_delta[legal_mask] < -1e-12).sum().item()
            )
        else:
            leakage_metrics["num_leakage_improved_cells"] = 0
        if "leakage_penalty" in result.score_terms:
            leakage_penalty = result.score_terms["leakage_penalty"][legal_mask]
            if leakage_penalty.numel() > 0:
                leakage_metrics["mean_leakage_penalty"] = float(leakage_penalty.mean().item())
                leakage_metrics["max_leakage_penalty"] = float(leakage_penalty.max().item())

        metrics = {
            "num_instances": int(request.inst_ids.numel()),
            "num_instances_with_candidates": int(has_any_candidate.sum().item()),
            "num_instances_with_legal_candidates": int(has_any_legal_candidate.sum().item()),
            "selected_legal_mask": selected_legal,
            "area": {
                "current_total_size": current_total_size,
                "continuous_total_size": continuous_total_size,
                "projected_total_size": projected_total_size,
                "num_projected_instances": int(legal_mask.sum().item()),
            },
            "density": {
                "projected_continuous_ratio": density_proxy_ratio,
                "max_allowed_projected_continuous_ratio": self.max_density_proxy_ratio,
            },
            "leakage": leakage_metrics,
            "timing": timing_metrics,
        }
        return ProjectionValidation(
            is_valid=not issues,
            issues=issues,
            metrics=metrics,
        )


class ArgminProjectionResolver(ProjectionResolver):
    def resolve(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        score: ProjectionScore,
        context: ProjectionContext | None = None,
    ) -> ProjectionResolution:
        del request, context
        selected_candidate_index = torch.argmin(score.total_score, dim=1)
        has_legal_candidate = candidates.candidate_legal_mask.any(dim=1)
        return ProjectionResolution(
            selected_candidate_index=selected_candidate_index,
            has_legal_candidate=has_legal_candidate,
            metadata={"resolver": "argmin"},
        )


class StableCurrentCellResolver(ProjectionResolver):
    def __init__(self, tie_epsilon: float = 1e-9):
        self.tie_epsilon = float(tie_epsilon)

    def resolve(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        score: ProjectionScore,
        context: ProjectionContext | None = None,
    ) -> ProjectionResolution:
        del context
        selected_candidate_index = torch.argmin(score.total_score, dim=1)
        has_legal_candidate = candidates.candidate_legal_mask.any(dim=1)
        min_score = score.total_score.gather(1, selected_candidate_index.unsqueeze(1)).squeeze(1)
        current_mask = (
            candidates.candidate_legal_mask
            & (candidates.candidate_cell_ids == request.current_cell_ids.unsqueeze(1))
            & (
                candidates.candidate_libcell_offsets
                == request.current_offsets.unsqueeze(1)
            )
        )
        tied_mask = torch.abs(score.total_score - min_score.unsqueeze(1)) <= self.tie_epsilon
        current_tie_mask = current_mask & tied_mask
        choose_current = current_tie_mask.any(dim=1)
        current_index = torch.argmax(current_tie_mask.int(), dim=1)
        selected_candidate_index = torch.where(
            choose_current,
            current_index,
            selected_candidate_index,
        )
        return ProjectionResolution(
            selected_candidate_index=selected_candidate_index,
            has_legal_candidate=has_legal_candidate,
            metadata={
                "resolver": "stable_current_cell",
                "tie_epsilon": self.tie_epsilon,
                "preserved_current_cell_mask": choose_current,
            },
        )


class NearestSizeProjectionResolver(ProjectionResolver):
    def __init__(self, tie_epsilon: float = 1e-9):
        self.tie_epsilon = float(tie_epsilon)

    def resolve(
        self,
        request: ProjectionRequest,
        candidates: ProjectionCandidates,
        score: ProjectionScore,
        context: ProjectionContext | None = None,
    ) -> ProjectionResolution:
        del score, context
        size_distance = torch.abs(
            request.continuous_sizes.unsqueeze(1) - candidates.candidate_sizes
        )
        size_distance = size_distance.masked_fill(
            ~candidates.candidate_legal_mask,
            float("inf"),
        )
        selected_candidate_index = torch.argmin(size_distance, dim=1)
        has_legal_candidate = candidates.candidate_legal_mask.any(dim=1)
        min_distance = size_distance.gather(1, selected_candidate_index.unsqueeze(1)).squeeze(1)
        current_mask = (
            candidates.candidate_legal_mask
            & (candidates.candidate_cell_ids == request.current_cell_ids.unsqueeze(1))
            & (
                candidates.candidate_libcell_offsets
                == request.current_offsets.unsqueeze(1)
            )
        )
        tied_mask = torch.abs(size_distance - min_distance.unsqueeze(1)) <= self.tie_epsilon
        current_tie_mask = current_mask & tied_mask
        choose_current = current_tie_mask.any(dim=1)
        current_index = torch.argmax(current_tie_mask.int(), dim=1)
        selected_candidate_index = torch.where(
            choose_current,
            current_index,
            selected_candidate_index,
        )
        return ProjectionResolution(
            selected_candidate_index=selected_candidate_index,
            has_legal_candidate=has_legal_candidate,
            metadata={
                "resolver": "nearest_size_round",
                "tie_epsilon": self.tie_epsilon,
                "preserved_current_cell_mask": choose_current,
            },
        )


class MainIdCandidateProvider:
    def _normalize_inst_ids(self, data_collections, inst_ids: torch.Tensor | None) -> torch.Tensor:
        device = data_collections.device
        if inst_ids is None:
            return torch.arange(
                data_collections.inst_main_id.numel(),
                device=device,
                dtype=torch.long,
            )
        return torch.as_tensor(inst_ids, device=device, dtype=torch.long)

    def enumerate(
        self,
        data_collections,
        inst_ids: torch.Tensor | None = None,
    ) -> ProjectionCandidates:
        inst_ids = self._normalize_inst_ids(data_collections, inst_ids)
        device = data_collections.device

        main_ids = data_collections.inst_main_id[inst_ids].long()
        current_cell_ids = data_collections.inst_cell_id[inst_ids].long()
        current_offsets = data_collections.inst_libcell_offset[inst_ids].long()
        is_sizeable = data_collections.inst_is_sizeable[inst_ids].bool()
        size_lower = data_collections.inst_size_lower[inst_ids].float()
        size_upper = data_collections.inst_size_upper[inst_ids].float()
        vt_mask = data_collections.inst_vt_mask[inst_ids].bool()
        cell_info = data_collections.flat_libcell_info
        getter = getattr(data_collections, "get_libcell_leakage", None)
        cell_leakage = getter() if callable(getter) else getattr(data_collections, "flat_libcell_leakage", None)
        main_id_2_cell_id_start = data_collections.main_id_2_cell_id_start.long()

        candidate_cell_lists: list[torch.Tensor] = []
        candidate_offset_lists: list[torch.Tensor] = []
        max_candidates = 1

        for row_idx, _ in enumerate(inst_ids.tolist()):
            main_id = int(main_ids[row_idx].item())
            if bool(is_sizeable[row_idx].item()) and main_id >= 0:
                cell_start = int(main_id_2_cell_id_start[main_id].item())
                cell_end = int(main_id_2_cell_id_start[main_id + 1].item())
                cell_ids = torch.arange(cell_start, cell_end, device=device, dtype=torch.long)
                offsets = torch.arange(0, cell_end - cell_start, device=device, dtype=torch.long)
            elif int(current_cell_ids[row_idx].item()) >= 0:
                cell_ids = current_cell_ids[row_idx].reshape(1)
                offsets = current_offsets[row_idx].reshape(1)
            else:
                cell_ids = torch.empty(0, device=device, dtype=torch.long)
                offsets = torch.empty(0, device=device, dtype=torch.long)
            candidate_cell_lists.append(cell_ids)
            candidate_offset_lists.append(offsets)
            max_candidates = max(max_candidates, int(cell_ids.numel()))

        num_insts = int(inst_ids.numel())
        candidate_cell_ids = torch.full(
            (num_insts, max_candidates),
            -1,
            device=device,
            dtype=torch.long,
        )
        candidate_offsets = torch.zeros_like(candidate_cell_ids)
        candidate_name_ids = torch.full_like(candidate_cell_ids, -1)
        candidate_sizes = torch.zeros((num_insts, max_candidates), device=device, dtype=torch.float32)
        candidate_vts = torch.zeros_like(candidate_cell_ids)
        candidate_leakages = torch.zeros((num_insts, max_candidates), device=device, dtype=torch.float32)
        candidate_mask = torch.zeros((num_insts, max_candidates), device=device, dtype=torch.bool)
        candidate_legal_mask = torch.zeros_like(candidate_mask)

        for row_idx, (cell_ids, offsets) in enumerate(zip(candidate_cell_lists, candidate_offset_lists)):
            count = int(cell_ids.numel())
            if count == 0:
                continue

            candidate_cell_ids[row_idx, :count] = cell_ids
            candidate_offsets[row_idx, :count] = offsets
            candidate_mask[row_idx, :count] = True
            candidate_name_ids[row_idx, :count] = cell_info[cell_ids, 0].long()
            candidate_sizes[row_idx, :count] = cell_info[cell_ids, 2].float()
            candidate_vts[row_idx, :count] = cell_info[cell_ids, 3].long()
            if cell_leakage is not None:
                candidate_leakages[row_idx, :count] = cell_leakage[cell_ids].float()

            if bool(is_sizeable[row_idx].item()) and int(main_ids[row_idx].item()) >= 0:
                legal = torch.ones(count, device=device, dtype=torch.bool)
                legal &= candidate_sizes[row_idx, :count] >= size_lower[row_idx]
                legal &= candidate_sizes[row_idx, :count] <= size_upper[row_idx]
                legal &= vt_mask[row_idx, candidate_vts[row_idx, :count]]
            else:
                legal = torch.ones(count, device=device, dtype=torch.bool)
            candidate_legal_mask[row_idx, :count] = legal

        return ProjectionCandidates(
            inst_ids=inst_ids,
            main_ids=main_ids,
            current_cell_ids=current_cell_ids,
            current_offsets=current_offsets,
            candidate_cell_ids=candidate_cell_ids,
            candidate_libcell_offsets=candidate_offsets,
            candidate_name_ids=candidate_name_ids,
            candidate_sizes=candidate_sizes,
            candidate_vts=candidate_vts,
            candidate_leakages=candidate_leakages,
            candidate_mask=candidate_mask,
            candidate_legal_mask=candidate_legal_mask,
        )


class VectorizedMainIdCandidateProvider(MainIdCandidateProvider):
    """Tensorized equivalent of :class:`MainIdCandidateProvider`.

    The reference provider performs one device synchronization per instance while
    it expands the main-id library range.  This provider constructs the same
    padded candidate table with row-wise tensor operations so a full-design
    projection does not cross the Python/device boundary once per instance.
    """

    def enumerate(
        self,
        data_collections,
        inst_ids: torch.Tensor | None = None,
    ) -> ProjectionCandidates:
        inst_ids = self._normalize_inst_ids(data_collections, inst_ids)
        device = data_collections.device
        num_instances = int(inst_ids.numel())

        main_ids = data_collections.inst_main_id[inst_ids].long()
        current_cell_ids = data_collections.inst_cell_id[inst_ids].long()
        current_offsets = data_collections.inst_libcell_offset[inst_ids].long()
        is_sizeable = data_collections.inst_is_sizeable[inst_ids].bool()
        size_lower = data_collections.inst_size_lower[inst_ids].float()
        size_upper = data_collections.inst_size_upper[inst_ids].float()
        vt_mask = data_collections.inst_vt_mask[inst_ids].bool()
        cell_info = data_collections.flat_libcell_info
        getter = getattr(data_collections, "get_libcell_leakage", None)
        cell_leakage = (
            getter()
            if callable(getter)
            else getattr(data_collections, "flat_libcell_leakage", None)
        )
        main_id_2_cell_id_start = data_collections.main_id_2_cell_id_start.long()

        num_main_ids = max(int(main_id_2_cell_id_start.numel()) - 1, 0)
        invalid_sizeable_main = is_sizeable & (main_ids >= num_main_ids)
        if bool(torch.any(invalid_sizeable_main).item()):
            bad_rows = torch.nonzero(invalid_sizeable_main, as_tuple=False).flatten()
            raise ValueError(
                "sizeable instances contain invalid main_id rows: "
                + ", ".join(str(int(row)) for row in bad_rows[:8].detach().cpu())
            )

        if num_main_ids:
            safe_main_ids = main_ids.clamp(min=0, max=num_main_ids - 1)
            main_starts = main_id_2_cell_id_start[safe_main_ids]
            main_ends = main_id_2_cell_id_start[safe_main_ids + 1]
        else:
            main_starts = torch.zeros_like(main_ids)
            main_ends = torch.zeros_like(main_ids)
        sizeable_counts = (main_ends - main_starts).clamp(min=0)
        sizeable_instance_rows = is_sizeable & (main_ids >= 0)
        fixed_counts = ((~sizeable_instance_rows) & (current_cell_ids >= 0)).long()
        counts = torch.where(
            sizeable_instance_rows,
            sizeable_counts,
            fixed_counts,
        )
        max_candidates = max(
            1,
            int(counts.max().item()) if num_instances else 1,
        )

        slots = torch.arange(max_candidates, device=device, dtype=torch.long)
        row_has_candidate = slots.unsqueeze(0) < counts.unsqueeze(1)
        sizeable_rows = sizeable_instance_rows.unsqueeze(1) & row_has_candidate
        fixed_rows = (
            ~sizeable_instance_rows.unsqueeze(1)
        ) & row_has_candidate

        candidate_cell_ids = torch.full(
            (num_instances, max_candidates),
            -1,
            device=device,
            dtype=torch.long,
        )
        candidate_cell_ids = torch.where(
            sizeable_rows,
            main_starts.unsqueeze(1) + slots.unsqueeze(0),
            candidate_cell_ids,
        )
        candidate_cell_ids = torch.where(
            fixed_rows,
            current_cell_ids.unsqueeze(1).expand(-1, max_candidates),
            candidate_cell_ids,
        )

        candidate_offsets = torch.zeros_like(candidate_cell_ids)
        candidate_offsets = torch.where(
            sizeable_rows,
            slots.unsqueeze(0).expand(num_instances, -1),
            candidate_offsets,
        )
        candidate_offsets = torch.where(
            fixed_rows,
            current_offsets.unsqueeze(1).expand(-1, max_candidates),
            candidate_offsets,
        )

        candidate_mask = row_has_candidate
        safe_cell_ids = candidate_cell_ids.clamp(min=0)
        candidate_name_ids = cell_info[safe_cell_ids, 0].long()
        candidate_sizes = cell_info[safe_cell_ids, 2].float()
        candidate_vts = cell_info[safe_cell_ids, 3].long()
        candidate_leakages = torch.zeros(
            (num_instances, max_candidates),
            device=device,
            dtype=torch.float32,
        )
        if cell_leakage is not None:
            candidate_leakages = cell_leakage[safe_cell_ids].float()

        # Values in padded cells are ignored by candidate_mask.  Keep their
        # metadata at the same defaults as the reference provider.
        candidate_name_ids = candidate_name_ids.masked_fill(~candidate_mask, -1)
        candidate_sizes = candidate_sizes.masked_fill(~candidate_mask, 0.0)
        candidate_vts = candidate_vts.masked_fill(~candidate_mask, 0)
        candidate_leakages = candidate_leakages.masked_fill(~candidate_mask, 0.0)

        candidate_legal_mask = candidate_mask.clone()
        legal_size_rows = sizeable_rows
        candidate_legal_mask &= ~legal_size_rows | (
            (candidate_sizes >= size_lower.unsqueeze(1))
            & (candidate_sizes <= size_upper.unsqueeze(1))
        )
        if vt_mask.size(1) == 0:
            candidate_legal_mask &= ~legal_size_rows
        else:
            invalid_vt = legal_size_rows & (candidate_vts >= vt_mask.size(1))
            if bool(torch.any(invalid_vt).item()):
                raise ValueError("candidate VT index exceeds vt_mask width")
            vt_legal = torch.zeros_like(candidate_legal_mask)
            safe_vt = candidate_vts.clamp(min=0, max=vt_mask.size(1) - 1)
            vt_legal = vt_mask.gather(1, safe_vt)
            candidate_legal_mask &= ~legal_size_rows | vt_legal

        return ProjectionCandidates(
            inst_ids=inst_ids,
            main_ids=main_ids,
            current_cell_ids=current_cell_ids,
            current_offsets=current_offsets,
            candidate_cell_ids=candidate_cell_ids,
            candidate_libcell_offsets=candidate_offsets,
            candidate_name_ids=candidate_name_ids,
            candidate_sizes=candidate_sizes,
            candidate_vts=candidate_vts,
            candidate_leakages=candidate_leakages,
            candidate_mask=candidate_mask,
            candidate_legal_mask=candidate_legal_mask,
        )


class GateProjectionOp(nn.Module):
    def __init__(
        self,
        data_collections,
        scorer: ProjectionScorer | None = None,
        validator: ProjectionValidator | None = None,
        candidate_provider: MainIdCandidateProvider | None = None,
        resolver: ProjectionResolver | None = None,
    ):
        super(GateProjectionOp, self).__init__()
        self.data_collections = data_collections
        self.scorer = scorer if scorer is not None else DistanceProjectionScorer()
        self.validator = validator if validator is not None else BasicProjectionValidator()
        self.candidate_provider = (
            candidate_provider if candidate_provider is not None else MainIdCandidateProvider()
        )
        self.resolver = resolver if resolver is not None else ArgminProjectionResolver()

    def _get_libcell_leakage_tensor(self):
        getter = getattr(self.data_collections, "get_libcell_leakage", None)
        if callable(getter):
            return getter()
        return getattr(self.data_collections, "flat_libcell_leakage", None)

    def _build_request(
        self,
        candidates: ProjectionCandidates,
        size_var: torch.Tensor | None = None,
        vt_var: torch.Tensor | None = None,
    ) -> ProjectionRequest:
        if size_var is None:
            size_var = self.data_collections.get_size_var()
        if vt_var is None:
            vt_var = self.data_collections.get_vt_var()
        if size_var is None or vt_var is None:
            raise ValueError("GateProjectionOp requires both size_var and vt_var")
        cell_leakage = self._get_libcell_leakage_tensor()

        size_var = size_var[candidates.inst_ids].float()
        vt_var = vt_var[candidates.inst_ids].float()
        current_leakage = torch.zeros_like(size_var)
        if cell_leakage is not None:
            current_valid = candidates.current_cell_ids >= 0
            if current_valid.any():
                current_leakage[current_valid] = cell_leakage[candidates.current_cell_ids[current_valid]].float()
        vt_expectations = torch.sum(
            vt_var
            * torch.arange(
                vt_var.shape[1],
                device=vt_var.device,
                dtype=vt_var.dtype,
            ).unsqueeze(0),
            dim=1,
        )

        return ProjectionRequest(
            inst_ids=candidates.inst_ids,
            main_ids=candidates.main_ids,
            current_cell_ids=candidates.current_cell_ids,
            current_offsets=candidates.current_offsets,
            current_leakage=current_leakage,
            continuous_sizes=size_var,
            vt_distributions=vt_var,
            vt_expectations=vt_expectations,
        )

    def forward(
        self,
        inst_ids: torch.Tensor | None = None,
        size_var: torch.Tensor | None = None,
        vt_var: torch.Tensor | None = None,
        context: ProjectionContext | None = None,
        candidates: ProjectionCandidates | None = None,
        profile: dict[str, Any] | None = None,
    ) -> ProjectionResult:
        projection_started_at = time.perf_counter()
        if candidates is None:
            candidate_started_at = time.perf_counter()
            candidates = self.candidate_provider.enumerate(self.data_collections, inst_ids)
            if profile is not None:
                profile["candidate_enumerate_ms"] = (
                    time.perf_counter() - candidate_started_at
                ) * 1000.0
        elif inst_ids is not None and not torch.equal(candidates.inst_ids, inst_ids):
            raise ValueError("provided projection candidates do not match inst_ids")

        request_started_at = time.perf_counter()
        request = self._build_request(candidates, size_var=size_var, vt_var=vt_var)
        if profile is not None:
            profile["projection_request_ms"] = (
                time.perf_counter() - request_started_at
            ) * 1000.0

        score_started_at = time.perf_counter()
        score = self.scorer.score(request, candidates, context=context)
        if profile is not None:
            profile["projection_score_ms"] = (
                time.perf_counter() - score_started_at
            ) * 1000.0

        resolve_started_at = time.perf_counter()
        resolution = self.resolver.resolve(request, candidates, score, context=context)
        if profile is not None:
            profile["projection_resolve_ms"] = (
                time.perf_counter() - resolve_started_at
            ) * 1000.0
        selected_candidate_index = resolution.selected_candidate_index
        has_legal_candidate = resolution.has_legal_candidate

        gather_index = selected_candidate_index.unsqueeze(1)
        current_name_id = torch.full_like(candidates.current_cell_ids, -1)
        current_size = torch.zeros_like(request.continuous_sizes)
        current_vt = torch.full_like(candidates.current_cell_ids, -1)
        current_valid = candidates.current_cell_ids >= 0
        if current_valid.any():
            current_cells = candidates.current_cell_ids[current_valid]
            cell_info = self.data_collections.flat_libcell_info
            current_name_id[current_valid] = cell_info[current_cells, 0].long()
            current_size[current_valid] = cell_info[current_cells, 2].float()
            current_vt[current_valid] = cell_info[current_cells, 3].long()

        projected_cell_id = candidates.candidate_cell_ids.gather(1, gather_index).squeeze(1)
        projected_libcell_offset = (
            candidates.candidate_libcell_offsets.gather(1, gather_index).squeeze(1)
        )
        projected_name_id = candidates.candidate_name_ids.gather(1, gather_index).squeeze(1)
        projected_size = candidates.candidate_sizes.gather(1, gather_index).squeeze(1)
        projected_vt = candidates.candidate_vts.gather(1, gather_index).squeeze(1)
        projected_leakage = candidates.candidate_leakages.gather(1, gather_index).squeeze(1)
        projection_cost = score.total_score.gather(1, gather_index).squeeze(1)

        projected_cell_id = projected_cell_id.masked_fill(~has_legal_candidate, -1)
        projected_libcell_offset = projected_libcell_offset.masked_fill(~has_legal_candidate, -1)
        projected_name_id = projected_name_id.masked_fill(~has_legal_candidate, -1)
        projected_size = projected_size.masked_fill(~has_legal_candidate, -1.0)
        projected_vt = projected_vt.masked_fill(~has_legal_candidate, -1)
        projected_leakage = projected_leakage.masked_fill(~has_legal_candidate, -1.0)
        projection_cost = projection_cost.masked_fill(~has_legal_candidate, float("inf"))
        leakage_delta = projected_leakage - request.current_leakage
        leakage_delta = leakage_delta.masked_fill(~has_legal_candidate, float("nan"))

        selected_terms = {
            name: tensor.gather(1, gather_index).squeeze(1).masked_fill(
                ~has_legal_candidate,
                float("inf"),
            )
            for name, tensor in score.terms.items()
        }
        current_candidate_mask = (
            candidates.candidate_mask
            & (candidates.candidate_cell_ids == candidates.current_cell_ids.unsqueeze(1))
            & (
                candidates.candidate_libcell_offsets
                == candidates.current_offsets.unsqueeze(1)
            )
        )
        current_candidate_present = current_candidate_mask.any(dim=1)
        current_candidate_index = torch.argmax(current_candidate_mask.int(), dim=1)
        current_gather_index = current_candidate_index.unsqueeze(1)
        current_candidate_legal = (
            candidates.candidate_legal_mask.gather(1, current_gather_index).squeeze(1)
            & current_candidate_present
        )
        current_candidate_total_score = (
            score.total_score.gather(1, current_gather_index).squeeze(1)
        ).masked_fill(~current_candidate_present, float("inf"))
        current_candidate_total_score_delta_vs_selected = (
            current_candidate_total_score - projection_cost
        ).masked_fill(~current_candidate_present, float("inf"))
        current_candidate_terms = {
            f"current_candidate_term_{name}": tensor.gather(1, current_gather_index).squeeze(1).masked_fill(
                ~current_candidate_present,
                float("inf"),
            )
            for name, tensor in score.terms.items()
        }
        resolution_metadata = dict(resolution.metadata)
        resolution_metadata.update(
            {
                "current_candidate_present": current_candidate_present,
                "current_candidate_legal": current_candidate_legal,
                "current_candidate_total_score": current_candidate_total_score,
                "current_candidate_total_score_delta_vs_selected": current_candidate_total_score_delta_vs_selected,
                **current_candidate_terms,
            }
        )

        provisional_result = ProjectionResult(
            inst_ids=candidates.inst_ids,
            main_ids=candidates.main_ids,
            current_cell_id=candidates.current_cell_ids,
            current_libcell_offset=candidates.current_offsets,
            current_name_id=current_name_id,
            current_size=current_size,
            current_vt=current_vt,
            current_leakage=request.current_leakage,
            continuous_size=request.continuous_sizes,
            continuous_vt_expectation=request.vt_expectations,
            projected_cell_id=projected_cell_id,
            projected_libcell_offset=projected_libcell_offset,
            projected_name_id=projected_name_id,
            projected_size=projected_size,
            projected_vt=projected_vt,
            projected_leakage=projected_leakage,
            leakage_delta=leakage_delta,
            has_legal_candidate=has_legal_candidate,
            selected_candidate_index=selected_candidate_index,
            projection_cost=projection_cost,
            score_terms=selected_terms,
            resolution_metadata=resolution_metadata,
            validation=ProjectionValidation(is_valid=False, issues=[], metrics={}),
        )
        validation_started_at = time.perf_counter()
        validation = self.validator.validate(request, candidates, provisional_result, context=context)
        provisional_result.validation = validation
        if profile is not None:
            profile["projection_validate_ms"] = (
                time.perf_counter() - validation_started_at
            ) * 1000.0
            profile["projection_finalize_ms"] = (
                time.perf_counter() - projection_started_at
            ) * 1000.0
            profile["projection_call_count"] = int(
                profile.get("projection_call_count", 0)
            ) + 1
        return provisional_result

    def _serializable_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                value = value.item()
            else:
                return value.tolist()
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return {key: self._serializable_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serializable_value(item) for item in value]
        return value

    def _summary_serializable_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0 or value.numel() == 1:
                return self._serializable_value(value.reshape(()))
            return None
        if isinstance(value, dict):
            serialized = {
                key: self._summary_serializable_value(item)
                for key, item in value.items()
            }
            return {key: item for key, item in serialized.items() if item is not None}
        if isinstance(value, (list, tuple)):
            if len(value) > 16:
                return None
            serialized = [self._summary_serializable_value(item) for item in value]
            if any(item is None for item in serialized):
                return None
            return serialized
        return self._serializable_value(value)

    def _projection_cost_stats(self, values: torch.Tensor) -> dict[str, float | None]:
        finite_values = values[torch.isfinite(values)]
        if finite_values.numel() == 0:
            return {
                "mean": None,
                "max": None,
                "sum": None,
            }
        return {
            "mean": float(finite_values.mean().item()),
            "max": float(finite_values.max().item()),
            "sum": float(finite_values.sum().item()),
        }

    def _objective_term_subset_summary(
        self,
        result: ProjectionResult,
        subset_mask: torch.Tensor,
        term_weights: dict[str, float],
    ) -> dict[str, Any]:
        subset_mask = subset_mask.bool()
        num_instances = int(subset_mask.sum().item())
        summary = {
            "num_instances": num_instances,
            "projection_cost": self._projection_cost_stats(result.projection_cost[subset_mask]),
            "terms": {},
            "dominant_terms_by_weighted_sum": [],
        }
        if num_instances == 0:
            return summary

        dominant_terms: list[dict[str, float | str]] = []
        for name in sorted(result.score_terms):
            values = result.score_terms[name][subset_mask]
            finite_values = values[torch.isfinite(values)]
            weight = float(term_weights.get(name, 1.0))
            if finite_values.numel() == 0:
                term_summary = {
                    "weight": weight,
                    "sum": None,
                    "mean": None,
                    "max": None,
                    "weighted_sum": None,
                    "weighted_mean": None,
                    "weighted_max": None,
                }
            else:
                weighted_values = finite_values * weight
                weighted_sum = float(weighted_values.sum().item())
                term_summary = {
                    "weight": weight,
                    "sum": float(finite_values.sum().item()),
                    "mean": float(finite_values.mean().item()),
                    "max": float(finite_values.max().item()),
                    "weighted_sum": weighted_sum,
                    "weighted_mean": float(weighted_values.mean().item()),
                    "weighted_max": float(weighted_values.max().item()),
                }
                dominant_terms.append(
                    {
                        "term": name,
                        "weighted_sum": weighted_sum,
                        "abs_weighted_sum": abs(weighted_sum),
                    }
                )
            summary["terms"][name] = term_summary

        dominant_terms.sort(
            key=lambda item: (
                -float(item["abs_weighted_sum"]),
                str(item["term"]),
            )
        )
        summary["dominant_terms_by_weighted_sum"] = [
            {
                "term": item["term"],
                "weighted_sum": item["weighted_sum"],
            }
            for item in dominant_terms
        ]
        return summary

    def _per_instance_metadata_columns(self, result: ProjectionResult) -> dict[str, list[Any]]:
        num_instances = int(result.inst_ids.numel())
        columns: dict[str, list[Any]] = {}
        for name, value in result.resolution_metadata.items():
            key = f"metadata_{name}"
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.ndim == 1 and value.numel() == num_instances:
                    columns[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and len(value) == num_instances:
                columns[key] = [self._serializable_value(item) for item in value]
        return columns

    def build_artifact_records(
        self,
        result: ProjectionResult,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        legal_mask = result.has_legal_candidate.bool()
        changed_cell = legal_mask & (
            (result.projected_cell_id != result.current_cell_id)
            | (result.projected_libcell_offset != result.current_libcell_offset)
        )
        changed_size = legal_mask & (result.projected_size != result.current_size)
        changed_vt = legal_mask & (result.projected_vt != result.current_vt)

        columns: dict[str, list[Any]] = {
            "inst_id": result.inst_ids.detach().cpu().tolist(),
            "main_id": result.main_ids.detach().cpu().tolist(),
            "current_cell_id": result.current_cell_id.detach().cpu().tolist(),
            "current_libcell_offset": result.current_libcell_offset.detach().cpu().tolist(),
            "current_name_id": result.current_name_id.detach().cpu().tolist(),
            "current_size": result.current_size.detach().cpu().tolist(),
            "current_vt": result.current_vt.detach().cpu().tolist(),
            "current_leakage": result.current_leakage.detach().cpu().tolist(),
            "continuous_size": result.continuous_size.detach().cpu().tolist(),
            "continuous_vt_expectation": result.continuous_vt_expectation.detach().cpu().tolist(),
            "projected_cell_id": result.projected_cell_id.detach().cpu().tolist(),
            "projected_libcell_offset": result.projected_libcell_offset.detach().cpu().tolist(),
            "projected_name_id": result.projected_name_id.detach().cpu().tolist(),
            "projected_size": result.projected_size.detach().cpu().tolist(),
            "projected_vt": result.projected_vt.detach().cpu().tolist(),
            "projected_leakage": result.projected_leakage.detach().cpu().tolist(),
            "leakage_delta": result.leakage_delta.detach().cpu().tolist(),
            "projection_cost": result.projection_cost.detach().cpu().tolist(),
            "has_legal_candidate": legal_mask.detach().cpu().tolist(),
            "changed_cell": changed_cell.detach().cpu().tolist(),
            "changed_size": changed_size.detach().cpu().tolist(),
            "changed_vt": changed_vt.detach().cpu().tolist(),
        }
        for name in sorted(result.score_terms):
            columns[f"term_{name}"] = result.score_terms[name].detach().cpu().tolist()
        columns.update(self._per_instance_metadata_columns(result))

        fieldnames = list(columns.keys())
        num_instances = int(result.inst_ids.numel())
        records = [
            {
                field: self._serializable_value(columns[field][row_idx])
                for field in fieldnames
            }
            for row_idx in range(num_instances)
        ]
        return records, fieldnames

    def build_pin_offset_artifacts(
        self,
        result: ProjectionResult,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectionPinOffsetArtifacts | None:
        required_names = (
            "pin_offset_x",
            "pin_offset_y",
            "flat_node2pin_map",
            "flat_node2pin_start_map",
            "pin_2_libpin_offset",
            "cell_id_2_libpin_id_start",
            "flat_lib_pin_offset_x",
            "flat_lib_pin_offset_y",
        )
        if any(getattr(self.data_collections, name, None) is None for name in required_names):
            return None

        pin_offset_result = self.compute_projected_pin_offsets(result)
        pin_offset_x = pin_offset_result["current_pin_offset_x"]
        pin_offset_y = pin_offset_result["current_pin_offset_y"]
        inst_ids = pin_offset_result["inst_ids"]
        current_cell_id = pin_offset_result["current_cell_id"]
        projected_cell_id = pin_offset_result["projected_cell_id"]
        legal_mask = pin_offset_result["legal_mask"]
        pin_ids = pin_offset_result["pin_ids"]
        libpin_offsets = pin_offset_result["libpin_offsets"]
        projected_x = pin_offset_result["projected_pin_offset_x"]
        projected_y = pin_offset_result["projected_pin_offset_y"]
        has_projected = pin_offset_result["has_projected_pin_offset"]
        changed_mask = pin_offset_result["changed_pin_offset"]

        records: list[dict[str, Any]] = []
        changed_pin_offset_count = int(changed_mask.sum().item())
        projected_pin_count = int(has_projected.sum().item())
        invalid_projected_pin_count = int((legal_mask & ~has_projected).sum().item())
        changed_inst_ids = set(inst_ids[changed_mask].tolist())
        mean_abs_dx = None
        mean_abs_dy = None
        if projected_pin_count > 0:
            mean_abs_dx = float(
                torch.abs(projected_x[has_projected] - pin_offset_x[has_projected]).mean().item()
            )
            mean_abs_dy = float(
                torch.abs(projected_y[has_projected] - pin_offset_y[has_projected]).mean().item()
            )

        for idx in range(pin_ids.numel()):
            px = None
            py = None
            if bool(has_projected[idx].item()):
                px = float(projected_x[idx].item())
                py = float(projected_y[idx].item())
            records.append(
                {
                    "inst_id": int(inst_ids[idx].item()),
                    "pin_id": int(pin_ids[idx].item()),
                    "current_cell_id": int(current_cell_id[idx].item()),
                    "projected_cell_id": int(projected_cell_id[idx].item()),
                    "has_legal_candidate": bool(legal_mask[idx].item()),
                    "libpin_offset": int(libpin_offsets[idx].item()),
                    "current_pin_offset_x": float(pin_offset_x[idx].item()),
                    "current_pin_offset_y": float(pin_offset_y[idx].item()),
                    "projected_pin_offset_x": px,
                    "projected_pin_offset_y": py,
                    "has_projected_pin_offset": bool(has_projected[idx].item()),
                    "changed_pin_offset": bool(changed_mask[idx].item()),
                }
            )

        fieldnames = [
            "inst_id",
            "pin_id",
            "current_cell_id",
            "projected_cell_id",
            "has_legal_candidate",
            "libpin_offset",
            "current_pin_offset_x",
            "current_pin_offset_y",
            "projected_pin_offset_x",
            "projected_pin_offset_y",
            "has_projected_pin_offset",
            "changed_pin_offset",
        ]
        summary = {
            "artifact_version": 1,
            "num_instances": int(result.inst_ids.numel()),
            "num_pin_records": len(records),
            "num_pins_with_projected_offset": projected_pin_count,
            "num_invalid_projected_pins": invalid_projected_pin_count,
            "num_changed_pin_offsets": changed_pin_offset_count,
            "num_instances_with_changed_pin_offsets": len(changed_inst_ids),
            "mean_abs_pin_offset_dx": mean_abs_dx,
            "mean_abs_pin_offset_dy": mean_abs_dy,
            "metadata": self._summary_serializable_value(metadata or {}),
        }
        return ProjectionPinOffsetArtifacts(
            records=records,
            fieldnames=fieldnames,
            summary=summary,
        )

    def build_artifact_summary(
        self,
        result: ProjectionResult,
        metadata: dict[str, Any] | None = None,
        pin_offset_artifacts: ProjectionPinOffsetArtifacts | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        legal_mask = result.has_legal_candidate.bool()
        changed_cell = legal_mask & (
            (result.projected_cell_id != result.current_cell_id)
            | (result.projected_libcell_offset != result.current_libcell_offset)
        )
        changed_size = legal_mask & (result.projected_size != result.current_size)
        changed_vt = legal_mask & (result.projected_vt != result.current_vt)
        finite_cost_mask = legal_mask & torch.isfinite(result.projection_cost)

        avg_projection_cost = None
        max_projection_cost = None
        mean_size_gap = None
        if finite_cost_mask.any():
            legal_costs = result.projection_cost[finite_cost_mask]
            avg_projection_cost = float(legal_costs.mean().item())
            max_projection_cost = float(legal_costs.max().item())
            mean_size_gap = float(
                torch.abs(
                    result.continuous_size[finite_cost_mask]
                    - result.projected_size[finite_cost_mask]
                ).mean().item()
            )
        total_current_leakage = float(result.current_leakage.clamp(min=0).sum().item())
        total_projected_leakage = float(result.projected_leakage[legal_mask].clamp(min=0).sum().item())
        total_leakage_delta = total_projected_leakage - total_current_leakage
        num_leakage_improved_cells = int(
            (result.leakage_delta[legal_mask] < -1e-12).sum().item()
        ) if legal_mask.any() else 0
        term_weights = dict(self.scorer.term_weights())
        raw_term_weights = metadata.get("projection_term_weights") or {}
        term_weights.update(
            {
                str(name): float(weight)
                for name, weight in raw_term_weights.items()
                if isinstance(name, str)
            }
        )
        objective_term_breakdown = {
            "all_legal": self._objective_term_subset_summary(
                result,
                legal_mask,
                term_weights,
            ),
            "changed_cells": self._objective_term_subset_summary(
                result,
                changed_cell,
                term_weights,
            ),
        }

        return {
            "artifact_version": 1,
            "num_instances": int(result.inst_ids.numel()),
            "num_legal_projections": int(legal_mask.sum().item()),
            "num_changed_cells": int(changed_cell.sum().item()),
            "num_changed_sizes": int(changed_size.sum().item()),
            "num_changed_vts": int(changed_vt.sum().item()),
            "avg_projection_cost": avg_projection_cost,
            "max_projection_cost": max_projection_cost,
            "mean_abs_continuous_discrete_size_gap": mean_size_gap,
            "total_current_leakage": total_current_leakage,
            "total_projected_leakage": total_projected_leakage,
            "total_leakage_delta": total_leakage_delta,
            "num_leakage_improved_cells": num_leakage_improved_cells,
            "score_term_names": sorted(result.score_terms.keys()),
            "objective_term_breakdown": objective_term_breakdown,
            "resolution_metadata": self._summary_serializable_value(result.resolution_metadata),
            "validation": {
                "is_valid": bool(result.validation.is_valid),
                "issues": list(result.validation.issues),
                "metrics": self._summary_serializable_value(result.validation.metrics),
            },
            "pin_offset_consistency": None
            if pin_offset_artifacts is None
            else self._summary_serializable_value(pin_offset_artifacts.summary),
            "metadata": self._summary_serializable_value(metadata),
        }

    def compute_projected_pin_offsets(
        self,
        result: ProjectionResult,
        row_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        required_names = (
            "pin_offset_x",
            "pin_offset_y",
            "flat_node2pin_map",
            "flat_node2pin_start_map",
            "pin_2_libpin_offset",
            "cell_id_2_libpin_id_start",
            "flat_lib_pin_offset_x",
            "flat_lib_pin_offset_y",
        )
        missing = [name for name in required_names if getattr(self.data_collections, name, None) is None]
        if missing:
            raise ValueError(
                "compute_projected_pin_offsets requires sizing pin-offset metadata: "
                + ", ".join(missing)
            )

        device = result.inst_ids.device
        if row_mask is None:
            row_mask = torch.ones(
                result.inst_ids.numel(),
                dtype=torch.bool,
                device=device,
            )
        else:
            row_mask = row_mask.bool().to(device=device)
            if row_mask.numel() != result.inst_ids.numel():
                raise ValueError("pin-offset row_mask does not match projection result")
        inst_ids = result.inst_ids[row_mask].long().to(device=device)
        current_cell_id = result.current_cell_id[row_mask].long().to(device=device)
        projected_cell_id = result.projected_cell_id[row_mask].long().to(device=device)
        legal_mask_by_inst = result.has_legal_candidate[row_mask].bool().to(device=device)

        flat_node2pin_map = self.data_collections.flat_node2pin_map.long().to(device=device)
        flat_node2pin_start_map = self.data_collections.flat_node2pin_start_map.long().to(device=device)
        pin_2_libpin_offset = self.data_collections.pin_2_libpin_offset.long().to(device=device)
        cell_id_2_libpin_id_start = self.data_collections.cell_id_2_libpin_id_start.long().to(device=device)
        flat_lib_pin_offset_x = self.data_collections.flat_lib_pin_offset_x.float().to(device=device)
        flat_lib_pin_offset_y = self.data_collections.flat_lib_pin_offset_y.float().to(device=device)
        current_pin_offset_x = self.data_collections.pin_offset_x.float().to(device=device)
        current_pin_offset_y = self.data_collections.pin_offset_y.float().to(device=device)

        if inst_ids.numel() > 0:
            if bool(torch.any(inst_ids < 0).item()) or bool(
                torch.any(inst_ids + 1 >= flat_node2pin_start_map.numel()).item()
            ):
                raise ValueError("projection instance IDs are outside node2pin metadata")

        pin_start_by_row = flat_node2pin_start_map[inst_ids]
        pin_count_by_row = flat_node2pin_start_map[inst_ids + 1] - pin_start_by_row
        if pin_count_by_row.numel() > 0:
            if bool(torch.any(pin_start_by_row < 0).item()) or bool(
                torch.any(pin_count_by_row < 0).item()
            ) or bool(
                torch.any(
                    pin_start_by_row + pin_count_by_row
                    > flat_node2pin_map.numel()
                ).item()
            ):
                raise ValueError("node2pin metadata contains an invalid pin range")
        row_ids = torch.repeat_interleave(
            torch.arange(inst_ids.numel(), device=device, dtype=torch.long),
            pin_count_by_row,
        )
        if row_ids.numel() == 0:
            empty_long = torch.zeros(0, device=device, dtype=torch.long)
            empty_bool = torch.zeros(0, device=device, dtype=torch.bool)
            empty_float = torch.zeros(0, device=device, dtype=torch.float32)
            return {
                "inst_ids": empty_long,
                "pin_ids": empty_long,
                "current_cell_id": empty_long,
                "projected_cell_id": empty_long,
                "legal_mask": empty_bool,
                "libpin_offsets": empty_long,
                "current_pin_offset_x": empty_float,
                "current_pin_offset_y": empty_float,
                "projected_pin_offset_x": empty_float,
                "projected_pin_offset_y": empty_float,
                "has_projected_pin_offset": empty_bool,
                "changed_pin_offset": empty_bool,
            }

        row_prefix = torch.cumsum(pin_count_by_row, dim=0) - pin_count_by_row
        flat_pin_indices = (
            torch.repeat_interleave(pin_start_by_row, pin_count_by_row)
            + torch.arange(row_ids.numel(), device=device, dtype=torch.long)
            - torch.repeat_interleave(row_prefix, pin_count_by_row)
        )
        pin_ids = flat_node2pin_map[flat_pin_indices]
        if bool(torch.any(pin_ids < 0).item()) or bool(
            torch.any(pin_ids >= pin_2_libpin_offset.numel()).item()
        ) or bool(torch.any(pin_ids >= current_pin_offset_x.numel()).item()) or bool(
            torch.any(pin_ids >= current_pin_offset_y.numel()).item()
        ):
            raise ValueError("node2pin metadata contains an invalid pin id")
        inst_ids_flat = inst_ids[row_ids]
        current_cell_flat = current_cell_id[row_ids]
        projected_cell_flat = projected_cell_id[row_ids]
        legal_mask = legal_mask_by_inst[row_ids]

        libpin_offsets = pin_2_libpin_offset[pin_ids]
        projected_libpin_id = torch.full_like(libpin_offsets, -1)
        valid_projected_cell = legal_mask & (projected_cell_flat >= 0)
        if bool(
            torch.any(
                valid_projected_cell
                & (projected_cell_flat + 1 >= cell_id_2_libpin_id_start.numel())
            ).item()
        ):
            raise ValueError("projected cell ID is outside libpin metadata")
        if bool(torch.any(valid_projected_cell & (libpin_offsets < 0)).item()):
            raise ValueError("projected pin has an invalid libpin offset")
        if valid_projected_cell.any():
            projected_libpin_id[valid_projected_cell] = (
                cell_id_2_libpin_id_start[projected_cell_flat[valid_projected_cell]]
                + libpin_offsets[valid_projected_cell]
            )
        if bool(
            torch.any(
                valid_projected_cell
                & (
                    (projected_libpin_id < 0)
                    | (projected_libpin_id >= flat_lib_pin_offset_x.numel())
                    | (projected_libpin_id >= flat_lib_pin_offset_y.numel())
                )
            ).item()
        ):
            raise ValueError("projected libpin ID is outside pin-offset metadata")

        has_projected_pin_offset = (
            valid_projected_cell
            & (libpin_offsets >= 0)
            & (projected_libpin_id >= 0)
            & (projected_libpin_id < flat_lib_pin_offset_x.numel())
        )
        projected_pin_offset_x = torch.zeros_like(current_pin_offset_x[pin_ids])
        projected_pin_offset_y = torch.zeros_like(current_pin_offset_y[pin_ids])
        if has_projected_pin_offset.any():
            projected_pin_offset_x[has_projected_pin_offset] = flat_lib_pin_offset_x[
                projected_libpin_id[has_projected_pin_offset]
            ]
            projected_pin_offset_y[has_projected_pin_offset] = flat_lib_pin_offset_y[
                projected_libpin_id[has_projected_pin_offset]
            ]

        current_selected_pin_offset_x = current_pin_offset_x[pin_ids]
        current_selected_pin_offset_y = current_pin_offset_y[pin_ids]
        changed_pin_offset = has_projected_pin_offset & (
            (torch.abs(projected_pin_offset_x - current_selected_pin_offset_x) > 1e-6)
            | (torch.abs(projected_pin_offset_y - current_selected_pin_offset_y) > 1e-6)
        )

        return {
            "inst_ids": inst_ids_flat,
            "pin_ids": pin_ids,
            "current_cell_id": current_cell_flat,
            "projected_cell_id": projected_cell_flat,
            "legal_mask": legal_mask,
            "libpin_offsets": libpin_offsets,
            "current_pin_offset_x": current_selected_pin_offset_x,
            "current_pin_offset_y": current_selected_pin_offset_y,
            "projected_pin_offset_x": projected_pin_offset_x,
            "projected_pin_offset_y": projected_pin_offset_y,
            "has_projected_pin_offset": has_projected_pin_offset,
            "changed_pin_offset": changed_pin_offset,
        }

    def write_artifacts(
        self,
        result: ProjectionResult,
        result_dir: str,
        design_name: str,
        metadata: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        os.makedirs(result_dir, exist_ok=True)
        profile_started_at = time.perf_counter()
        records, fieldnames = self.build_artifact_records(result)
        if profile is not None:
            profile["artifact_record_build_ms"] = (
                time.perf_counter() - profile_started_at
            ) * 1000.0
        pin_profile_started_at = time.perf_counter()
        pin_offset_artifacts = self.build_pin_offset_artifacts(result, metadata=metadata)
        if profile is not None:
            profile["pin_offset_compute_ms"] = (
                time.perf_counter() - pin_profile_started_at
            ) * 1000.0
        summary = self.build_artifact_summary(
            result,
            metadata=metadata,
            pin_offset_artifacts=pin_offset_artifacts,
        )

        jsonl_path = os.path.join(result_dir, f"{design_name}_projection.jsonl")
        csv_path = os.path.join(result_dir, f"{design_name}_projection.csv")
        summary_path = os.path.join(result_dir, f"{design_name}_projection_summary.json")
        pin_offset_jsonl_path = os.path.join(result_dir, f"{design_name}_projection_pin_offsets.jsonl")
        pin_offset_csv_path = os.path.join(result_dir, f"{design_name}_projection_pin_offsets.csv")
        pin_offset_summary_path = os.path.join(
            result_dir,
            f"{design_name}_projection_pin_offsets_summary.json",
        )

        artifact_write_started_at = time.perf_counter()
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

        artifact_paths = {
            "jsonl": jsonl_path,
            "csv": csv_path,
            "summary": summary_path,
        }
        if pin_offset_artifacts is not None:
            with open(pin_offset_jsonl_path, "w", encoding="utf-8") as f:
                for record in pin_offset_artifacts.records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            with open(pin_offset_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=pin_offset_artifacts.fieldnames)
                writer.writeheader()
                for record in pin_offset_artifacts.records:
                    writer.writerow(record)

            with open(pin_offset_summary_path, "w", encoding="utf-8") as f:
                json.dump(pin_offset_artifacts.summary, f, ensure_ascii=False, indent=2)
                f.write("\n")

            artifact_paths["pin_offset_jsonl"] = pin_offset_jsonl_path
            artifact_paths["pin_offset_csv"] = pin_offset_csv_path
            artifact_paths["pin_offset_summary"] = pin_offset_summary_path

        if profile is not None:
            profile["artifact_write_ms"] = (
                time.perf_counter() - artifact_write_started_at
            ) * 1000.0

        return artifact_paths
