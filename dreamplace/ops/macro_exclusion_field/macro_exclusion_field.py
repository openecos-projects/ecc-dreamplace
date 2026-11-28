"""Seamless Torsion-Dome + Distance Macro Exclusion Field and Secant MM Curvature.

Forward-derived continuous formulation for fixed-macro legalization-aware TDP:

1. Seamless 2D Elliptic-Streamline Obstacle Potential ``psi_M(x, y)`` (units ``[pos_unit]``):
   - Extracts all fixed multi-row macro shapes (grouping ``.DREAMPlace.Shape<k>``
     sub-rectangles by base instance so rectilinear macro notches are preserved
     while 1-row fixed standard/clock cells are never haloed) unified with
     uncovered ``row_gap_boxes``.
   - Solves the 2D Dirichlet Saint-Venant torsion PDE ``-Delta v = 1`` on the
     collar-dilated obstacle ``s > -ext_shift`` (``v = 0`` outside) via SPD
     5-point Conjugate Gradient, forming the scale-free roof dome / tent-rim profile
     whose strictly convex elliptic level sets extend through both the macro
     interior and exterior collar so off-center cells grazing a macro wall
     receive a smooth tangential deflection guiding them around the nearer
     macro corner instead of stalling at a flat normal saddle.
   - Applies a ``C^1`` quadratic onset and compact exterior cutoff, and evaluates
     on GPU via sparse-stencil uniform cubic B-spline interpolation
     (``C^2`` value, ``C^1`` analytical gradient, identically zero outside support).

2. Force-Matched Analytical Pull Weight ``w_i`` (units ``[Objective / pos_unit]``):
   - Directly matches each movable cell's exact Lipschitz wirelength + published
     Star timing inward slope, gated by self-calibrated physical evacuation delay
     slack reserve (``kappa_evac``) or uniform / ratchet modes.

3. Exact Secant MM Curvature Diagonal ``H_macro`` (units ``[Objective / pos_unit^2]``):
   - ``H_macro,i = w_i * ||grad psi_M(c_i)||_2 / max(psi_M(c_i) + delta, 2*delta)``
     added directly into ``PreconditionOp`` diagonal curvature, guaranteeing bounded
     single-step macro clearance without overshoot and triggering Nesterov
     ``transport_momentum`` damping automatically.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from dreamplace.ops.star_pin_pair.rowsum_curvature import rowsum_curvature_load
from dreamplace.row_gap_blockage import row_gap_boxes


def extract_macro_and_gap_boxes(
    placedb,
    pos: torch.Tensor,
    *,
    min_macro_rows: float = 2.0,
    include_row_gaps: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Extract ``(K, 4)`` ``(xl, yl, xh, yh)`` obstacle boxes in ``pos_unit``s."""
    n = int(placedb.num_nodes)
    m = int(placedb.num_movable_nodes)
    num_terminals = int(getattr(placedb, "num_terminals", 0))
    row_height = float(placedb.row_height)
    site_width = float(placedb.site_width)

    flat = pos.detach().reshape(-1).double().cpu().numpy()
    sx_raw = getattr(placedb, "node_size_x", None)
    sy_raw = getattr(placedb, "node_size_y", None)
    sx = np.asarray(sx_raw.detach().cpu().numpy() if torch.is_tensor(sx_raw) else sx_raw, dtype=np.float64)
    sy = np.asarray(sy_raw.detach().cpu().numpy() if torch.is_tensor(sy_raw) else sy_raw, dtype=np.float64)

    macro_boxes = []
    macro_base_count = 0
    if num_terminals > 0:
        f_start = m
        f_end = min(m + num_terminals, n, sx.size, sy.size)
        fx = flat[f_start:f_end]
        fy = flat[n + f_start : n + f_end]
        fw = sx[f_start:f_end]
        fh = sy[f_start:f_end]
        valid = (fw > 1e-6) & (fh > 1e-6)

        raw_names = getattr(placedb, "node_names", None)
        if raw_names is not None and len(raw_names) >= f_end:
            span_min: Dict[str, float] = {}
            span_max: Dict[str, float] = {}
            base_names = []
            for idx in range(f_end - f_start):
                raw_name = raw_names[f_start + idx]
                name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
                base = name.split(".DREAMPlace.Shape", 1)[0]
                base_names.append(base)
                if not valid[idx]:
                    continue
                yl_i = float(fy[idx])
                yh_i = float(fy[idx] + fh[idx])
                if base not in span_min or yl_i < span_min[base]:
                    span_min[base] = yl_i
                if base not in span_max or yh_i > span_max[base]:
                    span_max[base] = yh_i
            threshold = float(min_macro_rows) * row_height
            tall_bases = {
                b for b in span_min if (span_max[b] - span_min[b]) > threshold
            }
            macro_base_count = len(tall_bases)
            keep = np.array(
                [valid[idx] and (base_names[idx] in tall_bases) for idx in range(len(base_names))],
                dtype=bool,
            )
        else:
            keep = valid & (fh > float(min_macro_rows) * row_height)
            macro_base_count = int(keep.sum())

        if keep.any():
            macro_boxes.append(
                np.stack(
                    [fx[keep], fy[keep], fx[keep] + fw[keep], fy[keep] + fh[keep]],
                    axis=1,
                )
            )

    gap_count = 0
    rows = getattr(placedb, "rows", None)
    if include_row_gaps and rows is not None and len(rows):
        gaps = row_gap_boxes(
            rows,
            float(placedb.xl),
            float(placedb.yl),
            float(placedb.xh),
            float(placedb.yh),
            site_width,
            row_height,
        )
        if len(gaps):
            gap_count = int(len(gaps))
            macro_boxes.append(np.asarray(gaps, dtype=np.float64).reshape(-1, 4))

    if macro_boxes:
        all_boxes = np.concatenate(macro_boxes, axis=0)
    else:
        all_boxes = np.zeros((0, 4), dtype=np.float64)

    stats = {
        "macro_base_instances": int(macro_base_count),
        "macro_shape_boxes": int(len(all_boxes) - gap_count),
        "row_gap_boxes": int(gap_count),
        "total_obstacle_boxes": int(len(all_boxes)),
    }
    return all_boxes, stats


def _rasterize_boxes(
    boxes: np.ndarray,
    g_xl: float,
    g_yl: float,
    h_g: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Rasterize ``(K, 4)`` boxes onto grid centers ``(g_xl + (i + 0.5)*h_g, ...)``."""
    if len(boxes) == 0:
        return np.zeros((ny, nx), dtype=bool)
    ix0 = np.ceil((boxes[:, 0] - g_xl) / h_g - 0.5 - 1e-9).astype(np.int64)
    ix1 = np.floor((boxes[:, 2] - g_xl) / h_g - 0.5 + 1e-9).astype(np.int64) + 1
    iy0 = np.ceil((boxes[:, 1] - g_yl) / h_g - 0.5 - 1e-9).astype(np.int64)
    iy1 = np.floor((boxes[:, 3] - g_yl) / h_g - 0.5 + 1e-9).astype(np.int64) + 1

    ix0 = np.clip(ix0, 0, nx)
    ix1 = np.clip(ix1, 0, nx)
    iy0 = np.clip(iy0, 0, ny)
    iy1 = np.clip(iy1, 0, ny)
    valid = (ix1 > ix0) & (iy1 > iy0)
    if not valid.any():
        return np.zeros((ny, nx), dtype=bool)
    ix0, ix1, iy0, iy1 = ix0[valid], ix1[valid], iy0[valid], iy1[valid]

    diff = np.zeros((ny + 1, nx + 1), dtype=np.int32)
    np.add.at(diff, (iy0, ix0), 1)
    np.add.at(diff, (iy0, ix1), -1)
    np.add.at(diff, (iy1, ix0), -1)
    np.add.at(diff, (iy1, ix1), 1)
    cov = np.cumsum(np.cumsum(diff, axis=0), axis=1)[:ny, :nx]
    return cov > 0


def _solve_dirichlet_torsion(
    mask_np: np.ndarray,
    h_g: float,
    device: torch.device,
    *,
    max_cg_iters: int = 350,
    rtol: float = 1e-5,
) -> np.ndarray:
    """Solve ``-Delta v = 1`` on ``mask_np`` (``v = 0`` outside) via SPD 5-point CG."""
    if not mask_np.any():
        return np.zeros_like(mask_np, dtype=np.float64)

    try:
        return _solve_dirichlet_torsion_impl(
            mask_np, h_g, device, max_cg_iters=max_cg_iters, rtol=rtol
        )
    except torch.OutOfMemoryError:
        if device.type != "cpu":
            torch.cuda.empty_cache()
            return _solve_dirichlet_torsion_impl(
                mask_np, h_g, torch.device("cpu"), max_cg_iters=max_cg_iters, rtol=rtol
            )
        raise


def _solve_dirichlet_torsion_impl(
    mask_np: np.ndarray,
    h_g: float,
    device: torch.device,
    *,
    max_cg_iters: int = 350,
    rtol: float = 1e-5,
) -> np.ndarray:
    mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.float64)
    b = mask_t * (h_g * h_g)

    def matvec(u: torch.Tensor) -> torch.Tensor:
        up = torch.zeros_like(u)
        down = torch.zeros_like(u)
        left = torch.zeros_like(u)
        right = torch.zeros_like(u)
        up[1:, :] = u[:-1, :]
        down[:-1, :] = u[1:, :]
        left[:, 1:] = u[:, :-1]
        right[:, :-1] = u[:, 1:]
        return (4.0 * u - up - down - left - right) * mask_t

    v = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = float(torch.sum(r * r).item())
    if rs_old <= 0.0:
        return np.zeros_like(mask_np, dtype=np.float64)
    tol_sq = (rtol * rtol) * rs_old

    for _ in range(int(max_cg_iters)):
        ap = matvec(p)
        p_ap = float(torch.sum(p * ap).item())
        if p_ap <= 1e-30:
            break
        alpha = rs_old / p_ap
        v.add_(p, alpha=alpha)
        r.add_(ap, alpha=-alpha)
        rs_new = float(torch.sum(r * r).item())
        if rs_new <= tol_sq:
            break
        p.mul_(rs_new / rs_old).add_(r)
        rs_old = rs_new

    return v.clamp(min=0.0).cpu().numpy()


def _solve_dirichlet_torsion_multilevel(
    mask_np: np.ndarray,
    h_g: float,
    device: torch.device,
    *,
    coarse_stride: int = 4,
    coarse_iters: int = 600,
    fine_iters: int = 220,
    rtol: float = 1e-5,
) -> np.ndarray:
    """2-level coarse-to-fine CG for ``-Delta v = 1`` so 1000um+ macros converge fully in ~1.3s."""
    if not mask_np.any():
        return np.zeros_like(mask_np, dtype=np.float64)
    ny, nx = mask_np.shape
    if min(ny, nx) < 4 * coarse_stride:
        return _solve_dirichlet_torsion(
            mask_np, h_g, device=device, max_cg_iters=max(coarse_iters, fine_iters), rtol=rtol
        )
    try:
        return _solve_dirichlet_torsion_multilevel_impl(
            mask_np,
            h_g,
            device,
            coarse_stride=coarse_stride,
            coarse_iters=coarse_iters,
            fine_iters=fine_iters,
            rtol=rtol,
        )
    except torch.OutOfMemoryError:
        if device.type != "cpu":
            torch.cuda.empty_cache()
            return _solve_dirichlet_torsion_multilevel_impl(
                mask_np,
                h_g,
                torch.device("cpu"),
                coarse_stride=coarse_stride,
                coarse_iters=coarse_iters,
                fine_iters=fine_iters,
                rtol=rtol,
            )
        raise


def _solve_dirichlet_torsion_multilevel_impl(
    mask_np: np.ndarray,
    h_g: float,
    device: torch.device,
    *,
    coarse_stride: int,
    coarse_iters: int,
    fine_iters: int,
    rtol: float,
) -> np.ndarray:
    import torch.nn.functional as F

    ny, nx = mask_np.shape
    mask_t = torch.as_tensor(mask_np, device=device, dtype=torch.float64)
    pooled = F.avg_pool2d(
        mask_t[None, None], kernel_size=coarse_stride, stride=coarse_stride
    )[0, 0]
    mask_coarse = (pooled > 0.5).cpu().numpy()
    v_coarse = _solve_dirichlet_torsion_impl(
        mask_coarse,
        h_g * float(coarse_stride),
        device,
        max_cg_iters=coarse_iters,
        rtol=min(rtol, 1e-6),
    )
    v = (
        F.interpolate(
            torch.as_tensor(v_coarse, device=device, dtype=torch.float64)[None, None],
            size=(ny, nx),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        * mask_t
    )
    b = mask_t * (h_g * h_g)
    b_norm_sq = float(torch.sum(b * b).item())
    if b_norm_sq <= 0.0:
        return np.zeros_like(mask_np, dtype=np.float64)
    tol_sq = (rtol * rtol) * b_norm_sq

    def matvec(u: torch.Tensor) -> torch.Tensor:
        up = torch.zeros_like(u)
        down = torch.zeros_like(u)
        left = torch.zeros_like(u)
        right = torch.zeros_like(u)
        up[1:, :] = u[:-1, :]
        down[:-1, :] = u[1:, :]
        left[:, 1:] = u[:, :-1]
        right[:, :-1] = u[:, 1:]
        return (4.0 * u - up - down - left - right) * mask_t

    r = (b - matvec(v)) * mask_t
    p = r.clone()
    rs_old = float(torch.sum(r * r).item())
    if rs_old > tol_sq:
        for _ in range(int(fine_iters)):
            ap = matvec(p)
            p_ap = float(torch.sum(p * ap).item())
            if p_ap <= 1e-30:
                break
            alpha = rs_old / p_ap
            v.add_(p, alpha=alpha)
            r.add_(ap, alpha=-alpha)
            rs_new = float(torch.sum(r * r).item())
            if rs_new <= tol_sq:
                break
            p.mul_(rs_new / rs_old).add_(r)
            rs_old = rs_new

    return v.clamp(min=0.0).cpu().numpy()


def _solve_weighted_torsion_multilevel(
    active_np: np.ndarray,
    cond_np: np.ndarray,
    in_core_np: np.ndarray,
    h_g: float,
    device: torch.device,
    *,
    coarse_stride: int = 4,
    coarse_iters: int = 600,
    fine_iters: int = 240,
    rtol: float = 1e-5,
) -> np.ndarray:
    """Solve ``-div(cond * grad v) = 1`` on ``active_np`` with Neumann BC on ``in_core_np`` walls
    and Dirichlet ``v = 0`` on legal core whitespace ``in_core_np & (~active_np)``."""
    if not active_np.any():
        return np.zeros_like(active_np, dtype=np.float64)
    if not bool((in_core_np & (~active_np)).any()):
        return _solve_dirichlet_torsion_multilevel(
            active_np,
            h_g,
            device,
            coarse_stride=coarse_stride,
            coarse_iters=coarse_iters,
            fine_iters=fine_iters,
            rtol=rtol,
        )

    def _solve_single_level(
        act_m: np.ndarray,
        c_m: np.ndarray,
        core_m: np.ndarray,
        step_h: float,
        u_init: Optional[torch.Tensor],
        max_iters: int,
        tol: float,
        dev: torch.device,
    ) -> torch.Tensor:
        act_t = torch.as_tensor(act_m, device=dev, dtype=torch.float64)
        w_t = torch.as_tensor(c_m, device=dev, dtype=torch.float64).clamp(min=1e-4)
        core_t = torch.as_tensor(core_m, device=dev, dtype=torch.bool)
        wx = 0.5 * (w_t[:, :-1] + w_t[:, 1:]) * (core_t[:, :-1] & core_t[:, 1:]).to(torch.float64)
        wy = 0.5 * (w_t[:-1, :] + w_t[1:, :]) * (core_t[:-1, :] & core_t[1:, :]).to(torch.float64)

        b = act_t * (step_h * step_h)
        b_norm_sq = float(torch.sum(b * b).item())
        if b_norm_sq <= 0.0:
            return torch.zeros_like(b)
        tol_sq = (tol * tol) * b_norm_sq

        def matvec(u: torch.Tensor) -> torch.Tensor:
            u_eff = u * act_t
            fx = wx * (u_eff[:, :-1] - u_eff[:, 1:])
            fy = wy * (u_eff[:-1, :] - u_eff[1:, :])
            out = torch.zeros_like(u_eff)
            out[:, :-1].add_(fx)
            out[:, 1:].sub_(fx)
            out[:-1, :].add_(fy)
            out[1:, :].sub_(fy)
            return out * act_t

        v = torch.zeros_like(b) if u_init is None else (u_init * act_t)
        r = (b - matvec(v)) * act_t
        p = r.clone()
        rs_old = float(torch.sum(r * r).item())
        if rs_old > tol_sq:
            for _ in range(int(max_iters)):
                ap = matvec(p)
                p_ap = float(torch.sum(p * ap).item())
                if p_ap <= 1e-30:
                    break
                alpha = rs_old / p_ap
                v.add_(p, alpha=alpha)
                r.add_(ap, alpha=-alpha)
                rs_new = float(torch.sum(r * r).item())
                if rs_new <= tol_sq:
                    break
                p.mul_(rs_new / rs_old).add_(r)
                rs_old = rs_new
        return v.clamp(min=0.0)

    def _run_on_device(dev: torch.device) -> np.ndarray:
        import torch.nn.functional as F

        ny, nx = active_np.shape
        if min(ny, nx) < 4 * coarse_stride:
            return _solve_single_level(
                active_np, cond_np, in_core_np, h_g, None, max(coarse_iters, fine_iters), rtol, dev
            ).cpu().numpy()

        act_f = torch.as_tensor(active_np, device=dev, dtype=torch.float64)[None, None]
        cond_f = torch.as_tensor(cond_np, device=dev, dtype=torch.float64)[None, None]
        core_f = torch.as_tensor(in_core_np, device=dev, dtype=torch.float64)[None, None]
        act_c = (F.avg_pool2d(act_f, kernel_size=coarse_stride, stride=coarse_stride)[0, 0] > 0.5).cpu().numpy()
        core_c = (F.avg_pool2d(core_f, kernel_size=coarse_stride, stride=coarse_stride)[0, 0] > 0.5).cpu().numpy()
        cond_c = F.avg_pool2d(cond_f, kernel_size=coarse_stride, stride=coarse_stride)[0, 0].cpu().numpy()
        act_c = act_c & core_c
        if bool(act_c.any()) and bool((core_c & (~act_c)).any()):
            vc = _solve_single_level(
                act_c, cond_c, core_c, h_g * float(coarse_stride), None, coarse_iters, min(rtol, 1e-6), dev
            )
            v_init = F.interpolate(
                vc[None, None], size=(ny, nx), mode="bilinear", align_corners=False
            )[0, 0]
        else:
            v_init = None
        vf = _solve_single_level(active_np, cond_np, in_core_np, h_g, v_init, fine_iters, rtol, dev)
        return vf.cpu().numpy()

    try:
        return _run_on_device(device)
    except torch.OutOfMemoryError:
        if device.type != "cpu":
            torch.cuda.empty_cache()
            return _run_on_device(torch.device("cpu"))
        raise


def _exact_signed_distance(mask_np: np.ndarray, h_g: float) -> np.ndarray:
    """Signed inward Euclidean distance ``s`` (positive inside ``mask_np``, ``[pos_unit]``)."""
    from scipy.ndimage import distance_transform_edt

    d_in = distance_transform_edt(mask_np) * h_g
    d_out = distance_transform_edt(~mask_np) * h_g
    return np.where(mask_np, d_in - 0.5 * h_g, -(d_out - 0.5 * h_g))


def _gaussian_smooth_2d(field: np.ndarray, sigma_cells: float) -> np.ndarray:
    if sigma_cells <= 0.0:
        return field
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(field, sigma=float(sigma_cells), mode="nearest")


def _build_bspline_support_mask(psi_np: np.ndarray) -> np.ndarray:
    """Boolean ``(ny, nx)`` mask True iff any ``4x4`` B-spline stencil entry around ``(iy, ix)`` is > 0."""
    from scipy.ndimage import maximum_filter

    return maximum_filter(psi_np > 0.0, size=5, mode="constant", cval=False)


def build_macro_potential_grid(
    boxes: np.ndarray,
    xl: float,
    yl: float,
    xh: float,
    yh: float,
    row_height: float,
    device: torch.device,
    *,
    grid_pitch_rows: float = 1.0,
    collar_rows: float = 1.0,
    distance_blend: float = 0.15,
    smooth_sigma_cells: float = 0.5,
    profile_mode: str = "dome",
    ridge_smooth_rows: float = 3.0,
    bell_sigma_rows: float = 16.0,
    bell_bias: float = 0.80,
    max_grid_dim: int = 4096,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Precompute seamless elliptic-streamline ``psi_M`` grid of shape ``(ny, nx)`` in ``[pos_unit]``."""
    if len(boxes) == 0:
        return None, {"has_obstacles": 0.0}

    mode = str(profile_mode or "dome").strip().lower()
    if mode not in ("dome", "tent", "tent_rim", "tent_bell"):
        raise ValueError(f"Unsupported macro_exclusion profile_mode={profile_mode!r}")

    span_x = max(float(xh - xl), row_height)
    span_y = max(float(yh - yl), row_height)
    h_g = max(
        float(grid_pitch_rows) * row_height,
        span_x / float(max_grid_dim - 16),
        span_y / float(max_grid_dim - 16),
    )
    pad = max(6, int(math.ceil((float(collar_rows) * row_height) / h_g)) + 4)
    g_xl = float(xl) - pad * h_g
    g_yl = float(yl) - pad * h_g
    nx = int(math.ceil(span_x / h_g)) + 2 * pad
    ny = int(math.ceil(span_y / h_g)) + 2 * pad

    mask = _rasterize_boxes(boxes, g_xl, g_yl, h_g, nx, ny)
    if not mask.any():
        return None, {"has_obstacles": 0.0}

    s = _exact_signed_distance(mask, h_g)
    delta = max(float(collar_rows) * row_height, 0.5 * h_g)
    theta = float(np.clip(distance_blend, 0.0, 1.0))

    # Exact geometric cell-body overlap depth (positive iff s > -0.5 * row_height)
    pen_grid_np = np.maximum(s + 0.5 * float(row_height), 0.0).astype(np.float32)

    if mode == "dome":
        mask_collar = s > (-1.25 * delta)
        v = _solve_dirichlet_torsion(mask_collar, h_g, device=device)

        vx = np.zeros_like(v)
        vy = np.zeros_like(v)
        vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * h_g)
        vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * h_g)
        eps_v = 0.25 * h_g
        denom = np.sqrt(np.maximum(2.0 * v + vx * vx + vy * vy + eps_v * eps_v, 1e-12))
        u_dome = np.where(mask_collar, np.maximum(v / denom - 0.5 * h_g, 0.0), 0.0)

        s_col = np.maximum(s + 1.25 * delta, 0.0)
        u_raw = (1.0 - theta) * u_dome + theta * s_col

        psi_raw = np.where(
            u_raw <= 0.0,
            0.0,
            np.where(
                u_raw < 2.0 * delta,
                np.square(u_raw) / (4.0 * delta),
                u_raw - delta,
            ),
        )
        psi_smooth = _gaussian_smooth_2d(psi_raw, sigma_cells=smooth_sigma_cells)
        u_cut = np.clip((s + 1.5 * delta) / max(0.75 * delta, 1e-6), 0.0, 1.0)
        cutoff = u_cut * u_cut * (3.0 - 2.0 * u_cut)
        psi_smooth = np.where(s <= -1.5 * delta, 0.0, np.maximum(psi_smooth * cutoff, 0.0))
    elif mode == "tent_bell":
        ext_shift = 1.60 * delta
        ramp_width = 1.25 * delta
        cut_outer = 1.75 * delta
        cut_span = 0.50 * delta
        pedestal = 0.25 * h_g

        xs_c = g_xl + (np.arange(nx, dtype=np.float64) + 0.5) * h_g
        ys_c = g_yl + (np.arange(ny, dtype=np.float64) + 0.5) * h_g
        in_core = (
            (ys_c[:, None] >= float(yl))
            & (ys_c[:, None] <= float(yh))
            & (xs_c[None, :] >= float(xl))
            & (xs_c[None, :] <= float(xh))
        )
        legal_ws = in_core & (~mask)
        sigma_bell_cells = max(float(bell_sigma_rows) * float(row_height) / h_g, 1.0)
        non_legal_f = (~legal_ws).astype(np.float64)
        p_bell = np.clip(_gaussian_smooth_2d(non_legal_f, sigma_cells=sigma_bell_cells), 0.0, 1.0)

        if bool(legal_ws.any()):
            from scipy.ndimage import distance_transform_edt

            d_to_legal_px, near_idx = distance_transform_edt(
                ~legal_ws, return_distances=True, return_indices=True
            )
            d_to_legal = d_to_legal_px * h_g
            d_from_legal = distance_transform_edt(legal_ws) * h_g
            s_legal = np.where(legal_ws, -(d_from_legal - 0.5 * h_g), d_to_legal - 0.5 * h_g)
            p_near = p_bell[near_idx[0], near_idx[1]]
            p_exit_raw = np.clip((p_near - 0.50) / 0.50, 0.0, 1.0)
            smooth_seam = max(float(ridge_smooth_rows) * float(row_height) / h_g, 2.0)
            p_exit = np.clip(_gaussian_smooth_2d(p_exit_raw, sigma_cells=smooth_seam), 0.0, 1.0)
        else:
            s_legal = s
            p_exit = p_bell

        m_slope = 1.0 + max(float(bell_bias), 0.0) * p_exit
        cond = 1.0 / np.square(m_slope)

        active_core = in_core & (s_legal > (-ext_shift))
        v = _solve_weighted_torsion_multilevel(active_core, cond, in_core, h_g, device=device)

        ix0 = int(np.argmax(xs_c >= float(xl)))
        ix1 = int(nx - 1 - np.argmax((xs_c <= float(xh))[::-1]))
        iy0 = int(np.argmax(ys_c >= float(yl)))
        iy1 = int(ny - 1 - np.argmax((ys_c <= float(yh))[::-1]))
        if 0 <= iy0 <= iy1 < ny and 0 <= ix0 <= ix1 < nx:
            v_core = v[iy0 : iy1 + 1, ix0 : ix1 + 1]
            v = np.pad(
                v_core,
                ((iy0, ny - 1 - iy1), (ix0, nx - 1 - ix1)),
                mode="symmetric",
            )
            cond_core = cond[iy0 : iy1 + 1, ix0 : ix1 + 1]
            cond = np.pad(
                cond_core,
                ((iy0, ny - 1 - iy1), (ix0, nx - 1 - ix1)),
                mode="symmetric",
            )
            sl_core = s_legal[iy0 : iy1 + 1, ix0 : ix1 + 1]
            s_legal_ref = np.pad(
                sl_core,
                ((iy0, ny - 1 - iy1), (ix0, nx - 1 - ix1)),
                mode="symmetric",
            )
        else:
            s_legal_ref = s_legal

        vx = np.zeros_like(v)
        vy = np.zeros_like(v)
        vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * h_g)
        vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * h_g)
        r_ridge = max(float(ridge_smooth_rows) * row_height, 0.5 * h_g)
        w_grad_sq = cond * (vx * vx + vy * vy)
        r_sq = r_ridge * r_ridge
        denom = np.sqrt(np.maximum(2.0 * v + w_grad_sq + r_sq, 1e-12)) + np.sqrt(
            np.maximum(w_grad_sq + r_sq, 1e-12)
        )
        mask_ext = s_legal_ref > (-ext_shift)
        u_raw = np.where(mask_ext, np.maximum((2.0 * v) / denom - pedestal, 0.0), 0.0)

        psi_raw = np.where(
            u_raw <= 0.0,
            0.0,
            np.where(
                u_raw < ramp_width,
                np.square(u_raw) / (2.0 * ramp_width),
                u_raw - 0.5 * ramp_width,
            ),
        )
        psi_smooth = _gaussian_smooth_2d(psi_raw, sigma_cells=smooth_sigma_cells)
        u_cut = np.clip((s + cut_outer) / max(cut_span, 1e-6), 0.0, 1.0)
        cutoff = u_cut * u_cut * (3.0 - 2.0 * u_cut)
        psi_smooth = np.where(s <= (-cut_outer), 0.0, np.maximum(psi_smooth * cutoff, 0.0))
        pen_grid_np = np.where(
            s <= (-0.5 * float(row_height)),
            0.0,
            np.maximum(s_legal_ref + 0.5 * float(row_height), 0.0),
        ).astype(np.float32)
    else:
        ext_shift = 1.60 * delta if mode == "tent_rim" else 1.25 * delta
        ramp_width = 1.25 * delta if mode == "tent_rim" else 2.0 * delta
        cut_outer = 1.75 * delta if mode == "tent_rim" else 1.5 * delta
        cut_span = 0.50 * delta if mode == "tent_rim" else 0.75 * delta
        pedestal = 0.25 * h_g if mode == "tent_rim" else 0.5 * h_g

        mask_collar = s > (-ext_shift)
        v = _solve_dirichlet_torsion_multilevel(mask_collar, h_g, device=device)

        vx = np.zeros_like(v)
        vy = np.zeros_like(v)
        vx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2.0 * h_g)
        vy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * h_g)
        r_ridge = max(float(ridge_smooth_rows) * row_height, 0.5 * h_g)
        grad_sq = vx * vx + vy * vy
        r_sq = r_ridge * r_ridge
        denom = np.sqrt(np.maximum(2.0 * v + grad_sq + r_sq, 1e-12)) + np.sqrt(
            np.maximum(grad_sq + r_sq, 1e-12)
        )
        u_tent = np.where(mask_collar, np.maximum((2.0 * v) / denom - pedestal, 0.0), 0.0)

        s_col = np.maximum(s + ext_shift, 0.0)
        u_raw = (1.0 - theta) * u_tent + theta * s_col

        psi_raw = np.where(
            u_raw <= 0.0,
            0.0,
            np.where(
                u_raw < ramp_width,
                np.square(u_raw) / (2.0 * ramp_width),
                u_raw - 0.5 * ramp_width,
            ),
        )
        psi_smooth = _gaussian_smooth_2d(psi_raw, sigma_cells=smooth_sigma_cells)
        u_cut = np.clip((s + cut_outer) / max(cut_span, 1e-6), 0.0, 1.0)
        cutoff = u_cut * u_cut * (3.0 - 2.0 * u_cut)
        psi_smooth = np.where(s <= (-cut_outer), 0.0, np.maximum(psi_smooth * cutoff, 0.0))

    grid_tensor = torch.as_tensor(psi_smooth, device=device, dtype=torch.float32).contiguous()
    pen_tensor = torch.as_tensor(pen_grid_np, device=device, dtype=torch.float32).contiguous()
    meta: Dict[str, Any] = {
        "has_obstacles": 1.0,
        "profile_mode": mode,
        "ridge_smooth_rows": float(ridge_smooth_rows),
        "g_xl": float(g_xl),
        "g_yl": float(g_yl),
        "h_g": float(h_g),
        "delta": float(delta),
        "nx": float(nx),
        "ny": float(ny),
        "covered_fraction": float(mask[pad : ny - pad, pad : nx - pad].mean()),
        "psi_max": float(psi_smooth.max()),
        "pen_grid": pen_tensor,
    }
    return grid_tensor, meta


def _eval_cubic_bspline_2d(
    grid: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    g_xl: float,
    g_yl: float,
    h_g: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate uniform cubic B-spline value and analytical ``(d/dx, d/dy)`` on ``grid``."""
    ny, nx = int(grid.shape[0]), int(grid.shape[1])
    inv_h = 1.0 / float(h_g)
    u = ((cx - float(g_xl)) * inv_h - 0.5).clamp(1.0, float(nx - 2) - 1e-5)
    v = ((cy - float(g_yl)) * inv_h - 0.5).clamp(1.0, float(ny - 2) - 1e-5)

    ix = torch.floor(u)
    iy = torch.floor(v)
    tx = u - ix
    ty = v - iy
    ix_l = ix.long()
    iy_l = iy.long()

    omt_x = 1.0 - tx
    tx2 = tx * tx
    tx3 = tx2 * tx
    bx0 = (1.0 / 6.0) * omt_x * omt_x * omt_x
    bx1 = (1.0 / 6.0) * (3.0 * tx3 - 6.0 * tx2 + 4.0)
    bx2 = (1.0 / 6.0) * (-3.0 * tx3 + 3.0 * tx2 + 3.0 * tx + 1.0)
    bx3 = (1.0 / 6.0) * tx3

    dbx0 = -0.5 * omt_x * omt_x
    dbx1 = 0.5 * (3.0 * tx2 - 4.0 * tx)
    dbx2 = 0.5 * (-3.0 * tx2 + 2.0 * tx + 1.0)
    dbx3 = 0.5 * tx2

    omt_y = 1.0 - ty
    ty2 = ty * ty
    ty3 = ty2 * ty
    by0 = (1.0 / 6.0) * omt_y * omt_y * omt_y
    by1 = (1.0 / 6.0) * (3.0 * ty3 - 6.0 * ty2 + 4.0)
    by2 = (1.0 / 6.0) * (-3.0 * ty3 + 3.0 * ty2 + 3.0 * ty + 1.0)
    by3 = (1.0 / 6.0) * ty3

    dby0 = -0.5 * omt_y * omt_y
    dby1 = 0.5 * (3.0 * ty2 - 4.0 * ty)
    dby2 = 0.5 * (-3.0 * ty2 + 2.0 * ty + 1.0)
    dby3 = 0.5 * ty2

    bx = (bx0, bx1, bx2, bx3)
    dbx = (dbx0, dbx1, dbx2, dbx3)
    by = (by0, by1, by2, by3)
    dby = (dby0, dby1, dby2, dby3)

    flat_grid = grid.reshape(-1)
    val = torch.zeros_like(cx)
    gx = torch.zeros_like(cx)
    gy = torch.zeros_like(cx)

    for a in range(4):
        row_base = (iy_l + (a - 1)) * nx
        by_a = by[a]
        dby_a = dby[a]
        row_val = torch.zeros_like(cx)
        row_dx = torch.zeros_like(cx)
        for b in range(4):
            g_ab = flat_grid[row_base + (ix_l + (b - 1))].to(cx.dtype)
            row_val = row_val + bx[b] * g_ab
            row_dx = row_dx + dbx[b] * g_ab
        val = val + by_a * row_val
        gx = gx + by_a * row_dx
        gy = gy + dby_a * row_val

    gx = gx * inv_h
    gy = gy * inv_h
    return val, gx, gy


class _MacroFieldObjectiveFunction(Function):
    @staticmethod
    def forward(
        ctx,
        pos: torch.Tensor,
        grid: torch.Tensor,
        active_idx: torch.Tensor,
        active_weights: torch.Tensor,
        half_w: torch.Tensor,
        half_h: torch.Tensor,
        num_nodes: int,
        g_xl: float,
        g_yl: float,
        h_g: float,
    ) -> torch.Tensor:
        if active_idx.numel() == 0:
            ctx.has_active = False
            ctx.num_total = pos.numel()
            return pos.new_zeros(())
        cx = pos[active_idx] + half_w
        cy = pos[num_nodes + active_idx] + half_h
        val, gx, gy = _eval_cubic_bspline_2d(grid, cx, cy, g_xl, g_yl, h_g)
        w = active_weights.to(pos.dtype)
        ctx.has_active = True
        ctx.num_nodes = int(num_nodes)
        ctx.num_total = int(pos.numel())
        ctx.save_for_backward(active_idx, w * gx, w * gy)
        return torch.sum(w * val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not ctx.has_active:
            return (
                grad_output.new_zeros((ctx.num_total,)),
                None, None, None, None, None, None, None, None, None,
            )
        active_idx, fgx, fgy = ctx.saved_tensors
        grad_pos = grad_output.new_zeros((ctx.num_total,))
        scale = grad_output.reshape(())
        grad_pos[active_idx] = scale * fgx
        grad_pos[ctx.num_nodes + active_idx] = scale * fgy
        return grad_pos, None, None, None, None, None, None, None, None, None


def smoothstep01(u: float) -> float:
    x = max(0.0, min(1.0, float(u)))
    return x * x * (3.0 - 2.0 * x)


class MacroExclusionField(nn.Module):
    """Differentiable fixed-macro exclusion potential + Secant MM curvature bound."""

    def __init__(
        self,
        placedb,
        data_collections,
        pos: torch.Tensor,
        *,
        ov_start: float = 0.20,
        ov_full: float = 0.085,
        beta_max: float = 1.10,
        bg_wl_scale: float = 0.0,
        wl_gate_scale: float = 0.35,
        warmup_iters: int = 110,
        ramp_iters: int = 40,
        collar_rows: float = 1.0,
        grid_pitch_rows: float = 1.0,
        distance_blend: float = 0.15,
        smooth_sigma_cells: float = 0.5,
        profile_mode: str = "tent_rim",
        ridge_smooth_rows: float = 3.0,
        slack_gate_mode: str = "kappa_evac",
        slack_margin_ps: float = 300.0,
        slack_tau_ps: float = 100.0,
        depth_ps_per_um: float = 2.5,
        crit_wl_scale: float = 0.65,
        ratchet_decay: float = 1.0,
        kappa_slack: float = 1.0,
        row_margin: float = 5.0,
        min_macro_rows: float = 2.0,
        force_mode: str = "taylor_kkt",
        escape_slope_mode: str = "global",
        gate_release_mode: str = "ratchet",
        hyst_kappa_band: float = 0.75,
        interior_evac_boost: float = 0.50,
        bell_sigma_rows: float = 16.0,
        bell_bias: float = 0.80,
    ) -> None:
        super().__init__()
        self.num_nodes = int(placedb.num_nodes)
        self.num_movable = int(placedb.num_movable_nodes)
        self.row_height = float(placedb.row_height)
        self.ov_start = float(ov_start)
        self.ov_full = float(ov_full)
        self.beta_max = float(beta_max)
        self.bg_wl_scale = float(bg_wl_scale)
        self.wl_gate_scale = float(wl_gate_scale)
        self.warmup_iters = int(warmup_iters)
        self.ramp_iters = int(ramp_iters)
        self.profile_mode = str(profile_mode or "dome").strip().lower()
        self.ridge_smooth_rows = float(ridge_smooth_rows)
        self.bell_sigma_rows = max(float(bell_sigma_rows), 1.0)
        self.bell_bias = max(float(bell_bias), 0.0)
        self.slack_gate_mode = str(slack_gate_mode or "uniform").strip().lower()
        if self.slack_gate_mode not in (
            "uniform",
            "slack_ratchet",
            "slack_ramp",
            "kappa_evac",
            "kappa_evac_elmore2",
        ):
            raise ValueError(
                f"Unsupported macro_exclusion slack_gate_mode={slack_gate_mode!r}"
            )
        self.force_mode = str(force_mode or "taylor_kkt").strip().lower()
        if self.force_mode not in ("scalar", "normal_proj", "taylor_kkt"):
            raise ValueError(
                f"Unsupported macro_exclusion force_mode={force_mode!r}"
            )
        self.escape_slope_mode = str(escape_slope_mode or "global").strip().lower()
        if self.escape_slope_mode not in ("global", "liberty_cell"):
            raise ValueError(
                f"Unsupported macro_exclusion escape_slope_mode={escape_slope_mode!r}"
            )
        self.gate_release_mode = str(gate_release_mode or "ratchet").strip().lower()
        if self.gate_release_mode not in ("ratchet", "hysteresis"):
            raise ValueError(
                f"Unsupported macro_exclusion gate_release_mode={gate_release_mode!r}"
            )
        self.slack_margin_ps = float(slack_margin_ps)
        self.slack_tau_ps = float(slack_tau_ps)
        self.depth_ps_per_um = float(depth_ps_per_um)
        self.crit_wl_scale = float(crit_wl_scale)
        self.ratchet_decay = float(np.clip(ratchet_decay, 0.0, 1.0))
        self.kappa_slack = max(float(kappa_slack), 1e-4)
        self.row_margin = max(float(row_margin), 0.0)
        self.hyst_kappa_band = max(float(hyst_kappa_band), 0.05)
        self.interior_evac_boost = max(float(interior_evac_boost), 0.0)

        device = pos.device
        boxes, box_stats = extract_macro_and_gap_boxes(
            placedb, pos, min_macro_rows=min_macro_rows, include_row_gaps=True
        )
        grid, grid_meta = build_macro_potential_grid(
            boxes,
            float(placedb.xl),
            float(placedb.yl),
            float(placedb.xh),
            float(placedb.yh),
            self.row_height,
            device=device,
            grid_pitch_rows=grid_pitch_rows,
            collar_rows=collar_rows,
            distance_blend=distance_blend,
            smooth_sigma_cells=smooth_sigma_cells,
            profile_mode=self.profile_mode,
            ridge_smooth_rows=self.ridge_smooth_rows,
            bell_sigma_rows=self.bell_sigma_rows,
            bell_bias=self.bell_bias,
        )
        pen_grid = grid_meta.pop("pen_grid", None)
        self.has_obstacles = grid is not None
        self.metadata = {**box_stats, **grid_meta}

        if self.has_obstacles:
            self.register_buffer("grid", grid)
            self.register_buffer(
                "pen_grid",
                pen_grid if pen_grid is not None else torch.zeros_like(grid),
            )
            supp_np = _build_bspline_support_mask(grid.detach().cpu().numpy())
            self.register_buffer(
                "support_mask_flat",
                torch.as_tensor(supp_np.reshape(-1), device=device, dtype=torch.bool),
            )
            self.g_xl = float(grid_meta["g_xl"])
            self.g_yl = float(grid_meta["g_yl"])
            self.h_g = float(grid_meta["h_g"])
            self.delta = float(grid_meta["delta"])
            self.nx = int(grid.shape[1])
            self.ny = int(grid.shape[0])
        else:
            self.register_buffer("grid", torch.empty(0, device=device, dtype=torch.float32))
            self.register_buffer("pen_grid", torch.empty(0, device=device, dtype=torch.float32))
            self.register_buffer("support_mask_flat", torch.empty(0, device=device, dtype=torch.bool))
            self.g_xl = 0.0
            self.g_yl = 0.0
            self.h_g = 1.0
            self.delta = 1.0
            self.nx = 0
            self.ny = 0

        sx = data_collections.node_size_x[: self.num_movable].detach().to(device=device, dtype=torch.float32)
        sy = data_collections.node_size_y[: self.num_movable].detach().to(device=device, dtype=torch.float32)
        self.register_buffer("movable_half_w", 0.5 * sx)
        self.register_buffer("movable_half_h", 0.5 * sy)
        self.register_buffer("movable_area", sx * sy)

        self.register_buffer("movable_base_pull", torch.zeros(self.num_movable, device=device, dtype=torch.float32))
        self.register_buffer("movable_risk_gate", torch.zeros(self.num_movable, device=device, dtype=torch.float32))
        self.register_buffer("movable_peak_depth_um", torch.zeros(self.num_movable, device=device, dtype=torch.float32))
        self.register_buffer("active_idx", torch.empty(0, device=device, dtype=torch.long))
        self.register_buffer("active_f_time", torch.empty(0, device=device, dtype=torch.float32))
        self.register_buffer("active_chi_pwl", torch.empty(0, device=device, dtype=torch.float32))
        self._cached_key = None
        self._refresh_count = 0
        self.last_beta = 0.0
        self.last_overflow = None
        self.last_penalty_value = 0.0
        self.last_overlapping_cells = 0
        self.last_slack_gate_stats: Dict[str, float] = {}

    @torch.no_grad()
    def sync_movable_geometry(self, data_collections) -> None:
        """Refresh movable half-sizes after discrete sizing / inflation updates."""
        device = self.movable_half_w.device
        sx = data_collections.node_size_x[: self.num_movable].detach().to(device=device, dtype=torch.float32)
        sy = data_collections.node_size_y[: self.num_movable].detach().to(device=device, dtype=torch.float32)
        self.movable_half_w.copy_(0.5 * sx)
        self.movable_half_h.copy_(0.5 * sy)
        self.movable_area.copy_(sx * sy)
        self._cached_key = None

    @torch.no_grad()
    def reset_history(self, decay: float = 0.0) -> None:
        """Reset or decay latched per-cell risk/depth history (e.g. after discrete netlist mutation)."""
        keep = float(np.clip(decay, 0.0, 1.0))
        if keep <= 0.0:
            self.movable_risk_gate.zero_()
            self.movable_peak_depth_um.zero_()
        else:
            self.movable_risk_gate.mul_(keep)
            self.movable_peak_depth_um.mul_(keep)
        self._cached_key = None

    def compute_beta(self, overflow: Optional[float], owner=None) -> float:
        if not self.has_obstacles or self.beta_max <= 0.0 or overflow is None:
            return 0.0
        if not math.isfinite(float(overflow)):
            return 0.0
        ov = float(overflow)
        if ov >= self.ov_start:
            return 0.0
        if self.ov_start <= self.ov_full:
            g_ov = 1.0
        else:
            g_ov = smoothstep01((self.ov_start - ov) / (self.ov_start - self.ov_full))

        g_iter = 1.0
        if owner is not None and self.warmup_iters > 0:
            iter_fn = getattr(owner, "_segment_timing_cut_schedule_iteration", None)
            if callable(iter_fn):
                cur_iter = int(iter_fn())
                start_fn = getattr(
                    owner, "_segment_timing_snapshot_start_iter", None
                )
                if callable(start_fn):
                    t_opt_iter = int(start_fn())
                else:
                    t_opt_iter = int(
                        getattr(getattr(owner, "params", None), "timing_opt_iter", 500)
                    )
                t_start = t_opt_iter + self.warmup_iters
                if cur_iter <= t_start:
                    return 0.0
                if self.ramp_iters > 0:
                    g_iter = smoothstep01((cur_iter - t_start) / float(self.ramp_iters))
        return self.beta_max * g_ov * g_iter

    def _query_stencil_candidates(self, pos: torch.Tensor) -> torch.Tensor:
        """Return indices in ``[0, num_movable)`` whose center falls in ``support_mask_flat``."""
        m = self.num_movable
        if m <= 0 or not self.has_obstacles:
            return self.active_idx.new_empty(0)
        inv_h = 1.0 / float(self.h_g)
        cx = pos[:m].detach().float() + self.movable_half_w
        cy = pos[self.num_nodes : self.num_nodes + m].detach().float() + self.movable_half_h
        ix = torch.floor(((cx - float(self.g_xl)) * inv_h - 0.5).clamp(1.0, float(self.nx - 2) - 1e-5)).long()
        iy = torch.floor(((cy - float(self.g_yl)) * inv_h - 0.5).clamp(1.0, float(self.ny - 2) - 1e-5)).long()
        hit = self.support_mask_flat[iy * self.nx + ix]
        return torch.nonzero(hit, as_tuple=False).reshape(-1)

    @torch.no_grad()
    def _extract_pin_and_movable_slack_ps(
        self, owner, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """Extract per-pin slack and per-movable-cell worst incident pin slack ``min(RAT - AAT)`` in ``ps``."""
        op_coll = getattr(owner, "op_collections", None)
        tp_op = (
            getattr(owner, "timing_op", None)
            or (getattr(op_coll, "timing_propagation_op", None) if op_coll is not None else None)
        )
        if tp_op is None:
            return None, None, 0
        tp_gen = int(
            getattr(tp_op, "timing_state_generation", 0)
            or getattr(tp_op, "timing_forward_count", 0)
            or getattr(owner, "refresh_count", 0)
        )

        safe_inf = torch.tensor(1.0e9, device=device, dtype=torch.float32)
        sentinel = 8.0e7
        unconstrained_ps = 1.0e7

        # 1. Prefer SparseTimingRuntime's live pin slack snapshot (RAT - AAT in ps)
        raw_rs = getattr(tp_op, "pin_rslack_live_snapshot", None)
        raw_fs = getattr(tp_op, "pin_fslack_live_snapshot", None)
        pin_slack: Optional[torch.Tensor] = None
        if torch.is_tensor(raw_rs) and torch.is_tensor(raw_fs):
            rs_f = raw_rs.detach().to(device=device, dtype=torch.float32).reshape(-1)
            fs_f = raw_fs.detach().to(device=device, dtype=torch.float32).reshape(-1)
            n_pins = min(int(rs_f.numel()), int(fs_f.numel()))
            if n_pins > 0:
                rs_f, fs_f = rs_f[:n_pins], fs_f[:n_pins]
                valid_r = torch.isfinite(rs_f) & (rs_f.abs() < sentinel) & (rs_f < unconstrained_ps)
                valid_f = torch.isfinite(fs_f) & (fs_f.abs() < sentinel) & (fs_f < unconstrained_ps)
                pin_slack = torch.minimum(
                    torch.where(valid_r, rs_f, safe_inf),
                    torch.where(valid_f, fs_f, safe_inf),
                )

        # 2. Fallback to committed pin slack cache on StarTdpRuntime or full pin_rRAT - pin_rAAT
        if pin_slack is None:
            committed = getattr(owner, "_committed_pin_slack", None)
            if isinstance(committed, dict) and torch.is_tensor(committed.get("rise")) and torch.is_tensor(committed.get("fall")):
                rs_f = committed["rise"].detach().to(device=device, dtype=torch.float32).reshape(-1)
                fs_f = committed["fall"].detach().to(device=device, dtype=torch.float32).reshape(-1)
                n_pins = min(int(rs_f.numel()), int(fs_f.numel()))
                if n_pins > 0:
                    rs_f, fs_f = rs_f[:n_pins], fs_f[:n_pins]
                    valid_r = torch.isfinite(rs_f) & (rs_f.abs() < sentinel) & (rs_f < unconstrained_ps)
                    valid_f = torch.isfinite(fs_f) & (fs_f.abs() < sentinel) & (fs_f < unconstrained_ps)
                    pin_slack = torch.minimum(
                        torch.where(valid_r, rs_f, safe_inf),
                        torch.where(valid_f, fs_f, safe_inf),
                    )

        if pin_slack is None:
            snap = getattr(tp_op, "_timing_gradient_no_autograd_forward_snapshot", None)
            if isinstance(snap, dict) and not bool(snap.get("skip_full_rat", False)):
                r_aat = snap.get("pin_rAAT")
                f_aat = snap.get("pin_fAAT")
                r_rat = snap.get("pin_rRAT")
                f_rat = snap.get("pin_fRAT")
            else:
                r_aat = getattr(tp_op, "pin_rAAT", None)
                f_aat = getattr(tp_op, "pin_fAAT", None)
                r_rat = getattr(tp_op, "pin_rRAT", None)
                f_rat = getattr(tp_op, "pin_fRAT", None)
            if not all(torch.is_tensor(t) for t in (r_aat, f_aat, r_rat, f_rat)):
                return None, None, tp_gen

            r_aat_f = r_aat.detach().to(device=device, dtype=torch.float32).reshape(-1)
            f_aat_f = f_aat.detach().to(device=device, dtype=torch.float32).reshape(-1)
            r_rat_f = r_rat.detach().to(device=device, dtype=torch.float32).reshape(-1)
            f_rat_f = f_rat.detach().to(device=device, dtype=torch.float32).reshape(-1)
            n_pins = min(
                int(r_aat_f.numel()),
                int(f_aat_f.numel()),
                int(r_rat_f.numel()),
                int(f_rat_f.numel()),
            )
            if n_pins <= 0:
                return None, None, tp_gen
            r_aat_f, f_aat_f = r_aat_f[:n_pins], f_aat_f[:n_pins]
            r_rat_f, f_rat_f = r_rat_f[:n_pins], f_rat_f[:n_pins]

            valid_r = (
                torch.isfinite(r_aat_f)
                & torch.isfinite(r_rat_f)
                & (r_aat_f.abs() < sentinel)
                & (r_rat_f.abs() < sentinel)
            )
            valid_f = (
                torch.isfinite(f_aat_f)
                & torch.isfinite(f_rat_f)
                & (f_aat_f.abs() < sentinel)
                & (f_rat_f.abs() < sentinel)
            )
            pin_slack = torch.minimum(
                torch.where(valid_r, r_rat_f - r_aat_f, safe_inf),
                torch.where(valid_f, f_rat_f - f_aat_f, safe_inf),
            )

        pinmap_raw = getattr(getattr(owner, "data_collections", None), "pin2node_map", None)
        if not torch.is_tensor(pinmap_raw):
            return pin_slack, None, tp_gen
        pinmap = pinmap_raw.detach().to(device=device, dtype=torch.long).reshape(-1)
        n_use = min(int(pin_slack.numel()), int(pinmap.numel()))
        p_nodes = pinmap[:n_use]
        p_slack = pin_slack[:n_use]
        mov_mask = (p_nodes >= 0) & (p_nodes < self.num_movable)
        node_slack = torch.full((self.num_movable,), 1.0e9, device=device, dtype=torch.float32)
        if bool(mov_mask.any()):
            node_slack.scatter_reduce_(
                0, p_nodes[mov_mask], p_slack[mov_mask], reduce="amin", include_self=True
            )
        return pin_slack, node_slack, tp_gen

    @torch.no_grad()
    def _extract_movable_min_pin_slack_ps(
        self, owner, device: torch.device
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Extract per-movable-cell worst incident pin slack ``min(RAT - AAT)`` in ``ps``."""
        _, node_slack, tp_gen = self._extract_pin_and_movable_slack_ps(owner, device)
        return node_slack, tp_gen

    @torch.no_grad()
    def _extract_pin_drive_resistance_ps_per_cap(
        self,
        owner,
        num_pins: int,
        fallback_r_drv: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Extract per-output-pin Liberty drive sensitivity ``dD_cell / dC_load`` in ``[ps / cap_unit]``."""
        fb = max(float(fallback_r_drv), 0.0)
        if num_pins <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        op_coll = getattr(owner, "op_collections", None)
        tp_op = (
            getattr(owner, "timing_op", None)
            or (getattr(op_coll, "timing_propagation_op", None) if op_coll is not None else None)
        )
        tp_core = getattr(tp_op, "timing", tp_op) if tp_op is not None else None
        data_coll = getattr(owner, "data_collections", None)
        sources = tuple(s for s in (tp_op, tp_core, data_coll) if s is not None)
        if not sources:
            return torch.full((num_pins,), fb, device=device, dtype=torch.float32)

        for src in sources:
            explicit_r = getattr(src, "pin_drive_resistance_kohm", None)
            if torch.is_tensor(explicit_r):
                r_vec = explicit_r.detach().to(device=device, dtype=torch.float32).reshape(-1)
                if int(r_vec.numel()) >= num_pins:
                    return r_vec[:num_pins].clamp(min=0.0)

        pin_r = torch.full((num_pins,), -1.0, device=device, dtype=torch.float32)
        populated = False

        for src in sources:
            snap = getattr(src, "_timing_gradient_no_autograd_forward_snapshot", None)
            desc = snap.get("native_delay_lut_input_descriptor") if isinstance(snap, dict) else None
            if isinstance(desc, dict) and torch.is_tensor(desc.get("arc_out_pins")) and torch.is_tensor(desc.get("d_cap")):
                out_p = desc["arc_out_pins"].detach().to(device=device, dtype=torch.long).reshape(-1)
                d_cap = desc["d_cap"].detach().to(device=device, dtype=torch.float32).reshape(-1)
                n_a = min(int(out_p.numel()), int(d_cap.numel()))
                if n_a > 0:
                    out_p, d_cap = out_p[:n_a], d_cap[:n_a]
                    v_mask = (out_p >= 0) & (out_p < num_pins) & torch.isfinite(d_cap) & (d_cap >= 0.0)
                    if bool(v_mask.any()):
                        pin_r.scatter_reduce_(
                            0, out_p[v_mask], d_cap[v_mask], reduce="amax", include_self=True
                        )
                        populated = True
                break

        if not populated:
            inst_arcs = None
            arcs_info = None
            for src in sources:
                if inst_arcs is None and torch.is_tensor(getattr(src, "inst_flat_arcs", None)):
                    inst_arcs = getattr(src, "inst_flat_arcs")
                if arcs_info is None and getattr(src, "arcs_info", None) is not None:
                    arcs_info = getattr(src, "arcs_info")
            if (
                torch.is_tensor(inst_arcs)
                and inst_arcs.dim() == 2
                and int(inst_arcs.shape[1]) >= 4
                and arcs_info is not None
            ):
                out_p = inst_arcs[:, 1].detach().to(device=device, dtype=torch.long)
                lut_ids = inst_arcs[:, 3].detach().to(device=device, dtype=torch.long)
                for fam_name in ("r_delay_luts", "f_delay_luts"):
                    luts = getattr(arcs_info, fam_name, None)
                    if luts is None or not torch.is_tensor(getattr(luts, "flat_luts_values", None)):
                        continue
                    vals = luts.flat_luts_values.detach()
                    caps = luts.flat_luts_cap_table.detach()
                    c_dims = luts.cap_dims_actual.detach()
                    t_dims = luts.trans_dims_actual.detach()
                    n_luts = int(vals.shape[0]) if vals.dim() >= 2 else 0
                    if n_luts <= 0 or caps.dim() != 2 or int(caps.shape[1]) < 2:
                        continue
                    flat_v = vals.reshape(n_luts, -1).to(device=device, dtype=torch.float32)
                    caps_f = caps.to(device=device, dtype=torch.float32)
                    c_dims_l = c_dims.to(device=device, dtype=torch.long)
                    t_dims_l = t_dims.to(device=device, dtype=torch.long)
                    has_cap = c_dims_l >= 2
                    if not bool(has_cap.any()):
                        continue
                    c_last = (c_dims_l - 1).clamp(min=1, max=int(caps_f.shape[1]) - 1)
                    m_tr = ((t_dims_l.clamp(min=1) - 1) // 2).clamp(min=0)
                    stride = c_dims_l.clamp(min=1)
                    max_flat = int(flat_v.shape[1]) - 1
                    idx0 = (m_tr * stride).clamp(0, max_flat).unsqueeze(1)
                    idx1 = (m_tr * stride + c_last).clamp(0, max_flat).unsqueeze(1)
                    v0 = flat_v.gather(1, idx0).squeeze(1)
                    v1 = flat_v.gather(1, idx1).squeeze(1)
                    c0 = caps_f[:, 0]
                    c1 = caps_f.gather(1, c_last.unsqueeze(1)).squeeze(1)
                    lut_slope = torch.where(
                        has_cap & ((c1 - c0) > 1e-9),
                        ((v1 - v0) / (c1 - c0).clamp(min=1e-9)).clamp(min=0.0),
                        torch.full_like(v0, -1.0),
                    )
                    in_bounds = (out_p >= 0) & (out_p < num_pins) & (lut_ids >= 0) & (lut_ids < n_luts)
                    if bool(in_bounds.any()):
                        arc_s = lut_slope[lut_ids[in_bounds]]
                        ok_s = torch.isfinite(arc_s) & (arc_s >= 0.0)
                        if bool(ok_s.any()):
                            pin_r.scatter_reduce_(
                                0,
                                out_p[in_bounds][ok_s],
                                arc_s[ok_s],
                                reduce="amax",
                                include_self=True,
                            )
                            populated = True

        fallback_t = torch.tensor(fb, device=device, dtype=torch.float32)
        return torch.where(pin_r >= 0.0, pin_r, fallback_t)

    @torch.no_grad()
    def _sample_penetration_depth(self, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """Bilinearly sample exact cell-body overlap depth ``d_in`` in ``[pos_unit]`` (0 in legal rows)."""
        if not self.has_obstacles or self.pen_grid.numel() == 0 or cx.numel() == 0:
            return torch.zeros_like(cx)
        ny, nx = int(self.pen_grid.shape[0]), int(self.pen_grid.shape[1])
        inv_h = 1.0 / float(self.h_g)
        u = ((cx - float(self.g_xl)) * inv_h - 0.5).clamp(0.0, float(nx - 1) - 1e-5)
        v = ((cy - float(self.g_yl)) * inv_h - 0.5).clamp(0.0, float(ny - 1) - 1e-5)
        ix0 = torch.floor(u).long().clamp(0, max(nx - 2, 0))
        iy0 = torch.floor(v).long().clamp(0, max(ny - 2, 0))
        tx = (u - ix0.float()).clamp(0.0, 1.0)
        ty = (v - iy0.float()).clamp(0.0, 1.0)
        ix1 = (ix0 + 1).clamp(max=nx - 1)
        iy1 = (iy0 + 1).clamp(max=ny - 1)
        p00 = self.pen_grid[iy0, ix0]
        p01 = self.pen_grid[iy0, ix1]
        p10 = self.pen_grid[iy1, ix0]
        p11 = self.pen_grid[iy1, ix1]
        p0 = p00 + tx * (p01 - p00)
        p1 = p10 + tx * (p11 - p10)
        return (p0 + ty * (p1 - p0)).clamp(min=0.0)

    @torch.no_grad()
    def _compute_timing_hessian_and_gamma(
        self,
        owner,
        pos: torch.Tensor,
        star,
        coeff: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return ``(h_time_x[:m], h_time_y[:m], gamma_pos)`` in ``[Objective/pos_unit^2]`` and ``[pos_unit]``."""
        m = self.num_movable
        device = pos.device
        mock_h_fn = getattr(owner, "mock_hessian_and_gamma_fn", None)
        if callable(mock_h_fn):
            hx, hy, g_val = mock_h_fn(pos.detach())
            return (
                hx[:m].detach().to(device=device, dtype=torch.float32),
                hy[:m].detach().to(device=device, dtype=torch.float32),
                max(float(g_val), 1e-6),
            )

        gamma_pos = 4.0 * float(self.row_height)
        gamma_fn = getattr(owner, "_segment_timing_cut_effective_smoothing_gamma", None)
        if callable(gamma_fn):
            g_raw = gamma_fn()
            if g_raw is not None:
                gamma_pos = float(g_raw.detach().reshape(-1)[0].item()) if torch.is_tensor(g_raw) else float(g_raw)
        elif hasattr(owner, "gamma") or hasattr(getattr(owner, "model", None), "gamma"):
            g_raw = getattr(owner, "gamma", None)
            if g_raw is None:
                g_raw = getattr(getattr(owner, "model", None), "gamma", None)
            if g_raw is not None:
                gamma_pos = float(g_raw.detach().reshape(-1)[0].item()) if torch.is_tensor(g_raw) else float(g_raw)
        gamma_pos = max(float(gamma_pos), 1e-6)

        if (
            coeff > 0.0
            and star is not None
            and int(getattr(star, "snapshot_generation", 0)) > 0
            and hasattr(owner, "placedb")
            and hasattr(owner, "op_collections")
            and callable(getattr(owner.op_collections, "pin_pos_op", None))
        ):
            try:
                from dreamplace.ops.star_pin_pair.rowsum_curvature import (
                    clock_rowsum_hessian,
                    rowsum_hessian,
                )

                params = getattr(owner, "params", None)
                elmore_axes = int(
                    getattr(owner, "elmore_rowsum_axes", None)
                    or getattr(params, "segment_timing_elmore_rowsum_axes", 2)
                    or 2
                )
                full_slope_attr = getattr(owner, "full_linear_slope", None)
                if full_slope_attr is not None:
                    full_slope = bool(full_slope_attr)
                else:
                    full_slope = (
                        str(getattr(params, "segment_timing_rowsum_linear_slope", "full") or "full")
                        .strip()
                        .lower()
                        == "full"
                    )
                h_xy, g_calc = rowsum_hessian(
                    owner,
                    pos.detach(),
                    elmore_axes=elmore_axes,
                    full_slope=full_slope,
                )
                if g_calc is not None and float(g_calc) > 0.0:
                    gamma_pos = float(g_calc)
                if bool(getattr(owner, "lcb_ff_clock_enabled", True)) and bool(
                    getattr(params, "segment_timing_clock_curvature_precondition", 1)
                ):
                    h_xy = h_xy + clock_rowsum_hessian(
                        owner,
                        pos.detach(),
                        gamma=gamma_pos,
                        elmore_axes=elmore_axes,
                        full_slope=full_slope,
                    )
                return (
                    h_xy[0, :m].detach().to(device=device, dtype=torch.float32),
                    h_xy[1, :m].detach().to(device=device, dtype=torch.float32),
                    max(float(gamma_pos), 1e-6),
                )
            except Exception:
                pass

        zeros = torch.zeros(m, device=device, dtype=torch.float32)
        return zeros, zeros.clone(), gamma_pos

    @torch.no_grad()
    def _compute_total_pull_grad_xy(
        self,
        owner,
        pos: torch.Tensor,
        star,
        coeff: float,
        pair_slope: Optional[torch.Tensor],
        geom: Optional[Dict[str, torch.Tensor]],
        drv: Optional[torch.Tensor],
        snk: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``(d/dx_i, d/dy_i)`` of ``E_WL + coeff * E_time`` for movable cells ``[0, m)``."""
        m = self.num_movable
        device = pos.device
        mock_fn = getattr(owner, "mock_pull_grad_fn", None)
        if callable(mock_fn):
            gx, gy = mock_fn(pos.detach())
            return (
                gx[:m].detach().to(device=device, dtype=torch.float32),
                gy[:m].detach().to(device=device, dtype=torch.float32),
            )

        op_coll = getattr(owner, "op_collections", None)
        wl_op = getattr(op_coll, "wirelength_op", None) if op_coll is not None else None
        unit_surr_fn = getattr(owner, "_unit_surrogate_term", None)
        has_star_obj = callable(unit_surr_fn) or (
            star is not None and callable(getattr(star, "structured_delay_objective", None))
        )
        if callable(wl_op) or (coeff > 0.0 and has_star_obj):
            with torch.enable_grad():
                probe_pos = pos.detach().clone().requires_grad_(True)
                total_e = probe_pos.new_zeros(())
                if callable(wl_op):
                    uses_corr_fn = getattr(owner, "_segment_timing_uses_wl_base_correction", None)
                    uses_corr = bool(uses_corr_fn()) if callable(uses_corr_fn) else True
                    anc_fn = getattr(owner, "_segment_original_wl_anchor_weight", None)
                    anc_w = float(anc_fn()) if callable(anc_fn) else 0.0
                    wl_w = (1.0 + anc_w) if uses_corr else anc_w
                    if wl_w > 0.0:
                        total_e = total_e + float(wl_w) * wl_op(probe_pos)
                if coeff > 0.0:
                    if callable(unit_surr_fn):
                        total_e = total_e + float(coeff) * unit_surr_fn(probe_pos)
                    elif has_star_obj and op_coll is not None and callable(getattr(op_coll, "pin_pos_op", None)):
                        pin_pos_g = op_coll.pin_pos_op(probe_pos)
                        dist_fn = getattr(owner, "_segment_timing_cut_distance_model", None)
                        dist_model = str(dist_fn()) if callable(dist_fn) else "l1"
                        needs_gamma = (dist_model in ("wa", "lse")) or (getattr(star, "load_geometry", "star") == "wa_net")
                        gamma_fn = getattr(owner, "_segment_timing_cut_effective_smoothing_gamma", None)
                        data_gamma = gamma_fn() if (needs_gamma and callable(gamma_fn)) else None
                        pair_gamma = data_gamma if dist_model in ("wa", "lse") else None
                        base_caps = owner._driver_sink_star_base_cap_map()
                        t_obj = star.structured_delay_objective(
                            pin_pos_g,
                            base_caps=base_caps,
                            distance_model=dist_model,
                            smoothing_gamma=data_gamma,
                        )
                        if bool(getattr(getattr(owner, "params", None), "segment_timing_star_impulse_channel", False)) and callable(getattr(star, "structured_impulse_objective", None)):
                            t_obj = t_obj + star.structured_impulse_objective(
                                pin_pos_g,
                                base_caps=base_caps,
                                distance_model=dist_model,
                                smoothing_gamma=pair_gamma,
                            )
                        clk_fn = getattr(owner, "_capture_setup_clock_objective", None)
                        if callable(clk_fn):
                            t_obj = t_obj + clk_fn(
                                pin_pos_g,
                                distance_model=dist_model,
                                smoothing_gamma=pair_gamma,
                            )
                        total_e = total_e + float(coeff) * t_obj
                if total_e.requires_grad:
                    (grad_pos,) = torch.autograd.grad(
                        total_e, probe_pos, retain_graph=False, create_graph=False
                    )
                    gx_m = grad_pos[:m].detach().to(device=device, dtype=torch.float32).clone()
                    gy_m = grad_pos[self.num_nodes : self.num_nodes + m].detach().to(device=device, dtype=torch.float32).clone()
                    del probe_pos, total_e, grad_pos
                    return gx_m, gy_m

        gx_m = torch.zeros(m, device=device, dtype=torch.float32)
        gy_m = torch.zeros(m, device=device, dtype=torch.float32)
        if (
            pair_slope is not None
            and isinstance(geom, dict)
            and torch.is_tensor(geom.get("raw_dx"))
            and torch.is_tensor(geom.get("raw_dy"))
            and drv is not None
            and snk is not None
        ):
            sx_sign = torch.sign(geom["raw_dx"].detach().to(device=device, dtype=torch.float32))
            sy_sign = torch.sign(geom["raw_dy"].detach().to(device=device, dtype=torch.float32))
            distinct = drv != snk
            d_m = distinct & (drv >= 0) & (drv < m)
            s_m = distinct & (snk >= 0) & (snk < m)
            if bool(d_m.any()):
                gx_m.scatter_add_(0, drv[d_m], pair_slope[d_m] * sx_sign[d_m])
                gy_m.scatter_add_(0, drv[d_m], pair_slope[d_m] * sy_sign[d_m])
            if bool(s_m.any()):
                gx_m.scatter_add_(0, snk[s_m], -pair_slope[s_m] * sx_sign[s_m])
                gy_m.scatter_add_(0, snk[s_m], -pair_slope[s_m] * sy_sign[s_m])
        return gx_m, gy_m

    @torch.no_grad()
    def _compute_liberty_directional_evac_ps(
        self,
        *,
        owner,
        star,
        pin_pos: Optional[torch.Tensor],
        geom: Optional[Dict[str, torch.Tensor]],
        drv: Optional[torch.Tensor],
        snk: Optional[torch.Tensor],
        drv_mov: Optional[torch.Tensor],
        snk_mov: Optional[torch.Tensor],
        wire_slope: Optional[torch.Tensor],
        alpha_load_ps_per_um: float,
        pin_slack: Optional[torch.Tensor],
        node_slack: torch.Tensor,
        cand_idx: torch.Tensor,
        nx_cand: Optional[torch.Tensor],
        ny_cand: Optional[torch.Tensor],
        grad_norm_cand: Optional[torch.Tensor],
        disp_um: torch.Tensor,
        row_margin_um: float,
        pos_per_um: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-cell directional evacuation delay ``dtau_evac_ps`` and hysteresis floor ``dtau_min_ps``."""
        m = self.num_movable
        if (
            star is None
            or wire_slope is None
            or wire_slope.numel() == 0
            or drv is None
            or snk is None
            or drv_mov is None
            or snk_mov is None
        ):
            fb_alpha = max(float(getattr(star, "r_unit", 0.0) if star is not None else 0.0) * 0.01, 0.1)
            dtau = torch.full((m,), fb_alpha, device=device, dtype=torch.float32) * disp_um
            dtau_min = torch.full((m,), 0.5 * fb_alpha * row_margin_um, device=device, dtype=torch.float32)
            return dtau, dtau_min

        drv_pids = star.driver_pin_ids.long()
        snk_pids = star.sink_pin_ids.long()
        n_pins_est = int(max(int(drv_pids.max().item()), int(snk_pids.max().item())) + 1)
        if pin_pos is not None:
            n_pins_est = max(n_pins_est, int(pin_pos.numel()) // 2)
        if pin_slack is not None:
            n_pins_est = max(n_pins_est, int(pin_slack.numel()))

        c_unit = max(float(getattr(star, "c_unit", 0.0) or 0.0), 0.0)
        r_unit = max(float(getattr(star, "r_unit", 0.0) or 0.0), 0.0)
        fb_r_drv = alpha_load_ps_per_um / max(c_unit, 1e-12) if c_unit > 0.0 else 0.0
        pin_r_drv = self._extract_pin_drive_resistance_ps_per_cap(
            owner, n_pins_est, fb_r_drv, device
        )

        # By Hanan's RSMT strip theorem, displacing one cell outward by dhpwl
        # stretches a single boundary Steiner branch with unit slope 1.0 * c_unit.
        drv_load_slope = (pin_r_drv[drv_pids] * c_unit).clamp(min=0.0)
        rc_half = (
            max(r_unit * c_unit, 0.0)
            if self.slack_gate_mode == "kappa_evac_elmore2"
            else 0.5 * max(r_unit * c_unit, 0.0)
        )

        has_dir = torch.zeros(m, device=device, dtype=torch.bool)
        dx_cell = torch.zeros(m, device=torch.device(device), dtype=torch.float32)
        dy_cell = torch.zeros(m, device=torch.device(device), dtype=torch.float32)
        if (
            cand_idx.numel() > 0
            and nx_cand is not None
            and ny_cand is not None
            and grad_norm_cand is not None
        ):
            v_n = grad_norm_cand > 1e-8
            if bool(v_n.any()):
                c_v = cand_idx[v_n]
                has_dir[c_v] = True
                dx_cell[c_v] = disp_um[c_v] * nx_cand[v_n]
                dy_cell[c_v] = disp_um[c_v] * ny_cand[v_n]

        drv_safe = drv.clamp(0, max(m - 1, 0))
        snk_safe = snk.clamp(0, max(m - 1, 0))
        has_geom_xy = (
            isinstance(geom, dict)
            and torch.is_tensor(geom.get("raw_dx"))
            and torch.is_tensor(geom.get("raw_dy"))
        )
        if has_geom_xy:
            raw_dx_um = geom["raw_dx"].detach().to(device=device, dtype=torch.float32) / pos_per_um
            raw_dy_um = geom["raw_dy"].detach().to(device=device, dtype=torch.float32) / pos_per_um
            cur_len_um = raw_dx_um.abs() + raw_dy_um.abs()
            dx_s = dx_cell[snk_safe]
            dy_s = dy_cell[snk_safe]
            dl_snk_dir = ((raw_dx_um - dx_s).abs() + (raw_dy_um - dy_s).abs() - cur_len_um).clamp(min=0.0)
            dl_snk = torch.where(has_dir[snk_safe], dl_snk_dir, disp_um[snk_safe])

            dx_d = dx_cell[drv_safe]
            dy_d = dy_cell[drv_safe]
            dl_drv_dir = ((raw_dx_um + dx_d).abs() + (raw_dy_um + dy_d).abs() - cur_len_um).clamp(min=0.0)
            dl_drv = torch.where(has_dir[drv_safe], dl_drv_dir, disp_um[drv_safe])

            if (
                pin_pos is not None
                and int(pin_pos.numel()) >= 2 * n_pins_est
                and hasattr(star, "pair_net_ids")
                and int(getattr(star, "num_nets", 0)) > 0
            ):
                n_pp = int(pin_pos.numel()) // 2
                px_um = pin_pos[:n_pp].detach().to(device=device, dtype=torch.float32) / pos_per_um
                py_um = pin_pos[n_pp : 2 * n_pp].detach().to(device=device, dtype=torch.float32) / pos_per_um
                n_nets = int(star.num_nets)
                net_ids = star.pair_net_ids.long()
                net_xmin = torch.full((n_nets,), 1.0e30, device=device, dtype=torch.float32)
                net_xmax = torch.full((n_nets,), -1.0e30, device=device, dtype=torch.float32)
                net_ymin = torch.full((n_nets,), 1.0e30, device=device, dtype=torch.float32)
                net_ymax = torch.full((n_nets,), -1.0e30, device=device, dtype=torch.float32)
                for pids in (drv_pids, snk_pids):
                    net_xmin.scatter_reduce_(0, net_ids, px_um[pids], reduce="amin", include_self=True)
                    net_xmax.scatter_reduce_(0, net_ids, px_um[pids], reduce="amax", include_self=True)
                    net_ymin.scatter_reduce_(0, net_ids, py_um[pids], reduce="amin", include_self=True)
                    net_ymax.scatter_reduce_(0, net_ids, py_um[pids], reduce="amax", include_self=True)
                net_deg = torch.zeros(n_nets, device=device, dtype=torch.float32)
                net_deg.scatter_add_(0, net_ids, torch.ones_like(raw_dx_um))
                deg_pair = net_deg[net_ids]

                sx_new = px_um[snk_pids] + dx_s
                sy_new = py_um[snk_pids] + dy_s
                box_snk = (
                    (sx_new - net_xmax[net_ids]).clamp(min=0.0)
                    + (net_xmin[net_ids] - sx_new).clamp(min=0.0)
                    + (sy_new - net_ymax[net_ids]).clamp(min=0.0)
                    + (net_ymin[net_ids] - sy_new).clamp(min=0.0)
                )
                dhpwl_snk = torch.where(
                    has_dir[snk_safe],
                    torch.where(deg_pair <= 1.5, dl_snk_dir, box_snk),
                    disp_um[snk_safe],
                )

                dx_new = px_um[drv_pids] + dx_d
                dy_new = py_um[drv_pids] + dy_d
                box_drv = (
                    (dx_new - net_xmax[net_ids]).clamp(min=0.0)
                    + (net_xmin[net_ids] - dx_new).clamp(min=0.0)
                    + (dy_new - net_ymax[net_ids]).clamp(min=0.0)
                    + (net_ymin[net_ids] - dy_new).clamp(min=0.0)
                )
                dhpwl_drv = torch.where(
                    has_dir[drv_safe],
                    torch.where(deg_pair <= 1.5, dl_drv_dir, box_drv),
                    disp_um[drv_safe],
                )
            else:
                dhpwl_snk = dl_snk
                dhpwl_drv = dl_drv
        else:
            dl_snk = disp_um[snk_safe]
            dl_drv = disp_um[drv_safe]
            dhpwl_snk = dl_snk
            dhpwl_drv = dl_drv

        pack_um = 0.25 * row_margin_um
        dtau_snk = (wire_slope + rc_half * dl_snk) * dl_snk + drv_load_slope * dhpwl_snk
        dtau_snk_pack = (wire_slope + drv_load_slope) * pack_um
        dtau_drv_wire = (wire_slope + rc_half * dl_drv) * dl_drv
        dtau_drv_wire_pack = wire_slope * pack_um

        if pin_slack is not None and int(pin_slack.numel()) > int(max(int(drv_pids.max().item()), int(snk_pids.max().item()))):
            ps_snk = pin_slack[snk_pids]
            ps_drv = pin_slack[drv_pids]
            branch_excess = (ps_snk - ps_drv).clamp(min=0.0, max=1.0e8)
            dtau_drv_wire = (dtau_drv_wire - branch_excess).clamp(min=0.0)
            dtau_drv_wire_pack = (dtau_drv_wire_pack - branch_excess).clamp(min=0.0)

            snk_pin_excess = (ps_snk - node_slack[snk_safe]).clamp(min=0.0, max=1.0e8)
            dtau_snk = (dtau_snk - snk_pin_excess).clamp(min=0.0)
            dtau_snk_pack = (dtau_snk_pack - snk_pin_excess).clamp(min=0.0)

            drv_pin_excess = (ps_drv - node_slack[drv_safe]).clamp(min=0.0, max=1.0e8)
            dtau_drv = (drv_load_slope * dhpwl_drv + dtau_drv_wire - drv_pin_excess).clamp(min=0.0)
            dtau_drv_pack = (drv_load_slope * pack_um + dtau_drv_wire_pack - drv_pin_excess).clamp(min=0.0)
        else:
            dtau_drv = drv_load_slope * dhpwl_drv + dtau_drv_wire
            dtau_drv_pack = drv_load_slope * pack_um + dtau_drv_wire_pack

        dtau_cell = torch.zeros(m, device=device, dtype=torch.float32)
        dtau_pack_cell = torch.zeros(m, device=device, dtype=torch.float32)
        buf = torch.zeros(m, device=device, dtype=torch.float32)
        if bool(snk_mov.any()):
            buf.scatter_reduce_(0, snk[snk_mov], dtau_snk[snk_mov], reduce="amax", include_self=True)
            dtau_cell.add_(buf)
            buf.zero_()
            buf.scatter_reduce_(0, snk[snk_mov], dtau_snk_pack[snk_mov], reduce="amax", include_self=True)
            dtau_pack_cell.add_(buf)
        if bool(drv_mov.any()):
            buf.zero_()
            buf.scatter_reduce_(0, drv[drv_mov], dtau_drv[drv_mov], reduce="amax", include_self=True)
            dtau_cell.add_(buf)
            buf.zero_()
            buf.scatter_reduce_(0, drv[drv_mov], dtau_drv_pack[drv_mov], reduce="amax", include_self=True)
            dtau_pack_cell.add_(buf)
        del buf, pin_r_drv

        min_floor = max(alpha_load_ps_per_um, 1e-6) * max(0.1 * row_margin_um, 1e-6)
        dtau_evac = torch.maximum(dtau_cell, dtau_pack_cell).clamp(min=1e-9)
        dtau_min = torch.maximum(2.0 * dtau_pack_cell, torch.full_like(dtau_pack_cell, min_floor))
        return dtau_evac, dtau_min

    @torch.no_grad()
    def refresh_criticality_pulls(self, owner, pos: torch.Tensor) -> None:
        """Recompute per-movable-cell force-matched inward pull ``[Objective / pos_unit]``."""
        if not self.has_obstacles:
            return
        star = owner._star_pin_pair_op() if hasattr(owner, "_star_pin_pair_op") else None
        if star is None or int(star.snapshot_generation) <= 0:
            self.movable_base_pull.zero_()
            self.active_idx = self.active_idx.new_empty(0)
            self.active_f_time = self.active_f_time.new_empty(0)
            self.active_chi_pwl = self.active_chi_pwl.new_empty(0)
            return

        coeff = float(getattr(owner, "segment_timing_coefficient", 0.0) or 0.0)
        op_coll = getattr(owner, "op_collections", None)
        clk_star = getattr(owner, "clock_star", None) or (
            getattr(op_coll, "clock_star_pin_pair_op", None) if op_coll is not None else None
        )
        clk_gen = int(getattr(clk_star, "snapshot_generation", 0)) if clk_star is not None else 0
        if self.slack_gate_mode in (
            "slack_ratchet",
            "slack_ramp",
            "kappa_evac",
            "kappa_evac_elmore2",
        ):
            tp_op = getattr(owner, "timing_op", None) or (
                getattr(op_coll, "timing_propagation_op", None) if op_coll is not None else None
            )
            tp_gen = (
                int(
                    getattr(tp_op, "timing_state_generation", 0)
                    or getattr(tp_op, "timing_forward_count", 0)
                    or getattr(owner, "refresh_count", 0)
                )
                if tp_op is not None
                else int(getattr(owner, "refresh_count", 0))
            )
            key = (int(star.snapshot_generation), clk_gen, tp_gen, round(coeff, 12))
        else:
            key = (int(star.snapshot_generation), clk_gen, round(coeff, 12))
        if self._cached_key == key:
            return

        m = self.num_movable
        device = pos.device
        f_time = torch.zeros(m, device=device, dtype=torch.float32)
        pos_per_um = max(float(getattr(star, "coordinate_to_um", 0.0) or 1.0), 1e-6)
        alpha_load_ps_per_um = 0.0
        alpha_cell: Optional[torch.Tensor] = None
        pin_pos: Optional[torch.Tensor] = None
        geom: Optional[Dict[str, torch.Tensor]] = None
        pair_slope: Optional[torch.Tensor] = None
        wire_slope: Optional[torch.Tensor] = None
        drv: Optional[torch.Tensor] = None
        snk: Optional[torch.Tensor] = None
        drv_mov: Optional[torch.Tensor] = None
        snk_mov: Optional[torch.Tensor] = None

        if coeff > 0.0 and hasattr(star, "frozen_weight") and star.frozen_weight.numel() == star.sink_pin_ids.numel():
            pinmap = owner.data_collections.pin2node_map.long()
            pin_pos = owner.op_collections.pin_pos_op(pos.detach())
            geom = star.geometry(pin_pos)
            weight = star.frozen_weight.detach().float()
            curv_fn = getattr(owner, "_segment_timing_star_load_curvature", None)
            curv_on = bool(curv_fn()) if callable(curv_fn) else True
            load = rowsum_curvature_load(star, curv_on).float()
            if load.numel() != weight.numel():
                load = torch.zeros_like(weight)
            base_caps = owner._driver_sink_star_base_cap_map()
            sink_cap = base_caps["generic"][star.sink_pin_ids].float()
            length_um = geom["length_um"].detach().float()
            wire_slope = (
                float(star.r_unit) * (sink_cap + float(star.c_unit) * length_um)
            ).clamp(min=0.0)
            load_pos = load.clamp(min=0.0)
            pair_slope = (
                coeff * (weight.abs() * wire_slope + load_pos) / pos_per_um
            ).clamp(min=0.0)
            drv = pinmap[star.driver_pin_ids]
            snk = pinmap[star.sink_pin_ids]
            distinct = drv != snk
            drv_mov = distinct & (drv < m)
            snk_mov = distinct & (snk < m)
            if bool(drv_mov.any()):
                f_time.scatter_add_(0, drv[drv_mov], pair_slope[drv_mov])
            if bool(snk_mov.any()):
                f_time.scatter_add_(0, snk[snk_mov], pair_slope[snk_mov])

            if self.slack_gate_mode in ("kappa_evac", "kappa_evac_elmore2"):
                is_wa_net_load = getattr(star, "load_geometry", "star") == "wa_net"
                if hasattr(star, "pair_net_ids") and int(getattr(star, "num_nets", 0)) > 0:
                    net_ids = star.pair_net_ids.long()
                    net_abs_w = torch.zeros(
                        int(star.num_nets), device=device, dtype=torch.float32
                    )
                    net_abs_w.scatter_add_(0, net_ids, weight.abs())
                    pair_net_w = net_abs_w[net_ids]
                    del net_abs_w
                else:
                    pair_net_w = weight.abs()
                act_w = pair_net_w > 1e-9
                denom_vec = weight.abs() if is_wa_net_load else pair_net_w
                denom_w = (
                    float(denom_vec[act_w].sum().double().item())
                    if bool(act_w.any())
                    else 0.0
                )
                if denom_w > 1e-12:
                    numer_b = float(load_pos[act_w].sum().double().item())
                    alpha_load_ps_per_um = max(numer_b / denom_w, 0.0)
                else:
                    alpha_load_ps_per_um = (
                        float(wire_slope.mean().item())
                        if wire_slope.numel() > 0
                        else 0.0
                    )
                del pair_net_w, act_w

                if self.escape_slope_mode == "global":
                    alpha_cell = torch.zeros(m, device=device, dtype=torch.float32)
                    buf = torch.zeros(m, device=device, dtype=torch.float32)
                    if bool(snk_mov.any()):
                        buf.scatter_reduce_(
                            0,
                            snk[snk_mov],
                            wire_slope[snk_mov] + alpha_load_ps_per_um,
                            reduce="amax",
                            include_self=True,
                        )
                        alpha_cell.add_(buf)
                    if bool(drv_mov.any()):
                        buf.zero_()
                        buf.scatter_reduce_(
                            0,
                            drv[drv_mov],
                            wire_slope[drv_mov],
                            reduce="amax",
                            include_self=True,
                        )
                        alpha_cell.add_(buf)
                        buf.zero_()
                        buf.scatter_add_(
                            0,
                            drv[drv_mov],
                            torch.ones(
                                int(drv_mov.sum().item()),
                                device=device,
                                dtype=torch.float32,
                            ),
                        )
                        alpha_cell.add_(buf.sqrt_(), alpha=alpha_load_ps_per_um)
                    del buf
                    alpha_cell.clamp_(min=max(alpha_load_ps_per_um, 1e-9))

            mode_fn = getattr(owner, "_segment_timing_clock_weight_config", None)
            clk_mode, clk_scale = mode_fn() if callable(mode_fn) else ("off", 0.0)
            if (
                clk_star is not None
                and clk_mode != "off"
                and float(clk_scale) > 0.0
                and int(clk_star.frozen_weight.numel()) == int(clk_star.sink_pin_ids.numel())
            ):
                native = clk_star.compact_to_native_pin_ids.long()
                c_drv = pinmap[native[clk_star.driver_pin_ids.long()]]
                c_snk = pinmap[native[clk_star.sink_pin_ids.long()]]
                compact_pos = owner._lcb_ff_clock_compact_pin_pos(pin_pos)
                c_geom = clk_star.geometry(compact_pos)
                c_weight = clk_star.frozen_weight.detach().float()
                c_load = getattr(clk_star, "frozen_load_length_weight", None)
                c_load = torch.zeros_like(c_weight) if (c_load is None or c_load.numel() != c_weight.numel()) else c_load.detach().float()
                c_caps = owner._lcb_ff_clock_base_cap_map()
                c_sink_cap = c_caps["generic"][clk_star.sink_pin_ids.long()].float()
                c_len_um = c_geom["length_um"].detach().float()
                c_slope = (
                    coeff
                    * (
                        c_weight.abs() * float(clk_star.r_unit) * (c_sink_cap + float(clk_star.c_unit) * c_len_um)
                        + c_load.clamp(min=0.0)
                    )
                    / float(clk_star.coordinate_to_um)
                ).clamp(min=0.0)
                c_distinct = c_drv != c_snk
                for ep in (c_drv, c_snk):
                    mask = c_distinct & (ep < m)
                    if mask.any():
                        f_time.scatter_add_(0, ep[mask], c_slope[mask])

        pws_op = getattr(op_coll, "pws_op", None) if op_coll is not None else None
        if pws_op is not None and hasattr(owner.data_collections, "net_weights"):
            p_wl = pws_op(owner.data_collections.net_weights)[:m].detach().to(device=device, dtype=torch.float32)
        elif hasattr(owner.data_collections, "sum_pin_weights_in_nodes"):
            p_wl = owner.data_collections.sum_pin_weights_in_nodes[:m].detach().to(device=device, dtype=torch.float32)
        else:
            p_wl = owner.data_collections.num_pins_in_nodes[:m].detach().to(device=device, dtype=torch.float32)

        chi = f_time / (f_time + p_wl.clamp(min=1.0) + 1e-12)
        cand_idx = self._query_stencil_candidates(pos)
        psi_cand: Optional[torch.Tensor] = None
        nx_cand: Optional[torch.Tensor] = None
        ny_cand: Optional[torch.Tensor] = None
        grad_norm_cand: Optional[torch.Tensor] = None
        d_in_cand: Optional[torch.Tensor] = None
        if cand_idx.numel() > 0:
            cx = pos[cand_idx].detach().float() + self.movable_half_w[cand_idx]
            cy = (
                pos[self.num_nodes + cand_idx].detach().float()
                + self.movable_half_h[cand_idx]
            )
            psi_cand, gx_psi, gy_psi = _eval_cubic_bspline_2d(
                self.grid, cx, cy, self.g_xl, self.g_yl, self.h_g
            )
            d_in_cand = self._sample_penetration_depth(cx, cy)
            grad_norm_cand = torch.sqrt(gx_psi * gx_psi + gy_psi * gy_psi)
            denom_n = grad_norm_cand.clamp(min=1e-8)
            nx_cand = -gx_psi / denom_n
            ny_cand = -gy_psi / denom_n
            del gx_psi, gy_psi, denom_n

        if self.slack_gate_mode in (
            "slack_ratchet",
            "slack_ramp",
            "kappa_evac",
            "kappa_evac_elmore2",
        ):
            depth_um = torch.zeros(m, device=device, dtype=torch.float32)
            if cand_idx.numel() > 0 and psi_cand is not None:
                if self.profile_mode == "tent_bell" and d_in_cand is not None:
                    raw_depth_um = d_in_cand / pos_per_um
                else:
                    raw_depth_um = psi_cand.clamp(min=0.0) / pos_per_um
                if self.slack_gate_mode in ("kappa_evac", "kappa_evac_elmore2"):
                    depth_um[cand_idx] = raw_depth_um
                else:
                    depth_cap = 150.0 if self.slack_gate_mode == "slack_ramp" else 250.0
                    depth_um[cand_idx] = raw_depth_um.clamp(max=depth_cap)

            if self.gate_release_mode == "hysteresis":
                in_stencil = depth_um > 1e-3
                decayed_peak = 0.5 * self.movable_peak_depth_um
                decayed_peak = torch.where(decayed_peak > 0.1, decayed_peak, torch.zeros_like(decayed_peak))
                new_peak = torch.where(
                    in_stencil,
                    torch.maximum(self.movable_peak_depth_um, depth_um),
                    decayed_peak,
                )
                self.movable_peak_depth_um.copy_(new_peak)
                eff_depth_um = torch.maximum(depth_um, new_peak)
                del in_stencil, decayed_peak, new_peak
            else:
                eff_depth_um = depth_um

            pin_slack, node_slack, _ = self._extract_pin_and_movable_slack_ps(owner, device)
            med_dtau_in_macro = 0.0
            if self.profile_mode == "tent_bell":
                wall_um = (0.5 * float(self.row_height)) / pos_per_um
            elif self.profile_mode == "tent_rim":
                wall_um = (0.725 * float(self.delta)) / pos_per_um
            else:
                wall_um = (0.375 * float(self.delta)) / pos_per_um
            in_obs_mask = depth_um > wall_um
            s_th: Optional[torch.Tensor] = None
            dtau_min_ps: Optional[torch.Tensor] = None
            if node_slack is not None:
                if self.slack_gate_mode in ("kappa_evac", "kappa_evac_elmore2"):
                    row_margin_um = float(self.row_margin) * (
                        float(self.row_height) / pos_per_um
                    )
                    disp_um = eff_depth_um + row_margin_um
                    if self.escape_slope_mode == "liberty_cell":
                        dtau_evac_ps, dtau_min_ps = self._compute_liberty_directional_evac_ps(
                            owner=owner,
                            star=star,
                            pin_pos=pin_pos,
                            geom=geom,
                            drv=drv,
                            snk=snk,
                            drv_mov=drv_mov,
                            snk_mov=snk_mov,
                            wire_slope=wire_slope,
                            alpha_load_ps_per_um=alpha_load_ps_per_um,
                            pin_slack=pin_slack,
                            node_slack=node_slack,
                            cand_idx=cand_idx,
                            nx_cand=nx_cand,
                            ny_cand=ny_cand,
                            grad_norm_cand=grad_norm_cand,
                            disp_um=disp_um,
                            row_margin_um=row_margin_um,
                            pos_per_um=pos_per_um,
                            device=device,
                        )
                    else:
                        if alpha_cell is None:
                            fallback_alpha = max(
                                float(getattr(star, "r_unit", 0.0)) * 0.01, 0.1
                            )
                            alpha_cell = torch.full(
                                (m,), fallback_alpha, device=device, dtype=torch.float32
                            )
                        dtau_evac_ps = alpha_cell * disp_um
                        if self.slack_gate_mode == "kappa_evac_elmore2":
                            rc_quad = max(
                                float(getattr(star, "r_unit", 0.0))
                                * float(getattr(star, "c_unit", 0.0)),
                                0.0,
                            )
                            if rc_quad > 0.0:
                                dtau_evac_ps = dtau_evac_ps + rc_quad * (disp_um * disp_um)
                        dtau_min_ps = (0.5 * alpha_cell * row_margin_um).clamp(min=1e-6)
                    if bool(in_obs_mask.any()):
                        med_dtau_in_macro = float(
                            dtau_evac_ps[in_obs_mask].median().item()
                        )
                    s_th = (float(self.kappa_slack) * dtau_evac_ps).clamp(min=1e-9)
                    u_ramp = (1.0 - node_slack / s_th).clamp(0.0, 1.0)
                    gamma_slack = u_ramp * u_ramp * (3.0 - 2.0 * u_ramp)
                else:
                    s_th = (
                        float(self.slack_margin_ps)
                        + float(self.depth_ps_per_um) * eff_depth_um
                    ).clamp(min=1.0)
                    dtau_min_ps = torch.full_like(s_th, max(float(self.slack_tau_ps), 10.0))
                    if self.slack_gate_mode == "slack_ramp":
                        u_ramp = (1.0 - node_slack / s_th).clamp(0.0, 1.0)
                        gamma_slack = u_ramp * u_ramp * (3.0 - 2.0 * u_ramp)
                    else:
                        tau = max(float(self.slack_tau_ps), 1.0)
                        raw_sig = torch.sigmoid(
                            ((s_th - node_slack) / tau).clamp(-30.0, 30.0)
                        )
                        cut = 0.02
                        gamma_slack = ((raw_sig - cut) / (1.0 - cut)).clamp(0.0, 1.0)
            else:
                gamma_slack = torch.zeros_like(chi)
            del alpha_cell, pin_slack

            if self.force_mode in ("normal_proj", "taylor_kkt") and node_slack is not None:
                gamma_inst = gamma_slack
            else:
                gamma_inst = torch.maximum(gamma_slack, chi)

            if self.gate_release_mode == "hysteresis" and node_slack is not None and s_th is not None:
                band_base = torch.maximum(
                    s_th / max(float(self.kappa_slack), 1e-4),
                    dtau_min_ps if dtau_min_ps is not None else s_th,
                )
                s_band = (float(self.hyst_kappa_band) * band_base).clamp(min=1e-6)
                u_rel = ((node_slack - s_th) / s_band).clamp(0.0, 1.0)
                gamma_ceil = 1.0 - u_rel * u_rel * (3.0 - 2.0 * u_rel)
                retained = torch.minimum(
                    float(self.ratchet_decay) * self.movable_risk_gate, gamma_ceil
                )
                gamma_mem = torch.maximum(retained, gamma_inst)
                del band_base, s_band, u_rel, gamma_ceil, retained
            else:
                gamma_mem = torch.maximum(
                    float(self.ratchet_decay) * self.movable_risk_gate, gamma_inst
                )
            self.movable_risk_gate.copy_(gamma_mem)

            eff_crit_wl = (
                0.75
                if self.slack_gate_mode in ("kappa_evac", "kappa_evac_elmore2")
                else float(self.crit_wl_scale)
            )
            if self.force_mode == "taylor_kkt":
                gate_mult = float(self.bg_wl_scale) * (1.0 - gamma_mem) + gamma_mem
                gx_mov, gy_mov = self._compute_total_pull_grad_xy(
                    owner, pos, star, coeff, pair_slope, geom, drv, snk
                )
                h_tx, h_ty, gamma_pos = self._compute_timing_hessian_and_gamma(
                    owner, pos, star, coeff
                )
                base_pull = 0.5 * gate_mult * torch.sqrt(gx_mov * gx_mov + gy_mov * gy_mov)
                if (
                    cand_idx.numel() > 0
                    and d_in_cand is not None
                    and nx_cand is not None
                    and ny_cand is not None
                    and grad_norm_cand is not None
                ):
                    f_in_raw = gx_mov[cand_idx] * nx_cand + gy_mov[cand_idx] * ny_cand
                    h_wl_cand = (eff_crit_wl * p_wl[cand_idx]) / max(float(gamma_pos), 1e-6)
                    h_n_cand = (
                        h_wl_cand
                        + h_tx[cand_idx] * (nx_cand * nx_cand)
                        + h_ty[cand_idx] * (ny_cand * ny_cand)
                    )
                    f_cap_cand = torch.maximum(
                        f_in_raw.clamp(min=0.0),
                        eff_crit_wl * p_wl[cand_idx] + f_time[cand_idx],
                    )
                    f_kkt_cand = torch.minimum(
                        (f_in_raw + h_n_cand * d_in_cand).clamp(min=0.0),
                        f_cap_cand,
                    )
                    f_kkt_cand = torch.where(
                        grad_norm_cand > 1e-8, f_kkt_cand, torch.zeros_like(f_kkt_cand)
                    )
                    t_in = (d_in_cand / max(1.5 * float(self.delta), 1e-6)).clamp(0.0, 1.0)
                    u_in = t_in * t_in * (3.0 - 2.0 * t_in)
                    eff_norm = (1.0 - u_in) + u_in * grad_norm_cand.clamp(min=0.35)
                    base_pull[cand_idx] = gate_mult[cand_idx] * (f_kkt_cand / eff_norm)
                    del f_in_raw, h_wl_cand, h_n_cand, f_cap_cand, f_kkt_cand, t_in, u_in, eff_norm
                chi_pwl_term = eff_crit_wl * gamma_mem * p_wl
                del gate_mult, gx_mov, gy_mov, h_tx, h_ty
            elif self.force_mode == "normal_proj":
                gate_mult = float(self.bg_wl_scale) * (1.0 - gamma_mem) + gamma_mem
                gx_mov, gy_mov = self._compute_total_pull_grad_xy(
                    owner, pos, star, coeff, pair_slope, geom, drv, snk
                )
                base_pull = 0.5 * gate_mult * torch.sqrt(gx_mov * gx_mov + gy_mov * gy_mov)
                if (
                    cand_idx.numel() > 0
                    and psi_cand is not None
                    and nx_cand is not None
                    and ny_cand is not None
                    and grad_norm_cand is not None
                ):
                    f_in_cand = (
                        gx_mov[cand_idx] * nx_cand + gy_mov[cand_idx] * ny_cand
                    ).clamp(min=0.0)
                    f_in_cand = torch.where(
                        grad_norm_cand > 1e-8, f_in_cand, torch.zeros_like(f_in_cand)
                    )
                    psi_wall_pos = (
                        0.725 * float(self.delta)
                        if self.profile_mode == "tent_rim"
                        else 0.375 * float(self.delta)
                    )
                    t_in = (
                        (psi_cand - psi_wall_pos) / max(1.5 * float(self.delta), 1e-6)
                    ).clamp(0.0, 1.0)
                    u_in = t_in * t_in * (3.0 - 2.0 * t_in)
                    eff_norm = (1.0 - u_in) + u_in * grad_norm_cand.clamp(0.35, 1.0)
                    evac_mult = (1.0 + float(self.interior_evac_boost) * u_in) / eff_norm
                    base_pull[cand_idx] = gate_mult[cand_idx] * f_in_cand * evac_mult
                    del f_in_cand, t_in, u_in, eff_norm, evac_mult
                chi_pwl_term = gamma_mem * p_wl
                del gate_mult, gx_mov, gy_mov
            else:
                wl_mult = (
                    float(self.bg_wl_scale) * (1.0 - gamma_mem)
                    + eff_crit_wl * gamma_mem
                )
                base_pull = wl_mult * p_wl + f_time
                chi_pwl_term = eff_crit_wl * gamma_mem * p_wl

            self._refresh_count += 1
            self.last_slack_gate_stats = {
                "risk_active_cells": float((gamma_mem > 0.05).sum().item()),
                "in_macro_risk_cells": float(
                    (in_obs_mask & (gamma_mem > 0.05)).sum().item()
                ),
                "in_macro_safe_cells": float(
                    (in_obs_mask & (gamma_mem <= 0.05)).sum().item()
                ),
                "mean_gate_in_macro": float(
                    gamma_mem[in_obs_mask].mean().item()
                )
                if bool(in_obs_mask.any())
                else 0.0,
                "alpha_load_ps_per_um": float(alpha_load_ps_per_um),
                "median_dtau_in_macro_ps": float(med_dtau_in_macro),
            }
            if self._refresh_count == 1 or (self._refresh_count % 4 == 0):
                logging.info(
                    "MacroExclusionField[%s/%s/%s/%s] refresh#%d: risk_gt05=%d, in_macro_risk=%d, in_macro_safe=%d, mean_gate=%.3f, alpha_load=%.3fps/um, med_dtau=%.1fps",
                    self.slack_gate_mode,
                    self.force_mode,
                    self.escape_slope_mode,
                    self.gate_release_mode,
                    self._refresh_count,
                    int(self.last_slack_gate_stats["risk_active_cells"]),
                    int(self.last_slack_gate_stats["in_macro_risk_cells"]),
                    int(self.last_slack_gate_stats["in_macro_safe_cells"]),
                    self.last_slack_gate_stats["mean_gate_in_macro"],
                    self.last_slack_gate_stats["alpha_load_ps_per_um"],
                    self.last_slack_gate_stats["median_dtau_in_macro_ps"],
                )
        else:
            if self.force_mode == "taylor_kkt":
                gx_mov, gy_mov = self._compute_total_pull_grad_xy(
                    owner, pos, star, coeff, pair_slope, geom, drv, snk
                )
                h_tx, h_ty, gamma_pos = self._compute_timing_hessian_and_gamma(
                    owner, pos, star, coeff
                )
                gate_u = float(self.bg_wl_scale) + float(self.wl_gate_scale) * chi
                base_pull = 0.5 * gate_u * torch.sqrt(gx_mov * gx_mov + gy_mov * gy_mov)
                if (
                    cand_idx.numel() > 0
                    and d_in_cand is not None
                    and nx_cand is not None
                    and ny_cand is not None
                    and grad_norm_cand is not None
                ):
                    f_in_raw = gx_mov[cand_idx] * nx_cand + gy_mov[cand_idx] * ny_cand
                    h_wl_cand = (0.75 * p_wl[cand_idx]) / max(float(gamma_pos), 1e-6)
                    h_n_cand = (
                        h_wl_cand
                        + h_tx[cand_idx] * (nx_cand * nx_cand)
                        + h_ty[cand_idx] * (ny_cand * ny_cand)
                    )
                    f_cap_cand = torch.maximum(
                        f_in_raw.clamp(min=0.0),
                        0.75 * p_wl[cand_idx] + f_time[cand_idx],
                    )
                    f_kkt_cand = torch.minimum(
                        (f_in_raw + h_n_cand * d_in_cand).clamp(min=0.0),
                        f_cap_cand,
                    )
                    f_kkt_cand = torch.where(
                        grad_norm_cand > 1e-8, f_kkt_cand, torch.zeros_like(f_kkt_cand)
                    )
                    t_in = (d_in_cand / max(1.5 * float(self.delta), 1e-6)).clamp(0.0, 1.0)
                    u_in = t_in * t_in * (3.0 - 2.0 * t_in)
                    eff_norm = (1.0 - u_in) + u_in * grad_norm_cand.clamp(min=0.35)
                    base_pull[cand_idx] = gate_u[cand_idx] * (f_kkt_cand / eff_norm)
                chi_pwl_term = float(self.wl_gate_scale) * chi * p_wl
                del gx_mov, gy_mov, h_tx, h_ty, gate_u
            elif self.force_mode == "normal_proj":
                gx_mov, gy_mov = self._compute_total_pull_grad_xy(
                    owner, pos, star, coeff, pair_slope, geom, drv, snk
                )
                gate_u = float(self.bg_wl_scale) + float(self.wl_gate_scale) * chi
                base_pull = 0.5 * gate_u * torch.sqrt(gx_mov * gx_mov + gy_mov * gy_mov)
                if (
                    cand_idx.numel() > 0
                    and nx_cand is not None
                    and ny_cand is not None
                    and grad_norm_cand is not None
                ):
                    f_in_cand = (
                        gx_mov[cand_idx] * nx_cand + gy_mov[cand_idx] * ny_cand
                    ).clamp(min=0.0)
                    f_in_cand = torch.where(
                        grad_norm_cand > 1e-8, f_in_cand, torch.zeros_like(f_in_cand)
                    )
                    base_pull[cand_idx] = gate_u[cand_idx] * f_in_cand
                chi_pwl_term = float(self.wl_gate_scale) * chi * p_wl
                del gx_mov, gy_mov, gate_u
            else:
                base_pull = (
                    float(self.bg_wl_scale) * p_wl
                    + f_time
                    + float(self.wl_gate_scale) * chi * p_wl
                )
                chi_pwl_term = float(self.wl_gate_scale) * chi * p_wl
        del psi_cand, d_in_cand, nx_cand, ny_cand, grad_norm_cand, pin_pos, geom, pair_slope, wire_slope, drv, snk, drv_mov, snk_mov
        self.movable_base_pull.copy_(base_pull)

        nz = torch.nonzero(base_pull > 1e-9, as_tuple=False).reshape(-1)
        self.active_idx = nz
        self.active_f_time = f_time[nz]
        self.active_chi_pwl = chi_pwl_term[nz]
        self._cached_key = key

    def objective(self, owner, pos: torch.Tensor, overflow: Optional[float]) -> torch.Tensor:
        """Differentiable macro exclusion potential ``Phi_macro(pos)``."""
        if not self.has_obstacles:
            return pos.new_zeros(())
        beta = self.compute_beta(overflow, owner=owner)
        self.last_beta = float(beta)
        self.last_overflow = None if overflow is None else float(overflow)
        if beta <= 0.0:
            self.last_penalty_value = 0.0
            return pos.new_zeros(())

        self.refresh_criticality_pulls(owner, pos)
        cand_idx = self._query_stencil_candidates(pos)
        if cand_idx.numel() == 0:
            self.last_penalty_value = 0.0
            return pos.new_zeros(())

        base_w = self.movable_base_pull[cand_idx]
        nz_mask = base_w > 1e-9
        if not bool(nz_mask.any()):
            self.last_penalty_value = 0.0
            return pos.new_zeros(())
        cand_idx = cand_idx[nz_mask]
        weights = float(beta) * base_w[nz_mask]

        half_w = self.movable_half_w[cand_idx].to(pos.dtype)
        half_h = self.movable_half_h[cand_idx].to(pos.dtype)
        val = _MacroFieldObjectiveFunction.apply(
            pos,
            self.grid,
            cand_idx,
            weights,
            half_w,
            half_h,
            self.num_nodes,
            self.g_xl,
            self.g_yl,
            self.h_g,
        )
        self.last_penalty_value = float(val.detach().item())
        return val

    @torch.no_grad()
    def diagonal_hessian(self, owner, pos: torch.Tensor, overflow: Optional[float]) -> torch.Tensor:
        """Secant MM diagonal upper bound ``(2, num_nodes)`` in ``[Objective / pos_unit^2]``."""
        out = pos.new_zeros((2, self.num_nodes))
        if not self.has_obstacles:
            return out
        beta = self.compute_beta(overflow, owner=owner)
        if beta <= 0.0:
            return out

        self.refresh_criticality_pulls(owner, pos)
        cand_idx = self._query_stencil_candidates(pos)
        if cand_idx.numel() == 0:
            self.last_overlapping_cells = 0
            return out

        cx = pos[cand_idx].detach() + self.movable_half_w[cand_idx].to(pos.dtype)
        cy = pos[self.num_nodes + cand_idx].detach() + self.movable_half_h[cand_idx].to(pos.dtype)
        psi, gx, gy = _eval_cubic_bspline_2d(self.grid, cx, cy, self.g_xl, self.g_yl, self.h_g)
        grad_norm = torch.sqrt(gx * gx + gy * gy)
        base_w = self.movable_base_pull[cand_idx]
        inside = (psi > 1e-6) & (grad_norm > 1e-8) & (base_w > 1e-9)
        self.last_overlapping_cells = int((psi > 0.25 * self.delta).sum().item())
        if not bool(inside.any()):
            return out

        sub_idx = cand_idx[inside]
        w = (float(beta) * base_w[inside]).to(pos.dtype)
        denom = torch.clamp(psi[inside] + float(self.delta), min=2.0 * float(self.delta))
        h_val = (w * grad_norm[inside]) / denom
        out[0, sub_idx] = h_val
        out[1, sub_idx] = h_val
        return out
