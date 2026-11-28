"""Persistent pair-weight update helpers on the dense star tensor."""

from __future__ import annotations

import torch

EFFTDP_PAIR_W0 = 10.0
EFFTDP_PAIR_INCREMENT = 0.2
EFFTDP_PAIR_WMAX = 50.0
NEW_PAIR_POLICIES = ("efftdp", "full_evidence")
CAP_POLICIES = ("hard", "tanh")


def momentum_target_weights(raw_weight, scale=1.0, power=1.0):
    """Map the current adjoint to an unbounded EMA target."""
    raw = torch.as_tensor(raw_weight)
    if power == 1.0:
        return raw * raw.new_tensor(scale)
    return raw.sign() * raw.abs().pow(power) * raw.new_tensor(scale)


def momentum_pair_weights(
    previous_weight,
    current_weight,
    alpha=0.5,
    previous_seen=None,
    return_seen=False,
):
    """Apply an unbounded exponential history update.

    ``alpha`` is the coefficient on the old value, so the update is exactly
    ``alpha * old + (1 - alpha) * new``.  Unlike the EffTDP accumulator below,
    this helper deliberately has no floor, cap, admission clamp, or miss
    decay.  A pair that is absent from the current support therefore moves
    toward zero through the same EMA update as every other pair.

    ``previous_seen`` is retained only for provenance and diagnostics; it does
    not alter the numeric update.  Omitting it infers membership from a
    non-zero previous value.
    """
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("momentum alpha must be finite and in [0, 1]") from exc
    if not torch.isfinite(torch.tensor(alpha)).item() or not 0.0 <= alpha <= 1.0:
        raise ValueError("momentum alpha must be finite and in [0, 1]")

    previous = torch.as_tensor(previous_weight)
    current = torch.as_tensor(
        current_weight, dtype=previous.dtype, device=previous.device
    )
    if previous.shape != current.shape:
        raise ValueError("previous_weight and current_weight must have the same shape")
    if not bool(torch.isfinite(previous).all().item()):
        raise ValueError("previous_weight must be finite")
    if not bool(torch.isfinite(current).all().item()):
        raise ValueError("current_weight must be finite")

    if previous_seen is None:
        seen = previous != 0
    else:
        seen = torch.as_tensor(
            previous_seen, dtype=torch.bool, device=previous.device
        )
        if seen.shape != previous.shape:
            raise ValueError("previous_seen and previous_weight must have the same shape")

    updated = previous * previous.new_tensor(alpha) + current * previous.new_tensor(
        1.0 - alpha
    )
    updated_seen = seen | (current != 0)
    if return_seen:
        return updated, updated_seen
    return updated


def accumulate_efftdp_pair_weights(
    previous_weight,
    covering,
    w0=EFFTDP_PAIR_W0,
    increment=EFFTDP_PAIR_INCREMENT,
    wmax=EFFTDP_PAIR_WMAX,
    hit_threshold=0.0,
    miss_decay=1.0,
    new_pair_policy="efftdp",
    cap_policy="hard",
    previous_seen=None,
    return_seen=False,
):
    """Update dense star-pair weights with fast attack and leaky release.

    Selection is ``covering > hit_threshold``.  With ``new_pair_policy=efftdp``,
    a never-selected pair starts at ``w0`` and only covering beyond the first
    occurrence contributes an increment, matching EffTDP's insertion rule.
    ``full_evidence`` instead credits the complete first covering value as
    ``w0 + increment * covering``.
    ``cap_policy=hard`` applies the original upper clamp.  ``tanh`` treats
    ``wmax - w0`` as a smooth saturation scale while preserving additive,
    refresh-partition-independent evidence in the inverse-tanh coordinate.
    A missed historical pair is multiplied by ``miss_decay``.  The default
    value 1 exactly preserves the original ratchet behavior.  On reselection,
    a decayed pair is first restored to at least ``w0`` and then incremented.
    ``previous_seen`` distinguishes a historical pair whose weight is zero
    from a pair that has never entered the table.  For compatibility, omitting
    it infers membership from ``previous_weight > 0``.  Set ``return_seen=True``
    to receive the updated membership mask alongside the weights.
    """
    values = {
        "w0": float(w0),
        "increment": float(increment),
        "wmax": float(wmax),
        "hit_threshold": float(hit_threshold),
        "miss_decay": float(miss_decay),
    }
    if not all(torch.isfinite(torch.tensor(value)).item() for value in values.values()):
        raise ValueError("bounded pair-weight parameters must be finite")
    if values["w0"] < 0.0:
        raise ValueError("w0 must be non-negative")
    if values["wmax"] < values["w0"]:
        raise ValueError("wmax must be at least w0")
    if values["increment"] <= 0.0:
        raise ValueError("increment must be positive")
    if values["hit_threshold"] < 0.0:
        raise ValueError("hit_threshold must be non-negative")
    if not 0.0 <= values["miss_decay"] <= 1.0:
        raise ValueError("miss_decay must be in [0, 1]")
    new_pair_policy = str(new_pair_policy).strip().lower()
    if new_pair_policy not in NEW_PAIR_POLICIES:
        raise ValueError(
            "new_pair_policy must be one of: %s" % ", ".join(NEW_PAIR_POLICIES)
        )
    cap_policy = str(cap_policy).strip().lower()
    if cap_policy not in CAP_POLICIES:
        raise ValueError("cap_policy must be one of: %s" % ", ".join(CAP_POLICIES))

    previous = torch.as_tensor(previous_weight)
    hits = torch.as_tensor(covering, dtype=previous.dtype, device=previous.device)
    if previous.shape != hits.shape:
        raise ValueError("previous_weight and covering must have the same shape")
    if not bool(torch.isfinite(previous).all().item()):
        raise ValueError("previous_weight must be finite")
    if not bool(torch.isfinite(hits).all().item()):
        raise ValueError("covering must be finite")
    if bool(torch.any(previous < 0).item()):
        raise ValueError("previous_weight must be non-negative")

    if previous_seen is None:
        seen = previous > 0
    else:
        seen = torch.as_tensor(
            previous_seen, dtype=torch.bool, device=previous.device
        )
        if seen.shape != previous.shape:
            raise ValueError("previous_seen and previous_weight must have the same shape")

    selected = hits > values["hit_threshold"]
    extra_new = (
        hits
        if new_pair_policy == "full_evidence"
        else (hits - 1).clamp(min=0)
    )
    new_mask = selected & ~seen
    reselected_mask = selected & seen
    updated = previous.clone()
    w0_value = previous.new_tensor(values["w0"])
    increment_value = previous.new_tensor(values["increment"])
    wmax_value = previous.new_tensor(values["wmax"])
    active_mask = seen
    updated = torch.where(
        active_mask,
        torch.clamp(previous, min=w0_value, max=wmax_value),
        updated,
    )
    if values["miss_decay"] < 1.0:
        missed_mask = seen & ~selected
        decayed = (
            torch.clamp(previous, min=previous.new_tensor(0.0), max=wmax_value)
            * previous.new_tensor(values["miss_decay"])
        )
        updated = torch.where(missed_mask, decayed, updated)
    if cap_policy == "hard":
        updated = torch.where(
            new_mask,
            torch.clamp(w0_value + increment_value * extra_new, max=wmax_value),
            updated,
        )
        updated = torch.where(
            reselected_mask,
            torch.clamp(updated + increment_value * hits, max=wmax_value),
            updated,
        )
    else:
        span = wmax_value - w0_value
        if values["wmax"] == values["w0"]:
            new_weight = torch.full_like(previous, values["w0"])
            reselected_weight = new_weight
        else:
            def smooth_add(base, delta):
                q = (torch.clamp(base, min=w0_value, max=wmax_value) - w0_value) / span
                step = torch.tanh(delta / span)
                return torch.clamp(
                    w0_value + span * (q + step) / (1.0 + q * step),
                    min=w0_value,
                    max=wmax_value,
                )

            new_weight = smooth_add(
                torch.full_like(previous, values["w0"]),
                increment_value * extra_new,
            )
            reselected_weight = smooth_add(updated, increment_value * hits)
        updated = torch.where(new_mask, new_weight, updated)
        updated = torch.where(reselected_mask, reselected_weight, updated)
    updated_seen = seen | selected
    if return_seen:
        return updated, updated_seen
    return updated
