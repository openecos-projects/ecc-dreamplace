# Quad-Gradient Sizing Controller Plan

Status: Gate 1's shared cell budget is the `sizing` flow default at 1% after a single raw-DEF BM64 pilot; the 10% pilot regressed and remains an explicit sensitivity setting. Gate 2's oscillation veto remains opt-in; its 10% pilot regressed. Gate 3's non-Nesterov best-state snapshot timing was fixed, but its full state contract and Gate 4 are not implemented. This default is a working standard, not qualified post-legalization QoR. See [1% pilot](/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_regression_20260928_BM64_shared1_cap_ablation/RESULTS.md), [10% pilot](/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_regression_20260928_BM64_gzz_gate1_shared10/RESULTS.md), and [Gate 2](/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_regression_20260928_BM64_gzz_gate2_veto/RESULTS.md).

## Goal And Scope

Extend the existing discrete size/VT sizing controller in four gated steps:

1. Select size and VT actions under one global Top-K budget.
2. Suppress repeated per-instance master oscillation.
3. Preserve and verify whole-design best-state restoration as the controller changes.
4. Interleave timing updates with timing-safe power recovery.

The existing four-direction legal-neighbor table and gradient path are inputs to this work, not targets for a gradient rewrite. In-loop reference-STA calibration is deferred. The shared 1% cell budget is now the `sizing` (`size_only`) flow default; `--discrete-gradient-topk-shared-budget 0` retains the legacy per-direction policy, and `joint` flow defaults remain unchanged. These items are paper-inspired engineering work, not a claim of reproducing the complete DAC 2026 algorithm.

## Fixed Evidence Contract

- Pilot design: BM64, using the input DEF and LEF/Liberty/SDC/RC settings recorded in the [10% baseline](/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_regression_20260923_BM64_diff_tdp_size_vt_real_size_topk_up10/RESULTS.md). Check the input hash before every comparison: `dd0231ba72f7cd8f9bb24f29f709aabf890e7327e35c67b28e47b50a06d205e1`.
- Baseline: 50 steps, 49 discrete updates, size-up 10%, size-down 0%, VT 10%; raw post-sizing area `+4.498972%`, WNS `-1.56612893753 ns`, TNS `-453.895565092 ns`. This baseline ran on DreamPlace `932e8bc0`; record the current commit and rerun the control on it if implementation provenance changes.
- Each experiment gets a fresh output directory, effective parameters, commit and input hashes, exit status, action counts, final DEF/Verilog, external STA reports, and area/master/coordinate deltas. Do not overwrite prior baselines.
- Evaluate all variants with the same external OpenSTA contract. Report raw sizing separately from the result after the same legalization command; the existing raw baseline is not a legalized OpenROAD comparison. Record leakage separately from area and from total power.

## Gate 1: One Shared Top-K Budget

Owner: `dreamplace/ops/discrete_gradient_topk/quad_gradient_topk.py`, its local tests, and the sizing config/CLI boundary.

- In the `sizing` shared-budget policy, default to `K = ceil(0.01 * N_sizeable)` per round, where `N_sizeable` is the number of `inst_is_sizeable` cells, not the number of legal moves or cells with an improving move. The budget percentage may be overridden; 0% selects nothing. Keep the existing per-direction percentage policy available as an explicit control experiment.
- Confirm that scores from both axes have the same sign and comparable objective units. Check candidate rankings against independently evaluated single-move objective deltas on a small fixed sample. Do not combine incomparable size and VT scores just because both are numeric.
- For each cell, compare its legal improving `size_up`, `size_down`, `vt_slowdown`, and `vt_speedup` scores and retain only the best action; then rank these per-cell winners globally and take up to `K`. Use deterministic instance/action ordering for ties. Never spend two slots or apply two actions to the same cell; retain the existing runtime/PlaceDB/OpenROAD synchronization.
- Test ties, missing size/VT combinations, multiple attractive actions on one instance, fewer than `K` eligible cells, 0%/1% budget boundaries, and single-VT fallback. Verify `K` uses the full sizeable-cell denominator and the selected count never exceeds `K`.
- Pilot BM64 with the default shared 1% cell budget against a same-checkout 10% sensitivity run. Record `N_sizeable`, `K`, per-round selected/applied actions, cumulative action events, final unique master changes, area, external WNS/TNS, and runtime. The archived per-direction policy remains a reference, but predates the best-state timing fix and is not a matched post-fix control.

Exit: focused tests pass and the BM64 run proves both the budget invariant and full writeback/STA contract; no claim of QoR superiority is required.

## Gate 2: Per-Instance Oscillation Control

Owner: the discrete controller state and selection mask, with tests next to the Top-K operator.

- Track **committed** cell IDs per instance, not tentative candidates or continuous logits. Count reversals in the same optimization phase; first distinguish `A -> B -> A` from repeated `A -> B -> A -> B` before freezing.
- Start with an explicit short cooldown or immediate-reversal veto for repeatedly reversing instances. Insert the frozen/cooldown mask **before** global Top-K so the budget can refill. Record prevented reversals and replacement actions.
- Do not treat a timing-to-power phase transition as oscillation by default; record cross-phase reversals separately. Revisit this rule after Gate 4 experiments.
- The paper's `s_g^best` has no published gate-local evaluation rule. Do not label a state "best per gate" without an evaluated and reproducible choice rule. Test VT reversal, size reversal, legitimate one-time undo, cross-phase reversal, and history after whole-design rollback.

Exit: unit tests establish the state machine and a BM64 freeze-on/off run shows how reversal count, action budget, area, and external timing change. Keep it opt-in if QoR regresses or legitimate recovery is blocked.

## Gate 3: Whole-Design Best-State Contract

Owner: `NonLinearPlace` best-state snapshot/restore and focused integration tests. Reuse `early_stop_restore_best`; do not create a second independent snapshot system.

- In timing-only mode, preserve the current comparable timing-loss criterion and verify that a deliberately worse final step restores the earlier full state before projection/writeback.
- After Gate 2, ensure restored cell IDs, size/VT logits, leakage, geometry, pin offsets, timing arcs/cache state, PlaceDB mirrors, and oscillation history/masks agree. No stale freeze may refer to a master state that was rolled back.
- Before Gate 4, define a single, stable selection rule across phases: first enforce timing/DRV/area feasibility relative to the timing-stage anchor, then choose the best recovery state by the measured recovery objective. A timing-only best-loss comparison would otherwise erase valid power recovery. Keep objective values and the chosen anchor in the run artifact.
- Exercise a worse-last-step restoration and a timing-to-recovery-to-restoration case. Compare final DEF and external STA to the recorded chosen snapshot, not just a Python tensor.

Exit: the original timing-only contract still passes; post-restore DEF/master, geometry, VT, leakage, and STA evidence match the chosen snapshot under both timing and recovery modes.

## Gate 4: Timing-Safe Recovery Schedule

Owner: sizing flow scheduling and the existing timing/leakage objective boundary. Keep this out of placement-only runs and buffering.

- Make the schedule opt-in: several timing Top-K rounds followed by one recovery round, using separate gradients and a timing-safety mask. During recovery, permit only size-down and slower-VT moves that reduce the selected recovery metric; enforce one action per instance and the same `K` budget contract.
- The initial recovery objective is **Liberty leakage**, not paper-equivalent total power. First verify that its value/gradient is live and changes with actual master swaps. Do not substitute area for power or claim internal/switching power improvement without an independent total-power evaluator.
- Apply timing, slew/cap, and area acceptance guards to the candidate batch. Reject or roll back recovery that breaches the agreed timing-stage anchor; log the guard values, rejection reason, and objective before/after. Establish numeric tolerances and timing-safety threshold from a measured BM64 sweep instead of importing paper constants without checking units.
- Compare timing-only, VT-only recovery, size-down-only recovery, and combined recovery with the same input, step count, action budget, legalization command, and external STA. Report leakage and area alongside WNS/TNS and the number of accepted/rejected batches. Defer the paper's separate smoothing implementation unless the existing timing operator can provide a validated timing-safety signal.

Exit: a real BM64 run demonstrates accepted legal power-reducing actions with an independently checked timing boundary and a post-legalization STA result. If only leakage is validated, report "leakage recovery", not "total-power optimization".

## Release Boundary

After each gate, run narrow operator/flow tests plus the BM64 pilot before proceeding. Once Gate 4 passes in `size_only`, repeat the contract tests for `joint` and its size/VT/segment state restoration before enabling the controller there. The requested 1% sizing default is provisional on a single raw-DEF pilot; do not call it qualified QoR or compare against OpenROAD as an equal-budget win until inputs, actions, area, legalization, and STA are aligned.

Paper: [Pan et al., DAC 2026, Disentangled Differentiable Timing-Power Co-Optimization with Quad-Gradient Gate Sizing](https://guozz.cn/publication/phyoptdac-26/phyoptdac-26.pdf).
