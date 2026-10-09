# CPU coarse-guided maze routing

Current correction: CPU RRR completes the changing cost schedule and returns
the final route for RC evaluation. The measurements below record the earlier
`2b6cae3` policy, which stopped after a non-improving pass and restored an
intermediate overflow minimum. They are retained as historical evidence.
Corrected RRR3 executes/returns round 3, takes 27.58 s in the installed probe,
and has a 16.01% external TNS gap. See
[schedule-fix evidence](evidence/cpu-maze-rrr-schedule.json).

Implemented on 2026-10-09 against Xplace `59991041`. The implementation is
available in the primary checkout and rebuilt CPU native runtime. Qualification
artifacts are under
`/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_cpu_rrr_coarse_20261009`.
See [machine-readable evidence](evidence/cpu-maze-rrr-coarse.json).

Implementation commit: Xplace `2b6cae363ebd425a3973892386663901451be06b`.
Commit preparation passed both standards and specification review with no
actionable findings. The committed source hashes match the qualified runtime's
source snapshot.

## Search contract

CPU `cpu_pr_maze` now uses the existing GRDatabase `csrnScale` for hierarchical
search. The default scale is eight: a physical 512 by 512 routing grid has a
64 by 64 coarse grid. Padded fine-grid storage remains unchanged. Native
`load_gr_params({"csrn_scale": 1, ...})` selects the original full-grid search.
This is an existing native setting, not a new DreamPlace JSON parameter.
RRR-zero routing retains the pattern-route path.

The algorithm follows the CUDA coarse-search/fine-refinement architecture,
while using deterministic A* rather than CUDA's bounded parallel sweep kernels:

1. Aggregate legal fine wire costs by coarse cell, multiplying their integer
   mean by the scale. Aggregate legal via costs by their mean. Fine costs retain
   the existing iteration congestion schedule and routing-layer policy.
2. Search the coarse graph from the already connected tree to the next branch.
3. Refine that branch on the original fine grid inside the coarse path's cells,
   open on all enabled routing layers. Fine costs, capacities, explicit
   blockers, and pin-via legality are authoritative.
4. If refinement fails, expand the corridor by one and then two coarse cells.
   If the coarse graph fails or all refinements fail, use full-grid A*.
5. Export only validated fine-grid wire spans and vias through the canonical
   route state. Continue to require every branch and pin access to be connected.

Coarse wire costs are an approximation based on mean fine costs; they do not
reproduce CUDA's inverse remaining-resource aggregation or guarantee the
full-grid minimum-cost route. Coarse routes are guidance and are never exported
as physical routes. The fallback preserves reachability, not minimum-cost parity
when a restricted refinement already succeeds.

## Batch and cache ownership

Each fixed 32-net batch reads one immutable coarse snapshot. Fine searches use
per-worker scratch, and candidates commit in deterministic net order. Current
fine costs are used for serial re-search; the batch's coarse guidance remains
frozen until the next batch.

Serial cost updates invalidate only the coarse blocks whose fine integer costs
changed. Refresh those blocks before the next batch. Wire/via costs and their
coarse lower bounds match a full rebuild. No cache spans a new RRR iteration or
a separate GR call. Worker memory reservations include coarse search scratch;
shared storage includes coarse maps and invalidation flags.

The extracted `CpuMazeSearch` module owns the common grid encoding, legal edge
costs, A* search and scratch reset. `CpuMazeCoarseGrid` owns aggregation, cache
invalidation and corridor refinement. `CpuMazeRouter` continues to own branch
and pin connectivity, route construction and typed failures.

## BM64 evidence

All CPU measurements use the same DEF SHA256
`dd0231ba72f7cd8f9bb24f29f709aabf890e7327e35c67b28e47b50a06d205e1`,
LEFs, MET2-MET5, physical grid 512, requested RRR3, and CPU affinity 344-351.
Both full and coarse CPU routing execute two RRR iterations and restore the
first iteration. Times cover GR routing, excluding RC/STA.

| CPU implementation | Workers | Total GR seconds |
| --- | ---: | ---: |
| Archived full-grid A* baseline | 8 | 609.100 |
| Coarse guidance, complete rebuild per batch | 8 | 32.241 |
| Coarse guidance, incremental refresh | 8 | 16.004 |
| Installed coarse guidance, incremental refresh | 8 | 16.076 |
| Coarse guidance, incremental refresh | 1 | 29.227 |

The complete-rebuild, incremental-refresh and one-worker runs have identical
initial, per-iteration, final route and resource hashes, and byte-identical
route-pack NPZ files. BM64 uses eight overlapping searches when eight workers
are requested. No corridor widening or full-grid rescue is needed in this case.
The full-grid-to-coarse timing comparison is an archived baseline versus a new
algorithm; it is not a repeated statistical speedup study. The archived CUDA
RRR3 reference takes 14.287 seconds but runs three iterations with a different
algorithm and build.

The alignment check uses the same original DEF, libraries, SDC and ideal clock
in OpenROAD, with two parasitic sources:

| Parasitic source | WNS ns | TNS ns | Relative absolute-TNS gap to OpenROAD GR50 |
| --- | ---: | ---: | ---: |
| Full-grid CPU route exported through native RC tree | -2.803030 | -852.581421 | 20.36% |
| Coarse-guided CPU route exported through native RC tree | -2.385395 | -737.840576 | 31.08% |
| OpenROAD independent GR50 parasitics | -2.999287 | -1070.544556 | reference |

Both native-RC rows use OpenROAD read-only STA after SPEF export. They isolate
the routing/parasitic difference from differences in internal STA propagation.
The check establishes a substantial speed improvement on BM64 and valid RC/STA
coverage, but does not establish improved alignment with external OpenROAD.

## Validation

- Existing full-grid search checks and new coarse-search checks pass.
- Coarse checks cover rectangular/transposed grids, partial coarse cells,
  saturated finite costs, pin ladders, blocked corridors requiring full-grid
  rescue, stale coarse guidance, incremental/full rebuild equality, and worker
  counts 1/2/4/8.
- Coarse checks pass AddressSanitizer and UndefinedBehaviorSanitizer.
- Thirteen native tests cover RRR-zero parity, multi-batch deterministic
  routes/resource hashes, RC connectivity and finite nonzero gradients.
- A complete 50-step sizing fixture passes clock gradients, master/pin geometry
  changes, rerouting, best-state restoration, legalization and terminal STA.
- Original BM64 native RC export covers all 16,208 eligible nets and 56,299
  pins. Native and external evaluators retain exactly one excluded clock net.

CPU positive-RRR remains an explicit experimental backend. External alignment
and the impact on a complete BM64 S50 run are separate follow-up qualifications.
