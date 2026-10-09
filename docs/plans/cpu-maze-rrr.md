# CPU maze routing and rip-up/reroute plan

Status: CPU opt-in implementation and matched MET2-MET5 standalone qualification complete with performance and QoR limitations. The earlier MET2-RDL qualification is superseded for the intended window. Keep CPU RRR experimental and default RRR=0; see the [matched-window evidence](evidence/cpu-maze-rrr-m2-m5.json).
Plan version: v1, 2026-10-08.

Phase A implementation commits:

- Xplace route-state extraction: `b132020dd00fa165718a40b11bf2b5bad0750c7f`;
- DreamPlace gitlink update: `20c19ec83f3f5097c6170e77e4ba667a54de9db6`;
- Phase A evidence: [docs/plans/evidence/cpu-maze-rrr-phase-a.json](evidence/cpu-maze-rrr-phase-a.json).

Phase B implementation checkpoint:

- Xplace standalone maze search and deterministic batch API: `0dc5ec1586bb6b4b743ea58fdcc39db7e02033fa`;
- DreamPlace gitlink update: `4742989317a4a8f8f22a80528c555306e19415f1`;
- Phase B evidence: [docs/plans/evidence/cpu-maze-rrr-phase-b.json](evidence/cpu-maze-rrr-phase-b.json).

Phase C implementation checkpoint:

- Xplace CPU maze RRR fixture and rollback coverage: `44d05040d208c6ab0a398e5f76f88854c9679dcc`;
- DreamPlace gitlink update: `a1ed9bc8556565a025ce8480a02986a5463ca2b7`;
- Phase C evidence: [docs/plans/evidence/cpu-maze-rrr-phase-c.json](evidence/cpu-maze-rrr-phase-c.json).

Phase C follow-up checkpoints:

- Xplace conflict classification and maze worker metadata: `0fede4d677c2e1fffda769e1d27a847a7d101fcb`;
- DreamPlace GR sizing snapshot metadata and BM64 native diagnostic evidence: `66711335`.

Execution checkpoint (2026-10-09): Xplace `b93b5c06f599480448f992646679145e01abf413` separates RRR orchestration from `CPURouter.cpp`, validates route accounting and database lifetime, re-searches conflicts against current costs, preserves prepared Steiner order, applies the CUDA cost schedule, and measures real overlapping worker searches. See [worker scaling evidence](evidence/cpu-maze-rrr-worker-scaling.json). The historical BM64 qualification used MET2 to RDL at 512 by 512; the intended qualification window is MET2 to MET5, matching ECC ICsprout55 defaults and external GR50; the older MET1 and grid1024 interpretations in Phase C are corrected in its evidence file. Search remains full grid with an admissible A-star heuristic. Wire length in native statistics is the number of grid edges, not microns.

## Review verdict

The original draft was 7.5/10 for implementation readiness. The direction was right and the CUDA/CPU boundary was grounded in the source, but four decisions were still implicit: the immutable data passed to parallel searches, the multi-pin connection contract, the exact route-state checkpoint/rollback model, and the native/Python test boundary. This revision makes those decisions explicit. It is ready to drive an implementation split into route-state, maze search, RRR orchestration, and flow-integration changes; it is not an authorization to enable CPU RRR by default.

The acceptance bar is now separated into three independent claims:

1. **Correctness:** legal connected routes, exact usage accounting, deterministic candidates, rollback, and valid route/RC/STA export.
2. **Parallel execution:** candidate searches overlap, scale with worker count on a sufficiently large batch, and stay within a measured memory budget.
3. **QoR:** RRR improves or preserves the selected overflow/timing metrics under fixed provenance. A correctness pass or a faster route alone does not establish a QoR improvement.

## Long-term execution protocol

Freeze a baseline before Phase A and attach its identity to every phase record:

- DreamPlace commit and Xplace submodule commit;
- compiler, Python, Torch, build mode, and effective worker budget;
- input DEF, Verilog, LEF, Liberty, SDC hashes;
- routing layer range, grid dimensions, fanout/clock exclusions, cost policy, and RRR count;
- baseline route, route-pack, RC-tree, and readonly-STA artifact paths and hashes.

Each phase must produce a small machine-readable record containing the phase name, source commits, effective parameters, test commands, exit code, artifact paths, artifact hashes, and the exit-condition result. A failed experiment is recorded as failed or diagnostic; it does not advance the phase marker. Do not start the next phase until the current exit condition is recorded against the same baseline or the plan explicitly records a new baseline and why it changed.

Keep implementation changes separate from qualification changes: source/test commits may land in the feature branch, while BM64/worker-scaling/QoR reports belong in phase evidence files. This prevents a later experiment from silently changing the correctness oracle. The first implementation checkpoint is Phase A only; no positive CPU RRR or S50 default change is part of the initial delivery.

## Goal

Add a CPU implementation of the maze-routing and rip-up/reroute (RRR) phase used by the CUDA GPUGR backend. The first usable version should run the existing CPU pattern route for the initial solution, rip up nets whose current route crosses an overflow resource, and reroute those nets with a deterministic, batch-parallel CPU maze router for a bounded number of iterations.

The result should be callable from the existing GPUGR Python wrapper and usable by the standalone GR S50 sizing flow. The initial implementation must leave `auto`, `cpu_pr`, `cpu_pr_mt`, and the current S50 default behavior unchanged. Enabling CPU RRR is an explicit experiment setting.

## Current boundary

The CPU branch of `RouteForce::run_ggr()` calls `CPURouter::route()` or `routeParallel()` once and returns. The CUDA branch performs an initial pattern route, then selects overflow nets, removes their routes, and invokes maze routing in later iterations. CPU currently rejects every positive `rrr_iters` value.

The inspected CUDA revision is Xplace `dadc709cac779dd15f36f0f43c552f5c23880777`. Its rip-up and grid sweeps use parallel CUDA kernels, and routing proceeds through batches. However, `MazeRoute.cu::getResults()` sets `use_tf=false`; its active branch value-initializes `std::vector<cudaStream_t>` without creating streams. All net searches therefore launch on the default stream in order. Kernel-level grid parallelism is active; concurrent maze searches across nets are not established by this implementation. CPU inter-net parallelism is an explicit delivery requirement, independently of that CUDA limitation.

The existing CPU router already owns useful pieces that must remain the source of truth:

- `CpuRouteCandidate` stores raw route entries and wire/via resource deltas.
- `commitCandidate()` adds route usage and synchronizes `GrNet` and `rawRoutes`.
- `markOverflowNets()` computes wire/via overflow from the committed route map.
- `routePreparedNet()` performs cost-aware grid search for the prepared Steiner points.
- `GRDatabase` and `GrNet` provide the pin-access, route-layer, fixed-obstacle, and route-export contracts.

The missing pieces are route removal, a persistent route state across iterations, a true maze search for a selected net, deterministic RRR scheduling, and per-iteration reporting. Simply removing the current CPU guard would still execute only the one-pass pattern router.

The first implementation must not silently change the meaning of a route. A raw route is a sequence of positive wire spans and `-1` via markers in the existing encoded grid space. A successful candidate must connect every required pin-access group, preserve the existing routing-layer range, and pass the same route-pack/RC-tree connectivity checks used by `TimingRouteExport` and `GRParasiticsOp`.

## Design

### 1. Keep backend selection explicit

Add a distinct backend name such as `cpu_pr_maze` for the complete CPU PR+maze-RRR lane. Keep these semantics stable:

| Backend | Initial route | Later iterations | Current behavior |
| --- | --- | --- | --- |
| `cpu_pr` | serial pattern route | none | unchanged |
| `cpu_pr_mt` | deterministic parallel pattern route | none | unchanged |
| `cpu_pr_maze` | existing `cpu_pr_mt` pattern route | batch-parallel CPU maze RRR, with serial fallback | new, opt-in |
| `cuda` | CUDA pattern route | CUDA maze RRR | unchanged |
| `auto` | existing CUDA/CPU selection | no new implicit RRR | unchanged |

`cpu_pr_maze` with `rrr_iters=0` executes the existing `cpu_pr_mt` initial route and skips maze RRR. No separate maze-only mode is needed. Existing `cpu_pr` and `cpu_pr_mt` continue to reject positive RRR values rather than silently ignoring the setting. `num_threads` controls the worker pool in the new backend.

### 2. Extract shared CPU route state before adding the maze router

`CPURouter.cpp` is already a large high-touch module. Do not add the maze search and RRR state machine directly to it. Extract a small shared component, for example `CpuRouteState.{h,cpp}`, that owns:

- capacity, fixed usage, movable wire/via usage, wire distance, and cost maps;
- raw route storage and conversion to `GrNet` wire/via vectors;
- positive commit and exact remove operations;
- route-map/hash and usage-accounting helpers;
- overflow-resource marking and overflow-net collection.

`CPURouter` uses this state for the existing pattern path. `CpuMazeRouter` uses the same state for rip-up, search, commit, rollback, and reporting. A route removal operation must validate every segment before mutating usage, reject underflow, and leave the state unchanged on failure.

Use an explicit read-only search view rather than exposing mutable `CPURouter` members:

```text
CpuRouteState
  commit(net_id, candidate)
  remove(net_id) -> RouteSnapshot
  restore(net_id, RouteSnapshot)
  rebuild_costs(policy)
  overflow_report() -> OverflowReport
  checkpoint() -> RouteCheckpoint
  restore_checkpoint(RouteCheckpoint)

CpuMazeSearchView
  const capacity/fixed/usage/cost/via-cost/cost-sum maps
  routing-layer geometry and cost policy
  one net's prepared tree/pin-access data
```

`CpuMazeSearchView` is immutable for the lifetime of one candidate batch. The state owns all writes. This is the boundary that makes worker parallelism safe and prevents a search thread from accidentally changing usage while it is reading it.

The view should expose shared const storage for the capacity, fixed usage, committed usage, and cost maps; it must not copy full-chip maps per candidate. Each worker owns only its distance/predecessor frontier and candidate buffer.

`RouteCheckpoint` stores the route table needed to reconstruct a complete state, not a second copy of every derived map. Restoring a checkpoint rebuilds usage and costs from the route table and fixed-obstacle maps. Keep at most the current and best checkpoint live unless a measured memory budget permits more.

`OverflowReport` must carry both the selected net IDs and resource-level quantities: overflowing wire resource count, overflowing via resource count, summed positive wire excess, summed positive via excess, and the number of affected nets. The lexicographic best-state rule must name whether it uses excess first or resource count first; do not derive a floating “overflow amount” later from a boolean-only report.

The extracted state must preserve the current CPU route hashes before the maze path is added. Do this as a mechanical Phase A move: move accounting code first, keep the old `CPURouter` call order and tie-breaking, then compare serial and parallel pattern-route hashes. Only after that extraction is clean may `CpuMazeRouter` depend on the new state API.

### 3. Implement a deterministic CPU maze router

Add `CpuMazeRouter.{h,cpp}` under `thirdparty/xplace/cpp_to_py/gpugr/gr/`. The first version must support deterministic batch-parallel candidate generation. A serial execution mode remains necessary as a correctness oracle and fallback, but serial-only RRR is not the production target. It should:

1. Receive one selected `GrNet`, its prepared pin/tree points, and an immutable `CpuMazeSearchView`.
2. Build a multi-source search from the already connected portion of the net to the next unconnected tree branch. Reuse `PatternRoute::prepare()` only for the deterministic Steiner/tree-point order; use full grid expansion for each branch rather than restricting the result to the original pattern candidates.
3. Expand legal wire neighbours on the current routing layer and legal via transitions inside `routingLayerBegin..routingLayerEnd`.
4. Use the same integer cost model as the CUDA route for the first qualification: wire distance, via cost, fixed/movable usage, and overflow penalty. Expose the logistic slope, via multiplier, and short-violation discount as per-iteration inputs instead of duplicating constants.
5. Trace predecessors into the existing raw-route representation, coalesce adjacent wire cells, and record vias with the same encoding used by `TimingRouteExport.cpp`.
6. Check route-layer bounds, pin reachability, route-buffer limits, and all resource indices before returning a candidate.

The search must distinguish an unreachable net from a route-buffer allocation failure. An unreachable candidate is returned as a typed failure and leaves the shared state untouched; a malformed route or accounting violation is a hard error. The search must connect the prepared tree branch by branch, carry the already connected portion as the source set, and reject a candidate if any required branch remains disconnected. It must never infer connectivity by dropping zero-length spans or replacing a failed route with a previous route.

### 4. Implement the CPU RRR loop

Extend `RouteForce::run_ggr()` with a `cpu_pr_maze` branch:

1. Run the existing initial pattern route and commit all route usage.
2. Compute overflow resources and collect overflow net IDs, excluding `noroute`, clock, and high-fanout nets according to the existing `GRDatabase` policy.
3. Sort the selected nets deterministically using the CUDA-compatible primary order (descending bounding-box area, then stable net ID).
4. Remove all selected net routes from the shared usage state before calculating the next cost map.
5. Rebuild cost maps and reroute selected nets with the batch-parallel `CpuMazeRouter`.
6. Commit successful candidates; restore the old route for a failed candidate unless the old route was already invalid.
7. Record per-iteration route hash, wire/via usage hash, overflow-net count, overflow-wire/via count, total overflow amount, wirelength, via count, routed/unrouted counts, and elapsed time.
8. Retain the best complete route state according to lexicographic `(overflow amount, overflow resource count, wirelength, via count)`. On an exact metric tie, keep the earlier state (or use the lower route hash as a documented deterministic tie-break). Stop when there is no improvement, no overflow remains, or `rrr_iters` is exhausted.

Parallelism is required inside each RRR iteration. The RRR iterations themselves remain sequential because each iteration consumes the usage map produced by the previous iteration. Within an iteration:

1. Rip up the selected overflow nets and rebuild the shared cost map.
2. Use the existing batch scheduler as the starting point for a deterministic net order. Maze detours can leave the original net bounding box, so bounding-box separation alone does not establish resource independence.
3. Let the worker pool run independent maze searches with read-only cost/usage maps and per-worker predecessor/scratch storage.
4. Validate all candidate wire/via resources against one another and the current committed state.
5. Commit candidates in deterministic net order. If detours share wire/via resources or influence another candidate's cost through nearby via usage, rebuild the affected cost view and reroute conflicting candidates against the already committed state, committing each replacement before processing the next conflict. Record the fallback; repeating searches against the original stale cost map is insufficient.

The candidate search must never mutate shared usage. This gives parallel speedup without data races and preserves an auditable serial commit boundary. The serial mode should use the same candidate and commit functions, so it is an oracle for the parallel path rather than a second implementation.

Define an exact conflict footprint before implementation. At minimum it contains every wire resource and via resource in the candidate. Include the local via-neighbour resources read by the cost model when `twoCellsViaUsage()` or equivalent nonlocal terms are enabled. Two candidates with overlapping footprints cannot both be accepted from the same snapshot without a re-search. This rule is stricter than bounding-box disjointness and avoids silently accepting candidates evaluated against stale local costs.

Keep logical batch membership, net order, cost snapshots, tie-breaking, and commit policy independent of the worker count. A one-worker oracle searches the same batch against the same immutable snapshot as the parallel run; it must produce identical candidates and accepted route hashes. A different algorithm that refreshes costs after every net is a QoR comparator, not a byte-for-byte oracle.

Reuse `CpuRouteWorkerPool` and its worker-budget helper. Each worker owns its search frontier, distance/predecessor state, and candidate buffer. Account for shared snapshots, saved old routes, pending candidates, and per-worker scratch before selecting effective concurrency. A memory clamp limits concurrent searches without changing logical batch semantics; report requested/effective workers, clamp reason, and peak memory.

Mirror the CUDA RRR cost schedule where it is meaningful: iteration zero uses the initial pattern-route parameters; later iterations increase the logistic slope and adjust via/overflow penalties. The constants should live in one CPU/CUDA-shared policy helper or a documented parameter structure, not in two unrelated loops. Do not copy the CUDA implementation's stream/Taskflow assumptions into CPU code; CPU candidates use the immutable snapshot/worker-pool contract above.

### 5. Expose the flow without changing defaults

Add an explicit `cpu_pr_maze` backend to the Python backend normalization and native parameter validation. Add a standalone sizing parameter, `gr_sizing_rrr_iters`, defaulting to `0`; keep it separate from the existing `gpugr_area_adjust_rrr_iters`, post-legalization RRR, and final-evaluation RRR settings. `gr_sizing.py` should pass this value instead of hard-coding `rrr_iters=0`, but only accept a positive value with `gpugr_backend=cpu_pr_maze`.

The validation rule is explicit: when `gr_sizing_rrr_iters == 0`, preserve the current `auto -> cpu_pr_mt` GR sizing path; when it is positive, require `gpugr_backend == cpu_pr_maze`, `gpu == 0`, the ECC native backend, and `num_threads > 0`. A positive value must never silently downgrade to `cpu_pr_mt` or `cpu_pr`.

The exact integration points are:

- `thirdparty/xplace/cpp_to_py/gpugr/db/GRSetting.{h,cpp}`: add `CPU_PR_MAZE` parsing and reporting.
- `thirdparty/xplace/cpp_to_py/gpugr/gr/RouteForce.{h,cpp}`: select the new CPU orchestration branch; preserve the existing CPU and CUDA branches.
- `thirdparty/xplace/cpp_to_py/gpugr/gr/CPURouter.{h,cpp}`: route-state extraction and compatibility adapter; do not duplicate the maze algorithm here.
- `thirdparty/xplace/cpp_to_py/gpugr/gr/CpuRouteState.{h,cpp}` and `CpuMazeRouter.{h,cpp}`: new state and search ownership. The existing GGR CMake glob picks up these `.cpp` files; verify this in the generated build rather than adding a second build path.
- `thirdparty/xplace/cpp_to_py/gpugr/PyBindCppMain.cpp`: expose the extended stats only through the existing `run_stats()` contract.
- `dreamplace/ops/gpugr/xplace_backend.py`: backend normalization, positive-RRR validation, and metadata.
- `dreamplace/flows/gr_sizing_config.py`, `dreamplace/flows/gr_sizing.py`, `dreamplace/params.json`, and `dreamplace/Params.py`: parameter declaration, validation, and S50 handoff.
- `tests/ops/gpugr_backend_select_unittest.py`, `tests/flows/test_gr_sizing_reporting.py`, and the native GPUGR fixture tests: Python contract coverage.

Keep the current S50 profile at `iteration=50`, `rrr_iters=0`, and the current CPU backend until the new lane has separate QoR evidence. Do not change `gpugr_final_eval_rrr_iters`, placement inflation settings, or the `auto` resolution as part of this work.

The native metadata and Python result JSON must include:

- requested and resolved backend;
- requested RRR iterations and completed iterations;
- per-iteration overflow and route hashes;
- whether the best state was restored;
- CPU maze route failures and their reason categories.
- requested/effective workers, parallel batch count, conflict fallback count, peak memory, and time spent in search, cost rebuild, validation/commit, and rollback.

## Implementation phases

### Phase A — route-state extraction and accounting

- Extract shared route state without changing the `cpu_pr` or `cpu_pr_mt` route hashes.
- Add exact remove/restore operations and overflow-net collection.
- Add C++ unit tests for add/remove symmetry, no negative usage, raw-route round trips, and failed-removal atomicity.
- Run the existing CPU pattern-routing unit suite and the serial/parallel CPU dispatch qualification.
- Confirm the generated CPU and CUDA build manifests contain the new sources in the intended `ggr` target; no separate library or duplicate symbol path is allowed.

Exit condition: CPU PR and CPU PR MT outputs and hashes remain unchanged for the existing toy fixture, and route accounting passes randomized add/remove tests.

### Phase B — standalone CPU maze search

- Implement `CpuMazeRouter` with a small synthetic grid fixture covering wires, vias, obstacles, route-layer limits, and disconnected pins.
- Add per-worker search scratch and a batch-candidate API; compare the parallel candidate set with the serial oracle before committing either result.
- Validate predecessor tracing, route coalescing, and `GrNet` export.
- Add deterministic repeated-run checks and route legality checks.
- Add multi-pin cases with same-cell pin access, different legal layer ranges, a required via transition, and an intentionally disconnected pin group. Validate the raw-route representation with route-pack and RC-tree preparation, not only with an in-memory endpoint check.

Exit condition: every successful route connects all required pin groups and contains no out-of-range segment. For the same logical batches and cost snapshots, effective worker counts 1, 2, 4, and 8, where the memory budget permits, produce identical candidates and raw-route/usage hashes. A concurrency fixture proves that multiple candidate searches actually overlap in execution; a requested worker count that is clamped must be reported as clamped rather than counted as a scaling point.

### Phase C — one CPU RRR iteration

- Add the `cpu_pr_maze` backend and initial PR plus one overflow-only maze reroute.
- Exercise the same one-iteration fixture with one worker and multiple workers; require equivalent legality, overflow, and route accounting, with deterministic commit order.
- Add a fixture where the initial pattern route creates a known overflow and the maze detour removes it.
- Validate rollback when a detour is impossible and validate that non-overflow nets are not ripped up.
- Validate that candidates generated from one immutable snapshot are either all accepted in a conflict-free batch or selectively re-searched after the first deterministic conflict; no candidate is committed from a stale snapshot.

Exit condition: the fixture's overflow decreases or remains unchanged without resource-accounting errors; no improvement never makes the accepted result worse than the initial complete route.

### Phase D — bounded RRR and Python integration

- Add `rrr_iters` iteration control, early stopping, best-state restoration, native metadata, and `gr_sizing_rrr_iters`.
- Run CPU-only native build and the full focused GPUGR test set.
- Run the standalone S50 sizing flow with `RRR=0`, `RRR=1`, and `RRR=3` on BM64.
- Run the new backend with `gr_sizing_rrr_iters=0` and prove its initial route hash matches `cpu_pr_mt`; only then run positive RRR values.

Exit condition: `RRR=0` is identical to the current BM64 S50 baseline; positive RRR runs finish with complete artifacts and per-iteration evidence.

### Phase E — QoR qualification

- Compare CPU RRR against the current CPU `rrr=0` S50 and the qualified CUDA RRR route on BM64.
- Report routing overflow, wirelength, via count, route time, total S50 time, internal GR timing, and external GR+STA timing separately.
- Repeat on one case with a large known internal/external gap before considering a default change.
- For fixed DEF/netlist/SDC, grid, routing layers, RRR count, cost schedule, build, and initial route, measure workers 1/2/4/8. Record at least three timed runs after warm-up, report median maze-search time and total route time separately, and publish speedup `T1/Tp`, effective workers, peak memory, and fallback fraction. Exclude timing values from determinism hashes. Use a workload with enough rerouted nets to occupy the requested workers.

Exit condition: a decision record states whether CPU RRR is useful as an optional backend, whether it is fast enough for S50, and whether its route/timing provenance is suitable for production. Parallel qualification requires measured overlapping searches and repeatable maze-search acceleration at multiple workers; merely allocating several threads or passing a toy test does not complete the performance requirement. If total route time does not improve, profile the sequential cost rebuild/commit/fallback portion and retain that limitation explicitly.

## Tests and acceptance criteria

The implementation is complete only when all of the following hold:

- CPU-only CMake build succeeds with no CUDA symbols or runtime dependency.
- Existing `cpu_pr` and `cpu_pr_mt` tests pass unchanged; positive RRR remains rejected for those backends.
- New `cpu_pr_maze` tests cover initial route, rip-up, maze detour, via transition, overflow selection, rollback, disconnected pins, and route-layer bounds.
- Repeated runs and effective workers 1/2/4/8, where available, produce identical accepted raw-route, usage, and non-runtime QoR hashes under fixed logical batch semantics.
- The new backend demonstrates actual inter-net concurrency and includes measured worker scaling, memory usage, and conflict-fallback evidence.
- Timing-route export, eligible-pin coverage, native RC-tree preparation, and read-only STA all succeed before claiming the route is usable by S50.
- Every committed route has valid wire/via indices, non-negative usage, and matching `GrNet`/raw-route representations.
- RRR=0 preserves the current S50 BM64 metrics and decompressed DEF/Verilog outputs byte-for-byte.
- RRR=1 and RRR=3 produce per-iteration logs and complete route/timing artifacts; no improvement is accepted as a QoR gain.
- External GR+STA results are run only after internal route and RC artifacts pass the route identity and accounting checks.
- No placement default, `auto` backend, CUDA path, or ECC flow behavior changes in the implementation PR.

The following are explicit non-goals for this implementation: changing CUDA maze routing or enabling its disabled Taskflow path; parallel commit into shared usage; changing placement inflation or timing propagation; making `cpu_pr_maze` the `auto` backend; changing the existing S50 default; or claiming CPU RRR equivalence to OpenROAD routing.

## Risks and decisions to resolve during implementation

- A CPU maze search may improve overflow but produce longer routes or different pin-access topology. The lexicographic best-state policy and external STA check must prevent promoting a congestion-only regression.
- The CUDA implementation uses coarse-grid maze routing and a GPU route buffer. CPU route encoding and buffer growth must be validated independently; CUDA route hashes are not an expected byte-for-byte oracle.
- Multi-pin net connection order affects determinism and QoR. Use the existing prepared tree order initially and record it in route metadata.
- A serial CPU maze router may be too slow for the 42-case regression. Treat serial mode as the correctness oracle, but qualify the worker-pool candidate path in the same implementation. Profile search, batch scheduling, conflict fallback, rip-up, cost rebuild, and route export separately.
- CPU RRR should remain opt-in until BM64 and at least one congestion-heavy case show a repeatable improvement under the same GR and STA provenance.

## Historical qualification decision (2026-10-09, MET2-RDL; superseded)

[Historical Phase D](evidence/cpu-maze-rrr-phase-d.json) records MET2-RDL route compatibility and complete S50/terminal artifacts. [Phase E](evidence/cpu-maze-rrr-phase-e.json) records independent OpenROAD GR50 timing, matched original-DEF CUDA references, repeatable worker scaling, and limits. That MET2-RDL run accepted CPU RRR as an explicit standalone experiment without default promotion; its qualification decision is superseded by the matched-window evidence below. The regular ECC diff_sizing builder remains unchanged and still selects cpu_pr_mt. Experiments initialize the native ECC database and run the standalone flow_kind=sizing profile with an explicit cpu_pr_maze backend.

The 128-grid worker fixture shows 6.321 times maze-search speedup at eight workers and only 1.032 times total-route speedup: 99.872 percent of selected candidates need serial current-state re-search. On the matched S50 outputs, external GR50 TNS magnitude improves by 16.25 percent for BM64 RRR=1, 18.49 percent for BM64 RRR=3, and 68.50 percent for y_huff RRR=1. BM64 stage wall intervals are about 128, 1219, and 1871 seconds for RRR=0/1/3. These timings use file timestamps from initial config to final Verilog and exclude imports/cleanup. This runtime rejects default promotion despite the two-case external timing improvements.

Artifacts are archived at `/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_cpu_maze_rrr_20261009/`. Native statistics count wire grid edges; the fixed-input reference record separately measures physical wire length from exported graph coordinates. Peak process RSS is measured in the worker fixture; the native scratch estimate is not a process memory ceiling. This implementation changes neither CUDA maze routing nor placement/timing-propagation logic.

## Routing-window correction (2026-10-09)

The standalone runners explicitly selected RDL although ECC ICsprout55 defaults and the external OpenROAD evaluator use MET2-MET5. Corrected runs use MET2-MET5 at the same 512 grid, eight workers, padding zero and S50 controls. Historical results remain in their original archive and must not be relabeled or used as matched-window qualification. Corrected artifacts: `/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_cpu_maze_rrr_m2_m5_20261009`. No routing algorithm, generic layer defaults or ECC profile changes are needed.

[Matched-window qualification](evidence/cpu-maze-rrr-m2-m5.json) passes actual route-pack layer/coverage, RC-tree, read-only STA, full S50 compatibility and independent external GR50 gates. RRR=0 remains byte-identical to cpu_pr_mt. The 128-grid worker fixture measures 6.436 times search acceleration at eight workers but only 1.039 times total-route acceleration; 99.880 percent of candidates require serial current-state re-search. The S50 timing table is in the archive README. Assess WNS, TNS and electrical violations separately and retain the experimental/default-RRR-zero decision.
