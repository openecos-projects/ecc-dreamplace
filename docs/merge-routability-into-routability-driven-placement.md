# Merge Plan: Routability into `routability-driven-placement`

## Objective

Merge the routability work from `experiment/ggr-lshape-al` with the current
`main` architecture, resolve the interface changes deliberately, and publish
the resulting history on the `routability-driven-placement` branch of the
`ecc-dreamplace` repository.

This plan is implementation-only until its verification gates pass. It does
not authorize a push by itself; remote publication is a separate final action.

## Baseline (captured 2026-08-12)

- Repository: `git@gitee.com:ieda-ipd/ecc-dreamplace.git`
- Source routability branch: `experiment/ggr-lshape-al`
- Source SHA: `05d92c93f707223558b7cd4656258f106916942e`
- Architecture baseline: `main`
- Main SHA: `390a18c3af9adbc9aea3fda88b9d3636af2e3d5d`
- Existing local worktree: clean, checked out on `experiment/ggr-lshape-al`
- Final branch: `routability-driven-placement`
- Final branch policy: create/update from the completed merge result; do not
  overwrite an existing remote branch without an explicit comparison and
  approval.

## Merge strategy

Create a real Git merge commit with `main` as the first parent and
`experiment/ggr-lshape-al` as the second parent. Resolve that merge using the
`main` dependency/build-system architecture, then apply compatibility cleanup
and focused fixes in follow-up commits. This preserves branch ancestry while
making the behavioral port reviewable. Do not resolve the merge by choosing
one side for every conflict: the two branches changed the same placement files
and the same Steiner interfaces for different reasons.

The required local sequence is:

```bash
git switch experiment/ggr-lshape-al
git merge --no-ff --no-edit main
# resolve and stage conflicts, then commit the merge
# follow with separate cleanup, test, and verification commits
```

The final branch is created from the verified tip of this sequence, not from
either source branch directly.

The desired result must retain:

- L-shape routability operators, GGR L-shape topology, capacity-AL support,
  enhanced inflation, gpugr integration, and their parameter schema;
- the `main` vendored Limbo layout and current Flute LUT loading contract;
- the `main` ECC/iEDA database path and current `with_sta` timing path;
- the `main` removal of unsupported OpenTimer integration.

## Work packages

### 1. Freeze and stage the merge

1. Fetch `main` and `experiment/ggr-lshape-al` and verify the baseline SHAs.
2. Create the local safety branch
   `codex/pre-routability-main-merge-20260812` before changing the worktree.
3. Merge or rebase only in the child `ecc-dreamplace` repository. Keep the
   parent `ecc` gitlink untouched until the child result is complete.
4. Record the merge base and all conflict paths in the work log/commit
   message.

Gate: the child repository starts clean, both source refs are available, and
`codex/pre-routability-main-merge-20260812` resolves to
`05d92c93f707223558b7cd4656258f106916942e`.

### 2. Remove obsolete NCTUgr/place_io coupling

The old code is not the primary routability implementation, but it is a hard
initialization dependency today:

- `PlaceObj` imports `nctugr_binary` and constructs it whenever
  `routability_opt_flag` is enabled;
- `nctugr_binary` imports `place_io` and expects the external
  `thirdparty/NCTUgr.ICCAD2012/NCTUgr` binary;
- the old EGR wrapper has an unused `place_io` import;
- the old `Timer.py`/`ops/timing` path is unsupported by `main`.

Change the initialization path so that routability setup constructs only the
active ECC/iEDA EGR, gpugr, pin-utilization, and L-shape operators. Remove the
NCTUgr operator construction and its obsolete wrapper/imports.

For this merge, preserve the existing configuration key
`adjust_nctugr_area_flag` for backward compatibility, but make its current
meaning explicit: when enabled it selects the ECC/iEDA EGR route map for area
adjustment; it does not invoke the deleted NCTUgr binary. Add one migration
comment or log message documenting this compatibility mapping. Renaming the
key to an EGR-specific name is out of scope for this merge and must be a
separate follow-up change with configuration migration coverage.

Remove old OpenTimer-only construction and imports in the same compatibility
pass, while retaining `timing_propagation`, `rc_timing`, and ECC/iEDA STA
support used by the current flow.

Gate: importing the placement modules with the old directories absent does
not fail; `routability_opt_flag=1` builds the intended route operators without
accessing NCTUgr or `PlaceIOFunction`; `timing_opt_flag=1` follows `main`'s
explicit unsupported-mode behavior.

### 3. Reconcile the placement Python modules

Resolve `NonLinearPlace.py`, `PlaceObj.py`, `Placer.py`, `macroPlaceDB.py`,
`irt_egr.py`, and `params.json` by behavior, not by file-side selection.

- Start from the `main` signatures and lifecycle boundaries.
- Reapply the routability helpers and outer-loop state transitions from the
  experiment branch.
- Keep the ECC/iEDA `MacroPlaceDB` data manager contract.
- Preserve the L-shape telemetry, adaptive weighting, topology-cache, gpugr,
  and enhanced-inflation invariants.
- Remove dead `if False` orientation code that references old `place_io_cpp`.
- Keep parameter defaults stable unless a conflict proves that a default must
  change; add/retain tests for any changed default contract.

Gate: no conflict markers, no imports of deleted modules, and the public
constructor/call signatures match the current `main` caller chain.

### 4. Reconcile Steiner and EGR interfaces

Merge the C++/Python Steiner topology changes with these rules:

- use the `main` vendored Flute LUT location and argument contract;
- retain the experiment branch's deterministic/L-shape direction behavior;
- keep Python and C++ binding argument order/defaults identical;
- update all call sites and tests together;
- retain route-guide and topology-pack behavior only where it is backed by a
  current operator contract.

Gate: C++ configuration reaches the Steiner target, Python import reaches the
  generated extension, and the focused Steiner/GGR topology tests pass.

### 5. Build and focused verification

Run checks in increasing cost:

1. `git diff --check` and a repository-wide conflict-marker scan.
2. Python bytecode compilation for the touched Python modules.
3. CMake configure with the repository's supported environment.
4. Build the touched C++/CUDA extensions, especially Steiner topology and
   routability operators.
5. Run focused unit tests for enhanced bins/inflation, L-shape capacity,
   topology packing, EGR padding, and parameter defaults.
6. Run the repository's smallest ECC placement fixture, `test/simple.json`,
   as a non-routability smoke test and record its `flow.json`/placement output.
7. Run the same fixture through the routability path with an ephemeral JSON
   override setting `routability_opt_flag=1`,
   `l_shape_routability_flag=1`, `adjust_nctugr_area_flag=1`, and
   `adjust_gpugr_area_flag=0`. The run must construct EGR/L-shape operators
   without looking for `thirdparty/NCTUgr.ICCAD2012/NCTUgr`; record the exit
   status, route-map log, and generated placement/output artifact.
8. Run a negative compatibility check with `timing_opt_flag=1`; it must fail
   with `main`'s explicit unsupported-OpenTimer error rather than an import or
   missing-file error.

Gate: all required checks pass, or each failure is recorded with its exact
command, environment, and whether it is a pre-existing/toolchain limitation.
The smoke-test record must include the exact JSON path/override, command,
exit status, key log lines, and output artifact paths.

### 6. Commit and publish the child branch

1. Commit the real merge as a dedicated merge commit.
2. Commit obsolete-dependency cleanup separately.
3. Commit interface/test fixes separately, with verification evidence.
4. Create or fast-forward the local `routability-driven-placement` branch to
   the verified tip of those commits.
3. Compare the local branch with `origin/routability-driven-placement` using
   `git ls-remote` and a non-destructive log/diff review.
4. Push only after explicit publication approval, then verify the remote SHA.

Gate: the remote branch, if published, resolves to the verified local commit;
the final report includes the child commit SHA and the exact remote ref.

### 7. Update the parent ECC gitlink separately

After the child branch is complete and published (if publication is approved),
check out the desired child commit in the parent ECC worktree, stage only the
`chipcompiler/thirdparty/ecc-dreamplace` gitlink, and commit it separately.
Do not mix unrelated parent changes or `ecc-tools` changes into that commit.

Gate: parent `git diff --submodule=short` shows only the intended child gitlink
change, and parent/child status is reported separately.

## Abort and recovery rules

- Never use `git reset --hard`, force checkout, or force push as a conflict
  resolution shortcut.
- If a merge conflict becomes ambiguous, preserve the conflict state or abort
  the merge and return to the safety ref.
- Keep any moved submodule/vendor backup until the merged tree is built and a
  minimal routability smoke flow succeeds.
- Do not delete backup directories or unrelated worktree changes as part of
  this plan.

## Definition of done

The work is complete only when:

- `routability-driven-placement` contains the verified merge result;
- the merged tree builds against the `main` dependency architecture;
- L-shape routability behavior and focused tests pass;
- old NCTUgr/place_io/OpenTimer paths are either removed or explicitly isolated
  behind a tested, intentional compatibility boundary;
- the child remote state is verified, and the parent ECC gitlink is updated
  separately if requested.
