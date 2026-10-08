# ECC extension seam qualification

Date: 2026-10-01. Local branch: `codex/merge-main-routability-timing`.
Implementation baseline: `0004b329`. No remote publication.

This is a behavior-preserving extraction of ECC timing, real-size sizing/VT,
segment/joint, projection artifacts and handoff adapters. The official
optimizer, GP loop, stopping policy, legalization and final HPWL/RSMT stay in
`NonLinearPlace.py`. Its complete 5,140-line `__call__` AST is identical to
the baseline. The main module decreased from 14,587 to 11,815 lines; the
extracted owners range from 142 to 607 lines. These are ownership changes,
not an equivalent reduction in total repository code.

Local implementation commits:

| Seam | Commit | Owner |
| --- | --- | --- |
| A | e9205c18 | real-size transition and profile |
| B | 22d80e51 | timing policy, Pin2Pin, gradient projection and topology |
| C | 876ba4cf | segment/joint and terminal transactions |
| D | 4f80eae9 | timing reports, projection context/artifacts and snapshot refresh |
| E | b2664cdb | explicit continuation/restart and handoff transaction adapters |

The final qualification changes repair existing manual fixtures and source
inspection paths, and remove two imports made unused by extraction. There is
no algorithm, configuration, capability or native implementation change.

## Executed checks

Evidence root:
`/nfs/share/home/zhaoxueyan/dataset_cx55_ecc_workspace/ecc_nonlinearplace_split_20261001.QWF1fx`.
The parent plan and per-seam evidence are in
`docs/plans/ecc-dreamplace-nonlinearplace-ecc-extension-split.md` and its
`ecc-dreamplace-nonlinearplace-evidence/` sibling directory in the ECC parent.

Tests use installed candidate Python code from outside the source tree, with
Python 3.11, Torch 2.11.0+cpu and the existing verified native build. Prefix v28
was used for the real design run; v29 differs only by unused csv/resource
import removal. Every native extension hash matches between the two prefixes.
Source-first and install-first checks cover the engine and 15 flow modules.

| Check | Result | Artifact |
| --- | --- | --- |
| Final flow/transition/timing/mutation/plain-placement/buffering target set | 227 unaffected tests passed, 5 fixture/path failures; all five corrected and passed separately; 232 distinct tests now pass | final-target-tests-v2.log, final-fixture-retest.log |
| Final v29 import/fixture boundary check | 5 passed; overlaps the preceding set | final-v29-check.log |
| CUDA plain-placement branch | 1 explicit skip; CPU geometry-only and macro placement passed | final-target-tests-v2.log |
| Seam E handoff/restart | 101 passed, 13 subtests; separate set | seam-e-tests.log |
| Pre-E handoff replay | 35 existing tests per side, 61 records exactly equal | parent seam-e-handoff-parity.json |
| Existing native ECC timing export, buffer commit/rollback and geometry fixtures | 3 passed | native-tests.log, native/*/manifest.json |
| BM64 normal ECC CLI, same-process joint/refresh/continuation | success, observer passed | joint-ecc.log, ecc_61_3/ |
| External DEF/Verilog/native audit | 15,700 native masters, 57,601 connected pins, 56,573 Verilog signal pins; no mismatch | ac4_audit/ecc_61_3.json |
| Before/after BM64 comparison | selected semantic contracts exactly equal, no numeric tolerance | bm64-parity.json |
| Source/install identity, unchanged official loop, native hashes | passed | source-provenance.json, install-provenance.json |
| Changed-owner F401, py_compile and diff check | passed | parent final-qualification.md |

The three manual fixture failures also reproduce on pre-E v27: disabled
L-shape fields and `macro_only=False` were missing. Two old source inspections
assumed repository-root cwd; their paths now derive from the test file.
Assertions and product fallback were not weakened. A first test command named
a nonexistent generation test file and ran no tests; that attempt is retained
and not counted as a pass.

## Real BM64 behavior

The unchanged archived config uses the same post-placement
`BM64_pins_tdp_fillers_noise0025.def`, seed 3000, float32, CPU, two threads,
density 0.8, padding 0, clock 1.538 ns, MET1 placement RC, up/VT budgets 1%,
BUFX4H7L, initial z=0.1 and threshold=0.5.

The first window performs 61 steps and commits 1,203 sizing/VT actions plus
6 buffers at iteration 60. Fresh PyDB/placer continuation performs 3 more
steps and commits 21 sizing/VT actions. The continuation has finite nonzero
placement, sizing and buffer gradients; no optimizer actions are injected.
The entire run stays in one worker process. Continuation topology generation
is 2 and the final native refresh generation is 3.

Compared with the pre-extraction merge qualification, both action digests,
all sizing action records, gradient metrics, final native master map and
complete PyDB snapshot match exactly. Refresh counts, capabilities, generation,
placement-RC status, objective, overflow and timing diagnostics also match.
Only process identities, output paths and wall-clock fields are outside this
semantic comparison. Native, DEF and Verilog agree on all expected changed
masters and signal connectivity. All six new buffers use BUFX4H7L.

## Capability and limits

Timing export, sizing/VT native writeback, buffer commit, DEF/Verilog output,
full PyDB replacement and subsequent optimizer continuation remain available.
`supports_committed_refresh=false`,
`sta_state_status=native_rebuilt_without_fresh_rc`, and
`atomic_across_sizing_and_buffer=false` are preserved.
Native STA without SPEF remains diagnostic; DreamPlace recomputes placement
RC from LEF/FLUTE after mutation.

This run disables legalization. The existing 4 out-of-core instances and
18,227 overlap pairs remain; all six buffers overlap existing placement.
The artifacts establish refactor parity and mutation/continuation behavior,
not signoff QoR or a legal placement result. CUDA, OpenROAD QoR parity and
all-case regression are outside this plan. Prior independent routability
qualification remains applicable because its source/native boundaries and
the complete placement-loop AST were not changed.
