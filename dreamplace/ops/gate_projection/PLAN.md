# Gate Projection Plan

This directory-level plan tracks only the projection operator work. It is
separate from the repository-wide RLCR plan.

## Objective

Build a reusable projection contract that can support multiple discrete gate
sizing strategies without rewriting the legality and data-wrangling path.

## Current Step

The current implementation lands the contract layer:

- candidate enumeration
- pluggable scoring
- pluggable resolving
- pluggable validation
- structured projection result
- standalone projection artifact writer
- post-optimization integration in `NonLinearPlace`

## Next Steps

1. Add a timing-aware or sensitivity-guided scorer so later experiments can
   reuse this operator without changing the legality layer.
2. Add richer projection context producers so timing, area, density, or
   sensitivity terms can be attached without wiring them through the op API.
3. Add a resolver that supports top-k repair or beam search if stability-aware
   tie-breaking is still too weak.
4. Expand projection validation reports beyond legality into timing-quality
   conclusions after legalization.

## Integration Notes

- Keep the operator outside `PlaceObj`'s gradient path.
- Treat `projected_libcell_offset` as the discrete output, not as the
  continuous optimization variable.
- Keep score terms decomposed so later experiments can inspect why one library
  cell won over another.
- Keep projection artifacts stable across scorer swaps so downstream analysis
  scripts do not need to know which sizing strategy produced the result.

## Acceptance For This Directory

The contract layer is considered in place when:

- a caller can enumerate candidates from `inst_main_id`
- a caller can replace the scorer without touching the provider
- a caller can replace the resolver without touching the provider or scorer
- the result contains both the chosen discrete mapping and the score breakdown
- a post-optimization path can emit projection artifacts under `result_dir`
- unit tests cover default behavior, custom scorer/resolver behavior, and
  artifact emission
