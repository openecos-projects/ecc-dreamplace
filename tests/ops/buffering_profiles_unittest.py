from dreamplace.flows.buffering_profiles import (
    BUFFERING_DEFAULTS,
    CANDIDATE_BUFFERING_DEFAULTS,
    CANDIDATE_PHYSICAL_ECO_DEFAULTS,
    PHYSICAL_ECO_DEFAULTS,
    SEGMENT_BUFFERING_DEFAULTS,
    SEGMENT_PHYSICAL_ECO_DEFAULTS,
    buffering_default_dict,
    physical_eco_default_dict,
)


def test_buffering_profile_contains_canonical_relaxed_defaults():
    defaults = buffering_default_dict()

    # The canonical flow discriminator replaced the legacy dry-run flag.
    assert defaults["buffering_mode"] == "segment"
    assert defaults["buffering_continuous_relaxed_optimization"] == 1
    assert defaults["buffering_segment_count_tns_gradient"] == 1
    assert defaults["buffering_candidate_policy"] == "segment_only"
    assert defaults["buffering_max_repeaters_per_segment"] == 3
    assert defaults["buffering_include_tree_node_candidates"] == 0
    assert (
        defaults["buffering_relaxed_timing_integration_mode"]
        == "dynamic_net_provider"
    )


def test_physical_eco_profile_extends_buffering_with_current_defaults():
    defaults = physical_eco_default_dict()

    for name, value in BUFFERING_DEFAULTS:
        assert defaults[name] == value
    for name, value in SEGMENT_BUFFERING_DEFAULTS:
        assert defaults[name] == value
    for name, value in PHYSICAL_ECO_DEFAULTS:
        assert defaults[name] == value
    for name, value in SEGMENT_PHYSICAL_ECO_DEFAULTS:
        assert defaults[name] == value
    # The legacy real-commit / legal-state-loop defaults were removed together
    # with the legacy buffering runners, so the physical-eco profile currently
    # adds no dedicated keys beyond the buffering profile.
    assert defaults == buffering_default_dict()


def test_candidate_profile_overrides_segment_count_flags():
    candidate_defaults = dict(CANDIDATE_BUFFERING_DEFAULTS)

    assert candidate_defaults["buffering_segment_count_tns_gradient"] == 0
    # The former segment-count projection-commit override was removed with the
    # legacy runners; the candidate physical-eco profile now carries no keys.
    assert dict(CANDIDATE_PHYSICAL_ECO_DEFAULTS) == {}
