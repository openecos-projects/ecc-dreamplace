from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BackendCaps:
    backend: str
    has_sta: bool
    supports_sta_pydb: bool
    has_libcell_timing: bool
    has_diff_sizing_metadata: bool
    has_buffer_optimizer_metadata: bool
    has_diff_optimizer_metadata: bool
    supports_apply_placement: bool
    supports_apply_sizing: bool
    supports_buffer_commit: bool
    supports_committed_refresh: bool
    supports_tcl_save: bool
    coordinate_source: str
    sta_reference_role: str
    parasitics_initialization: str
    sta_state_status: str

    @classmethod
    def ieda(cls):
        return cls(
            backend="ieda",
            has_sta=True,
            supports_sta_pydb=True,
            has_libcell_timing=True,
            has_diff_sizing_metadata=True,
            has_buffer_optimizer_metadata=False,
            has_diff_optimizer_metadata=False,
            supports_apply_placement=True,
            supports_apply_sizing=True,
            supports_buffer_commit=False,
            supports_committed_refresh=False,
            supports_tcl_save=True,
            coordinate_source="ieda_dm",
            sta_reference_role="diagnostic_exporter",
            parasitics_initialization="unverified",
            sta_state_status="diagnostic_unverified",
        )

    @classmethod
    def openroad(cls):
        return cls(
            backend="openroad",
            has_sta=True,
            supports_sta_pydb=True,
            has_libcell_timing=True,
            has_diff_sizing_metadata=True,
            has_buffer_optimizer_metadata=True,
            has_diff_optimizer_metadata=True,
            supports_apply_placement=True,
            supports_apply_sizing=True,
            supports_buffer_commit=True,
            supports_committed_refresh=True,
            supports_tcl_save=False,
            coordinate_source="openroad_db",
            sta_reference_role="golden_opensta",
            parasitics_initialization="placement",
            sta_state_status="validated_golden",
        )

    @classmethod
    def ecc(cls, *, timing=False, parasitics_initialization=None):
        """Capability contract for the native ecc-tools Python module."""
        if timing:
            return cls(
                backend="ecc",
                has_sta=True,
                supports_sta_pydb=True,
                has_libcell_timing=True,
                has_diff_sizing_metadata=True,
                has_buffer_optimizer_metadata=True,
                has_diff_optimizer_metadata=True,
                supports_apply_placement=True,
                # These are promoted on the live backend instance only after
                # the corresponding native round-trip succeeds.
                supports_apply_sizing=False,
                supports_buffer_commit=False,
                supports_committed_refresh=False,
                supports_tcl_save=True,
                coordinate_source="ecc_idb",
                sta_reference_role="native_current_main",
                parasitics_initialization=(
                    parasitics_initialization or "configured_spef_or_none"
                ),
                sta_state_status="validated_native_export",
            )
        return cls(
            backend="ecc",
            has_sta=False,
            supports_sta_pydb=False,
            has_libcell_timing=False,
            has_diff_sizing_metadata=False,
            has_buffer_optimizer_metadata=False,
            has_diff_optimizer_metadata=False,
            supports_apply_placement=True,
            supports_apply_sizing=False,
            supports_buffer_commit=False,
            supports_committed_refresh=False,
            supports_tcl_save=True,
            coordinate_source="ecc_idb",
            sta_reference_role="none",
            parasitics_initialization="none",
            sta_state_status="unavailable",
        )

    def to_dict(self):
        return asdict(self)


def ieda_backend_caps():
    return BackendCaps.ieda()


def openroad_backend_caps():
    return BackendCaps.openroad()


def ecc_backend_caps(*, timing=False, parasitics_initialization=None):
    return BackendCaps.ecc(
        timing=timing,
        parasitics_initialization=parasitics_initialization,
    )
