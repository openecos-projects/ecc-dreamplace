class DynamicNetProvider:
    """Protocol boundary for TimingPropagation-integrated net arc providers."""

    def virtual_buffer_drv_tensors(self):
        """Additional virtual output-pin violations in ns/pF, or unsupported."""
        return None

    def has_dynamic_nets(self, net_ids, *, level_id=None, level_view_epoch=None):
        raise NotImplementedError

    def propagate_net_aat_level(
        self,
        *,
        net_ids,
        pin_rAAT,
        pin_fAAT,
        pin_rtran,
        pin_ftran,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
        base_pin_net_impulse_rise,
        base_pin_net_impulse_fall,
        static_calculate_net_aat_level,
        level_id=None,
        level_view_epoch=None,
    ):
        raise NotImplementedError

    def apply_dynamic_net_cap_overlay(
        self,
        *,
        pin_net_cap_rise,
        pin_net_cap_fall,
    ):
        return pin_net_cap_rise, pin_net_cap_fall

    def begin_critical_path_net_delay_snapshot(
        self,
        *,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
    ):
        return None

    def consume_critical_path_net_delays(
        self,
        *,
        base_pin_net_delay_rise,
        base_pin_net_delay_fall,
    ):
        return base_pin_net_delay_rise, base_pin_net_delay_fall
