import logging


def _to_float(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric_value(metric, name):
    if metric is None or not hasattr(metric, name):
        return None
    return _to_float(getattr(metric, name))


class OpenRoadHandoffController:
    """
    Decide whether the placement loop should hand off to the persistent
    OpenROAD bridge. This object owns trigger policy only; it does not own
    OpenROAD mutation or post-mutation synchronization.
    """

    def __init__(self, config, runtime_state=None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.trigger = self.config.get("trigger", {}) or {}
        state = runtime_state if isinstance(runtime_state, dict) else {}
        self._triggered_keys = state.setdefault("triggered_keys", set())

    @classmethod
    def from_params(cls, params):
        config = getattr(params, "openroad_handoff", None)
        if config is None:
            config = {
                "enabled": False,
                "trigger": {
                    "mode": "disabled",
                    "interval": None,
                    "phases": [],
                    "metrics": [],
                },
                "buffer_insertion": {
                    "enabled": False,
                    "strategy": "repair_design",
                    "options": {},
                },
            }
        runtime_state = getattr(params, "_openroad_handoff_runtime_state", None)
        if not isinstance(runtime_state, dict):
            runtime_state = {}
            setattr(params, "_openroad_handoff_runtime_state", runtime_state)
        return cls(config, runtime_state=runtime_state)

    def should_handoff(self, event):
        if not self.enabled:
            return False, None

        decision = self._match_trigger(event)
        if decision is None:
            return False, None

        return True, decision

    def _match_trigger(self, event):
        mode = (self.trigger.get("mode") or "disabled").lower()
        if mode == "disabled":
            return None

        decision = None
        if mode in ("interval", "composite"):
            decision = self._match_interval(event)

        if decision is None and mode in ("phase", "composite"):
            decision = self._match_phase(event)

        if decision is None and mode in ("metric", "composite"):
            decision = self._match_metric(event)

        if decision is None:
            return None

        guard_reason = self._match_guards(event)
        if guard_reason is None:
            return None
        if guard_reason:
            decision["trigger_reason"] = "%s@%s" % (
                decision["trigger_reason"],
                guard_reason,
            )

        dedupe_key = self._dedupe_key(event)
        if dedupe_key is not None:
            if dedupe_key in self._triggered_keys:
                return None
            self._triggered_keys.add(dedupe_key)

        return decision

    def _match_guards(self, event):
        guards = self.trigger.get("guards") or []
        if not guards:
            return ""
        reasons = []
        for guard in guards:
            if not isinstance(guard, dict):
                return None
            metric_name = guard.get("name")
            compare = guard.get("op", "<=")
            threshold = _to_float(guard.get("value"))
            metric_value = _extract_metric_value(event.get("metric"), metric_name)
            if threshold is None or metric_value is None:
                return None
            if not self._compare_metric(metric_value, compare, threshold):
                return None
            reasons.append("%s%s%s" % (metric_name, compare, threshold))
        return "@".join(reasons)

    def _compare_metric(self, metric_value, compare, threshold):
        if compare == ">=":
            return metric_value >= threshold
        if compare == "<=":
            return metric_value <= threshold
        if compare == ">":
            return metric_value > threshold
        if compare == "<":
            return metric_value < threshold
        return False

    def _dedupe_key(self, event):
        key_name = self.trigger.get("dedupe_by")
        if not key_name:
            return None
        key_value = event.get(key_name)
        if key_value is None and key_name == "absolute_iteration":
            key_value = event.get("iteration")
        if key_value is None:
            return None
        return (key_name, key_value)

    def _match_interval(self, event):
        interval = self.trigger.get("interval")
        iteration = event.get("absolute_iteration", event.get("iteration"))
        if not interval or iteration is None:
            return None
        if iteration > 0 and iteration % int(interval) == 0:
            return {
                "trigger_mode": "interval",
                "trigger_reason": "interval=%s@iter=%s" % (int(interval), int(iteration)),
            }
        return None

    def _match_phase(self, event):
        phases = self.trigger.get("phases") or []
        event_location = event.get("phase")
        if event_location is None:
            event_location = event.get("stage")
        if event_location is None:
            return None
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            phase_location = phase.get("phase")
            if phase_location is None:
                phase_location = phase.get("stage")
            at_start = bool(phase.get("at_stage_start", False))
            if phase_location != event_location:
                continue
            if at_start and event.get("iteration_in_stage") == 0:
                return {
                    "trigger_mode": "phase",
                    "trigger_reason": "phase=%s@start" % event_location,
                }
        return None

    def _match_metric(self, event):
        metrics = self.trigger.get("metrics") or []
        for metric_rule in metrics:
            if not isinstance(metric_rule, dict):
                continue
            metric_name = metric_rule.get("name")
            compare = metric_rule.get("op", ">=")
            threshold = _to_float(metric_rule.get("value"))
            metric_value = _extract_metric_value(event.get("metric"), metric_name)
            if threshold is None or metric_value is None:
                continue
            if compare == ">=" and metric_value >= threshold:
                return {
                    "trigger_mode": "metric",
                    "trigger_reason": "%s%s%s" % (metric_name, compare, threshold),
                }
            if compare == "<=" and metric_value <= threshold:
                return {
                    "trigger_mode": "metric",
                    "trigger_reason": "%s%s%s" % (metric_name, compare, threshold),
                }
            if compare == ">" and metric_value > threshold:
                return {
                    "trigger_mode": "metric",
                    "trigger_reason": "%s%s%s" % (metric_name, compare, threshold),
                }
            if compare == "<" and metric_value < threshold:
                return {
                    "trigger_mode": "metric",
                    "trigger_reason": "%s%s%s" % (metric_name, compare, threshold),
                }
        return None

    def build_handoff_event(self, event, decision):
        handoff_event = dict(event)
        if "phase" not in handoff_event and "stage" in handoff_event:
            handoff_event["phase"] = handoff_event["stage"]
        if "absolute_iteration" not in handoff_event and "iteration" in handoff_event:
            handoff_event["absolute_iteration"] = handoff_event["iteration"]
        if "stage" not in handoff_event and "phase" in handoff_event:
            handoff_event["stage"] = handoff_event["phase"]
        if "iteration" not in handoff_event and "absolute_iteration" in handoff_event:
            handoff_event["iteration"] = handoff_event["absolute_iteration"]
        handoff_event["trigger_mode"] = decision["trigger_mode"]
        handoff_event["trigger_reason"] = decision["trigger_reason"]
        handoff_event["reason"] = decision["trigger_reason"]
        metric = event.get("metric")
        handoff_event["metric_snapshot"] = {
            "hpwl": _extract_metric_value(metric, "hpwl"),
            "overflow": _extract_metric_value(metric, "overflow"),
            "objective": _extract_metric_value(metric, "objective"),
            "wns": _extract_metric_value(metric, "wns"),
            "tns": _extract_metric_value(metric, "tns"),
            "ws": _extract_metric_value(metric, "ws"),
            "ts": _extract_metric_value(metric, "ts"),
            "max_slew_violation": _extract_metric_value(metric, "max_slew_violation"),
            "max_load_cap_violation": _extract_metric_value(metric, "max_load_cap_violation"),
        }
        logging.info(
            "OpenROAD handoff triggered: mode=%s reason=%s phase=%s absolute_iteration=%s",
            handoff_event.get("trigger_mode"),
            handoff_event.get("trigger_reason"),
            handoff_event.get("phase"),
            handoff_event.get("absolute_iteration"),
        )
        return handoff_event
