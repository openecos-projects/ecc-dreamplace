"""Lifecycle seam for routability-specific placement policy.

The controller deliberately does not own optimizer state, route backends, or
placement writeback.  It owns routability timing and policy decisions while
leaving the ordinary DreamPlace loop and objective evaluation in place.
"""

from collections.abc import Callable
from typing import Any
import logging


class RoutabilityController:
    """Coordinate routability lifecycle events without changing placement.

    ``event_sink`` is intentionally private-facing and optional.  Tests use it
    to capture the ordering contract while production callers leave it unset,
    leaving policy state independent of event recording.
    """

    _EVENT_NAMES = (
        "before_stage",
        "before_iteration",
        "after_iteration",
        "before_area_adjust",
        "after_geometry_change",
        "before_legalization",
        "after_legalization",
        "finalize",
    )

    def __init__(
        self,
        params: Any,
        *,
        event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.enabled = bool(getattr(params, "routability_opt_flag", False))
        self.inflation_min_interval = int(getattr(params, "inflation_min_interval", 0))
        if self.inflation_min_interval < 0:
            raise ValueError("inflation_min_interval must be non-negative")
        self._gp_steps = 0
        self._last_inflation_step = None
        self._event_sink = event_sink
        self.inflation = None
        if self.enabled:
            from dreamplace.ops.routability import enhanced_inflation_controller

            self.inflation = enhanced_inflation_controller

    @staticmethod
    def resolve_route_map_source(params):
        """Select the active route map without constructing legacy operators."""
        if getattr(params, "adjust_gpugr_area_flag", False):
            return "gpugr"
        if getattr(params, "adjust_nctugr_area_flag", False):
            logging.info(
                "adjust_nctugr_area_flag is a legacy compatibility key; "
                "using ECC/iRT EGR route map for area adjustment "
                "(NCTUgr is not invoked)"
            )
            return "irt_egr"
        return "rudy"

    @staticmethod
    def is_modularity_inflation_enabled(params):
        return bool(getattr(params, "modularity_inflation_flag", False))

    @staticmethod
    def ensure_modularity_inflation_contract(params, route_map_source=None):
        if not bool(getattr(params, "modularity_inflation_flag", False)):
            return
        if not getattr(params, "routability_opt_flag", False):
            raise RuntimeError("modularity_inflation_flag=1 requires routability_opt_flag=1")
        if not getattr(params, "modularity_require_gpugr_flag", 1):
            return
        if not getattr(params, "adjust_gpugr_area_flag", False):
            raise RuntimeError("modularity_inflation_flag=1 requires adjust_gpugr_area_flag=1")
        if getattr(params, "adjust_nctugr_area_flag", False):
            raise RuntimeError("modularity_inflation_flag=1 does not support adjust_nctugr_area_flag=1")
        if getattr(params, "adjust_rudy_area_flag", False):
            raise RuntimeError("modularity_inflation_flag=1 does not support adjust_rudy_area_flag=1")
        if route_map_source is not None and route_map_source != "gpugr":
            raise RuntimeError(
                "modularity_inflation_flag=1 requires gpugr route source, got %s"
                % route_map_source
            )

    @staticmethod
    def ensure_active_modularity_clusters(params, placedb, model, pos, num_area_adjust):
        if not bool(getattr(params, "modularity_inflation_flag", False)):
            return
        if int(num_area_adjust) != 0:
            return
        if getattr(placedb, "modularity_active_clustering_result", None) is not None:
            return

        from dreamplace.ops.routability.leiden_clustering import (
            build_active_leiden_clusters,
            plot_modularity_clusters,
        )

        active_result = build_active_leiden_clusters(placedb, params, pos)
        placedb.modularity_active_clustering_result = active_result
        if hasattr(model, "data_collections") and model.data_collections is not None:
            model.data_collections.refresh_modularity_clusters_from_placedb(placedb)
        logging.info(
            "Prepared modularity active clusters at first inflation trigger: levels=%d counts=%s",
            len(active_result.cluster_ids_by_level),
            active_result.num_clusters_by_level,
        )
        if getattr(params, "modularity_plot_flag", False):
            saved_paths = plot_modularity_clusters(
                placedb=placedb,
                params=params,
                pos=pos,
                clustering_result=active_result,
                source="active",
                round_idx=int(num_area_adjust),
            )
            logging.info("Saved modularity debug plots: %s", saved_paths)
            if getattr(params, "modularity_plot_exit_flag", False):
                logging.info("modularity_plot_exit_flag=1, exit(0) after saving modularity debug plots")
                raise SystemExit(0)

    @staticmethod
    def should_defer_stop_for_legacy_inflation(
        params, num_area_adjust, max_area_adjust_rounds, overflow
    ):
        from dreamplace.ops.routability import inflation_legalization

        return bool(
            getattr(params, "routability_opt_flag", False)
            and inflation_legalization.is_enabled(params)
            and inflation_legalization.should_trigger_legacy_inflation(
                params, num_area_adjust, max_area_adjust_rounds, overflow
            )
        )

    def inflation_triggers(
        self, params, num_area_adjust, max_area_adjust_rounds, overflow
    ):
        if (
            self._last_inflation_step is not None
            and self._gp_steps - self._last_inflation_step < self.inflation_min_interval
        ):
            return False, False
        from dreamplace.ops.routability import enhanced_inflation_controller
        from dreamplace.ops.routability import inflation_legalization

        enhanced = enhanced_inflation_controller.should_trigger_enhanced_inflation(
            params,
            num_area_adjust=num_area_adjust,
            overflow=overflow,
        )
        legacy = inflation_legalization.should_trigger_legacy_inflation(
            params,
            num_area_adjust,
            max_area_adjust_rounds,
            overflow,
            trigger_enhanced_inflation=enhanced,
        )
        return bool(enhanced), bool(legacy)

    def record_inflation(self) -> None:
        """Start the interval only after an actual area change."""
        self._last_inflation_step = self._gp_steps
        logging.info(
            "Inflation applied after GP step %d; next round requires at least %d more GP steps",
            self._gp_steps, self.inflation_min_interval,
        )

    @staticmethod
    def area_adjust_flags(params, *, use_enhanced, default_flags):
        if use_enhanced:
            from dreamplace.ops.routability import enhanced_inflation_controller

            return enhanced_inflation_controller.get_area_adjust_flags(params)
        return dict(default_flags)

    def _emit(self, name: str, **payload: Any) -> None:
        if self.enabled and self._event_sink is not None:
            self._event_sink(name, payload)

    def before_stage(self, *, model: Any, stage_idx: int) -> None:
        self._emit("before_stage", model=model, stage_idx=stage_idx)

    def before_iteration(
        self,
        *,
        model: Any,
        iteration: int,
        position: Any,
    ) -> None:
        self._emit(
            "before_iteration",
            model=model,
            iteration=iteration,
            position=position,
        )

    def after_iteration(
        self,
        *,
        model: Any,
        iteration: int,
        metrics: Any,
    ) -> None:
        if self.enabled:
            self._gp_steps += 1
        self._emit(
            "after_iteration",
            model=model,
            iteration=iteration,
            metrics=metrics,
        )

    def before_area_adjust(
        self,
        *,
        model: Any,
        position: Any,
        maps: Any = None,
    ) -> None:
        self._emit(
            "before_area_adjust",
            model=model,
            position=position,
            maps=maps,
        )

    def after_geometry_change(self, *, model: Any, position: Any) -> None:
        self._emit(
            "after_geometry_change",
            model=model,
            position=position,
        )

    def before_legalization(
        self,
        *,
        model: Any,
        position: Any,
    ) -> None:
        self._emit(
            "before_legalization",
            model=model,
            position=position,
        )

    def after_legalization(
        self,
        *,
        model: Any,
        position: Any,
    ) -> None:
        self._emit(
            "after_legalization",
            model=model,
            position=position,
        )

    def finalize(self, *, model: Any, position: Any) -> None:
        self._emit("finalize", model=model, position=position)


__all__ = ["RoutabilityController"]
