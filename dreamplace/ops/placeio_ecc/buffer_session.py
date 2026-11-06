"""Commit boundary for a shadow buffer overlay in the ECC placement backend."""

from .buffer_overlay import BufferOverlay, action_digest
from .place_io import PlaceIOFunction

ARTIFACT = "placeio_ecc_buffer_session"
ARTIFACT_VERSION = 1


class PlaceIOBufferSession:
    """Shadow transaction over one ``ECCPlaceIOBackend``.

    ``apply``/``evaluate``/``checkpoint``/``restore``/``rollback`` only touch
    the Python overlay.  The native database changes only in ``commit``.
    """

    def __init__(self, raw_db, *, timing_model=None):
        if raw_db is None or not hasattr(raw_db, "pydb"):
            raise ValueError("PlaceIOBufferSession requires an ECC raw_db")
        self.raw_db = raw_db
        self.timing_model = timing_model
        self.base_pydb = None
        self.overlay = None
        self.state = "new"
        self._commit_summary = None

    def _require_open(self):
        if self.state != "open":
            raise RuntimeError(f"buffer session is {self.state}; begin() must be called first")

    def begin(self):
        if self.state in {"open", "committed"}:
            return {
                "status": "already_begun" if self.state == "open" else "already_committed",
                "base_pydb_id": id(self.base_pydb),
                "state": self.state,
            }
        self.base_pydb = self.raw_db.pydb
        self.overlay = BufferOverlay(
            self.base_pydb,
            timing_model=self.timing_model,
        )
        if hasattr(self.timing_model, "bind_overlay"):
            self.timing_model.bind_overlay(self.overlay)
        self.state = "open"
        self._commit_summary = None
        return {
            "status": "ok",
            "base_pydb_id": id(self.base_pydb),
            "state": self.state,
            "refresh_generation": int(getattr(self.raw_db, "refresh_generation", 0) or 0),
        }

    def apply(self, actions, *, transactional=True):
        self._require_open()
        return self.overlay.apply(actions, transactional=transactional)

    def record_candidate(self, net_id, observation, state):
        self._require_open()
        return self.overlay.record_native_candidate(net_id, observation, state)

    def record_action_state(self, net_id, action_id, state):
        self._require_open()
        return self.overlay.record_native_action_state(net_id, action_id, state)

    def seed(self, actions):
        self._require_open()
        return self.overlay.seed(actions)

    def strip_net(self, net_name=None):
        """Remove session-inserted buffers before commit (P6 foundation)."""

        self._require_open()
        return self.overlay.remove_session_buffers(net_name)

    def replace_session_tree(self, net_name, actions):
        self._require_open()
        return self.overlay.replace_session_tree(net_name, actions)

    def remove_base_buffers(self, instance_names, *, net_name=None):
        """Plan removal of buffers that already exist in the base PyPlaceDB."""

        self._require_open()
        return self.overlay.remove_base_buffers(instance_names, net_name=net_name)

    def evaluate(self):
        self._require_open()
        if self.timing_model is None:
            return {
                "status": "unsupported",
                "reason": "timing_model_required",
                "epoch": self.overlay.epoch,
            }
        if callable(self.timing_model):
            result = self.timing_model(self.base_pydb, self.overlay)
        elif hasattr(self.timing_model, "evaluate"):
            result = self.timing_model.evaluate(self.base_pydb, self.overlay)
        else:
            raise TypeError("timing_model must be callable or provide evaluate()")
        return dict(result or {})

    def checkpoint(self, label=""):
        self._require_open()
        return self.overlay.checkpoint(label)

    def release_checkpoint(self, checkpoint_id):
        self._require_open()
        return self.overlay.release_checkpoint(checkpoint_id)

    def restore(self, checkpoint_id):
        self._require_open()
        result = self.overlay.restore(checkpoint_id)
        if hasattr(self.timing_model, "sync_overlay_state"):
            self.timing_model.sync_overlay_state(self.overlay)
        return result

    def export_actions(self):
        self._require_open()
        return self.overlay.commit_actions()

    def rollback(self):
        if self.state == "committed":
            return {
                "status": "already_committed",
                "reason": "native_commit_has_already_happened",
            }
        if self.overlay is None:
            return {"status": "not_started"}
        result = self.overlay.rollback()
        result["base_pydb_id"] = id(self.base_pydb)
        result["pydb_identity_preserved"] = self.raw_db.pydb is self.base_pydb
        self.state = "rolled_back"
        return result

    def commit(self):
        if self.state == "committed":
            return {
                "status": "already_committed",
                "reason": "buffer_session_is_idempotent",
                "commit_summary": dict(self._commit_summary or {}),
            }
        self._require_open()
        if self.overlay.base_removals and self.overlay.actions:
            raise ValueError(
                "mixed base buffer removal and insert commit is not supported; "
                "commit removals, refresh, then start a new session"
            )
        base_removals = list(self.overlay.base_removals)
        actions = base_removals if base_removals else self.overlay.commit_actions()
        digest = action_digest(actions)
        if not actions:
            self.state = "committed"
            self._commit_summary = {
                "status": "skipped",
                "reason": "no_actions",
                "accepted_count": 0,
                "action_digest": digest,
            }
            return dict(self._commit_summary)

        from .timing_carrier import refresh_python_timing_carrier, require_carrier_schema

        require_carrier_schema(self.raw_db.pydb)
        summary = dict(
            PlaceIOFunction.apply_buffer_actions(
                self.raw_db,
                actions,
                digest,
            )
            or {}
        )
        status = str(summary.get("status") or "")
        accepted = int(summary.get("accepted_count", 0) or 0)
        if status not in {"accepted", "ok"} or accepted != len(actions):
            self.state = "commit_failed"
            summary.update(
                {
                    "status": status or "rejected",
                    "action_digest": digest,
                    "native_db_mutated": accepted > 0,
                }
            )
            return summary

        # The native transaction has already completed. A failed graph
        # refresh must not allow this session to insert the same batch again.
        self.state = "committed"
        try:
            new_pydb, refresh_summary = refresh_python_timing_carrier(self.raw_db)
        except (RuntimeError, ValueError, LookupError, TypeError, AttributeError) as error:
            summary.update(
                status="refresh_failed",
                action_digest=digest,
                native_db_mutated=True,
                refresh={"status": "failed", "message": str(error)},
            )
            self._commit_summary = summary
            return dict(summary)
        self.base_pydb = new_pydb
        summary.update(
            {
                "status": "committed",
                "action_digest": digest,
                "accepted_count": accepted,
                "removed_count": len(base_removals),
                "inserted_count": 0 if base_removals else accepted,
                "refresh_generation": int(getattr(self.raw_db, "refresh_generation", 0) or 0),
                "refresh": dict(refresh_summary or {}),
                "pydb_replaced": self.raw_db.pydb is new_pydb,
            }
        )
        self._commit_summary = summary
        return dict(summary)

    def summary(self):
        return {
            "artifact": ARTIFACT,
            "artifact_version": ARTIFACT_VERSION,
            "state": self.state,
            "base_pydb_id": None if self.base_pydb is None else id(self.base_pydb),
            "overlay": None if self.overlay is None else self.overlay.summary(),
            "commit": dict(self._commit_summary or {}),
        }
