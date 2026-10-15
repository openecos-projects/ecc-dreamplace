from copy import deepcopy

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dreamplace.ops.openroad_handoff import PlacementHandoffSession
sys.path.pop()


class PlacementHandoffSessionTest(unittest.TestCase):
    def _metric_snapshot(self):
        return {
            "hpwl": 1.5,
            "overflow": 0.25,
            "objective": 9.0,
            "wns": -0.11,
            "tns": -1.2,
            "ws": -0.09,
            "ts": 0.0,
            "max_slew_violation": 0.03,
            "max_load_cap_violation": 0.04,
        }

    def _buffer_churn_counts(self):
        return {
            "added_buffer_count": 2,
            "removed_buffer_count": 1,
            "surviving_buffer_count": 7,
        }

    def test_commit_advances_epoch_only_after_successful_continuation(self):
        session = PlacementHandoffSession(session_id="run-1")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 10, "phase": 0},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=10"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 100, "pins": 200, "nets": 50},
            pre_snapshot_fingerprint="nodes=100|pins=200|nets=50",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 105, "pins": 210, "nets": 55},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )
        self.assertEqual(session.topology_epoch, 0)
        session.commit_continuation(
            seq,
            {
                "runtimedb_rebuild_performed": True,
                "runtimedb_generation_after": 1,
            },
        )
        self.assertEqual(session.topology_epoch, 1)
        self.assertEqual(session.runtimedb_generation, 1)

    def test_failed_continuation_moves_session_to_failed(self):
        session = PlacementHandoffSession(session_id="run-2")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 20, "phase": 1},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=20"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 105, "pins": 210, "nets": 55},
            pre_snapshot_fingerprint="nodes=105|pins=210|nets=55",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 110, "pins": 220, "nets": 60},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )
        session.fail_current_handoff(seq, "rebuild", "boom")
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.event_history[-1]["status"], "rebuild_failed")
        self.assertEqual(session.topology_epoch, 0)
        self.assertEqual(session.runtimedb_generation, 0)

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 2,
                },
            )

        self.assertEqual(session.topology_epoch, 0)
        self.assertEqual(session.runtimedb_generation, 0)
        self.assertEqual(session.status, "failed")

    def test_invalid_transition_order_raises_runtime_error(self):
        session = PlacementHandoffSession(session_id="run-3")

        with self.assertRaises(RuntimeError):
            session.record_mutation_result(
                1,
                {
                    "mutation_kind": "topology_changed",
                    "post_counts": {"nodes": 1, "pins": 1, "nets": 1},
                    "requires_runtimedb_rebuild": True,
                },
            )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                1,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 1,
                },
            )

    def test_commit_before_mutation_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-4")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 30, "phase": 2},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=30"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 120, "pins": 240, "nets": 70},
            pre_snapshot_fingerprint="nodes=120|pins=240|nets=70",
        )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 1,
                },
            )

        self.assertEqual(session.topology_epoch, 0)
        self.assertEqual(session.runtimedb_generation, 0)
        self.assertEqual(session.status, "handoff_in_progress")

    def test_duplicate_mutation_recording_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-5")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 40, "phase": 3},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=40"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 130, "pins": 260, "nets": 80},
            pre_snapshot_fingerprint="nodes=130|pins=260|nets=80",
        )

        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 135, "pins": 270, "nets": 85},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        with self.assertRaises(RuntimeError):
            session.record_mutation_result(
                seq,
                {
                    "mutation_kind": "topology_changed",
                    "post_counts": {"nodes": 140, "pins": 280, "nets": 90},
                    "requires_runtimedb_rebuild": True,
                },
            )

        self.assertEqual(session.event_history[-1]["post_counts"], {"nodes": 135, "pins": 270, "nets": 85})

    def test_begin_handoff_preserves_metric_snapshot_from_run_event(self):
        session = PlacementHandoffSession(session_id="run-5a")

        seq = session.begin_handoff(
            run_event={
                "absolute_iteration": 41,
                "phase": 3,
                "metric_snapshot": self._metric_snapshot(),
            },
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=41"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 130, "pins": 260, "nets": 80},
            pre_snapshot_fingerprint="nodes=130|pins=260|nets=80",
        )

        self.assertEqual(seq, 1)
        self.assertEqual(session.event_history[-1]["metric_snapshot"], self._metric_snapshot())

    def test_commit_continuation_preserves_restart_probe_payloads(self):
        session = PlacementHandoffSession(session_id="run-probes")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 50, "phase": 0, "metric_snapshot": self._metric_snapshot()},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=50"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 100, "pins": 200, "nets": 50},
            pre_snapshot_fingerprint="nodes=100|pins=200|nets=50",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 101, "pins": 202, "nets": 51},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )
        session.commit_continuation(
            seq,
            {
                "runtimedb_rebuild_performed": True,
                "runtimedb_generation_after": 1,
                "restart_policy": "warm_schedule",
                "restart_probes": [
                    {"phase": "restart_iter0", "overflow": 0.2},
                    {"phase": "restart_iter50", "overflow": 0.4},
                ],
            },
        )
        event = session.event_history[-1]
        self.assertEqual(event["restart_policy"], "warm_schedule")
        self.assertEqual(event["restart_probes"][0]["phase"], "restart_iter0")
        self.assertEqual(event["restart_probes"][1]["overflow"], 0.4)

    def test_continuation_payload_cannot_overwrite_metric_snapshot(self):
        session = PlacementHandoffSession(session_id="run-5b")
        seq = session.begin_handoff(
            run_event={
                "absolute_iteration": 42,
                "phase": 3,
                "metric_snapshot": self._metric_snapshot(),
            },
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=42"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 130, "pins": 260, "nets": 80},
            pre_snapshot_fingerprint="nodes=130|pins=260|nets=80",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "placement_only",
                "requires_runtimedb_rebuild": False,
            },
        )

        before = deepcopy(session.event_history[-1])
        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                    "metric_snapshot": {"hpwl": 99.0},
                },
            )

        self.assertEqual(session.event_history[-1], before)

    def test_failure_payload_cannot_overwrite_metric_snapshot(self):
        session = PlacementHandoffSession(session_id="run-5c")
        seq = session.begin_handoff(
            run_event={
                "absolute_iteration": 43,
                "phase": 3,
                "metric_snapshot": self._metric_snapshot(),
            },
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=43"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 130, "pins": 260, "nets": 80},
            pre_snapshot_fingerprint="nodes=130|pins=260|nets=80",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "placement_only",
                "requires_runtimedb_rebuild": False,
            },
        )

        before = deepcopy(session.event_history[-1])
        with self.assertRaises(RuntimeError):
            session.fail_current_handoff(
                seq,
                "mutation",
                "boom",
                failure_payload={"metric_snapshot": {"hpwl": 99.0}},
            )

        self.assertEqual(session.event_history[-1], before)

    def test_late_failure_cannot_rewrite_committed_history(self):
        session = PlacementHandoffSession(session_id="run-6")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 50, "phase": 4},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=50"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 140, "pins": 280, "nets": 90},
            pre_snapshot_fingerprint="nodes=140|pins=280|nets=90",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 145, "pins": 290, "nets": 95},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )
        session.commit_continuation(
            seq,
            {
                "runtimedb_rebuild_performed": True,
                "runtimedb_generation_after": 2,
            },
        )

        committed_record = dict(session.event_history[-1])

        with self.assertRaises(RuntimeError):
            session.fail_current_handoff(seq, "rebuild", "late boom")

        self.assertEqual(session.status, "active")
        self.assertEqual(session.topology_epoch, 1)
        self.assertEqual(session.runtimedb_generation, 2)
        self.assertEqual(session.event_history[-1], committed_record)

    def test_rebuild_required_but_not_performed_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-7")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 60, "phase": 5},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=60"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 150, "pins": 300, "nets": 100},
            pre_snapshot_fingerprint="nodes=150|pins=300|nets=100",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 155, "pins": 310, "nets": 105},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                    "runtimedb_generation_after": 3,
                },
            )

        self.assertEqual(session.topology_epoch, 0)
        self.assertEqual(session.runtimedb_generation, 0)
        self.assertEqual(session.status, "rebuild_in_progress")

    def test_topology_changed_commit_missing_post_counts_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-8")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 70, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=70"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 160, "pins": 320, "nets": 110},
            pre_snapshot_fingerprint="nodes=160|pins=320|nets=110",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        before = deepcopy(session.event_history[-1])

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "post_counts": {"nodes": 165, "pins": 330, "nets": 115},
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 1,
                },
            )

        self.assertEqual(session.event_history[-1], before)

    def test_mutation_payload_continuation_owned_fields_are_rejected(self):
        session = PlacementHandoffSession(session_id="run-8b")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 72, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=72"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 161, "pins": 321, "nets": 111},
            pre_snapshot_fingerprint="nodes=161|pins=321|nets=111",
        )

        with self.assertRaises(RuntimeError):
            session.record_mutation_result(
                seq,
                {
                    "mutation_kind": "topology_changed",
                    "post_counts": {"nodes": 166, "pins": 331, "nets": 116},
                    "requires_runtimedb_rebuild": True,
                    "runtimedb_rebuild_performed": True,
                },
            )

    def test_unknown_mutation_kind_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-8c")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 74, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=74"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 162, "pins": 322, "nets": 112},
            pre_snapshot_fingerprint="nodes=162|pins=322|nets=112",
        )

        with self.assertRaises(RuntimeError):
            session.record_mutation_result(
                seq,
                {
                    "mutation_kind": "misspelled_kind",
                    "requires_runtimedb_rebuild": False,
                },
            )

    def test_topology_changed_mutation_requires_runtimedb_rebuild(self):
        session = PlacementHandoffSession(session_id="run-8d")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 74, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=74"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 162, "pins": 322, "nets": 112},
            pre_snapshot_fingerprint="nodes=162|pins=322|nets=112",
        )

        with self.assertRaises(RuntimeError):
            session.record_mutation_result(
                seq,
                {
                    "mutation_kind": "topology_changed",
                    "post_counts": {"nodes": 165, "pins": 325, "nets": 115},
                    "requires_runtimedb_rebuild": False,
                },
            )

    def test_topology_changed_mutation_requires_buffer_churn_counts(self):
        session = PlacementHandoffSession(session_id="run-8d2")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 75, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=75"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 162, "pins": 322, "nets": 112},
            pre_snapshot_fingerprint="nodes=162|pins=322|nets=112",
        )

        with self.assertRaisesRegex(RuntimeError, "buffer churn"):
            session.record_mutation_result(
                seq,
                {
                    "mutation_kind": "topology_changed",
                    "post_counts": {"nodes": 165, "pins": 325, "nets": 115},
                    "requires_runtimedb_rebuild": True,
                    "added_buffer_count": 1,
                    "removed_buffer_count": 0,
                },
            )

    def test_topology_changed_mutation_rejects_invalid_buffer_churn_counts(self):
        invalid_values = (None, True, -1, 1.5, "2")
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                session = PlacementHandoffSession(session_id="run-8d3")
                seq = session.begin_handoff(
                    run_event={"absolute_iteration": 75, "phase": 6},
                    trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=75"},
                    openroad_session_identity="bridge-1",
                    pre_counts={"nodes": 162, "pins": 322, "nets": 112},
                    pre_snapshot_fingerprint="nodes=162|pins=322|nets=112",
                )
                payload = {
                    "mutation_kind": "topology_changed",
                    "post_counts": {"nodes": 165, "pins": 325, "nets": 115},
                    "requires_runtimedb_rebuild": True,
                    **self._buffer_churn_counts(),
                }
                payload["added_buffer_count"] = invalid_value

                with self.assertRaisesRegex(RuntimeError, "buffer churn"):
                    session.record_mutation_result(seq, payload)

    def test_geometry_changed_mutation_requires_runtimedb_rebuild(self):
        session = PlacementHandoffSession(session_id="run-8e")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 76, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=76"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 163, "pins": 323, "nets": 113},
            pre_snapshot_fingerprint="nodes=163|pins=323|nets=113",
        )

        with self.assertRaisesRegex(RuntimeError, "geometry_changed mutation requires runtimedb rebuild"):
            session.record_mutation_result(
                seq,
                {
                    "mutation_kind": "geometry_changed",
                    "requires_runtimedb_rebuild": False,
                },
            )

    def test_geometry_changed_mutation_with_rebuild_is_accepted_and_advances_generation(self):
        session = PlacementHandoffSession(session_id="run-8e2")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 77, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=77"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 163, "pins": 323, "nets": 113},
            pre_snapshot_fingerprint="nodes=163|pins=323|nets=113",
        )

        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "geometry_changed",
                "requires_runtimedb_rebuild": True,
                "post_snapshot_fingerprint": "nodes=163|pins=323|nets=113|geom=2",
            },
        )

        self.assertEqual(session.status, "rebuild_in_progress")
        before = deepcopy(session.event_history[-1])
        with self.assertRaisesRegex(RuntimeError, "required runtimedb rebuild was not performed"):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                },
            )
        self.assertEqual(session.event_history[-1], before)

        with self.assertRaisesRegex(RuntimeError, "runtimedb_generation_after must advance"):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 0,
                },
            )
        self.assertEqual(session.event_history[-1], before)

        session.commit_continuation(
            seq,
            {
                "runtimedb_rebuild_performed": True,
                "runtimedb_generation_after": 1,
            },
        )

        self.assertEqual(session.event_history[-1]["mutation_kind"], "geometry_changed")
        self.assertEqual(session.event_history[-1]["status"], "success")
        self.assertEqual(session.status, "active")
        self.assertEqual(session.topology_epoch, 0)
        self.assertEqual(session.runtimedb_generation, 1)
        self.assertEqual(
            session.active_topology_summary["counts"],
            {"nodes": 163, "pins": 323, "nets": 113},
        )
        self.assertEqual(
            session.active_topology_summary["snapshot_fingerprint"],
            "nodes=163|pins=323|nets=113|geom=2",
        )

    def test_geometry_changed_commit_revalidates_runtimedb_rebuild_contract(self):
        session = PlacementHandoffSession(session_id="run-8e3")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 77, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=77"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 163, "pins": 323, "nets": 113},
            pre_snapshot_fingerprint="nodes=163|pins=323|nets=113",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "geometry_changed",
                "requires_runtimedb_rebuild": True,
            },
        )
        session.event_history[-1]["requires_runtimedb_rebuild"] = False
        before = deepcopy(session.event_history[-1])

        with self.assertRaisesRegex(RuntimeError, "geometry_changed mutation requires runtimedb rebuild"):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                },
            )

        self.assertEqual(session.event_history[-1], before)

    def test_buffer_count_fields_are_mutation_owned(self):
        session = PlacementHandoffSession(session_id="run-8f")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 78, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=78"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 164, "pins": 324, "nets": 114},
            pre_snapshot_fingerprint="nodes=164|pins=324|nets=114",
        )

        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "geometry_changed",
                "requires_runtimedb_rebuild": True,
                "added_buffer_count": 2,
                "removed_buffer_count": 1,
                "surviving_buffer_count": 7,
            },
        )
        self.assertEqual(session.event_history[-1]["added_buffer_count"], 2)
        self.assertEqual(session.event_history[-1]["removed_buffer_count"], 1)
        self.assertEqual(session.event_history[-1]["surviving_buffer_count"], 7)
        before = deepcopy(session.event_history[-1])

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                    "added_buffer_count": 99,
                },
            )

        self.assertEqual(session.event_history[-1], before)

    def test_strategy_command_fields_are_mutation_owned(self):
        session = PlacementHandoffSession(session_id="run-8g")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 79, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=79"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 164, "pins": 324, "nets": 114},
            pre_snapshot_fingerprint="nodes=164|pins=324|nets=114",
        )

        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
                "buffer_insertion_strategy": "repair_design",
                "buffer_insertion_command": "repair_design -max_wire_length {10}",
            },
        )
        before = deepcopy(session.event_history[-1])

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                    "buffer_insertion_command": "repair_design",
                },
            )

        self.assertEqual(session.event_history[-1], before)

    def test_mutation_failure_can_record_attempted_strategy_command(self):
        session = PlacementHandoffSession(session_id="run-8h")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 80, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=80"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 164, "pins": 324, "nets": 114},
            pre_snapshot_fingerprint="nodes=164|pins=324|nets=114",
        )

        session.fail_current_handoff(
            seq,
            "mutation",
            "openroad failed",
            failure_payload={
                "buffer_insertion_strategy": "repair_design",
                "buffer_insertion_command": "repair_design -max_wire_length {10}",
            },
        )

        record = session.event_history[-1]
        self.assertEqual(record["status"], "mutation_failed")
        self.assertEqual(record["buffer_insertion_strategy"], "repair_design")
        self.assertEqual(
            record["buffer_insertion_command"],
            "repair_design -max_wire_length {10}",
        )

    def test_missing_runtimedb_generation_after_is_rejected_when_rebuild_performed(self):
        session = PlacementHandoffSession(session_id="run-9")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 75, "phase": 6},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=75"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 160, "pins": 320, "nets": 110},
            pre_snapshot_fingerprint="nodes=160|pins=320|nets=110",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 165, "pins": 330, "nets": 115},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                },
            )

    def test_regressive_runtimedb_generation_after_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-10")
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 80, "phase": 7},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=80"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 170, "pins": 340, "nets": 120},
            pre_snapshot_fingerprint="nodes=170|pins=340|nets=120",
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 175, "pins": 350, "nets": 125},
                "post_snapshot_fingerprint": "nodes=175|pins=350|nets=125",
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": True,
                "runtimedb_generation_after": 2,
            },
        )

        second_seq = session.begin_handoff(
            run_event={"absolute_iteration": 90, "phase": 8},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=90"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 175, "pins": 350, "nets": 125},
            pre_snapshot_fingerprint="nodes=175|pins=350|nets=125",
        )
        session.record_mutation_result(
            second_seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 185, "pins": 370, "nets": 135},
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                second_seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 1,
                },
            )

    def test_non_advancing_runtimedb_generation_after_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-10b")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 85, "phase": 7},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=85"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 172, "pins": 344, "nets": 122},
            pre_snapshot_fingerprint="nodes=172|pins=344|nets=122",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 178, "pins": 356, "nets": 128},
                "post_snapshot_fingerprint": "nodes=178|pins=356|nets=128",
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 0,
                },
            )

    def test_non_integer_runtimedb_generation_after_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-10c")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 86, "phase": 7},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=86"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 173, "pins": 346, "nets": 123},
            pre_snapshot_fingerprint="nodes=173|pins=346|nets=123",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 179, "pins": 358, "nets": 129},
                "post_snapshot_fingerprint": "nodes=179|pins=358|nets=129",
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": True,
                },
            )

    def test_failed_commit_leaves_event_record_unchanged(self):
        session = PlacementHandoffSession(session_id="run-12")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 120, "phase": 11},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=120"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 210, "pins": 420, "nets": 160},
            pre_snapshot_fingerprint="nodes=210|pins=420|nets=160",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )
        before = deepcopy(session.event_history[-1])

        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": False,
                    "pre_counts": {"nodes": 1, "pins": 1, "nets": 1},
                },
            )

        self.assertEqual(session.event_history[-1], before)

    def test_second_handoff_mismatched_bridge_identity_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-13")
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 130, "phase": 12},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=130"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 220, "pins": 440, "nets": 170},
            pre_snapshot_fingerprint="nodes=220|pins=440|nets=170",
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": False,
            },
        )

        with self.assertRaises(RuntimeError):
            session.begin_handoff(
                run_event={"absolute_iteration": 140, "phase": 13},
                trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=140"},
                openroad_session_identity="bridge-2",
                pre_counts={"nodes": 220, "pins": 440, "nets": 170},
                pre_snapshot_fingerprint="nodes=220|pins=440|nets=170",
            )

    def test_second_handoff_mismatched_pre_counts_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-14")
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 150, "phase": 14},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=150"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 230, "pins": 460, "nets": 180},
            pre_snapshot_fingerprint="nodes=230|pins=460|nets=180",
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": False,
            },
        )

        with self.assertRaises(RuntimeError):
            session.begin_handoff(
                run_event={"absolute_iteration": 160, "phase": 15},
                trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=160"},
                openroad_session_identity="bridge-1",
                pre_counts={"nodes": 231, "pins": 460, "nets": 180},
                pre_snapshot_fingerprint="nodes=231|pins=460|nets=180",
            )

    def test_second_handoff_mismatched_pre_snapshot_fingerprint_is_rejected(self):
        session = PlacementHandoffSession(session_id="run-14b")
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 155, "phase": 14},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=155"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 232, "pins": 464, "nets": 182},
            pre_snapshot_fingerprint="nodes=232|pins=464|nets=182",
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "no_mutation",
                "post_snapshot_fingerprint": "nodes=232|pins=464|nets=182",
                "requires_runtimedb_rebuild": False,
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": False,
            },
        )

        with self.assertRaises(RuntimeError):
            session.begin_handoff(
                run_event={"absolute_iteration": 165, "phase": 15},
                trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=165"},
                openroad_session_identity="bridge-1",
                pre_counts={"nodes": 232, "pins": 464, "nets": 182},
                pre_snapshot_fingerprint="nodes=232|pins=464|nets=182|digest=changed",
            )

    def test_none_post_snapshot_fingerprint_falls_back_to_pre_snapshot(self):
        session = PlacementHandoffSession(session_id="run-14c")
        first_counts = {"nodes": 233, "pins": 466, "nets": 183}
        first_fingerprint = "nodes=233|pins=466|nets=183"
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 166, "phase": 15},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=166"},
            openroad_session_identity="bridge-1",
            pre_counts=first_counts,
            pre_snapshot_fingerprint=first_fingerprint,
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "no_mutation",
                "post_snapshot_fingerprint": None,
                "requires_runtimedb_rebuild": False,
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": False,
            },
        )

        self.assertEqual(
            session.active_topology_summary["snapshot_fingerprint"],
            first_fingerprint,
        )
        second_seq = session.begin_handoff(
            run_event={"absolute_iteration": 167, "phase": 15},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=167"},
            openroad_session_identity="bridge-1",
            pre_counts=first_counts,
            pre_snapshot_fingerprint=first_fingerprint,
        )

        self.assertEqual(second_seq, 2)

    def test_failure_stage_is_restricted(self):
        session = PlacementHandoffSession(session_id="run-15")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 170, "phase": 16},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=170"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 240, "pins": 480, "nets": 190},
            pre_snapshot_fingerprint="nodes=240|pins=480|nets=190",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )

        with self.assertRaises(RuntimeError):
            session.fail_current_handoff(seq, "bogus", "boom")

    def test_rebuild_failure_label_is_rejected_for_non_rebuild_mutation(self):
        session = PlacementHandoffSession(session_id="run-16")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 180, "phase": 17},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=180"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 250, "pins": 500, "nets": 200},
            pre_snapshot_fingerprint="nodes=250|pins=500|nets=200",
        )
        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )

        with self.assertRaises(RuntimeError):
            session.fail_current_handoff(seq, "rebuild", "boom")

    def test_topology_changing_follow_up_uses_preserved_active_summary_counts(self):
        session = PlacementHandoffSession(session_id="run-11")
        first_seq = session.begin_handoff(
            run_event={"absolute_iteration": 100, "phase": 9},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=100"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 190, "pins": 380, "nets": 140},
            pre_snapshot_fingerprint="nodes=190|pins=380|nets=140",
        )
        session.record_mutation_result(
            first_seq,
            {
                "mutation_kind": "no_mutation",
                "requires_runtimedb_rebuild": False,
            },
        )
        session.commit_continuation(
            first_seq,
            {
                "runtimedb_rebuild_performed": False,
            },
        )
        first_summary = dict(session.active_topology_summary)

        second_seq = session.begin_handoff(
            run_event={"absolute_iteration": 110, "phase": 10},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=110"},
            openroad_session_identity="bridge-1",
            pre_counts=first_summary["counts"],
            pre_snapshot_fingerprint="nodes=190|pins=380|nets=140",
        )
        self.assertEqual(session.event_history[-1]["pre_counts"], first_summary["counts"])
        session.record_mutation_result(
            second_seq,
            {
                "mutation_kind": "topology_changed",
                "post_counts": {"nodes": 195, "pins": 390, "nets": 145},
                "post_snapshot_fingerprint": "nodes=195|pins=390|nets=145",
                "requires_runtimedb_rebuild": True,
                **self._buffer_churn_counts(),
            },
        )
        session.commit_continuation(
            second_seq,
            {
                "runtimedb_rebuild_performed": True,
                "runtimedb_generation_after": 1,
            },
        )

        self.assertIsNotNone(session.active_topology_summary)
        self.assertEqual(session.active_topology_summary["counts"], {"nodes": 195, "pins": 390, "nets": 145})
        self.assertEqual(session.active_topology_summary["topology_epoch"], 1)
        self.assertEqual(session.active_topology_summary["runtimedb_generation"], 1)

    def test_fail_handoff_start_records_failed_event(self):
        session = PlacementHandoffSession(session_id="run-17")
        seq = session.fail_handoff_start(
            run_event={"absolute_iteration": 190, "phase": 18},
            trigger_decision={"trigger_mode": "interval", "trigger_reason": "interval=190"},
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 260, "pins": 520, "nets": 210},
            pre_snapshot_fingerprint="nodes=260|pins=520|nets=210",
            failure_stage="refresh",
            error_summary="active topology mismatch",
        )

        self.assertEqual(seq, 1)
        self.assertEqual(session.status, "failed")
        self.assertEqual(session.event_history[-1]["status"], "refresh_failed")
        self.assertEqual(session.event_history[-1]["failure_stage"], "refresh")
        self.assertEqual(
            session.event_history[-1]["error_summary"],
            "active topology mismatch",
        )

    def test_buffer_only_policy_is_mutation_owned(self):
        session = PlacementHandoffSession(session_id="run-policy")
        seq = session.begin_handoff(
            run_event={"absolute_iteration": 100, "phase": 0},
            trigger_decision={
                "trigger_mode": "exact_iteration",
                "trigger_reason": "exact_iteration@iter=100",
            },
            openroad_session_identity="bridge-1",
            pre_counts={"nodes": 2, "pins": 2, "nets": 1},
            pre_snapshot_fingerprint="nodes=2|pins=2|nets=1",
        )
        policy = {
            "status": "violated",
            "violation_counts": {"total_violation_count": 1},
            "allowed_change_counts": {},
            "summary": {},
            "violation_samples": [],
            "unknown_reasons": [],
            "sample_limit": 50,
            "samples_truncated": False,
        }

        session.record_mutation_result(
            seq,
            {
                "mutation_kind": "geometry_changed",
                "requires_runtimedb_rebuild": True,
                "buffer_only_policy": policy,
            },
        )

        self.assertEqual(session.event_history[-1]["buffer_only_policy"], policy)
        with self.assertRaises(RuntimeError):
            session.commit_continuation(
                seq,
                {
                    "runtimedb_rebuild_performed": True,
                    "runtimedb_generation_after": 1,
                    "buffer_only_policy": {"status": "clean"},
                },
            )


if __name__ == "__main__":
    unittest.main()
