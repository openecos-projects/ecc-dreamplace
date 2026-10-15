import json
import tempfile
import unittest
from pathlib import Path

from dreamplace.ops.physical_action import (
    VirtualBufferState,
    build_commit_action_candidates,
    write_final_state,
    write_virtual_update_trace,
)


class PhysicalActionSchemaTest(unittest.TestCase):
    def _candidate(self):
        return {
            "candidate_id": 3,
            "net_id": 9,
            "net_name": "n9",
            "x_dbu": 1200,
            "y_dbu": 3400,
            "x_um": 1.2,
            "y_um": 3.4,
            "buffer_main_type_index": 5,
            "predicted_delta_obj": -0.25,
            "segment_id": 12,
            "segment_parent_node_id": 4,
            "segment_child_node_id": 5,
            "segment_tree_depth": 2,
            "segment_split_index": 1,
            "segment_split_count_on_edge": 3,
            "segment_split_ratio": 0.5,
        }

    def _legal_table(self):
        return {
            "legal_cell_ids": [17],
            "legal_master_names": ["BUF_X1"],
            "legal_size_values": [1.0],
        }

    def test_action_layer_builds_commit_schema(self):
        state = VirtualBufferState()
        state.insert(self._candidate(), bsu=0)

        artifact = build_commit_action_candidates(
            state,
            legal_table=self._legal_table(),
            executable_by_compact_batch_v1=False,
        )

        self.assertEqual(artifact["artifact"], "buffering_commit_action_candidates")
        self.assertEqual(artifact["candidate_commit_action_count"], 1)
        self.assertEqual(artifact["actions"][0]["buffer_master_name"], "BUF_X1")
        self.assertEqual(artifact["actions"][0]["segment_id"], 12)
        self.assertEqual(artifact["actions"][0]["segment_parent_node_id"], 4)
        self.assertEqual(artifact["actions"][0]["segment_child_node_id"], 5)
        self.assertEqual(artifact["actions"][0]["segment_tree_depth"], 2)
        self.assertEqual(artifact["actions"][0]["segment_split_index"], 1)
        self.assertEqual(artifact["actions"][0]["segment_split_count_on_edge"], 3)
        self.assertAlmostEqual(artifact["actions"][0]["segment_split_ratio"], 0.5)

    def test_action_layer_writes_trace_schema(self):
        state = VirtualBufferState()
        state.insert(self._candidate(), bsu=0)
        state.resize(self._candidate(), target_bsu=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            trace_summary = write_virtual_update_trace(state, path / "trace.jsonl")
            final_summary = write_final_state(state, path / "state.json")

            trace_lines = [
                json.loads(line)
                for line in (path / "trace.jsonl").read_text().splitlines()
            ]
            final_state = json.loads((path / "state.json").read_text())

        self.assertEqual(trace_summary["trace_event_count"], 2)
        self.assertEqual(trace_lines[0]["action"], "insert")
        self.assertEqual(trace_lines[1]["action"], "noop")
        self.assertEqual(final_summary["artifact"], "buffering_final_state")
        self.assertEqual(final_state["candidate_commit_action_count"], 1)


if __name__ == "__main__":
    unittest.main()
