import json
import tempfile
import unittest
from pathlib import Path

from dreamplace.ops.physical_action import (
    VirtualBufferState,
    write_final_state,
    write_virtual_update_trace,
)


class PhysicalActionTraceSchemaTest(unittest.TestCase):
    def test_writes_jsonl_trace_and_final_state(self):
        state = VirtualBufferState()
        candidate = {
            "candidate_id": 4,
            "net_id": 8,
            "x_dbu": 100,
            "y_dbu": 200,
            "predicted_delta_obj": -1.5,
        }
        state.insert(candidate, bsu=0)
        state.resize(candidate, target_bsu=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "buffering_virtual_update_trace.jsonl"
            state_path = Path(tmpdir) / "buffering_final_state.json"
            trace_summary = write_virtual_update_trace(state, trace_path)
            final_summary = write_final_state(state, state_path)

            trace_lines = [json.loads(line) for line in trace_path.read_text().splitlines()]
            final_state = json.loads(state_path.read_text())

        self.assertEqual(trace_summary["trace_event_count"], 2)
        self.assertEqual(trace_lines[0]["action"], "insert")
        self.assertEqual(trace_lines[1]["action"], "upsize")
        self.assertEqual(final_summary["virtual_buffer_count"], 1)
        self.assertEqual(final_state["committed_buffer_count"], 0)
        self.assertEqual(final_state["entries"][0]["candidate_id"], 4)


if __name__ == "__main__":
    unittest.main()
