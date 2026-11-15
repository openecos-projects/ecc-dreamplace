"""Coordinate capture tests; no CUDA, solver, or rasterization is required."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dreamplace.replay_capture import PlacementReplayCapture


class ReplayCaptureTest(unittest.TestCase):
    def test_optimizer_and_committed_vectors_exclude_fillers_and_preserve_dbu(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = PlacementReplayCapture(tmp, time.monotonic_ns())
            params = SimpleNamespace(scale_factor=0.5, shift_factor=[100, 200], macro_only=False)
            db = SimpleNamespace(num_physical_nodes=2, node_names=[b'a', b'm'], node_is_hard_macro=[False, True],
                                 num_place_blockages=0, fixed_slice=slice(1, 2),
                                 pydb=SimpleNamespace(node_size_x=[10, 40], node_size_y=[20, 50]),
                                 xl=0, yl=0, xh=100, yh=100)
            capture.sample(params, db, np.array([5, 10, 999, 15, 20, 999]), 'iteration')
            capture.sample(params, db, np.array([5, 10, 15, 20]), 'writeback', committed=True)
            frames = [json.loads(p.read_text()) for p in sorted(Path(tmp).glob('*.json'))]
            expected = {
                'inst:a': {'kind': 'instance', 'name': 'a', 'bbox': [110, 230, 120, 250]},
                'inst:m': {'kind': 'macro', 'name': 'm', 'bbox': [120, 240, 160, 290]},
                'die': {'kind': 'die', 'name': 'placement region', 'bbox': [100, 200, 300, 400]},
            }
            self.assertEqual([frame['objects'] for frame in frames], [expected, expected])
            self.assertLessEqual(frames[0]['elapsed_ns'], frames[1]['elapsed_ns'])
            self.assertEqual(frames[1]['state_domain'], 'native_committed')

    def test_solver_blockages_and_zero_area_terminals_are_not_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = PlacementReplayCapture(tmp, time.monotonic_ns())
            params = SimpleNamespace(scale_factor=1, shift_factor=[0, 0], macro_only=False)
            db = SimpleNamespace(num_physical_nodes=3, node_names=[b'a', b'obstacle', b'port'],
                                 node_is_hard_macro=[False, False, False],
                                 num_place_blockages=1, fixed_slice=slice(1, 2),
                                 pydb=SimpleNamespace(node_size_x=[10, 40, 0], node_size_y=[20, 50, 0]),
                                 xl=0, yl=0, xh=100, yh=100)
            capture.sample(params, db, np.array([5, 10, 20, 15, 20, 30]), 'iteration')
            frame = json.loads((Path(tmp) / 'dp-000000.json').read_text())
            self.assertEqual(set(frame['objects']), {'inst:a', 'die'})


if __name__ == '__main__':
    unittest.main()
