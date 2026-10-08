import unittest

from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState


class CellOscillationStateTest(unittest.TestCase):
    def test_one_undo_is_allowed_but_repeated_reversal_freezes_two_rounds(self):
        state = CellOscillationState()

        for iteration, previous, target in ((0, 1, 2), (1, 2, 1), (2, 1, 2)):
            self.assertFalse(
                state.blocked(phase="timing", iteration=iteration,
                              count=2, device="cpu")[0]
            )
            state.record(phase="timing", iteration=iteration,
                         instance_ids=[0], previous_ids=[previous], target_ids=[target])

        self.assertEqual(state.reversals, 2)
        self.assertEqual(state.repeated_reversals, 1)
        self.assertTrue(state.blocked(phase="timing", iteration=3, count=2, device="cpu")[0])
        self.assertTrue(state.blocked(phase="timing", iteration=4, count=2, device="cpu")[0])
        self.assertFalse(state.blocked(phase="timing", iteration=5, count=2, device="cpu")[0])

    def test_snapshot_restore_removes_stale_freeze_and_preserves_phase_boundary(self):
        state = CellOscillationState()
        state.record(phase="timing", iteration=0,
                     instance_ids=[0], previous_ids=[1], target_ids=[2])
        snapshot = state.snapshot()
        for iteration, previous, target in ((1, 2, 1), (2, 1, 2)):
            state.record(phase="timing", iteration=iteration,
                         instance_ids=[0], previous_ids=[previous], target_ids=[target])
        self.assertTrue(state.blocked(phase="timing", iteration=3, count=1, device="cpu")[0])

        state.restore(snapshot)
        self.assertEqual(state.snapshot(), snapshot)
        self.assertFalse(state.blocked(phase="timing", iteration=3, count=1, device="cpu")[0])
        state.record(phase="recovery", iteration=4,
                     instance_ids=[0], previous_ids=[2], target_ids=[1])
        self.assertEqual(state.repeated_reversals, 0)
        self.assertFalse(state.blocked(phase="recovery", iteration=5, count=1, device="cpu")[0])


if __name__ == "__main__":
    unittest.main()
