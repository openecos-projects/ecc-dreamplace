import unittest
from types import SimpleNamespace

import torch

from torch import nn

from dreamplace.BasicPlace import BasicPlace, PlaceDataCollection
from dreamplace.ops.buffer_insertion.optimization_state import (
    buffer_optimization_grad_stats,
    build_buffer_optimization_state,
    capture_buffer_optimization_snapshot,
    restore_buffer_optimization_snapshot,
)
from dreamplace.ops.buffer_insertion.segment_count_state import build_segment_count_state


class BufferOptimizerOwnershipTest(unittest.TestCase):
    def _line_net(self):
        return {
            "net_id": 41,
            "net_name": "line",
            "driver_pin_id": 0,
            "coordinates": {0: (0, 0), 1: (100, 0)},
            "rc_tree": {
                "root_node_id": 0,
                "children_by_node": {0: [1], 1: []},
                "edge_rc": {(0, 1): {"r": 10.0, "c": 4.0}},
                "node_cap": {0: 0.0, 1: 6.0},
                "sink_nodes": [1],
            },
        }

    def test_place_data_collection_exposes_pydb_metadata_for_buffering_state_builder(self):
        data = PlaceDataCollection.__new__(PlaceDataCollection)
        pydb = SimpleNamespace(dbu=2000, buffer_main_type_index=7)
        placedb = SimpleNamespace(
            pydb=pydb,
            dbu=1000,
            buffer_main_type_index=7,
            buffer_main_type_status="ok",
        )

        data._initialize_buffering_database_handles(placedb)

        self.assertIs(data.pydb, pydb)
        self.assertIs(data.buffering_metadata, pydb)
        self.assertEqual(data.dbu, 1000)
        self.assertEqual(data.buffer_main_type_index, 7)
        self.assertEqual(data.buffer_main_type_status, "ok")

    def test_place_data_collection_tolerates_pydb_without_buffer_attributes(self):
        class FixedPydb:
            __slots__ = ("dbu",)

            def __init__(self):
                self.dbu = 2000

        data = PlaceDataCollection.__new__(PlaceDataCollection)
        pydb = FixedPydb()
        placedb = SimpleNamespace(
            pydb=pydb,
            dbu=1000,
            buffer_main_type_index=-1,
            buffer_main_type_status="unsupported",
        )

        data._initialize_buffering_database_handles(placedb)

        self.assertIs(data.pydb, pydb)
        self.assertIs(data.buffering_metadata, pydb)
        self.assertEqual(data.buffer_main_type_index, -1)
        self.assertEqual(data.buffer_main_type_status, "unsupported")

    def test_place_data_collection_exposes_buffer_variable_getters(self):
        data = PlaceDataCollection.__new__(PlaceDataCollection)
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 2, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.0,
            initial_bsu_index=1.5,
        )

        data.set_buffer_optimization_state(state)

        self.assertIs(data.buffer_optimization_state, state)
        self.assertIs(data.buffer_bu_logits, state.bu_logits)
        self.assertIs(data.buffer_bsu_index_param, state.bsu_index_param)
        self.assertTrue(torch.allclose(data.get_buffer_bu_var(), torch.sigmoid(state.bu_logits)))
        self.assertTrue(torch.allclose(data.get_buffer_bsu_index_var(), torch.tensor([1.5])))
        self.assertEqual(list(data.get_buffer_candidate_node_id().tolist()), [1])

    def test_place_data_collection_buffer_getters_return_none_without_state(self):
        data = PlaceDataCollection.__new__(PlaceDataCollection)
        data.buffer_optimization_state = None
        data.buffer_bu_logits = None
        data.buffer_z_param = None
        data.buffer_bsu_index_param = None

        self.assertIsNone(data.get_buffer_bu_var())
        self.assertIsNone(data.get_buffer_z_var())
        self.assertIsNone(data.get_buffer_bsu_index_var())
        self.assertIsNone(data.get_buffer_candidate_node_id())

    def test_place_data_collection_exposes_segment_count_state(self):
        data = PlaceDataCollection.__new__(PlaceDataCollection)
        state = build_segment_count_state(
            [self._line_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=3,
            initial_z=0.25,
            initial_bsu_index=2.0,
        )

        data.set_buffer_optimization_state(state)

        self.assertIs(data.buffer_optimization_state, state)
        self.assertIsNone(data.buffer_bu_logits)
        self.assertIs(data.buffer_z_param, state.z_param)
        self.assertIs(data.buffer_bsu_index_param, state.bsu_index_param)
        self.assertIsNone(data.get_buffer_bu_var())
        self.assertIsNone(data.get_buffer_candidate_node_id())
        self.assertTrue(torch.allclose(data.get_buffer_z_var(), torch.tensor([0.25])))
        self.assertTrue(torch.allclose(data.get_buffer_bsu_index_var(), torch.tensor([2.0])))

    def test_segment_timing_epoch_advances_only_for_new_state_or_payload(self):
        data = PlaceDataCollection.__new__(PlaceDataCollection)
        data.buffer_segment_timing_topology_epoch = 0
        data.buffer_segment_timing_topology_epoch_reason = "initial"
        state = build_segment_count_state(
            [self._line_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=3,
        )
        payload = {
            "nets": [self._line_net()],
            "per_size_input_cap": torch.ones((1, 2)),
            "per_size_delay": torch.ones((1, 2)),
            "per_size_output_slew": torch.ones((1, 2)),
        }

        self.assertTrue(
            data.install_buffer_segment_timing_state(
                state,
                payload,
                reason="first_segment_state",
            )
        )
        self.assertEqual(data.buffer_segment_timing_topology_epoch, 1)
        self.assertEqual(
            data.buffer_segment_timing_topology_epoch_reason,
            "first_segment_state",
        )
        self.assertFalse(
            data.install_buffer_segment_timing_state(
                state,
                payload,
                reason="duplicate_attach",
            )
        )
        self.assertEqual(data.buffer_segment_timing_topology_epoch, 1)

        rebuilt_payload = dict(payload)
        self.assertTrue(
            data.install_buffer_segment_timing_state(
                state,
                rebuilt_payload,
                reason="rebuilt_payload",
            )
        )
        self.assertEqual(data.buffer_segment_timing_topology_epoch, 2)
        self.assertEqual(
            data.buffer_segment_timing_topology_epoch_reason,
            "rebuilt_payload",
        )

    def test_basic_place_registers_buffer_state_parameters(self):
        basic_place = BasicPlace.__new__(BasicPlace)
        nn.Module.__init__(basic_place)
        basic_place.buffer_params = nn.ParameterList()
        basic_place.data_collections = PlaceDataCollection.__new__(PlaceDataCollection)
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 2, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
        )

        basic_place.register_buffer_optimization_state(state)

        self.assertIs(basic_place.data_collections.buffer_optimization_state, state)
        self.assertEqual(len(list(basic_place.buffer_params.parameters())), 2)
        self.assertIs(list(basic_place.buffer_params.parameters())[0], state.bu_logits)
        self.assertIs(list(basic_place.buffer_params.parameters())[1], state.bsu_index_param)

    def test_basic_place_registers_segment_count_buffer_parameters(self):
        basic_place = BasicPlace.__new__(BasicPlace)
        nn.Module.__init__(basic_place)
        basic_place.buffer_params = nn.ParameterList()
        basic_place.data_collections = PlaceDataCollection.__new__(PlaceDataCollection)
        state = build_segment_count_state(
            [self._line_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=3,
        )

        basic_place.register_buffer_optimization_state(state)

        self.assertIs(basic_place.data_collections.buffer_optimization_state, state)
        self.assertEqual(len(list(basic_place.buffer_params.parameters())), 2)
        self.assertIs(list(basic_place.buffer_params.parameters())[0], state.z_param)
        self.assertIs(list(basic_place.buffer_params.parameters())[1], state.bsu_index_param)

    def test_buffer_snapshot_and_grad_stats_include_both_trainable_variables(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 2, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.25,
            initial_bsu_index=1.5,
        )

        loss = state.bu_logits.sum() + state.bsu_index_param.sum()
        loss.backward()

        snapshot = capture_buffer_optimization_snapshot(state)
        grad_stats = buffer_optimization_grad_stats(state)

        self.assertTrue(torch.allclose(snapshot["buffer_bu_logits"], state.bu_logits.detach()))
        self.assertTrue(
            torch.allclose(
                snapshot["buffer_bsu_index_param"],
                state.bsu_index_param.detach(),
            )
        )
        self.assertTrue(torch.allclose(snapshot["buffer_bu_var"], state.relaxed_bu().detach()))
        self.assertTrue(
            torch.allclose(
                snapshot["buffer_bsu_index_var"],
                state.bsu_index().detach(),
            )
        )
        self.assertGreater(grad_stats["buffer_bu_grad_norm"], 0.0)
        self.assertIsNone(grad_stats["buffer_z_grad_norm"])
        self.assertGreater(grad_stats["buffer_bsu_index_grad_norm"], 0.0)

    def test_segment_snapshot_and_grad_stats_include_z_parameter(self):
        state = build_segment_count_state(
            [self._line_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=3,
            initial_z=0.25,
            initial_bsu_index=2.0,
        )

        loss = state.z_param.sum() + state.bsu_index_param.sum()
        loss.backward()

        snapshot = capture_buffer_optimization_snapshot(state)
        grad_stats = buffer_optimization_grad_stats(state)

        self.assertIsNone(snapshot["buffer_bu_logits"])
        self.assertIsNone(snapshot["buffer_bu_var"])
        self.assertTrue(torch.allclose(snapshot["buffer_z_param"], state.z_param.detach()))
        self.assertTrue(torch.allclose(snapshot["buffer_z_var"], state.z_value().detach()))
        self.assertTrue(
            torch.allclose(
                snapshot["buffer_bsu_index_param"],
                state.bsu_index_param.detach(),
            )
        )
        self.assertIsNone(grad_stats["buffer_bu_grad_norm"])
        self.assertGreater(grad_stats["buffer_z_grad_norm"], 0.0)
        self.assertGreater(grad_stats["buffer_bsu_index_grad_norm"], 0.0)

    def test_buffer_snapshot_restore_restores_both_trainable_variables(self):
        state = build_buffer_optimization_state(
            [
                {"candidate_id": 0, "tree_node_id": 1, "net_id": 2, "buffer_main_type_index": 1},
            ],
            buffer_main_type_index=1,
            legal_buffer_count=3,
            initial_bu_logit=0.25,
            initial_bsu_index=1.5,
        )

        best_state = capture_buffer_optimization_snapshot(state)
        with torch.no_grad():
            state.bu_logits.add_(10.0)
            state.bsu_index_param.add_(10.0)

        summary = restore_buffer_optimization_snapshot(state, best_state)

        self.assertTrue(summary["restored"])
        self.assertTrue(torch.allclose(state.bu_logits, best_state["buffer_bu_logits"]))
        self.assertTrue(
            torch.allclose(
                state.bsu_index_param,
                best_state["buffer_bsu_index_param"],
            )
        )

    def test_segment_snapshot_restore_restores_z_and_bsu_variables(self):
        state = build_segment_count_state(
            [self._line_net()],
            buffer_main_type_index=7,
            legal_buffer_count=8,
            max_repeater_count=3,
            initial_z=0.25,
            initial_bsu_index=2.0,
        )

        best_state = capture_buffer_optimization_snapshot(state)
        with torch.no_grad():
            state.z_param.add_(2.0)
            state.bsu_index_param.add_(3.0)

        summary = restore_buffer_optimization_snapshot(state, best_state)

        self.assertTrue(summary["restored"])
        self.assertIn("buffer_z_param", summary["restored_keys"])
        self.assertTrue(torch.allclose(state.z_param, best_state["buffer_z_param"]))
        self.assertTrue(
            torch.allclose(
                state.bsu_index_param,
                best_state["buffer_bsu_index_param"],
            )
        )


if __name__ == "__main__":
    unittest.main()
