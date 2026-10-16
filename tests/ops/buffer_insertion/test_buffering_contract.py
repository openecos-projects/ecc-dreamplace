import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from dreamplace.ops.buffer_insertion.contract import BufferFamilyContract


def _ok_metadata():
    return SimpleNamespace(
        buffer_main_type_index=7,
        buffer_main_type_status="ok",
        flat_libcell_info=[
            [0, 7, 1.0, 0],
            [1, 7, 2.0, 0],
            [2, 9, 1.0, 0],
        ],
        flat_libcell_names=["BUF_X1", "BUF_X2", "NAND_X1"],
        flat_libcell_width=[10, 12, 14],
        flat_libcell_height=[20, 20, 20],
        flat_libcell_leakage=[0.1, 0.2, 0.3],
        cell_id_2_libpin_id_start=[0, 2, 4, 6],
        flat_lib_pin_cap=[0.01, 0.0, 0.02, 0.0, 0.03, 0.0],
        cell_id_2_arc_id_start=[0, 1, 2, 3],
        flat_libarc_info=[[0, 1], [0, 1], [0, 1]],
        f_delay_flat_luts_values=[1.0, 1.0],
        r_delay_flat_luts_values=[1.0, 1.0],
        f_trans_flat_luts_values=[1.0, 1.0],
        r_trans_flat_luts_values=[1.0, 1.0],
    )


class BufferFamilyContractTest(unittest.TestCase):
    def test_ok_contract_reuses_family_local_legal_index(self):
        artifact = BufferFamilyContract(_ok_metadata()).build_artifact()

        self.assertEqual(artifact["status"], "ok")
        self.assertEqual(artifact["buffer_main_type_index"], 7)
        self.assertEqual(artifact["buffer_vt"], 0)
        self.assertEqual(artifact["legal_cell_ids"], [0, 1])
        self.assertEqual(artifact["legal_master_names"], ["BUF_X1", "BUF_X2"])
        self.assertEqual(artifact["legal_size_values"], [1.0, 2.0])
        self.assertEqual(artifact["bsu_semantics"], "diff_sizing_family_local_legal_index")
        self.assertEqual(artifact["legal_cell_ids"][1], artifact["bsu_to_cell_id"]["1"])

    def test_missing_unique_buffer_family_is_unsupported(self):
        metadata = _ok_metadata()
        metadata.buffer_main_type_index = -1
        metadata.buffer_main_type_status = "unsupported_multiple_buffer_families"

        artifact = BufferFamilyContract(metadata).build_artifact()

        self.assertEqual(artifact["status"], "unsupported")
        self.assertIn("unsupported_multiple_buffer_families", artifact["unsupported_reasons"])

    def test_empty_legal_table_is_unsupported(self):
        metadata = _ok_metadata()
        metadata.buffer_main_type_index = 123

        artifact = BufferFamilyContract(metadata).build_artifact()

        self.assertEqual(artifact["status"], "unsupported")
        self.assertIn("empty_buffer_legal_table", artifact["unsupported_reasons"])

    def test_explicit_master_resolves_one_of_the_exported_buffer_families(self):
        metadata = _ok_metadata()
        metadata.buffer_main_type_index = -1
        metadata.buffer_main_type_status = "unsupported_multiple_buffer_families"
        metadata.buffer_main_type_candidate_indices = [7, 11]
        artifact = BufferFamilyContract(metadata, preferred_master_name="BUF_X2").build_artifact()
        expected = BufferFamilyContract(_ok_metadata()).build_artifact()
        self.assertEqual(artifact, expected)

    def test_writes_auditable_json_artifact(self):
        metadata = _ok_metadata()
        output_path = Path(self.id().replace(".", "_") + ".json")
        try:
            artifact = BufferFamilyContract(metadata).write_artifact(output_path)
            loaded = json.loads(output_path.read_text())
        finally:
            output_path.unlink(missing_ok=True)

        self.assertEqual(loaded["status"], "ok")
        self.assertEqual(loaded["legal_cell_ids"], artifact["legal_cell_ids"])


if __name__ == "__main__":
    unittest.main()
