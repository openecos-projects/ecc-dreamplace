"""The fixed ICS55 drive-strength input shared by the CLI and ECC step."""

DRIVE_TYPE_TABLE_SCHEMA = "ecc.discrete_buffer_drive_strength.v1"
EXPECTED_DRIVE_TYPE_MASTERS = ("BUFX8H7L", "BUFX12H7L", "BUFX16H7L", "BUFX20H7L")


def build_drive_strength_table(pdk_name, buffer_masters, source_pdk):
    missing = [name for name in EXPECTED_DRIVE_TYPE_MASTERS if name not in buffer_masters]
    if missing:
        raise ValueError(f"case PDK is missing approved buffers: {missing}")
    return {
        "schema": DRIVE_TYPE_TABLE_SCHEMA,
        "source": "case-pdk-buffer-list",
        "library": str(pdk_name),
        "units": {"drive_strength": "relative"},
        "source_pdk": str(source_pdk),
        "buffers": [
            {"drive_type_id": index, "master_name": name, "drive_strength": strength}
            for index, (name, strength) in enumerate(
                zip(EXPECTED_DRIVE_TYPE_MASTERS, (8.0, 12.0, 16.0, 20.0))
            )
        ],
    }
