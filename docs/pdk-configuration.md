# Process-specific placement inputs

Use explicit configuration for `buffering_fixed_buffer_master`,
`buffering_candidate_drive_strength_table`, and `openroad_vt_suffixes`.
No ICS55 masters or H7/9T suffixes are inferred. The external drive table carries
its library identity and ordered `{drive_type_id, master_name, drive_strength}`
rows; ECC writes it from the frozen PDK profile. Masters must exist in the legal
buffer library, with finite positive strengths. Typed candidate buffering
accepts two or more rows and sizes its state tensors from the table.

The legacy option spelling `four_strength_table` is retained for configuration
compatibility; its row count is no longer fixed at four. The two-row regression
exercises candidate-state construction and selection of the configured master.

The OpenROAD bridge uses explicitly configured suffixes only; unmatched cells
remain separate families. The native classifier has a standalone compiled test.
The optional full OpenROAD bridge and CUDA paths still require builds/tests on
machines with those backends enabled. ECC-backend ICS55 v2 is the real-process
integration qualification for this change; arbitrary foundry qualification is
not implied.
