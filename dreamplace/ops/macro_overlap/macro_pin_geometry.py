"""Restore physical macro-pin geometry at both placement lifecycle boundaries."""

import numpy as np
import torch


def restore_macro_pin_halo(params, placedb, data_collections, pos):
    if params.macro_pin_halo_x >= 0:
        with torch.no_grad():
            macro_pin_to_macro = np.searchsorted(
                placedb.movable_macro_idx,
                placedb.pin2node_map[placedb.movable_macro_pins],
            )
            data_collections.node_size_x[placedb.movable_macro_idx] -= torch.tensor(
                placedb.is_pin_lower_x * params.macro_pin_halo_x
                + placedb.is_pin_upper_x * params.macro_pin_halo_x,
                device=pos.device,
            )
            data_collections.node_size_y[placedb.movable_macro_idx] -= torch.tensor(
                placedb.is_pin_lower_y * params.macro_pin_halo_y
                + placedb.is_pin_upper_y * params.macro_pin_halo_y,
                device=pos.device,
            )

            data_collections.pin_offset_x[placedb.movable_macro_pins] -= torch.tensor(
                placedb.is_pin_lower_x[macro_pin_to_macro] * params.macro_pin_halo_x,
                device=pos.device,
            )
            data_collections.pin_offset_y[placedb.movable_macro_pins] -= torch.tensor(
                placedb.is_pin_lower_y[macro_pin_to_macro] * params.macro_pin_halo_y,
                device=pos.device,
            )
            # macro locations

            pos[placedb.movable_slice][placedb.movable_macro_mask] += torch.tensor(
                placedb.is_pin_lower_x * params.macro_pin_halo_x, device=pos.device
            )

            pos[placedb.num_nodes : placedb.num_nodes + placedb.num_movable_nodes][
                placedb.movable_macro_mask
            ] += torch.tensor(placedb.is_pin_lower_y * params.macro_pin_halo_y, device=pos.device)
            params.macro_pin_halo_x = 0
            params.macro_pin_halo_y = 0
