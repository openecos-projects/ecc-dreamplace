"""Register sparse Liberty quantities with the existing piecewise evaluator."""

from .cell_modeling import DATASET_SPECS, PIECEWISE_LINEAR_SCHEMA


def register_piecewise_lut_dataset(model, name, luts_attr, *, arc_indices, **metadata):
    """Keep global Liberty arc IDs when a quantity covers only selected rows."""
    if model.cell_model_schema != PIECEWISE_LINEAR_SCHEMA:
        raise ValueError("additional Liberty datasets require the piecewise cell model")
    specs = getattr(model, "dataset_specs", DATASET_SPECS)
    if name in specs:
        raise ValueError(f"cell-model dataset is already registered: {name}")
    state = model._build_piecewise_state(**metadata)
    table = state["arc_index_table"]
    valid = table >= 0
    table[valid] = arc_indices.to(table.device)[table[valid]]
    model.dataset_specs = {**specs, name: {"luts_attr": luts_attr}}
    model.piecewise_state[name] = state
