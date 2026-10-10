import json
import logging
import os
from dataclasses import dataclass

from .backend_caps import DEFAULT_BACKEND_CAPS


DEFAULT_VT_CONFIG = [
    ("H7H", "HVT"),
    ("H7L", "LVT"),
    ("H7R", "RVT"),
]

IEDAIO = None
IEDASta = None
IEDADesign = None


def _load_ieda_io_class():
    global IEDAIO
    if IEDAIO is None:
        os.environ["eda_tool"] = "iEDA"
        from tools.iEDA.module.io import IEDAIO as ieda_io_class

        IEDAIO = ieda_io_class
    return IEDAIO


def _load_ieda_sta_class():
    global IEDASta
    if IEDASta is None:
        os.environ["eda_tool"] = "iEDA"
        from tools.iEDA.module.sta import IEDASta as ieda_sta_class

        IEDASta = ieda_sta_class
    return IEDASta


def _load_ieda_design_class():
    global IEDADesign
    if IEDADesign is None:
        os.environ["eda_tool"] = "iEDA"
        from tools.iEDA.data.design import IEDADesign as ieda_design_class

        IEDADesign = ieda_design_class
    return IEDADesign


def bind_ieda_module(ieda_module):
    from tools.iEDA.utility import base as ieda_base

    ieda_base.ieda = ieda_module
    return ieda_module


@dataclass
class IEDAPlaceIOBackend:
    workspace: str
    io: object
    dm_inst: object
    pydb: object
    backend_caps: object = DEFAULT_BACKEND_CAPS


def _read_process_node_from_workspace(workspace_dir):
    config_path = os.path.join(workspace_dir, "config", "workspace.json")
    if not os.path.exists(config_path):
        return ""

    try:
        with open(config_path, "r") as f:
            return str(json.load(f).get("workspace", {}).get("process_node", ""))
    except Exception as ex:
        logging.warning("Failed to read process_node from %s: %s", config_path, ex)
        return ""


def _create_pydb(ieda_module, dm_inst, with_sta, vt_config, process_node):
    try:
        return ieda_module.pydb(dm_inst, with_sta, vt_config, process_node)
    except TypeError as ex:
        if "incompatible function arguments" not in str(ex):
            raise
        logging.warning(
            "ieda.pydb does not accept process_node yet; falling back to legacy 3-arg signature"
        )
        return ieda_module.pydb(dm_inst, with_sta, vt_config)


def _workspace_from_backend_or_path(raw_db_or_workspace):
    if isinstance(raw_db_or_workspace, IEDAPlaceIOBackend):
        return raw_db_or_workspace.workspace
    return str(raw_db_or_workspace)


def _def_input_from_params(params):
    design_inputs = getattr(params, "design_inputs", None) or {}
    return design_inputs.get("def") or getattr(params, "def_input", "")


class PlaceIOFunction:
    @staticmethod
    def make_io(workspace, *args, **kwargs):
        try:
            return _load_ieda_io_class()(workspace, *args, **kwargs)
        except SystemExit as ex:
            raise RuntimeError(
                "failed to initialize iEDA backend; ensure eda_tool=iEDA and "
                "AiEDA/third_party/iEDA/bin is visible in PYTHONPATH"
            ) from ex

    @staticmethod
    def read(params, workspace):
        # iEDA is workspace-first: IEDAIO owns workspace interpretation and
        # produces dm_inst/pydb. params.design_inputs may override the DEF
        # consumed by read_def, but AutoDMP should not reimplement the full
        # AiEDA workspace schema here.
        with_sta = getattr(params, "with_sta", False)
        ieda_io = PlaceIOFunction.make_io(workspace)
        if hasattr(ieda_io, "read_def"):
            ieda_io.read_def(_def_input_from_params(params))
        dm_inst = ieda_io.get_dmInst_ptr()
        vt_config = getattr(params, "ieda_vt_config", DEFAULT_VT_CONFIG)
        process_node = _read_process_node_from_workspace(workspace)
        # with_sta enables iEDA pydb timing export, but backend_caps still mark
        # this timing state as diagnostic until placement parasitics are proven
        # equivalent to the OpenROAD/OpenSTA golden path.
        pydb = _create_pydb(
            ieda_io.ieda,
            dm_inst,
            with_sta,
            vt_config,
            process_node,
        )
        return IEDAPlaceIOBackend(
            workspace=workspace,
            io=ieda_io,
            dm_inst=dm_inst,
            pydb=pydb,
        )

    @staticmethod
    def pydb(raw_db):
        return raw_db.pydb

    @staticmethod
    def dm_inst(raw_db):
        return raw_db.dm_inst

    @staticmethod
    def backend_caps(raw_db=None):
        if isinstance(raw_db, IEDAPlaceIOBackend):
            return raw_db.backend_caps.to_dict()
        return DEFAULT_BACKEND_CAPS.to_dict()

    @staticmethod
    def init_sta(raw_db_or_workspace):
        workspace = _workspace_from_backend_or_path(raw_db_or_workspace)
        ieda_sta = _load_ieda_sta_class()(workspace)
        ieda_sta.init_sta()
        return ieda_sta

    @staticmethod
    def make_design(raw_db_or_workspace):
        workspace = _workspace_from_backend_or_path(raw_db_or_workspace)
        return _load_ieda_design_class()(workspace)

    @staticmethod
    def write_def(raw_db_or_workspace, filename):
        if isinstance(raw_db_or_workspace, IEDAPlaceIOBackend):
            return raw_db_or_workspace.io.def_save(filename)
        return _load_ieda_io_class()(_workspace_from_backend_or_path(raw_db_or_workspace)).def_save(
            filename
        )

    @staticmethod
    def write_tcl(raw_db_or_workspace, filename):
        if isinstance(raw_db_or_workspace, IEDAPlaceIOBackend):
            return raw_db_or_workspace.io.tcl_save(filename)
        return _load_ieda_io_class()(_workspace_from_backend_or_path(raw_db_or_workspace)).tcl_save(
            filename
        )

    @staticmethod
    def apply(raw_db, node_x, node_y):
        return raw_db.io.write_placement_back(raw_db.dm_inst, node_x, node_y)

    @staticmethod
    def apply_sizing(raw_db, cell_ids, cell_master_names):
        return raw_db.io.write_sizing_back(raw_db.dm_inst, cell_ids, cell_master_names)
