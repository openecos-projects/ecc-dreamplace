"""Built-in GPUGR backend for ecc-dreamplace, driven by the bundled Xplace fork.

This replaces the external ``tools.iEDA.module.gpugr`` dependency. The Xplace
fork lives at ``<ecc-dreamplace>/thirdparty/xplace`` (git submodule, branch
feature/ggr-lshape); wheel builds compile its native extensions against the
same Torch as DreamPlace. ``thirdparty/build_xplace_gpugr.sh`` is available
for standalone native development.

Ported from the pinned AiEDA reference implementation
(tools/iEDA/module/gpugr.py @ 9b8aa46, sha256 b3981e71...) with three
architectural replacements:

1. DEF handoff is explicit: the current ECC database is exported via
   ``ecc_module.def_save(tmp_def)`` instead of relying on a same-process
   iEDA session. On a parser-cache hit neither the caller's write-back nor
   this export happens, so repeated inflation rounds do not re-serialize DEF.
2. LEF list / design name / output dirs come from ECC params/placedb
   (``params.lef_input``, ``params.design_name()``, ``params.result_dir``);
   no AiEDA workspace modules are involved.
3. Xplace is located at the submodule path and imported lazily with sys.path
   injection; missing build artifacts raise a clear error pointing at the
   build script.
"""

import gzip
import json
import logging
import os
import shutil
import sys
import sysconfig
import tempfile
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .result_maps import aggregate_layer_maps, summarize_congestion_tensor, validate_map_planes
from .xplace_native_output import XplaceNativeOutputMixin
from .xplace_parser_cache import XplaceParserCacheMixin

logger = logging.getLogger(__name__)

SUPPORTED_GPUGR_BACKENDS = ("cuda", "cpu_pr", "cpu_pr_mt", "auto")


def normalize_gpugr_backend(backend: str = "auto") -> str:
    normalized = str(backend or "auto").strip().lower()
    if normalized not in SUPPORTED_GPUGR_BACKENDS:
        raise ValueError(
            f"Unsupported gpugr backend '{backend}'. "
            f"Supported backends: {', '.join(SUPPORTED_GPUGR_BACKENDS)}"
        )
    return normalized


def _extension_cuda_enabled(gpugr_module) -> bool:
    if gpugr_module is None:
        return bool(torch.backends.cuda.is_built() and torch.version.cuda is not None)
    if hasattr(gpugr_module, "cuda_enabled"):
        return bool(gpugr_module.cuda_enabled())
    if hasattr(gpugr_module, "is_cuda_enabled"):
        return bool(gpugr_module.is_cuda_enabled())
    return bool(torch.backends.cuda.is_built() and torch.version.cuda is not None)


def resolve_gpugr_backend(
    backend: str = "auto", cuda_available: bool = None, extension_cuda_enabled: bool = None
) -> str:
    normalized = normalize_gpugr_backend(backend)
    cuda_available = torch.cuda.is_available() if cuda_available is None else bool(cuda_available)
    extension_cuda_enabled = (
        bool(torch.backends.cuda.is_built() and torch.version.cuda is not None)
        if extension_cuda_enabled is None
        else bool(extension_cuda_enabled)
    )

    cuda_ready = cuda_available and extension_cuda_enabled
    if normalized == "auto":
        resolved = "cuda" if cuda_ready else "cpu_pr_mt"
        if resolved == "cpu_pr_mt" and not extension_cuda_enabled:
            logger.warning(
                "gpugr backend=auto resolved to cpu_pr_mt because the gpugr extension "
                "was built without CUDA"
            )
        return resolved
    if normalized == "cuda" and not cuda_ready:
        raise RuntimeError(
            "gpugr backend=cuda requested, but CUDA support is unavailable "
            f"(torch_cuda_available={cuda_available}, "
            f"extension_cuda_enabled={extension_cuda_enabled})"
        )
    return normalized


def validate_gpugr_backend_request(backend: str, rrr_iters: int):
    normalized = normalize_gpugr_backend(backend)
    if normalized in ("cpu_pr", "cpu_pr_mt") and int(rrr_iters) > 0:
        raise RuntimeError(
            f"gpugr backend={normalized} only supports one CPU routing pass with "
            f"rrr_iters=0; got rrr_iters={int(rrr_iters)}. CPU RRR is unsupported "
            "in this phase."
        )


def gpugr_run_metadata(
    requested_backend: str,
    resolved_backend: str,
    route_xsize: int,
    route_ysize: int,
    rrr_iters: int,
    skip_m1_route: bool,
    routing_layer_range=None,
    routing_layer_names=None,
) -> dict:
    """Return JSON-safe configuration evidence for one GPUGR invocation."""
    metadata = {
        "route_xsize": int(route_xsize),
        "route_ysize": int(route_ysize),
        "rrr_iters": int(rrr_iters),
        "skip_m1_route": bool(skip_m1_route),
        "gpugr_backend_requested": str(requested_backend),
        "gpugr_backend": str(resolved_backend),
    }
    if routing_layer_range is not None:
        begin, end = routing_layer_range
        metadata["routing_layer_begin"] = int(begin)
        metadata["routing_layer_end"] = int(end)
    if routing_layer_names is not None:
        metadata["routing_layer_names"] = [str(name) for name in routing_layer_names]
        if routing_layer_range is not None:
            begin, end = routing_layer_range
            metadata["enabled_routing_layer_names"] = [
                str(name) for name in routing_layer_names[int(begin) : int(end) + 1]
            ]
    return metadata


def _install_optional_route_force_stubs():
    """Stub seaborn/torchvision when absent so src.core.route_force imports.

    route_force.py imports both at module level but only uses them in the
    plotting helper and the placement-only filler pseudo force — neither is
    reachable from the GPUGR backend path (calc_gr_wl_via /
    estimate_num_shorts are pure torch). Stubbing keeps the ECC venv free of
    two heavy unused dependencies. Never overwrites a real installation.
    """
    try:
        import seaborn  # noqa: F401
    except ImportError:
        sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
    try:
        import torchvision.transforms  # noqa: F401
    except ImportError:
        torchvision = sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
        transforms = types.ModuleType("torchvision.transforms")
        torchvision.transforms = transforms
        sys.modules.setdefault("torchvision.transforms", transforms)


def _preload_libpython():
    """Preload the shared libpython so extensions that link it can be dlopened.

    xplace builds io_parser with pybind11_add_module(... SHARED), which records
    a DT_NEEDED on libpython3.11.so.1.0. The uv-managed ECC venv interpreter is
    statically linked, so the loader cannot resolve that soname unless the
    library is already present in the process. Mirrors torch's own global
    preload strategy; a no-op when libpython is static or already loaded.
    """
    import ctypes
    import sysconfig

    if str(sysconfig.get_config_var("Py_ENABLE_SHARED")) != "1":
        return
    libdir = sysconfig.get_config_var("LIBDIR")
    ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if not libdir or not ldlibrary:
        return
    candidate = Path(libdir) / ldlibrary
    if candidate.exists():
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


class XplaceGPUGR(XplaceParserCacheMixin, XplaceNativeOutputMixin):
    """Run Xplace gpugr on the current ECC placement via an explicit DEF handoff."""

    def __init__(self, params, placedb):
        self.params = params
        self.placedb = placedb
        self._parser_db_cache = None
        self._xplace_root = None
        self._xplace_modules = None
        self._flute_lut_paths = None
        self._flute_register_cache = set()
        self._last_routing_layer_range = None
        self._last_routing_layer_names = None

    @property
    def _ecc_module(self):
        module = self.placedb.data_manager
        if not hasattr(module, "def_save"):
            raise RuntimeError(
                "Xplace GPUGR backend requires the ECC runtime module (with def_save); "
                "placedb.data_manager does not provide one"
            )
        return module

    @staticmethod
    def _xplace_root_candidates():
        # 1. explicit override
        env_root = os.environ.get("ECC_XPLACE_ROOT")
        if env_root:
            yield Path(env_root).expanduser()
        # 2. wheel runtime, including native artifacts for inplace editable sources
        yield Path(sysconfig.get_path("platlib")) / "thirdparty" / "xplace"
        # 3. standalone development without an installed wheel
        yield Path(__file__).resolve().parents[3] / "thirdparty" / "xplace"
        # 4. running from the installed (editable) package: the cmake-configured
        #    source dir recorded in dreamplace/configure.py
        try:
            import dreamplace.configure as configure

            source_dir = configure.compile_configurations.get("DREAMPLACE_SOURCE_DIR")
            if source_dir:
                yield Path(source_dir) / "thirdparty" / "xplace"
        except Exception:
            pass

    def _ensure_xplace_python_path(self):
        if self._xplace_root is not None:
            return self._xplace_root
        tried = []
        source_root = None
        for candidate in self._xplace_root_candidates():
            candidate = candidate.resolve()
            if candidate in tried:
                continue
            tried.append(candidate)
            if (candidate / "cpp_to_py").is_dir():
                if source_root is None:
                    source_root = candidate
                if not any((candidate / "cpp_to_py" / "cpybin").glob("gpugr*.so")):
                    continue
                xplace_root_str = str(candidate)
                if xplace_root_str not in sys.path:
                    sys.path.insert(0, xplace_root_str)
                self._xplace_root = candidate
                return candidate
        if source_root is not None:
            self._xplace_root = source_root
            return source_root
        raise RuntimeError(
            "Xplace submodule not found; tried: "
            + ", ".join(str(path) for path in tried)
            + ". Run `git submodule update --init --recursive`, or set "
            "ECC_XPLACE_ROOT to the xplace checkout."
        )

    def _resolve_flute_lut_paths(self):
        if self._flute_lut_paths is not None:
            return self._flute_lut_paths
        xplace_root = self._ensure_xplace_python_path()
        powv_path = (xplace_root / "thirdparty" / "flute" / "POWV9.dat").resolve()
        post_path = (xplace_root / "thirdparty" / "flute" / "POST9.dat").resolve()
        if not powv_path.exists():
            raise FileNotFoundError(f"Flute POWV LUT not found: {powv_path}")
        if not post_path.exists():
            raise FileNotFoundError(f"Flute POST LUT not found: {post_path}")
        self._flute_lut_paths = (str(powv_path), str(post_path))
        return self._flute_lut_paths

    @staticmethod
    def _load_xplace_python_modules(
        xplace_root, gpugr_module=None, use_cuda=None
    ):
        """Load the xplace Python helpers without executing package __init__ files.

        ``src/__init__.py`` and ``utils/__init__.py`` pull in placement-only
        dependencies (pulp, visualization stacks) that the GPUGR path never
        uses and the ECC venv does not install. ``route_force.py`` reaches
        ``utils.logger`` through an absolute import, so register a shell for
        the top-level ``utils`` name as well: the import machinery then loads
        ``utils/logger.py`` directly without executing ``utils/__init__.py``.
        The modules we need are self-contained apart from relative imports
        inside ``src/core``, so we load them under private package names whose
        __path__ points at the real directories.
        """
        import importlib.util

        if gpugr_module is None:
            from cpp_to_py import gpugr as gpugr_module
        if use_cuda is None:
            use_cuda = _extension_cuda_enabled(gpugr_module)

        def load_file(qualname, path):
            spec = importlib.util.spec_from_file_location(qualname, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualname] = module
            spec.loader.exec_module(module)
            return module

        def shell_package(qualname, path):
            module = sys.modules.get(qualname)
            if module is None:
                module = types.ModuleType(qualname)
                module.__path__ = [str(path)]
                sys.modules[qualname] = module
            return module

        io_parser_module = load_file(
            "_xplace_utils_io_parser", xplace_root / "utils" / "io_parser.py"
        )
        shell_package("utils", xplace_root / "utils")
        shell_package("_xplace_src", xplace_root / "src")
        shell_package("_xplace_src.core", xplace_root / "src" / "core")
        # The native GPUGR extension can be built with CUDA support while the
        # active Torch runtime is CPU-only.  RouteForce imports dct_cuda at
        # module load time, so key this compatibility stub off the resolved
        # backend instead of the extension's compile-time flag.
        if not use_cuda:
            dct_module = types.ModuleType("_xplace_src.core.dct2_fft2")

            def cuda_only_dct(*_args, **_kwargs):
                raise RuntimeError("Xplace DCT operators require a CUDA-enabled torch build")

            for name in ("dct2", "idct2", "idxst_idct", "idct_idxst"):
                setattr(dct_module, name, cuda_only_dct)
            sys.modules[dct_module.__name__] = dct_module
        flute_module = load_file(
            "_xplace_src.core.flute", xplace_root / "src" / "core" / "flute.py"
        )
        route_force_module = load_file(
            "_xplace_src.core.route_force", xplace_root / "src" / "core" / "route_force.py"
        )

        return (
            io_parser_module.IOParser,
            gpugr_module,
            flute_module.Flute,
            route_force_module.calc_gr_wl_via,
            route_force_module.estimate_num_shorts,
        )

    def _import_xplace_modules(self, resolved_backend=None):
        if self._xplace_modules is not None:
            return self._xplace_modules
        xplace_root = self._ensure_xplace_python_path()
        cpybin = xplace_root / "cpp_to_py" / "cpybin"
        if not cpybin.is_dir() or not any(cpybin.glob("gpugr*.so")):
            raise RuntimeError(
                f"Xplace gpugr extensions are not built (missing gpugr*.so under {cpybin}); "
                "build them with thirdparty/build_xplace_gpugr.sh against the ECC venv torch"
            )
        _install_optional_route_force_stubs()
        _preload_libpython()
        try:
            # Import only the native GPUGR extension first.  Resolving the
            # requested backend must happen before loading route_force.py,
            # whose import-time dct_cuda dependency is CUDA-only.
            from cpp_to_py import gpugr as gpugr_module

            if resolved_backend is None:
                resolved_backend = resolve_gpugr_backend(
                    "auto",
                    cuda_available=torch.cuda.is_available(),
                    extension_cuda_enabled=_extension_cuda_enabled(gpugr_module),
                )
            self._xplace_modules = self._load_xplace_python_modules(
                xplace_root,
                gpugr_module,
                use_cuda=resolved_backend == "cuda",
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import Xplace gpugr modules. "
                "Make sure thirdparty/xplace is built (thirdparty/build_xplace_gpugr.sh) "
                "and cpp_to_py/cpybin is available."
            ) from exc
        return self._xplace_modules

    def _flute_register_key(self, threads: int, powv_path: str, post_path: str):
        return (int(threads), str(powv_path), str(post_path))

    def _flute_register_cached(self, threads: int, powv_path: str, post_path: str):
        return self._flute_register_key(threads, powv_path, post_path) in self._flute_register_cache

    def _register_flute_once(self, Flute, threads: int, powv_path: str, post_path: str):
        key = self._flute_register_key(threads, powv_path, post_path)
        if key in self._flute_register_cache:
            if getattr(Flute, "num_threads", None) != int(threads):
                Flute.num_threads = int(threads)
            return False
        Flute.register(int(threads), powv_path, post_path)
        self._flute_register_cache.add(key)
        return True

    def _dedup_paths(self, paths):
        unique_paths = []
        seen = set()
        for path in paths:
            if not path:
                continue
            resolved = str(Path(path).expanduser().resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_paths.append(resolved)
        return unique_paths

    def _resolve_lefs(self):
        lef_input = getattr(self.params, "lef_input", None) or []
        if isinstance(lef_input, (str, Path)):
            lef_input = [lef_input]
        lefs = self._dedup_paths(lef_input)
        if not lefs:
            raise ValueError("No LEF files found in params.lef_input.")
        for lef in lefs:
            if not Path(lef).exists():
                raise FileNotFoundError(f"LEF not found: {lef}")
        return lefs

    def _infer_design_name(self, def_path: str):
        name = Path(def_path).name
        if name.endswith(".def.gz"):
            return name[:-7]
        if name.endswith(".def"):
            return name[:-4]
        return Path(def_path).stem

    def _resolve_output_dir(self, out_dir: str = ""):
        if out_dir:
            result_dir = Path(out_dir).expanduser().resolve()
        else:
            result_dir = Path(self.params.result_dir).expanduser().resolve() / "gpugr_operator"
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir

    def _materialize_def(self, def_path: str, work_dir: Path):
        def_file = Path(def_path).expanduser().resolve()
        if not def_file.exists():
            raise FileNotFoundError(f"DEF not found: {def_path}")
        if def_file.suffix != ".gz":
            return str(def_file), None

        dst_path = work_dir / def_file.name[:-3]
        with gzip.open(def_file, "rb") as f_src, open(dst_path, "wb") as f_dst:
            shutil.copyfileobj(f_src, f_dst)
        return str(dst_path), str(def_file)

    def _export_current_def(self, work_dir: Path, design_name: str):
        export_path = work_dir / f"{design_name}_gpugr.def"
        self._ecc_module.def_save(str(export_path))
        if not export_path.exists() or export_path.stat().st_size == 0:
            raise RuntimeError(f"Failed to export DEF from current ECC DB: {export_path}")
        return str(export_path)

    def _build_params(self, benchmark: str, design_name: str, def_path: str, lefs):
        return {
            "benchmark": benchmark,
            "design_name": design_name,
            "def": def_path,
            "lefs": lefs,
        }

    def _normalize_name(self, name):
        if isinstance(name, bytes):
            return name.decode("utf-8")
        if hasattr(name, "decode"):
            try:
                return name.decode("utf-8")
            except Exception:
                pass
        return str(name)

    @staticmethod
    def _routing_layer_metadata(routeforce, layer_count):
        default_range = (0, max(0, int(layer_count) - 1))
        range_getter = getattr(routeforce, "routing_layer_range", None)
        if not callable(range_getter):
            return default_range, []
        raw_range = range_getter()
        if raw_range is None or len(raw_range) != 2:
            raise RuntimeError(f"Invalid GPUGR routing layer range: {raw_range!r}")
        begin, end = int(raw_range[0]), int(raw_range[1])
        if begin < 0 or end >= int(layer_count) or begin > end:
            raise RuntimeError(
                f"GPUGR routing layer range [{begin}, {end}] is invalid for "
                f"{int(layer_count)} layers"
            )
        names_getter = getattr(routeforce, "routing_layer_names", None)
        names = list(names_getter()) if callable(names_getter) else []
        if names and len(names) != int(layer_count):
            raise RuntimeError(
                "GPUGR routing layer name count does not match map layer count: "
                f"{len(names)} != {int(layer_count)}"
            )
        return (begin, end), names

    @staticmethod
    def _aggregate_layer_maps(demand_map, capacity_map, layer_ids):
        return aggregate_layer_maps(demand_map, capacity_map, layer_ids)

    def _compute_maps(self, routeforce, gpdb, skip_m1_route=True):
        dmd_map, wire_dmd_map, via_dmd_map = routeforce.dmd_map()
        cap_map = routeforce.cap_map()
        raw_wire_dmd_map = routeforce.raw_wire_dmd_map()
        fix_usage_map = routeforce.fix_usage_map()
        mov_usage_map = routeforce.mov_usage_map()

        map_device = dmd_map.device
        wire_dmd_map = wire_dmd_map.to(device=map_device)
        via_dmd_map = via_dmd_map.to(device=map_device)
        cap_map = cap_map.to(device=map_device)
        raw_wire_dmd_map = raw_wire_dmd_map.to(device=map_device)
        fix_usage_map = fix_usage_map.to(device=map_device)
        mov_usage_map = mov_usage_map.to(device=map_device)

        routing_layer_range, routing_layer_names = self._routing_layer_metadata(
            routeforce, dmd_map.size(0)
        )
        self._last_routing_layer_range = routing_layer_range
        self._last_routing_layer_names = routing_layer_names
        layer_ids = torch.arange(dmd_map.size(0), device=map_device)
        enabled_layers = (layer_ids >= routing_layer_range[0]) & (
            layer_ids <= routing_layer_range[1]
        )

        # Keep the layer dimension stable for callers, but make the native
        # routing window explicit in every map consumed by DreamPlace.
        layer_mask = enabled_layers.view(-1, 1, 1)
        zero = torch.zeros_like(dmd_map)
        dmd_map = torch.where(layer_mask, dmd_map, zero)
        wire_dmd_map = torch.where(layer_mask, wire_dmd_map, torch.zeros_like(wire_dmd_map))
        via_dmd_map = torch.where(layer_mask, via_dmd_map, torch.zeros_like(via_dmd_map))
        raw_wire_dmd_map = torch.where(
            layer_mask, raw_wire_dmd_map, torch.zeros_like(raw_wire_dmd_map)
        )
        fix_usage_map = torch.where(layer_mask, fix_usage_map, torch.zeros_like(fix_usage_map))
        mov_usage_map = torch.where(layer_mask, mov_usage_map, torch.zeros_like(mov_usage_map))
        cap_map = torch.where(layer_mask, cap_map, torch.zeros_like(cap_map))

        m1direction = gpdb.m1direction()
        h_id = 1 if m1direction else 0
        v_id = 0 if m1direction else 1
        route_layers = enabled_layers.clone()
        if skip_m1_route:
            route_layers &= layer_ids != 0
            if h_id == 0:
                h_id += 2
            if v_id == 0:
                v_id += 2
        h_layers = route_layers & ((layer_ids % 2) == (h_id % 2))
        v_layers = route_layers & ((layer_ids % 2) == (v_id % 2))

        h_ids = torch.nonzero(h_layers, as_tuple=False).flatten()
        v_ids = torch.nonzero(v_layers, as_tuple=False).flatten()
        union_ids = torch.nonzero(route_layers, as_tuple=False).flatten()
        cg_map_h_raw = self._aggregate_layer_maps(dmd_map, cap_map, h_ids)
        cg_map_v_raw = self._aggregate_layer_maps(dmd_map, cap_map, v_ids)
        cg_map_union_raw = self._aggregate_layer_maps(dmd_map, cap_map, union_ids)
        cg_map_h_overflow = torch.clamp(cg_map_h_raw - 1.0, min=0.0)
        cg_map_v_overflow = torch.clamp(cg_map_v_raw - 1.0, min=0.0)
        cg_map_union_overflow = torch.clamp(cg_map_union_raw - 1.0, min=0.0)

        effective_demand_map = wire_dmd_map + via_dmd_map
        capacity_floor = torch.finfo(cap_map.dtype).eps
        effective_capacity_map = torch.clamp(
            cap_map - fix_usage_map - mov_usage_map,
            min=capacity_floor,
        )
        cg_map_h_effective_raw = self._aggregate_layer_maps(
            effective_demand_map, effective_capacity_map, h_ids
        )
        cg_map_v_effective_raw = self._aggregate_layer_maps(
            effective_demand_map, effective_capacity_map, v_ids
        )
        cg_map_h_effective_overflow = torch.clamp(
            cg_map_h_effective_raw - 1.0,
            min=0.0,
        )
        cg_map_v_effective_overflow = torch.clamp(
            cg_map_v_effective_raw - 1.0,
            min=0.0,
        )

        maps = {
            "cg_map_h_raw": cg_map_h_raw,
            "cg_map_v_raw": cg_map_v_raw,
            "cg_map_union_raw": cg_map_union_raw,
            "cg_map_h_overflow": cg_map_h_overflow,
            "cg_map_v_overflow": cg_map_v_overflow,
            "cg_map_union_overflow": cg_map_union_overflow,
            "cg_map_h_effective_raw": cg_map_h_effective_raw,
            "cg_map_v_effective_raw": cg_map_v_effective_raw,
            "cg_map_h_effective_overflow": cg_map_h_effective_overflow,
            "cg_map_v_effective_overflow": cg_map_v_effective_overflow,
            "dmd_map": dmd_map,
            "demand_map": dmd_map,
            "raw_wire_demand_map": raw_wire_dmd_map,
            "wire_demand_map": wire_dmd_map,
            "via_demand_map": via_dmd_map,
            "fix_usage_map": fix_usage_map,
            "mov_usage_map": mov_usage_map,
            "capacity_map": cap_map,
        }
        validate_map_planes(maps, expected_layers=int(dmd_map.size(0)))
        return maps

    def _add_metric_aliases(self, metrics: dict):
        if "gr_num_vias" in metrics:
            metrics.setdefault("gr_vias", metrics["gr_num_vias"])
        return metrics

    def _summarize_congestion_tensor(self, tensor: torch.Tensor, overflow_threshold: float):
        return summarize_congestion_tensor(tensor, overflow_threshold)

    def _format_profile_fields(self, fields: dict):
        parts = []
        for key, value in fields.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return " " + " ".join(parts) if parts else ""

    def _emit_profile_record(self, name: str, elapsed_ms: float, fields: dict):
        logging.info(
            "[L-shape profile] %s elapsed=%.3fms%s",
            name,
            elapsed_ms,
            self._format_profile_fields(fields),
        )

    def _flush_profile_records(self, records: list):
        while records:
            name, elapsed_ms, fields = records.pop(0)
            self._emit_profile_record(name, elapsed_ms, fields)

    def _flush_grdatabase_setup_profile(self, grdb, enabled: bool, profile_prefix: str):
        if not enabled or not hasattr(grdb, "setup_profile"):
            return
        try:
            records = grdb.setup_profile()
        except Exception:
            logging.exception("failed to read gpugr GRDatabase setup profile")
            return
        for record in records or []:
            if not isinstance(record, dict):
                continue
            phase_name = record.get("name")
            if not phase_name:
                continue
            elapsed_ms = float(record.get("elapsed_ms", 0.0))
            fields = {
                key: value for key, value in record.items() if key not in ("name", "elapsed_ms")
            }
            self._emit_profile_record(
                f"{profile_prefix}.create_grdatabase.{phase_name}",
                elapsed_ms,
                fields,
            )

    @contextmanager
    def _profile_phase(self, enabled: bool, name: str, profile_records: list = None, **fields):
        if not enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if profile_records is not None:
                profile_records.append((name, elapsed_ms, fields))
            else:
                self._emit_profile_record(name, elapsed_ms, fields)

    def _profile_phase_kwargs(self, enabled: bool, profile_records: list):
        if not enabled:
            return {}
        return {"profile_records": profile_records}

    def _save_maps(self, maps_path: Path, maps: dict):
        np.savez_compressed(
            maps_path, **{key: value.detach().cpu().numpy() for key, value in maps.items()}
        )

    def _save_metrics(self, metrics_path: Path, metrics: dict):
        with open(metrics_path, "w", encoding="utf-8") as f_metrics:
            json.dump(metrics, f_metrics, indent=2)

    def _save_route_entries(self, route_entries_path: Path, route_entries: list):
        with open(route_entries_path, "w", encoding="utf-8") as f_routes:
            json.dump(route_entries, f_routes, indent=2)

    def _save_heatmap_png(self, png_path: Path, tensor: torch.Tensor, title: str):
        import matplotlib.pyplot as plt

        png_path.parent.mkdir(parents=True, exist_ok=True)
        image = tensor.detach().cpu()
        if image.dim() == 3:
            image = image.sum(dim=0)
        image = image.t().flip(0).numpy()

        plt.figure(figsize=(8, 6))
        plt.imshow(image, origin="lower", aspect="auto", cmap="YlGnBu")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    def _save_pngs(self, png_dir: Path, design_name: str, maps: dict):
        png_dir.mkdir(parents=True, exist_ok=True)
        png_specs = (
            ("cg_map_h_raw", f"{design_name}_cg_map_h_raw.png", "Horizontal Raw Congestion"),
            ("cg_map_v_raw", f"{design_name}_cg_map_v_raw.png", "Vertical Raw Congestion"),
            ("cg_map_union_raw", f"{design_name}_cg_map_union_raw.png", "Union Raw Congestion"),
            ("cg_map_h_overflow", f"{design_name}_cg_map_h_overflow.png", "Horizontal Overflow"),
            ("cg_map_v_overflow", f"{design_name}_cg_map_v_overflow.png", "Vertical Overflow"),
            ("cg_map_union_overflow", f"{design_name}_cg_map_union_overflow.png", "Union Overflow"),
            (
                "cg_map_h_effective_raw",
                f"{design_name}_cg_map_h_effective_raw.png",
                "Horizontal Effective Congestion",
            ),
            (
                "cg_map_v_effective_raw",
                f"{design_name}_cg_map_v_effective_raw.png",
                "Vertical Effective Congestion",
            ),
            (
                "cg_map_h_effective_overflow",
                f"{design_name}_cg_map_h_effective_overflow.png",
                "Horizontal Effective Overflow",
            ),
            (
                "cg_map_v_effective_overflow",
                f"{design_name}_cg_map_v_effective_overflow.png",
                "Vertical Effective Overflow",
            ),
            ("capacity_map", f"{design_name}_capacity_sum.png", "Capacity Sum"),
            ("dmd_map", f"{design_name}_demand_sum.png", "Demand Sum"),
            ("wire_demand_map", f"{design_name}_wire_demand_sum.png", "Wire Demand Sum"),
            ("via_demand_map", f"{design_name}_via_demand_sum.png", "Via Demand Sum"),
            (
                "raw_wire_demand_map",
                f"{design_name}_raw_wire_demand_sum.png",
                "Raw Wire Demand Sum",
            ),
            ("fix_usage_map", f"{design_name}_fix_usage_sum.png", "Fixed Usage Sum"),
            ("mov_usage_map", f"{design_name}_mov_usage_sum.png", "Movable Usage Sum"),
        )
        png_paths = {}
        for key, filename, title in png_specs:
            png_path = (png_dir / filename).resolve()
            self._save_heatmap_png(png_path, maps[key], title)
            png_paths[key] = str(png_path)
        return png_paths

    def run_gpugr(
        self,
        input_def: str = "",
        out_dir: str = "",
        design_name: str = "",
        benchmark: str = "",
        gpu: int = 0,
        threads: int = 20,
        route_xsize: int = 0,
        route_ysize: int = 0,
        rrr_iters: int = 1,
        guide_path: str = "",
        skip_m1_route: bool = True,
        verbose_parser_log: bool = False,
        cpp_log_level: int = 2,
        export_current_db: bool = False,
        keep_temp_def: bool = False,
        save_artifacts: bool = False,
        include_route_entries: bool = False,
        include_topology_pack: bool = False,
        include_l_shape_topology_pack: bool = False,
        topology_net_name_to_id: dict = None,
        topology_pin_name_to_id: dict = None,
        topology_flat_net2pin_map=None,
        topology_flat_net2pin_start_map=None,
        topology_num_pins: int = 0,
        topology_num_nets: int = 0,
        topology_ignored_net_names=None,
        topology_max_gap: int = 1,
        topology_xl: float = 0.0,
        topology_yl: float = 0.0,
        topology_bin_size_x: float = 1.0,
        topology_bin_size_y: float = 1.0,
        parser_cache_enable: bool = False,
        parser_cache_node_lpos=None,
        parser_cache_node_names=None,
        parser_cache_fallback_before_export=None,
        profile_enabled: bool = False,
        profile_prefix: str = "gpugr.run",
        backend: str = "auto",
        bottom_routing_layer: str = None,
        top_routing_layer: str = None,
    ):
        """Run gpugr and return maps plus metrics.

        If `input_def` is empty, the operator exports the current ECC DB to a
        temporary DEF first. On a parser-cache hit the DEF is neither
        re-exported nor re-parsed.
        """
        requested_backend = normalize_gpugr_backend(backend)
        del topology_ignored_net_names
        if bottom_routing_layer is None:
            bottom_routing_layer = getattr(self.params, "gpugr_bottom_routing_layer", "")
        if top_routing_layer is None:
            top_routing_layer = getattr(self.params, "gpugr_top_routing_layer", "")
        bottom_routing_layer = str(bottom_routing_layer or "")
        top_routing_layer = str(top_routing_layer or "")
        profile_enabled = bool(profile_enabled)
        topology_net_name_to_id = topology_net_name_to_id or {}
        topology_pin_name_to_id = topology_pin_name_to_id or {}
        topology_flat_net2pin_map = np.asarray(
            [] if topology_flat_net2pin_map is None else topology_flat_net2pin_map,
            dtype=np.int64,
        )
        topology_flat_net2pin_start_map = np.asarray(
            [] if topology_flat_net2pin_start_map is None else topology_flat_net2pin_start_map,
            dtype=np.int64,
        )
        parser_cache_lpos = None
        parser_cache_names = None
        parser_cache_requested = False
        if parser_cache_enable and not input_def:
            parser_cache_lpos, parser_cache_names = self._normalize_parser_cache_lpos(
                parser_cache_node_lpos,
                parser_cache_node_names,
            )
            if parser_cache_lpos is not None and parser_cache_names is None:
                logging.warning(
                    "gpugr parser cache disabled because parser_cache_node_names was not provided"
                )
            else:
                parser_cache_requested = parser_cache_lpos is not None
        native_profile_records = []
        with self._profile_phase(profile_enabled, f"{profile_prefix}.resolve_inputs"):
            lefs = self._resolve_lefs()
            result_dir = self._resolve_output_dir(out_dir)

            if not design_name:
                design_name = self.params.design_name()
            if not benchmark:
                benchmark = "custom"

        with self._profile_phase(profile_enabled, f"{profile_prefix}.make_temp_dir"):
            temp_dir = Path(tempfile.mkdtemp(prefix="gpugr_", dir=str(result_dir)))
        decompressed_def = None
        native_output_capture = temp_dir / "gpugr_native.log"

        try:
            parser_cache_key = self._build_parser_cache_key(benchmark, design_name, lefs)
            use_cached_parser = parser_cache_requested and self._parser_cache_matches(
                parser_cache_key,
                parser_cache_names,
                int(parser_cache_lpos.size(0)) if parser_cache_lpos is not None else None,
            )
            with self._profile_phase(
                profile_enabled,
                f"{profile_prefix}.materialize_def",
                input_def=1 if input_def else 0,
                parser_cache_hit=1 if use_cached_parser else 0,
            ):
                if use_cached_parser:
                    materialized_def = self._parser_db_cache.get(
                        "source_def", "<gpugr-parser-cache>"
                    )
                elif input_def:
                    materialized_def, decompressed_def = self._materialize_def(input_def, temp_dir)
                    design_name = design_name or self._infer_design_name(materialized_def)
                else:
                    if callable(parser_cache_fallback_before_export):
                        parser_cache_fallback_before_export()
                    materialized_def = self._export_current_def(temp_dir, design_name)
                    export_current_db = True

            with self._profile_phase(profile_enabled, f"{profile_prefix}.import_xplace_modules"):
                # Resolve the backend before importing route_force.py.  In a
                # CPU Torch environment an explicitly requested CPU backend
                # must never pull in the CUDA DCT extension.
                self._ensure_xplace_python_path()
                from cpp_to_py import gpugr as gpugr_module

                resolved_backend = resolve_gpugr_backend(
                    requested_backend,
                    cuda_available=torch.cuda.is_available(),
                    extension_cuda_enabled=_extension_cuda_enabled(gpugr_module),
                )
                IOParser, gpugr, Flute, calc_gr_wl_via, estimate_num_shorts = (
                    self._import_xplace_modules(resolved_backend=resolved_backend)
                )
                validate_gpugr_backend_request(resolved_backend, rrr_iters)

            with self._profile_phase(profile_enabled, f"{profile_prefix}.resolve_flute_luts"):
                powv_path, post_path = self._resolve_flute_lut_paths()
            try:
                with self._capture_native_output(native_output_capture):
                    native_profile_kwargs = self._profile_phase_kwargs(
                        profile_enabled,
                        native_profile_records,
                    )
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.flute_register",
                        cache_hit=1
                        if self._flute_register_cached(threads, powv_path, post_path)
                        else 0,
                        **native_profile_kwargs,
                    ):
                        self._register_flute_once(Flute, threads, powv_path, post_path)
                        if resolved_backend == "cuda":
                            torch.cuda.synchronize(f"cuda:{gpu}")

                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.build_params",
                        **native_profile_kwargs,
                    ):
                        params = self._build_params(benchmark, design_name, materialized_def, lefs)
                        parser = None
                    start_time = time.time()
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.parser_read",
                        threads=threads,
                        parser_cache_hit=1 if use_cached_parser else 0,
                        **native_profile_kwargs,
                    ):
                        if use_cached_parser:
                            rawdb = self._parser_db_cache["rawdb"]
                            gpdb = self._parser_db_cache["gpdb"]
                        else:
                            parser = IOParser()
                            rawdb, gpdb = parser.read(
                                params,
                                verbose_log=verbose_parser_log,
                                log_level=cpp_log_level,
                                lite_mode=True,
                                random_place=False,
                                num_threads=threads,
                            )

                    route_grid_edges_dbu = None
                    if route_xsize > 0 and route_ysize > 0:
                        _, die_xh, _, die_yh = gpdb.dieInfo()
                        # GRDatabase uses integer DIEAREA pitches and a final edge at dieHX/HY.
                        route_grid_edges_dbu = (
                            np.append(np.arange(route_xsize) * (die_xh // route_xsize), die_xh),
                            np.append(np.arange(route_ysize) * (die_yh // route_ysize), die_yh),
                        )

                    parser_cache_active = False
                    if parser_cache_requested:
                        try:
                            if not use_cached_parser:
                                cache = self._store_parser_cache(
                                    parser_cache_key,
                                    rawdb,
                                    gpdb,
                                    parser_cache_names,
                                    parser_cache_lpos,
                                )
                                cache["source_def"] = materialized_def
                                parser_cache_active = False
                            else:
                                with self._profile_phase(
                                    profile_enabled,
                                    f"{profile_prefix}.parser_cache_apply_node_lpos",
                                    parser_cache_hit=1,
                                    node_count=int(parser_cache_lpos.size(0)),
                                    **native_profile_kwargs,
                                ):
                                    self._apply_parser_cache_node_lpos(
                                        gpdb,
                                        parser_cache_lpos,
                                        parser_cache_names,
                                    )
                                parser_cache_active = True
                        except Exception:
                            logging.exception(
                                "gpugr parser cache update failed; falling back to DEF "
                                "parser for this run"
                            )
                            self._parser_db_cache = None
                            if use_cached_parser:
                                with self._profile_phase(
                                    profile_enabled,
                                    f"{profile_prefix}.parser_cache_fallback_materialize_def",
                                    fallback_callback=1
                                    if callable(parser_cache_fallback_before_export)
                                    else 0,
                                    **native_profile_kwargs,
                                ):
                                    if callable(parser_cache_fallback_before_export):
                                        parser_cache_fallback_before_export()
                                    materialized_def = self._export_current_def(
                                        temp_dir, design_name
                                    )
                                    export_current_db = True
                                    params = self._build_params(
                                        benchmark, design_name, materialized_def, lefs
                                    )
                                with self._profile_phase(
                                    profile_enabled,
                                    f"{profile_prefix}.parser_cache_fallback_parser_read",
                                    threads=threads,
                                    **native_profile_kwargs,
                                ):
                                    parser = IOParser()
                                    rawdb, gpdb = parser.read(
                                        params,
                                        verbose_log=verbose_parser_log,
                                        log_level=cpp_log_level,
                                        lite_mode=True,
                                        random_place=False,
                                        num_threads=threads,
                                    )

                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.resolve_route_guide",
                        **native_profile_kwargs,
                    ):
                        route_guide = ""
                        if guide_path:
                            route_guide = str(Path(guide_path).expanduser().resolve())
                            Path(route_guide).parent.mkdir(parents=True, exist_ok=True)

                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.load_gr_params",
                        **native_profile_kwargs,
                    ):
                        gpugr.load_gr_params(
                            {
                                "device_id": gpu,
                                "route_xSize": route_xsize,
                                "route_ySize": route_ysize,
                                "rrrIters": rrr_iters,
                                "threads": int(threads),
                                "route_guide": route_guide,
                                "backend": resolved_backend,
                                "bottom_routing_layer": bottom_routing_layer,
                                "top_routing_layer": top_routing_layer,
                            }
                        )
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.create_grdatabase",
                        route_grid=f"{route_xsize}x{route_ysize}",
                        **native_profile_kwargs,
                    ):
                        grdb = gpugr.create_grdatabase(rawdb, gpdb)
                        self._flush_grdatabase_setup_profile(
                            grdb,
                            profile_enabled,
                            profile_prefix,
                        )
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.create_routeforce",
                        **native_profile_kwargs,
                    ):
                        routeforce = gpugr.create_routeforce(grdb)
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.run_ggr",
                        rrr_iters=rrr_iters,
                        **native_profile_kwargs,
                    ):
                        routeforce.run_ggr()
                        if resolved_backend == "cuda":
                            torch.cuda.synchronize(f"cuda:{gpu}")
                    native_stats = (
                        dict(routeforce.run_stats())
                        if resolved_backend in ("cpu_pr", "cpu_pr_mt")
                        and hasattr(routeforce, "run_stats")
                        else {}
                    )
                    elapsed = time.time() - start_time
                    topology_pack_result = {}
                    l_shape_topology_pack_result = {}
                    fallback_route_entries_required = False
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.same_net_topology_pack",
                        topology_pack_enabled=1 if include_topology_pack else 0,
                        **native_profile_kwargs,
                    ):
                        if include_topology_pack:
                            if hasattr(routeforce, "same_net_topology_pack"):
                                try:
                                    topology_pack_result = routeforce.same_net_topology_pack(
                                        topology_net_name_to_id,
                                        int(topology_max_gap),
                                        int(route_xsize),
                                        int(route_ysize),
                                        float(topology_xl),
                                        float(topology_yl),
                                        float(topology_bin_size_x),
                                        float(topology_bin_size_y),
                                    )
                                except Exception:
                                    logging.exception(
                                        "gpugr same-net topology pack failed; "
                                        "falling back to route_entries"
                                    )
                                    fallback_route_entries_required = True
                            else:
                                logging.warning(
                                    "gpugr extension does not expose "
                                    "same_net_topology_pack; falling back to route_entries"
                                )
                                fallback_route_entries_required = True
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.l_shape_topology_pack",
                        topology_pack_enabled=1 if include_l_shape_topology_pack else 0,
                        **native_profile_kwargs,
                    ):
                        if include_l_shape_topology_pack:
                            if not hasattr(routeforce, "l_shape_topology_pack"):
                                raise RuntimeError(
                                    "gpugr extension does not expose l_shape_topology_pack"
                                )
                            l_shape_topology_pack_result = routeforce.l_shape_topology_pack(
                                topology_pin_name_to_id,
                                topology_net_name_to_id,
                                topology_flat_net2pin_map,
                                topology_flat_net2pin_start_map,
                                int(topology_num_pins),
                                int(topology_num_nets),
                                int(route_xsize),
                                int(route_ysize),
                                float(topology_xl),
                                float(topology_yl),
                                float(topology_bin_size_x),
                                float(topology_bin_size_y),
                            )
                    with self._profile_phase(
                        profile_enabled,
                        f"{profile_prefix}.route_entries",
                        route_entries_enabled=1
                        if include_route_entries or fallback_route_entries_required
                        else 0,
                        **native_profile_kwargs,
                    ):
                        route_entries = (
                            routeforce.route_entries()
                            if include_route_entries or fallback_route_entries_required
                            else []
                        )
            except Exception as exc:
                self._flush_profile_records(native_profile_records)
                self._append_native_output_to_place_logs(native_output_capture)
                native_log_path = self._persist_native_log(
                    native_output_capture,
                    result_dir,
                    design_name,
                    temp_dir.name,
                )
                message = (
                    f"gpugr failed while routing design {design_name}: {type(exc).__name__}: {exc}"
                )
                if native_log_path:
                    message += f"; native output captured at {native_log_path}"
                raise RuntimeError(message) from exc

            self._flush_profile_records(native_profile_records)
            with self._profile_phase(profile_enabled, f"{profile_prefix}.append_native_log"):
                self._append_native_output_to_place_logs(native_output_capture)
            with self._profile_phase(profile_enabled, f"{profile_prefix}.scan_native_log"):
                native_output_lines = self._read_native_output_lines(native_output_capture)
                suspicious_native_lines = self._find_suspicious_native_lines(native_output_lines)
            native_log_path = ""
            if save_artifacts or suspicious_native_lines:
                with self._profile_phase(profile_enabled, f"{profile_prefix}.persist_native_log"):
                    native_log_path = self._persist_native_log(
                        native_output_capture,
                        result_dir,
                        design_name,
                        temp_dir.name,
                    )
            if suspicious_native_lines:
                log_ref = native_log_path or str(native_output_capture.resolve())
                logging.warning(
                    "gpugr native output reported %d suspicious lines for %s; "
                    "see %s; first line: %s",
                    len(suspicious_native_lines),
                    design_name,
                    log_ref,
                    suspicious_native_lines[0],
                )

            with self._profile_phase(
                profile_enabled,
                f"{profile_prefix}.compute_maps",
                skip_m1_route=1 if skip_m1_route else 0,
            ):
                maps = self._compute_maps(routeforce, gpdb, skip_m1_route=skip_m1_route)

            with self._profile_phase(profile_enabled, f"{profile_prefix}.metrics_core"):
                num_ovfl_nets = int(routeforce.num_ovfl_nets())
                gr_wirelength, gr_num_vias = calc_gr_wl_via(grdb, routeforce)
                gr_est_shorts = estimate_num_shorts(
                    routeforce,
                    gpdb,
                    maps["capacity_map"],
                    maps["wire_demand_map"],
                    maps["via_demand_map"],
                )
            with self._profile_phase(profile_enabled, f"{profile_prefix}.metrics_summaries"):
                cg_h_raw_summary = self._summarize_congestion_tensor(
                    maps["cg_map_h_raw"], overflow_threshold=1.0
                )
                cg_v_raw_summary = self._summarize_congestion_tensor(
                    maps["cg_map_v_raw"], overflow_threshold=1.0
                )
                cg_union_raw_summary = self._summarize_congestion_tensor(
                    maps["cg_map_union_raw"], overflow_threshold=1.0
                )
                cg_h_overflow_summary = self._summarize_congestion_tensor(
                    maps["cg_map_h_overflow"], overflow_threshold=0.0
                )
                cg_v_overflow_summary = self._summarize_congestion_tensor(
                    maps["cg_map_v_overflow"], overflow_threshold=0.0
                )
                cg_union_overflow_summary = self._summarize_congestion_tensor(
                    maps["cg_map_union_overflow"], overflow_threshold=0.0
                )
                cg_h_effective_raw_summary = self._summarize_congestion_tensor(
                    maps["cg_map_h_effective_raw"], overflow_threshold=1.0
                )
                cg_v_effective_raw_summary = self._summarize_congestion_tensor(
                    maps["cg_map_v_effective_raw"], overflow_threshold=1.0
                )
                cg_h_effective_overflow_summary = self._summarize_congestion_tensor(
                    maps["cg_map_h_effective_overflow"], overflow_threshold=0.0
                )
                cg_v_effective_overflow_summary = self._summarize_congestion_tensor(
                    maps["cg_map_v_effective_overflow"], overflow_threshold=0.0
                )

            metrics = {
                "design_name": design_name,
                "benchmark": benchmark,
                "def_path": materialized_def,
                "guide_path": route_guide,
                "elapsed_sec": elapsed,
                "parser_threads": int(threads),
                "native_route_stats": native_stats,
                "num_overflow_nets": num_ovfl_nets,
                "gr_wirelength": float(gr_wirelength),
                "gr_num_vias": float(gr_num_vias),
                "gr_est_shorts": float(gr_est_shorts),
                "cg_map_h_raw_shape": list(maps["cg_map_h_raw"].shape),
                "cg_map_v_raw_shape": list(maps["cg_map_v_raw"].shape),
                "cg_map_union_raw_shape": list(maps["cg_map_union_raw"].shape),
                "cg_map_h_raw_max": cg_h_raw_summary["max"],
                "cg_map_v_raw_max": cg_v_raw_summary["max"],
                "cg_map_union_raw_max": cg_union_raw_summary["max"],
                "cg_map_h_raw_mean": cg_h_raw_summary["mean"],
                "cg_map_v_raw_mean": cg_v_raw_summary["mean"],
                "cg_map_union_raw_mean": cg_union_raw_summary["mean"],
                "cg_map_h_raw_top1pct_mean": cg_h_raw_summary["top1pct_mean"],
                "cg_map_v_raw_top1pct_mean": cg_v_raw_summary["top1pct_mean"],
                "cg_map_union_raw_top1pct_mean": cg_union_raw_summary["top1pct_mean"],
                "cg_map_h_raw_overflow_bin_ratio": cg_h_raw_summary["overflow_bin_ratio"],
                "cg_map_v_raw_overflow_bin_ratio": cg_v_raw_summary["overflow_bin_ratio"],
                "cg_map_union_raw_overflow_bin_ratio": cg_union_raw_summary["overflow_bin_ratio"],
                "cg_map_h_overflow_max": cg_h_overflow_summary["max"],
                "cg_map_v_overflow_max": cg_v_overflow_summary["max"],
                "cg_map_union_overflow_max": cg_union_overflow_summary["max"],
                "cg_map_h_overflow_mean": cg_h_overflow_summary["mean"],
                "cg_map_v_overflow_mean": cg_v_overflow_summary["mean"],
                "cg_map_union_overflow_mean": cg_union_overflow_summary["mean"],
                "cg_map_h_overflow_top1pct_mean": cg_h_overflow_summary["top1pct_mean"],
                "cg_map_v_overflow_top1pct_mean": cg_v_overflow_summary["top1pct_mean"],
                "cg_map_union_overflow_top1pct_mean": cg_union_overflow_summary["top1pct_mean"],
                "cg_map_h_overflow_bin_ratio": cg_h_overflow_summary["overflow_bin_ratio"],
                "cg_map_v_overflow_bin_ratio": cg_v_overflow_summary["overflow_bin_ratio"],
                "cg_map_union_overflow_bin_ratio": cg_union_overflow_summary["overflow_bin_ratio"],
                "cg_map_h_effective_raw_max": cg_h_effective_raw_summary["max"],
                "cg_map_v_effective_raw_max": cg_v_effective_raw_summary["max"],
                "cg_map_h_effective_raw_mean": cg_h_effective_raw_summary["mean"],
                "cg_map_v_effective_raw_mean": cg_v_effective_raw_summary["mean"],
                "cg_map_h_effective_raw_top1pct_mean": cg_h_effective_raw_summary[
                    "top1pct_mean"
                ],
                "cg_map_v_effective_raw_top1pct_mean": cg_v_effective_raw_summary[
                    "top1pct_mean"
                ],
                "cg_map_h_effective_raw_overflow_bin_ratio": cg_h_effective_raw_summary[
                    "overflow_bin_ratio"
                ],
                "cg_map_v_effective_raw_overflow_bin_ratio": cg_v_effective_raw_summary[
                    "overflow_bin_ratio"
                ],
                "cg_map_h_effective_overflow_max": cg_h_effective_overflow_summary["max"],
                "cg_map_v_effective_overflow_max": cg_v_effective_overflow_summary["max"],
                "cg_map_h_effective_overflow_mean": cg_h_effective_overflow_summary["mean"],
                "cg_map_v_effective_overflow_mean": cg_v_effective_overflow_summary["mean"],
                "native_issue_line_count": len(suspicious_native_lines),
                "export_current_db": bool(export_current_db),
                "decompressed_def_from": decompressed_def or "",
                "parser_cache_requested": bool(parser_cache_requested),
                "parser_cache_hit": bool(use_cached_parser),
                "parser_cache_active": bool(parser_cache_active),
                **gpugr_run_metadata(
                    requested_backend=requested_backend,
                    resolved_backend=resolved_backend,
                    route_xsize=route_xsize,
                    route_ysize=route_ysize,
                    rrr_iters=rrr_iters,
                    skip_m1_route=skip_m1_route,
                    routing_layer_range=self._last_routing_layer_range,
                    routing_layer_names=self._last_routing_layer_names,
                ),
            }
            self._add_metric_aliases(metrics)

            artifact_paths = {}
            if save_artifacts:
                with self._profile_phase(profile_enabled, f"{profile_prefix}.save_artifacts"):
                    maps_path = (result_dir / f"{design_name}_gpugr_map.npz").resolve()
                    metrics_path = (result_dir / f"{design_name}_gpugr_metrics.json").resolve()
                    png_dir = (result_dir / "png").resolve()
                    self._save_maps(maps_path, maps)
                    self._save_metrics(metrics_path, metrics)
                    png_paths = self._save_pngs(png_dir, design_name, maps)
                    artifact_paths = {
                        "maps_path": str(maps_path),
                        "metrics_path": str(metrics_path),
                        "png_dir": str(png_dir),
                    }
                    if include_route_entries:
                        route_entries_path = (
                            result_dir / f"{design_name}_gpugr_routes.json"
                        ).resolve()
                        self._save_route_entries(route_entries_path, route_entries)
                        artifact_paths["route_entries_path"] = str(route_entries_path)
                    artifact_paths.update(png_paths)
            if native_log_path:
                artifact_paths["native_log_path"] = native_log_path
                metrics["native_log_path"] = native_log_path

            return {
                "metrics": metrics,
                "native_stats": native_stats,
                "maps": maps,
                "route_grid_edges_dbu": route_grid_edges_dbu,
                "artifact_paths": artifact_paths,
                "route_entries": route_entries,
                "same_net_topology_cache": dict(topology_pack_result.get("cache", {})),
                "same_net_topology_stats": dict(topology_pack_result.get("stats", {})),
                "l_shape_topology_pack": dict(l_shape_topology_pack_result),
            }
        finally:
            # Preserve the complete native workspace when routing raises.  It
            # contains the DEF and native log needed to diagnose GPU/maze
            # routing failures; successful calls still use the requested
            # cleanup policy.
            if not keep_temp_def and sys.exc_info()[0] is None:
                with self._profile_phase(profile_enabled, f"{profile_prefix}.cleanup_temp_dir"):
                    shutil.rmtree(temp_dir, ignore_errors=True)
