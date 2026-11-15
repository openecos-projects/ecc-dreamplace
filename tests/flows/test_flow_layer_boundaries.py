"""Architectural boundary invariants for dreamplace.flows subpackages."""

import ast
from collections import defaultdict
from pathlib import Path

import dreamplace

DREAMPLACE_ROOT = Path(dreamplace.__file__).resolve().parent
FLOWS_ROOT = DREAMPLACE_ROOT / "flows"
OPS_ROOT = DREAMPLACE_ROOT / "ops"

ALLOWED_SUBPACKAGE_DEPS = {
    "config": set(),
    "gp": {"config"},
    "discrete": {"config", "gp"},
    "s5b1": {"config", "gp", "discrete"},
    "post_gp": {"config", "gp", "discrete"},
    "baselines": {"config", "gp", "discrete", "s5b1", "post_gp"},
}


def _iter_flow_modules():
    for path in sorted(FLOWS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(FLOWS_ROOT)
        if rel.parts == ("__init__.py",):
            continue
        yield rel, path


def _extract_flow_imports(path: Path, current_subpkg: str) -> list[tuple[int, str, str]]:
    """Return (lineno, target_subpkg, target_dotted) for all dreamplace.flows imports in `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("dreamplace.flows."):
                    parts = alias.name.split(".")
                    if len(parts) >= 3:
                        hits.append((node.lineno, parts[2], alias.name))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level == 1:
                target = f"dreamplace.flows.{current_subpkg}" + (f".{mod}" if mod else "")
                hits.append((node.lineno, current_subpkg, target))
            elif node.level == 2 and mod:
                sub = mod.split(".")[0]
                hits.append((node.lineno, sub, f"dreamplace.flows.{mod}"))
            elif mod == "dreamplace.flows":
                for alias in node.names:
                    if alias.name in ALLOWED_SUBPACKAGE_DEPS:
                        hits.append((node.lineno, alias.name, f"dreamplace.flows.{alias.name}"))
            elif mod.startswith("dreamplace.flows."):
                parts = mod.split(".")
                if len(parts) >= 3:
                    target_dotted = mod
                    if len(parts) == 3 and parts[2] in ALLOWED_SUBPACKAGE_DEPS:
                        for alias in node.names:
                            hits.append(
                                (node.lineno, parts[2], f"{mod}.{alias.name}")
                            )
                    else:
                        hits.append((node.lineno, parts[2], target_dotted))
    return hits


def test_flows_has_no_flat_top_level_modules():
    flat_files = [
        p.name
        for p in sorted(FLOWS_ROOT.glob("*.py"))
        if p.name != "__init__.py"
    ]
    assert flat_files == []

    subpackages = {
        p.name
        for p in FLOWS_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith("__")
    }
    assert subpackages == set(ALLOWED_SUBPACKAGE_DEPS)


def test_flows_subpackage_dependencies_are_strictly_layered():
    violations = []
    module_graph = defaultdict(set)

    for rel, path in _iter_flow_modules():
        subpkg = rel.parts[0]
        assert subpkg in ALLOWED_SUBPACKAGE_DEPS, f"Unexpected subpackage: {rel}"
        if rel.name == "__init__.py":
            continue
        src_mod = "dreamplace.flows." + ".".join(rel.with_suffix("").parts)
        allowed = ALLOWED_SUBPACKAGE_DEPS[subpkg] | {subpkg}
        for lineno, target_subpkg, target_dotted in _extract_flow_imports(path, subpkg):
            if target_subpkg not in allowed:
                violations.append(
                    f"{rel}:{lineno} ({subpkg}) -> {target_dotted} ({target_subpkg})"
                )
            if target_dotted != src_mod and target_dotted.count(".") >= 3:
                module_graph[src_mod].add(target_dotted)

    assert violations == []

    visited = set()
    stack = []
    cycles = []

    def _dfs(node: str):
        visited.add(node)
        stack.append(node)
        for nxt in sorted(module_graph.get(node, ())):
            if nxt in stack:
                cycles.append(stack[stack.index(nxt) :] + [nxt])
            elif nxt not in visited:
                _dfs(nxt)
        stack.pop()

    for mod in sorted(module_graph):
        if mod not in visited:
            _dfs(mod)

    assert cycles == []


def test_ops_layer_does_not_import_flows_layer():
    violations = []
    for path in sorted(OPS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "dreamplace.flows" or alias.name.startswith(
                        "dreamplace.flows."
                    ):
                        violations.append(
                            f"{path.relative_to(DREAMPLACE_ROOT)}:{node.lineno} -> {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "dreamplace.flows" or mod.startswith("dreamplace.flows."):
                    violations.append(
                        f"{path.relative_to(DREAMPLACE_ROOT)}:{node.lineno} -> {mod}"
                    )
    assert violations == []
