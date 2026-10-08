"""Import stubs shared by the unit tests that exercise a product module in isolation.

Several tests import one product module while replacing its heavy dependencies
(CUDA operators, OpenROAD, the flow layer) with stand-ins, so the test can run
without a full build.  Two things trip those hand-written stubs up whenever the
product grows a new import:

* a stub replaces a parent package with a plain module, so every
  `from <parent>.<child> import name` that is not in the list fails with
  "no module named ... ; <parent> is not a package";
* `import <parent>.<child> as child` additionally needs the `<child>` attribute
  on the parent, which hand-written stubs never set, so the import fails with
  "cannot import name 'child' from 'parent'".

`auto_stub` fixes both for the deliberately stubbed layer: everything below the
given prefixes is fabricated on demand (as a package, wired onto its parent),
while imports outside those prefixes still fail for real, so a genuinely broken
product import is not hidden.  `register_stub_modules` does the same wiring for
hand-written stub dicts.
"""

import importlib.abc
import importlib.util
import sys
import types

__all__ = ["auto_stub", "register_stub_modules"]


def _placeholder(name):
    """A permissive stand-in: subclassable, callable, attribute-transparent."""

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return None

        def __getattr__(self, item):
            return _placeholder("%s.%s" % (name, item))

    _Placeholder.__name__ = name.rsplit(".", 1)[-1]
    _Placeholder.__qualname__ = _Placeholder.__name__
    return _Placeholder


def _module_getattr(name):
    def __getattr__(item):
        return _placeholder("%s.%s" % (name, item))

    return __getattr__


def _make_stub_package(name):
    """A package-shaped stub, so imports below it can be resolved."""
    module = types.ModuleType(name)
    module.__path__ = []
    module.__getattr__ = _module_getattr(name)
    return module


def _is_importable(name):
    """True when `name` resolves to a real module or package."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _wire_parent(module):
    """Make `import parent.child as child` work: child must be an attribute."""
    parent_name, _, child = module.__name__.rpartition(".")
    if not child:
        return
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, module)


def _unwire_parent(module):
    parent_name, _, child = module.__name__.rpartition(".")
    if not child:
        return
    parent = sys.modules.get(parent_name)
    if parent is not None and getattr(parent, child, None) is module:
        try:
            delattr(parent, child)
        except AttributeError:
            pass


class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Fabricate modules under the stubbed prefixes, packages included."""

    def __init__(self, prefixes, except_):
        self._prefixes = tuple(prefixes)
        self._except = tuple(except_)

    def _covers(self, fullname):
        if any(
            fullname == name or fullname.startswith(name + ".")
            for name in self._except
        ):
            return False
        return any(
            fullname == prefix or fullname.startswith(prefix + ".")
            for prefix in self._prefixes
        )

    def find_spec(self, fullname, path=None, target=None):
        if self._covers(fullname):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        module.__path__ = []
        module.__getattr__ = _module_getattr(module.__name__)
        _wire_parent(module)


class auto_stub:
    """Context manager installing auto-fabricating stubs for `prefixes`.

    >>> with auto_stub("dreamplace.ops"):
    ...     import dreamplace.ops.anything.deep as deep  # fabricated
    ...     from dreamplace.ops.other.thing import Whatever  # fabricated
    """

    def __init__(self, *prefixes, except_=()):
        self._prefixes = prefixes
        self._finder = _AutoStubFinder(prefixes, except_)
        self._previous = {}

    def __enter__(self):
        self._previous = {
            name: module
            for name, module in sys.modules.items()
            if self._finder._covers(name)
        }
        for name in self._previous:
            del sys.modules[name]
        sys.meta_path.insert(0, self._finder)
        # Materialise the stubbed packages right away and attach them to their
        # parents: `import a.b.c as x` short-circuits on sys.modules and then
        # needs the `a.b` attribute, which is only set when the parent was really
        # imported.  Ancestors above the prefix are imported for real (they are
        # not part of the stubbed layer); only the prefix itself is replaced.
        for prefix in self._prefixes:
            parts = prefix.split(".")
            for depth in range(1, len(parts)):
                ancestor = ".".join(parts[:depth])
                try:
                    importlib.import_module(ancestor)
                except ImportError:
                    if ancestor not in sys.modules:
                        sys.modules[ancestor] = _make_stub_package(ancestor)
                module = sys.modules.get(ancestor)
                if module is not None:
                    _wire_parent(module)
            if prefix not in sys.modules:
                sys.modules[prefix] = _make_stub_package(prefix)
            _wire_parent(sys.modules[prefix])
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.meta_path.remove(self._finder)
        except ValueError:
            pass
        for name, module in list(sys.modules.items()):
            if self._finder._covers(name):
                _unwire_parent(module)
        for name in [n for n in sys.modules if self._finder._covers(n)]:
            del sys.modules[name]
        sys.modules.update(self._previous)
        for module in self._previous.values():
            _wire_parent(module)
        return False


def register_stub_modules(modules):
    """Register hand-written stub modules and wire the import tree.

    Registering the deepest module alone is not enough: `import a.b.c as x`
    finds `a.b.c` in sys.modules and therefore never imports its ancestors, so
    it then fails on the missing `a.b` attribute.  Every stub gets `__path__`,
    every missing ancestor package is synthesized, and every stub is attached
    to its parent under its own name.  Returns the previous sys.modules
    entries so callers can restore.
    """
    pending = dict(modules)
    prefixes = tuple(
        prefix
        for finder in sys.meta_path
        if isinstance(finder, _AutoStubFinder)
        for prefix in finder._prefixes
    )
    for name in list(modules):
        parts = name.split(".")
        for depth in range(1, len(parts)):
            ancestor = ".".join(parts[:depth])
            if ancestor in pending or ancestor in sys.modules:
                continue
            if any(
                ancestor == prefix or ancestor.startswith(prefix + ".")
                for prefix in prefixes
            ):
                continue  # the auto stub finder resolves this one on demand
            if _is_importable(ancestor):
                continue  # a real package: leave it to the normal import path
            pending[ancestor] = _make_stub_package(ancestor)
    previous = {}
    for name, module in pending.items():
        previous[name] = sys.modules.get(name)
        if isinstance(module, types.ModuleType):
            module.__path__ = []
        sys.modules[name] = module
    for module in pending.values():
        _wire_parent(module)
    return previous


def install(prefixes):
    """Non-context-manager form for tests that keep the stubs for the session."""
    return auto_stub(*prefixes).__enter__()