"""Immutable transport records shared by native overlay journal checkpoints."""

import json
import math


def _readonly(*_args, **_kwargs):
    raise TypeError("native overlay records are immutable; replace the record instead")


class FrozenRecord(dict):
    def __deepcopy__(self, memo):
        return self


class FrozenValues(list):
    def __deepcopy__(self, memo):
        return self


for _name in (
    "__setitem__",
    "__delitem__",
    "clear",
    "pop",
    "popitem",
    "setdefault",
    "update",
    "__ior__",
):
    setattr(FrozenRecord, _name, _readonly)
for _name in (
    "__setitem__",
    "__delitem__",
    "append",
    "extend",
    "insert",
    "clear",
    "pop",
    "remove",
    "sort",
    "reverse",
    "__iadd__",
    "__imul__",
):
    setattr(FrozenValues, _name, _readonly)


def _freeze(value):
    if isinstance(value, dict):
        return FrozenRecord((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return FrozenValues(_freeze(item) for item in value)
    return value


def _fast_freeze(value):
    t = type(value)
    if t is str or t is int or t is bool or value is None:
        return value
    if t is float:
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant")
        return value
    if t is dict or t is FrozenRecord:
        for k in value:
            if type(k) is not str:
                raise TypeError("non-string dict key")
        return FrozenRecord((k, _fast_freeze(value[k])) for k in sorted(value))
    if t is list or t is tuple or t is FrozenValues:
        return FrozenValues(_fast_freeze(item) for item in value)
    raise TypeError("fallback")


def freeze_record(value):
    if isinstance(value, (FrozenRecord, FrozenValues)):
        return value
    try:
        return _fast_freeze(value)
    except TypeError:
        # Match the existing JSON transport contract once when publishing a record.
        return _freeze(
            json.loads(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True))
        )


def frozen_mutation_arguments(name, arguments):
    if name == "append":
        return (freeze_record(arguments[0]),)
    if name == "insert":
        return (arguments[0], freeze_record(arguments[1]))
    if name in ("extend", "__iadd__"):
        return ([freeze_record(value) for value in arguments[0]],)
    if name == "__setitem__":
        index, value = arguments
        return (
            index,
            [freeze_record(item) for item in value]
            if isinstance(index, slice)
            else freeze_record(value),
        )
    return arguments


class RecordList(list):
    def __init__(self, values=(), *, already_frozen=False):
        if already_frozen:
            super().__init__(values)
        else:
            super().__init__(freeze_record(value) for value in values)


def _record_mutation(name):
    method = getattr(list, name)

    def mutate(self, *args):
        return method(self, *frozen_mutation_arguments(name, args))

    return mutate


for _name in ("append", "extend", "insert", "__setitem__", "__iadd__"):
    setattr(RecordList, _name, _record_mutation(_name))


class NativeOverlayMetadata:
    """Own record publication and shared checkpoint roots; no repair decisions."""

    def __init__(self):
        self._metadata_journal_enabled = False

    def enable_metadata_journal(self):
        if self._metadata_journal_enabled:
            return
        self._metadata_journal_enabled = True
        self.actions = self.actions
        self.base_removals = self.base_removals
        self.native_topology_events = self.native_topology_events
        self.projection_audit = self.projection_audit
        self.net_patches = self.net_patches

    @property
    def base_removals(self):
        return self._base_removals

    @base_removals.setter
    def base_removals(self, values):
        self._base_removals = RecordList(values) if self._metadata_journal_enabled else values

    @property
    def native_topology_events(self):
        return self._native_topology_events

    @native_topology_events.setter
    def native_topology_events(self, values):
        self._native_topology_events = (
            RecordList(values) if self._metadata_journal_enabled else values
        )

    @property
    def projection_audit(self):
        return self._projection_audit

    @projection_audit.setter
    def projection_audit(self, value):
        self._projection_audit = freeze_record(value) if self._metadata_journal_enabled else value

    @property
    def net_patches(self):
        return self._net_patches

    @net_patches.setter
    def net_patches(self, value):
        self._net_patches = freeze_record(value) if self._metadata_journal_enabled else value

    def metadata_checkpoint(self):
        return {
            "metadata_journal": True,
            "actions": tuple(self.actions),
            "actions_digest": getattr(self.actions, "_digest", None),
            "base_removals": tuple(self.base_removals),
            "native_topology_events": tuple(self.native_topology_events),
            "projection_audit": self.projection_audit,
            "net_patches": self.net_patches,
        }
