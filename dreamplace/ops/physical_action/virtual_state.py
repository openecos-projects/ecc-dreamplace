def select_best_insert_candidate(candidates):
    candidates = list(candidates or [])
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item.get("predicted_delta_obj", 0.0)))


def select_best_insert_per_net(candidates):
    best_by_net = {}
    for candidate in candidates or []:
        net_id = int(candidate.get("net_id", candidate.get("affected_net_id", -1)))
        if net_id < 0:
            continue
        current = best_by_net.get(net_id)
        if current is None or float(candidate.get("predicted_delta_obj", 0.0)) < float(current.get("predicted_delta_obj", 0.0)):
            best_by_net[net_id] = candidate
    return [
        best_by_net[net_id]
        for net_id in sorted(best_by_net)
    ]


class VirtualBufferState:
    def __init__(self):
        self.entries = {}
        self.trace = []
        self.committed_buffer_count = 0

    @property
    def virtual_buffer_count(self):
        return sum(1 for entry in self.entries.values() if int(entry.get("bu", 0)) == 1)

    def _candidate_id(self, candidate):
        candidate_id = int(candidate.get("candidate_id", -1))
        if candidate_id < 0:
            raise ValueError("candidate_id must be non-negative")
        return candidate_id

    def insert(self, candidate, bsu):
        candidate_id = self._candidate_id(candidate)
        if candidate_id in self.entries and int(self.entries[candidate_id].get("bu", 0)) == 1:
            raise ValueError("candidate already has a virtual buffer")
        entry = dict(candidate)
        entry.update({"candidate_id": candidate_id, "bu": 1, "bsu": int(bsu)})
        self.entries[candidate_id] = entry
        trace_entry = {"action": "insert", "candidate_id": candidate_id, "bu": 1, "bsu": int(bsu)}
        self.trace.append(trace_entry)
        return entry

    def resize(self, candidate, target_bsu):
        candidate_id = self._candidate_id(candidate)
        entry = self.entries.get(candidate_id)
        if entry is None or int(entry.get("bu", 0)) != 1:
            raise ValueError("cannot resize a candidate without a virtual buffer")
        current_bsu = int(entry.get("bsu", -1))
        target_bsu = int(target_bsu)
        if target_bsu == current_bsu:
            action = "noop"
        elif target_bsu > current_bsu:
            action = "upsize"
        else:
            action = "downsize"
        entry["bsu"] = target_bsu
        trace_entry = {"action": action, "candidate_id": candidate_id, "old_bsu": current_bsu, "bsu": target_bsu}
        self.trace.append(trace_entry)
        return trace_entry

    def remove(self, candidate, allow_remove=False):
        if not allow_remove:
            raise ValueError("virtual buffer removal is disabled before warmup")
        candidate_id = self._candidate_id(candidate)
        entry = self.entries.get(candidate_id)
        if entry is None or int(entry.get("bu", 0)) != 1:
            raise ValueError("cannot remove a candidate without a virtual buffer")
        entry["bu"] = 0
        trace_entry = {"action": "remove", "candidate_id": candidate_id, "bu": 0}
        self.trace.append(trace_entry)
        return trace_entry
