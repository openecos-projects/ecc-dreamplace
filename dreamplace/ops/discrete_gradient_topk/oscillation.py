import torch


class CellOscillationState:
    def __init__(self, cooldown_rounds=2):
        self.cooldown_rounds = cooldown_rounds
        self.phase = None
        self.history = {}
        self.frozen_until = {}
        self.reversals = 0
        self.repeated_reversals = 0

    def _begin_phase(self, phase):
        if self.phase != phase:
            self.phase = phase
            self.history.clear()
            self.frozen_until.clear()

    def blocked(self, *, phase, iteration, count, device):
        self._begin_phase(phase)
        mask = torch.zeros(count, dtype=torch.bool, device=device)
        for instance_id, until in self.frozen_until.items():
            if iteration <= until:
                mask[instance_id] = True
        return mask

    def record(self, *, phase, iteration, instance_ids, previous_ids, target_ids):
        self._begin_phase(phase)
        for instance_id, previous, target in zip(
            instance_ids, previous_ids, target_ids, strict=True
        ):
            history = self.history.setdefault(instance_id, [previous])
            if history[-1] != previous:
                history = [previous]
            history = (history + [target])[-4:]
            if len(history) >= 3 and history[-1] == history[-3]:
                self.reversals += 1
            if len(history) == 4 and history[0] == history[2] and history[1] == history[3]:
                self.repeated_reversals += 1
                self.frozen_until[instance_id] = iteration + self.cooldown_rounds
            self.history[instance_id] = history

    def snapshot(self):
        return {
            "phase": self.phase,
            "history": {key: list(value) for key, value in self.history.items()},
            "frozen_until": dict(self.frozen_until),
            "reversals": self.reversals,
            "repeated_reversals": self.repeated_reversals,
        }

    def restore(self, snapshot):
        self.phase = snapshot["phase"]
        self.history = {key: list(value) for key, value in snapshot["history"].items()}
        self.frozen_until = dict(snapshot["frozen_until"])
        self.reversals = snapshot["reversals"]
        self.repeated_reversals = snapshot["repeated_reversals"]
