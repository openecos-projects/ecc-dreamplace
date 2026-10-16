import json
from pathlib import Path


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_virtual_update_trace(virtual_state, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for index, event in enumerate(virtual_state.trace):
            payload = {"event_index": index}
            payload.update(event)
            stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return {
        "artifact": "buffering_virtual_update_trace",
        "path": str(path),
        "trace_event_count": len(virtual_state.trace),
    }


def write_final_state(virtual_state, output_path):
    entries = [
        dict(entry)
        for _, entry in sorted(virtual_state.entries.items(), key=lambda item: int(item[0]))
    ]
    payload = {
        "artifact": "buffering_final_state",
        "artifact_version": 1,
        "committed_buffer_count": int(virtual_state.committed_buffer_count),
        "virtual_buffer_count": int(virtual_state.virtual_buffer_count),
        "candidate_commit_action_count": int(virtual_state.virtual_buffer_count),
        "entries": entries,
    }
    _write_json(output_path, payload)
    return payload
