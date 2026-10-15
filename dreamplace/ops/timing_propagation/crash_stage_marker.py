import json
import os
import time
from pathlib import Path


STAGE_MARKER_ENV = "AUTODMP_CUDA_CRASH_STAGE_MARKER_PATH"
RUN_ID_ENV = "AUTODMP_CUDA_CRASH_RUN_ID"
DEVICE_MODE_ENV = "AUTODMP_CUDA_CRASH_DEVICE_MODE"
GLOBAL_GPU_ENV = "AUTODMP_CUDA_CRASH_GLOBAL_GPU"
GLOBAL_GPU_ID_ENV = "AUTODMP_CUDA_CRASH_GLOBAL_GPU_ID"


def _safe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_crash_stage_marker(stage, status, **details):
    marker_path = os.environ.get(STAGE_MARKER_ENV)
    if not marker_path:
        return
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "time": time.time(),
        "run_id": os.environ.get(RUN_ID_ENV),
        "stage": str(stage),
        "status": str(status),
        "device_mode": os.environ.get(DEVICE_MODE_ENV),
        "global_gpu": _safe_int(os.environ.get(GLOBAL_GPU_ENV)),
        "global_gpu_id": _safe_int(os.environ.get(GLOBAL_GPU_ID_ENV)),
    }
    if details:
        row["details"] = {str(key): _json_safe(value) for key, value in details.items()}
    try:
        path = Path(marker_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception:
        return
