"""Native stdout/stderr capture helpers for the Xplace GPUGR backend.

The gpugr pybind extension logs straight to fd 1/2; these helpers tee that
output into a file so failures can be diagnosed and the log persisted next to
the other run artifacts. Mixed into XplaceGPUGR.

Ported from the pinned AiEDA reference
(tools/iEDA/module/gpugr.py @ 9b8aa46, sha256 b3981e71...), unchanged except
for logging module usage.
"""

import ctypes
import logging
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


class XplaceNativeOutputMixin:
    def _get_logging_file_handlers(self):
        handlers = []
        seen = set()
        logger = logging.getLogger()
        for handler in logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            log_path = getattr(handler, "baseFilename", "")
            if not log_path:
                continue
            resolved_path = str(Path(log_path).resolve())
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            handlers.append(handler)
        return handlers

    @contextmanager
    def _capture_native_output(self, capture_path: Path):
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        tee_read_fd, tee_write_fd = os.pipe()
        tee_cmd = ["tee", str(capture_path)]
        tee_stdin = os.fdopen(tee_read_fd, "rb", closefd=True)
        tee_proc = subprocess.Popen(
            tee_cmd,
            stdin=tee_stdin,
            stdout=None,
            stderr=None,
            close_fds=True,
        )
        tee_stdin.close()
        try:
            os.dup2(tee_write_fd, 1)
            os.dup2(tee_write_fd, 2)
            yield
        finally:
            try:
                try:
                    ctypes.CDLL(None).fflush(None)
                except Exception:
                    pass
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os.dup2(stdout_fd, 1)
                os.dup2(stderr_fd, 2)
                os.close(stdout_fd)
                os.close(stderr_fd)
                os.close(tee_write_fd)
                tee_proc.wait()

    def _append_native_output_to_place_logs(self, capture_path: Path):
        if not capture_path.exists():
            return
        text = capture_path.read_text(encoding="utf-8", errors="replace")
        if not text:
            return
        for handler in self._get_logging_file_handlers():
            stream = getattr(handler, "stream", None)
            if stream is None:
                continue
            handler.acquire()
            try:
                stream.seek(0, os.SEEK_END)
                stream.write(text)
                if not text.endswith("\n"):
                    stream.write("\n")
                stream.flush()
            finally:
                handler.release()

    def _read_native_output_lines(self, capture_path: Path):
        if not capture_path.exists():
            return []
        return capture_path.read_text(encoding="utf-8", errors="replace").splitlines()

    def _find_suspicious_native_lines(self, lines):
        keywords = (
            "failed",
            "error",
            "out of boundary",
            "lll too small",
        )
        suspicious_lines = []
        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in keywords):
                suspicious_lines.append(line.strip())
        return suspicious_lines

    def _persist_native_log(
        self, capture_path: Path, result_dir: Path, design_name: str, suffix: str
    ):
        if not capture_path.exists():
            return ""
        dst_path = (result_dir / f"{design_name}_gpugr_native_{suffix}.log").resolve()
        shutil.copyfile(capture_path, dst_path)
        return str(dst_path)
