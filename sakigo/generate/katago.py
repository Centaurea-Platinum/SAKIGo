"""KataGo analysis-engine subprocess client (extracted from the legacy run())."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def katago_executable_names(platform_name: str | None = None) -> tuple[str, ...]:
    platform_key = (platform_name or sys.platform).lower()
    if platform_key.startswith("win"):
        return ("katago.exe", "katago")
    return ("katago", "katago.exe")


def find_katago_path(engine_root: Path, platform_name: str | None = None) -> Path:
    candidates: list[Path] = []
    for executable_name in katago_executable_names(platform_name):
        candidates.extend(sorted(engine_root.glob(f"*/{executable_name}")))
    if not candidates:
        names = ", ".join(katago_executable_names(platform_name))
        raise FileNotFoundError(f"could not find Distillation/engine/*/{{{names}}}")
    return candidates[0]


def default_katago_path() -> Path:
    return find_katago_path(ROOT / "Distillation" / "engine")


def default_config_path(katago_path: Path) -> Path:
    return katago_path.parent / "analysis_example.cfg"


def default_model_path() -> Path:
    candidates = sorted((ROOT / "Distillation" / "models").glob("*.bin.gz"))
    if not candidates:
        raise FileNotFoundError("could not find Distillation/models/*.bin.gz")
    return candidates[0]


def analysis_override(
    *, analysis_threads: int, nn_batch_size: int, analysis_log_dir: Path
) -> str:
    return (
        f"numAnalysisThreads={analysis_threads},"
        "numSearchThreadsPerAnalysisThread=1,"
        f"nnMaxBatchSize={nn_batch_size},"
        "nnCacheSizePowerOfTwo=12,"
        f"logDir={analysis_log_dir},"
        "logToStderr=true,"
        "reportAnalysisWinratesAs=BLACK"
    )


class KataGoAnalysisClient:
    """Owns the KataGo subprocess, its reader threads, and the response queue."""

    def __init__(
        self,
        *,
        katago: Path,
        model: Path,
        config: Path,
        analysis_threads: int,
        nn_batch_size: int,
        run_dir: Path,
        ready_timeout: float = 300.0,
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.stderr_log = run_dir / "katago_stderr.log"
        analysis_log_dir = run_dir / "analysis_logs"
        override = analysis_override(
            analysis_threads=analysis_threads,
            nn_batch_size=nn_batch_size,
            analysis_log_dir=analysis_log_dir,
        )
        self.process = subprocess.Popen(
            [
                str(katago),
                "analysis",
                "-model",
                str(model),
                "-config",
                str(config),
                "-override-config",
                override,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.process.kill()
            self.process.wait(timeout=30)
            raise RuntimeError("failed to open KataGo pipes")
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._ready = threading.Event()
        self._startup_finished = threading.Event()
        self._startup_error: str | None = None
        self._stderr_tail: list[str] = []
        threading.Thread(target=self._stderr_reader, daemon=True).start()
        threading.Thread(target=self._stdout_reader, daemon=True).start()
        if not self._startup_finished.wait(ready_timeout):
            self.process.kill()
            self.process.wait(timeout=30)
            raise RuntimeError("KataGo did not become ready: " + " | ".join(self._stderr_tail))
        if not self._ready.is_set():
            returncode = self.process.poll()
            self.shutdown()
            detail = self._startup_error or " | ".join(self._stderr_tail)
            raise RuntimeError(
                f"KataGo exited before becoming ready (returncode={returncode}): "
                + detail
            )

    def _stderr_reader(self) -> None:
        try:
            with self.stderr_log.open("w", encoding="utf-8") as handle:
                for line in self.process.stderr:
                    handle.write(line)
                    handle.flush()
                    text = line.rstrip("\n")
                    self._stderr_tail.append(text)
                    del self._stderr_tail[:-20]
                    if "Started, ready" in text:
                        self._ready.set()
                        self._startup_finished.set()
        except Exception as error:
            self._startup_error = f"stderr reader failed: {error}"
            self._startup_finished.set()

    def _stdout_reader(self) -> None:
        for line in self.process.stdout:
            if line.startswith("{"):
                try:
                    self.responses.put(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # EOF: KataGo exited. Wake the consumer instead of letting it block forever.
        self.responses.put(
            {
                "_engine_exited": True,
                "returncode": self.process.poll(),
                "stderr_tail": list(self._stderr_tail),
            }
        )
        startup_finished = getattr(self, "_startup_finished", None)
        if startup_finished is not None:
            startup_finished.set()

    def send(self, query: str) -> None:
        self.process.stdin.write(query + "\n")

    def flush(self) -> None:
        self.process.stdin.flush()

    def kill(self) -> None:
        self.process.kill()

    def shutdown(self) -> None:
        """Must run on every exit path, or the KataGo process is orphaned."""
        if self.process.poll() is not None:
            self.process.wait()
            return
        try:
            self.process.stdin.close()
        except OSError:
            pass
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)
