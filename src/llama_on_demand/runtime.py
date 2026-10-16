"""One llama.cpp process, held only while serving or briefly idle."""

import asyncio
import logging
import os
import re
import signal
import socket
import time
from pathlib import Path

import httpx

from .models import model_path

logger = logging.getLogger(__name__)
OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")


class Runtime:
    def __init__(self, config: dict):
        self.config = config
        self.process = None
        self.model = None
        self.port = None
        self.warning = None
        self.offloaded_layers = None
        self.total_layers = None
        self.last_used = time.monotonic()
        self.state_dir = (
            Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "llama-on-demand"
        )
        self._reader = None
        self._loaded_spec = None

    async def start(self, model: str) -> None:
        path, record = model_path(model, self.config)
        if (
            self.process
            and self.process.returncode is None
            and self.model == model
            and self._loaded_spec == (path, record)
        ):
            return
        await self.stop()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.state_dir.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.get(
                "server_command", os.environ.get("LLAMA_SERVER_COMMAND", "llama-server")
            ),
            "--model",
            str(path),
            "--alias",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--parallel",
            "1",
            "--fit",
            "on",
            "--fit-target",
            str(self.config.get("fit_target_mib", 1024)),
            "--no-ui",
            "--verbosity",
            "4",
        ]
        # Do not set gpu-layers: llama.cpp must fit it to currently free VRAM.
        options = record.get("options", {})
        for key, value in options.items():
            normalized_key = key.replace("_", "-")
            if normalized_key in {"gpu-layers", "n-gpu-layers"}:
                logger.warning(
                    "Ignoring model %s option %s=%s; GPU layers are managed by automatic fitting",
                    model,
                    key,
                    value,
                )
                continue
            if normalized_key in {
                "model",
                "alias",
                "host",
                "port",
                "parallel",
                "fit",
                "sleep-idle-seconds",
            }:
                raise ValueError(f"reserved llama.cpp option: {key}")
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            else:
                command.extend([f"--{key}", str(value)])
        self.model = model
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        assert self.process.stdout is not None
        self._reader = asyncio.create_task(self._read_logs(self.process.stdout))
        deadline = time.monotonic() + self.config.get("startup_timeout", 300)
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                while time.monotonic() < deadline:
                    if self.process.returncode is not None:
                        raise RuntimeError(
                            f"llama.cpp exited with {self.process.returncode}; see {self.state_dir / 'llama.log'}"
                        )
                    try:
                        response = await client.get(f"{self.base_url}/health")
                        if response.status_code == 200:
                            self._loaded_spec = (path, record)
                            self.last_used = time.monotonic()
                            return
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.2)
            raise TimeoutError("llama.cpp startup timed out")
        except BaseException:
            await self.stop()
            raise

    async def _read_logs(self, stream: asyncio.StreamReader) -> None:
        pending = b""
        with (self.state_dir / "llama.log").open("wb") as output:
            while chunk := await stream.read(8192):
                output.write(chunk)
                output.flush()
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._inspect_log(line)
                # Keep memory bounded even if the worker never writes a newline.
                # The overlap preserves offload messages split across chunks.
                if len(pending) > 8192:
                    self._inspect_log(pending)
                    pending = pending[-256:]
            if pending:
                self._inspect_log(pending)

    def _inspect_log(self, line: bytes) -> None:
        text = line.decode(errors="replace")
        match = OFFLOAD.search(text)
        if match:
            layers = tuple(map(int, match.groups()))
            if layers != (self.offloaded_layers, self.total_layers):
                self.offloaded_layers, self.total_layers = layers
                self.warning = None
                if self.offloaded_layers < self.total_layers:
                    self.warning = (
                        f"cpu-offload; gpu-layers={self.offloaded_layers}/{self.total_layers}"
                    )
                    logger.warning("Model %s uses CPU offload (%s)", self.model, self.warning)
        if "error" in text.lower():
            logger.error("llama.cpp: %s", text.strip())

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        process = self.process
        if process is not None and process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 15)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        try:
            if self._reader:
                await self._reader
        except Exception:
            logger.exception("Failed to read llama.cpp logs")
        finally:
            self.process = self.model = self.port = self._reader = None
            self.warning = self.offloaded_layers = self.total_layers = None
            self._loaded_spec = None

    def status(self) -> dict:
        return {
            "model": self.model,
            "running": self.process is not None and self.process.returncode is None,
            "warning": self.warning,
            "gpu_layers": self.offloaded_layers,
            "total_layers": self.total_layers,
        }
