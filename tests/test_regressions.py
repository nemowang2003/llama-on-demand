import asyncio
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from starlette.requests import ClientDisconnect

from llama_on_demand import models
from llama_on_demand.runtime import Runtime
from llama_on_demand.service import create_app


def update_registry(root, barrier, prefix):
    os.environ["XDG_DATA_HOME"] = root
    read_text = Path.read_text

    def slow_read(path, *args, **kwargs):
        text = read_text(path, *args, **kwargs)
        time.sleep(0.002)
        return text

    barrier.wait(timeout=30)
    with patch.object(Path, "read_text", slow_read):
        for index in range(10):
            models.save(f"{prefix}-{index}", {"path": f"{prefix}/{index}.gguf"})


class RegistryTests(unittest.TestCase):
    def test_parallel_processes_preserve_all_records(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as root:
            barrier = context.Barrier(4)
            processes = [
                context.Process(target=update_registry, args=(root, barrier, index))
                for index in range(4)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=40)
                    self.assertEqual(process.exitcode, 0)
                with patch.dict(os.environ, {"XDG_DATA_HOME": root}):
                    records = models.catalog()
                self.assertEqual(
                    records,
                    {
                        f"{prefix}-{index}": {"path": f"{prefix}/{index}.gguf"}
                        for prefix in range(4)
                        for index in range(10)
                    },
                )
                self.assertEqual(list(Path(root).rglob("models.json.*")), [])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(timeout=5)

    def test_failed_replace_preserves_registry_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "models.json"
            with patch("llama_on_demand.models.registry_path", return_value=path):
                models.save("original", {"path": "original.gguf"})
                with patch.object(Path, "replace", side_effect=OSError("disk error")):
                    with self.assertRaises(OSError):
                        models.save("new", {"path": "new.gguf"})
                self.assertEqual(models.catalog(), {"original": {"path": "original.gguf"}})
                self.assertEqual(list(Path(root).glob("models.json.*")), [])
                models.save("new", {"path": "new.gguf"})
                self.assertEqual(set(models.catalog()), {"original", "new"})


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_long_logs_preserve_bytes_and_detect_split_offload_message(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = Runtime({})
            runtime.state_dir = Path(root)
            stream = asyncio.StreamReader()
            # Put the offload marker across a read boundary within an oversized line.
            data = b"x" * (8192 * 9 - 10) + b"offloaded 10/33 layers to GPU\n" + "中文".encode()
            stream.feed_data(data)
            stream.feed_eof()
            await runtime._read_logs(stream)
            self.assertEqual((Path(root) / "llama.log").read_bytes(), data)
            self.assertEqual(runtime.warning, "cpu-offload; gpu-layers=10/33")

    async def test_failed_log_reader_does_not_prevent_stop_or_later_start(self):
        runtime = Runtime({})
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True
        )
        runtime.process = process
        runtime.model = "old"

        async def broken_reader():
            raise OSError("log write failed")

        runtime._reader = asyncio.create_task(broken_reader())
        with self.assertLogs("llama_on_demand.runtime", level="ERROR"):
            await runtime.stop()
        self.assertIsNotNone(process.returncode)
        self.assertIsNone(runtime.process)
        self.assertIsNone(runtime._reader)
        await runtime.stop()
        await self.check_options(runtime)

    async def check_options(self, runtime):
        stream = asyncio.StreamReader()
        stream.feed_eof()
        process = SimpleNamespace(returncode=None, stdout=stream)
        spawn = AsyncMock(return_value=process)
        client_type = httpx.AsyncClient
        transport = httpx.MockTransport(lambda req: httpx.Response(200))
        runtime.config = {}
        with tempfile.TemporaryDirectory() as root:
            runtime.state_dir = Path(root)
            with (
                patch(
                    "llama_on_demand.runtime.model_path",
                    return_value=(
                        Path("model.gguf"),
                        {"options": {"embedding": True, "mlock": False, "ctx-size": 4096}},
                    ),
                ),
                patch("llama_on_demand.runtime.asyncio.create_subprocess_exec", spawn),
                patch(
                    "llama_on_demand.runtime.httpx.AsyncClient",
                    side_effect=lambda **kwargs: client_type(transport=transport),
                ),
            ):
                await runtime.start("new")
            await runtime._reader
            command = spawn.call_args.args
            self.assertIn("--embedding", command)
            self.assertNotIn("True", command)
            self.assertNotIn("False", command)
            self.assertNotIn("--mlock", command)
            self.assertEqual(command[command.index("--ctx-size") + 1], "4096")
            self.assertEqual(runtime.model, "new")
            runtime.process = None
            await runtime.stop()

    async def test_boolean_options_are_flags_and_false_is_omitted(self):
        await self.check_options(Runtime({}))


class StreamingFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_header_disconnect_closes_upstream_and_releases_slot(self):
        for version in ("2.3", "2.4"):
            with self.subTest(asgi=version):
                await self.check_disconnect(version)

    async def check_disconnect(self, version):
        runtime = SimpleNamespace(
            last_used=0,
            base_url="http://fake",
            warning=None,
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        with patch.dict(os.environ, {"LLAMA_ON_DEMAND_API_KEY": "test"}):
            app = create_app({"models": {"m": {}}}, runtime)
        upstream = httpx.Response(200, content=b"data: [DONE]\n\n")
        client_type = httpx.AsyncClient
        client = client_type(transport=httpx.MockTransport(lambda req: upstream))
        header_started = asyncio.Event()
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {
                    "type": "http.request",
                    "body": b'{"model":"m","stream":true}',
                    "more_body": False,
                }
            await header_started.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.start":
                header_started.set()
                if version == "2.4":
                    raise OSError("disconnected")
                await asyncio.Event().wait()

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": version},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [(b"authorization", b"Bearer test")],
        }
        with patch("llama_on_demand.service.httpx.AsyncClient", return_value=client):
            if version == "2.4":
                with self.assertRaises(ClientDisconnect):
                    await app(scope, receive, send)
            else:
                await app(scope, receive, send)
        self.assertTrue(upstream.is_closed)
        self.assertTrue(client.is_closed)
        runtime.stop.assert_awaited_once()
        self.assertFalse(app.state.scheduler.lock.locked())
        await asyncio.wait_for(app.state.scheduler.acquire(), timeout=1)
        app.state.scheduler.release()
