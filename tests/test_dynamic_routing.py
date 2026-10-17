import asyncio
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

import httpx
import uvicorn

from llama_on_demand import models
from llama_on_demand.runtime import Runtime
from llama_on_demand.server import unix_listener
from llama_on_demand.service import create_app


@asynccontextmanager
async def serving(app, listener):
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical"))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("test server exited before startup")
                await asyncio.sleep(0.01)
        yield
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


class DynamicRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_direct_api_switches_and_reuses_real_worker_processes_over_unix_sockets(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "LLAMA_ON_DEMAND_API_KEY": "test",
                    "XDG_RUNTIME_DIR": str(root),
                    "XDG_STATE_HOME": str(root),
                },
            )
        )
        self.enterContext(
            patch("llama_on_demand.models.registry_path", return_value=root / "models.json")
        )
        worker = root / "worker"
        source = Path(__file__).parent / "fixtures/fake_worker.py"
        worker.write_text(f"#!{sys.executable}\n" + source.read_text())
        worker.chmod(0o700)
        runtime = Runtime(
            {"server_command": str(worker), "startup_timeout": 5, "idle_seconds": 0.1}
        )
        app = create_app(runtime.config, runtime)
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(root / "api.sock")),
            base_url="http://localhost",
            headers={"Authorization": "Bearer test"},
        ) as client:
            with unix_listener(root / "api.sock") as listener:
                async with serving(app, listener):
                    self.assertFalse((await client.get("/status")).json()["running"])
                    self.assertEqual((await client.get("/v1/models")).json()["data"], [])
                    # Both aliases are registered after the service has started.
                    for name in ("first", "owner/second"):
                        weights = root / (name.replace("/", "-") + ".gguf")
                        weights.write_bytes(b"GGUFtest")
                        models.import_model(name, weights)
                    self.assertEqual(
                        {m["id"] for m in (await client.get("/v1/models")).json()["data"]},
                        {"first", "owner/second"},
                    )
                    self.assertIsNone(runtime.process)
                    result = await client.post("/v1/chat/completions", json={"model": "first"})
                    self.assertEqual(result.status_code, 200, result.text)
                    first_pid = result.json()["pid"]
                    first_socket = runtime.socket_path
                    assert first_socket is not None
                    self.assertTrue(first_socket.is_socket())
                    self.assertEqual(stat.S_IMODE(first_socket.parent.stat().st_mode), 0o700)
                    self.assertIn("cpu-offload", result.headers["x-local-llm-warning"])
                    result = await client.post("/v1/completions", json={"model": "first"})
                    self.assertEqual(result.json()["pid"], first_pid)
                    result = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "owner/second",
                            "stream": True,
                        },
                    )
                    self.assertEqual(result.status_code, 200)
                    chunk = json.loads(result.text.splitlines()[0].removeprefix("data: "))
                    self.assertEqual(chunk["model"], "owner/second")
                    self.assertNotEqual(chunk["pid"], first_pid)
                    self.assertFalse(first_socket.exists())
                    result = await client.post(
                        "/v1/embeddings", json={"model": "first", "input": "hi"}
                    )
                    self.assertEqual(result.status_code, 200, result.text)
                    self.assertEqual(result.json()["model"], "first")
                    result = await client.post("/v1/chat/completions", json={"model": "missing"})
                    self.assertEqual(result.status_code, 404)
                    async with asyncio.timeout(5):
                        while runtime.process is not None:
                            await asyncio.sleep(0.05)
                    self.assertIsNone(runtime.socket_path)
                    self.assertEqual(list((root / "llama-on-demand").glob("w-*")), [])
                    models.save("unavailable", {"path": str(root / "missing.gguf")})
                    result = await client.post(
                        "/v1/chat/completions", json={"model": "unavailable"}
                    )
                    self.assertEqual(result.status_code, 503)
                    self.assertFalse(app.state.scheduler.lock.locked())
