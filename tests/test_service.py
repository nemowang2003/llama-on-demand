import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

from llama_on_demand import models
from llama_on_demand.runtime import OFFLOAD
from llama_on_demand.service import Scheduler, create_app


class FakeRuntime:
    def __init__(self):
        self.last_used = time.monotonic()
        self.warning = "cpu-offload; gpu-layers=10/33"
        self.base_url = "http://fake"
        self.socket_path = "/fake.sock"
        self.calls = []

    async def start(self, model):
        self.calls.append(("start", model))

    async def stop(self):
        self.calls.append(("stop", None))

    def status(self):
        return {"warning": self.warning}


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fifo_and_timeout_do_not_steal_lock(self):
        runtime = FakeRuntime()
        scheduler = Scheduler({"queue_timeout": 0.05}, runtime)
        await scheduler.acquire()
        with self.assertRaises(HTTPException) as error:
            await scheduler.acquire()
        self.assertEqual(error.exception.status_code, 504)
        self.assertTrue(scheduler.lock.locked())
        self.assertEqual(scheduler.waiting, 0)
        scheduler.release()
        order = []

        async def worker(i):
            await scheduler.acquire()
            order.append(i)
            await asyncio.sleep(0.005)
            scheduler.release()

        await asyncio.gather(*(worker(i) for i in range(5)))
        self.assertEqual(order, list(range(5)))

    async def test_cancelled_waiter_does_not_release_active_request(self):
        scheduler = Scheduler({}, FakeRuntime())
        await scheduler.acquire()
        task = asyncio.create_task(scheduler.acquire())
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(scheduler.lock.locked())
        self.assertEqual(scheduler.waiting, 0)
        scheduler.release()

    async def test_idle_exit_never_stops_active_inference(self):
        runtime = FakeRuntime()
        scheduler = Scheduler({"idle_seconds": 0}, runtime)
        await scheduler.acquire()
        task = asyncio.create_task(scheduler.reap())
        await asyncio.sleep(0.6)
        self.assertEqual(runtime.calls, [])
        scheduler.release()
        await asyncio.sleep(0.6)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIn(("stop", None), runtime.calls)

    async def test_backend_header_and_metadata_dont_start_model(self):
        runtime = FakeRuntime()
        with patch.dict(os.environ, {"LLAMA_ON_DEMAND_API_KEY": "test"}):
            app = create_app({"models": {"sakura": {}}}, runtime)
        real_client = httpx.AsyncClient
        upstream = httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []}))

        def client(*args, **kwargs):
            return real_client(transport=upstream)

        async with real_client(
            transport=httpx.ASGITransport(app=app),
            base_url="http://local",
            headers={"Authorization": "Bearer test"},
        ) as browser:
            self.assertEqual((await browser.get("/v1/models")).status_code, 200)
            self.assertEqual(runtime.calls, [])
            with patch("llama_on_demand.service.httpx.AsyncClient", side_effect=client):
                response = await browser.post(
                    "/v1/chat/completions", json={"model": "sakura", "messages": []}
                )
            self.assertEqual(response.status_code, 200)
            self.assertIn("cpu-offload", response.headers["x-local-llm-warning"])
            self.assertFalse(app.state.scheduler.lock.locked())

    async def test_start_failure_releases_queue(self):
        runtime = FakeRuntime()

        self.enterContext(
            patch.object(runtime, "start", side_effect=RuntimeError("model cannot load"))
        )
        with patch.dict(os.environ, {"LLAMA_ON_DEMAND_API_KEY": "test"}):
            app = create_app({"models": {"sakura": {}}}, runtime)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://local",
            headers={"Authorization": "Bearer test"},
        ) as browser:
            response = await browser.post("/v1/chat/completions", json={"model": "sakura"})
            self.assertEqual(response.status_code, 503)
            self.assertFalse(app.state.scheduler.lock.locked())


class ModelTests(unittest.TestCase):
    def test_import_validates_gguf_and_does_not_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "sakura.gguf"
            model.write_bytes(b"GGUFtest")
            with patch("llama_on_demand.models.registry_path", return_value=root / "models.json"):
                models.import_model("sakura", model)
                self.assertEqual(models.model_path("sakura", {})[0], model)
            model.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                models.validate(model)

    def test_offload_log_recognizes_partial_and_full(self):
        match = OFFLOAD.search("load_tensors: offloaded 10/33 layers to GPU")
        assert match is not None
        self.assertEqual(match.groups(), ("10", "33"))


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_holds_slot_until_final_chunk(self):
        runtime = FakeRuntime()
        gate = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'data: {"choices":[]}\n\n'
                await gate.wait()
                yield b"data: [DONE]\n\n"

        transport = httpx.MockTransport(
            lambda req: httpx.Response(
                200, stream=Body(), headers={"content-type": "text/event-stream"}
            )
        )
        client_type = httpx.AsyncClient
        with patch.dict(os.environ, {"LLAMA_ON_DEMAND_API_KEY": "test"}):
            app = create_app({"models": {"sakura": {}, "other": {}}}, runtime)
        async with client_type(
            transport=httpx.ASGITransport(app=app),
            base_url="http://local",
            headers={"Authorization": "Bearer test"},
        ) as client:
            with patch(
                "llama_on_demand.service.httpx.AsyncClient",
                side_effect=lambda **kwargs: client_type(transport=transport),
            ):
                first = asyncio.create_task(
                    client.post("/v1/chat/completions", json={"model": "sakura", "stream": True})
                )
                for _ in range(50):
                    if runtime.calls:
                        break
                    await asyncio.sleep(0.01)
                second = asyncio.create_task(
                    client.post("/v1/chat/completions", json={"model": "other", "stream": True})
                )
                await asyncio.sleep(0.05)
                self.assertEqual(runtime.calls, [("start", "sakura")])
                self.assertEqual(app.state.scheduler.waiting, 1)
                gate.set()
                responses = await asyncio.gather(first, second)
                self.assertEqual([r.status_code for r in responses], [200, 200])
                self.assertEqual(runtime.calls, [("start", "sakura"), ("start", "other")])
                self.assertFalse(app.state.scheduler.lock.locked())

    async def test_cancelled_stream_stops_worker_before_unlock(self):
        runtime = FakeRuntime()
        entered = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await asyncio.Event().wait()
                yield b""

        transport = httpx.MockTransport(lambda req: httpx.Response(200, stream=Body()))
        client_type = httpx.AsyncClient
        with patch.dict(os.environ, {"LLAMA_ON_DEMAND_API_KEY": "test"}):
            app = create_app({"models": {"sakura": {}}}, runtime)
        async with client_type(
            transport=httpx.ASGITransport(app=app),
            base_url="http://local",
            headers={"Authorization": "Bearer test"},
        ) as client:
            with patch(
                "llama_on_demand.service.httpx.AsyncClient",
                side_effect=lambda **kwargs: client_type(transport=transport),
            ):
                task = asyncio.create_task(
                    client.post("/v1/chat/completions", json={"model": "sakura", "stream": True})
                )
                await entered.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIn(("stop", None), runtime.calls)
                self.assertFalse(app.state.scheduler.lock.locked())
