import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from llama_on_demand.service import create_app


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"LLAMA_ON_DEMAND_API_KEY": "test"}))
        self.enterContext(patch("llama_on_demand.service.catalog", return_value={"local": {}}))
        self.runtime = SimpleNamespace(
            last_used=0,
            base_url="http://worker",
            socket_path="/fake.sock",
            warning="cpu-offload; gpu-layers=10/33",
            start=AsyncMock(),
            stop=AsyncMock(),
        )
        self.app = create_app({}, self.runtime)
        self.client_type = httpx.AsyncClient

    def browser(self, headers=None):
        return self.client_type(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://local",
            headers=headers or {"Authorization": "Bearer test"},
        )

    async def test_native_payloads_responses_and_anthropic_headers_are_preserved(self):
        messages = {
            "model": "local",
            "max_tokens": 128,
            "system": "Be helpful",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
            "tools": [{"name": "weather", "input_schema": {"type": "object"}}],
        }
        responses = {
            "model": "local",
            "instructions": "Be helpful",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            "tools": [{"type": "function", "name": "weather", "parameters": {"type": "object"}}],
        }
        for endpoint, payload, output in (
            (
                "messages",
                messages,
                {"type": "message", "content": [{"type": "text", "text": "Hi"}]},
            ),
            (
                "responses",
                responses,
                {
                    "object": "response",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}
                    ],
                },
            ),
            ("messages/count_tokens", messages, {"input_tokens": 10}),
            (
                "responses/input_tokens",
                responses,
                {"object": "response.input_tokens", "input_tokens": 10},
            ),
        ):
            with self.subTest(endpoint=endpoint):
                captured = []

                def upstream(request):
                    captured.append(request)
                    return httpx.Response(200, json=output)

                async with self.browser(
                    {
                        "x-api-key": "test",
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "test-beta",
                    }
                ) as client:
                    with patch(
                        "llama_on_demand.service.httpx.AsyncClient",
                        side_effect=lambda **kwargs: self.client_type(
                            transport=httpx.MockTransport(upstream)
                        ),
                    ):
                        response = await client.post("/v1/" + endpoint, json=payload)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json(), output)
                self.assertIn("cpu-offload", response.headers["x-local-llm-warning"])
                self.assertEqual(captured[0].url.path, "/v1/" + endpoint)
                self.assertEqual(json.loads(captured[0].content), payload)
                self.assertEqual(captured[0].headers["anthropic-version"], "2023-06-01")
                self.assertEqual(captured[0].headers["anthropic-beta"], "test-beta")
                self.assertNotIn("authorization", captured[0].headers)
                self.assertNotIn("x-api-key", captured[0].headers)
                self.assertFalse(self.app.state.scheduler.lock.locked())

    async def test_native_stream_events_are_not_rewritten(self):
        for endpoint, data in (
            (
                "messages",
                b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1"}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
            ),
            (
                "responses",
                b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_1"}}\n\nevent: response.completed\ndata: {"type":"response.completed"}\n\n',
            ),
        ):
            with self.subTest(endpoint=endpoint):
                transport = httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, content=data, headers={"content-type": "text/event-stream"}
                    )
                )
                async with self.browser() as client:
                    with patch(
                        "llama_on_demand.service.httpx.AsyncClient",
                        side_effect=lambda **kwargs: self.client_type(transport=transport),
                    ):
                        response = await client.post(
                            "/v1/" + endpoint, json={"model": "local", "stream": True}
                        )
                self.assertEqual(response.content, data)
                self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
                self.assertFalse(self.app.state.scheduler.lock.locked())

    async def test_local_errors_match_the_protocol_and_do_not_load_unauthorized_models(self):
        for endpoint in ("messages", "responses", "chat/completions"):
            with self.subTest(endpoint=endpoint):
                self.runtime.start.reset_mock()
                async with self.browser(
                    {"Authorization": "Bearer wrong", "x-api-key": "wrong"}
                ) as client:
                    denied = await client.post("/v1/" + endpoint, json={"model": "local"})
                self.assertEqual(denied.status_code, 401)
                async with self.browser() as client:
                    invalid = await client.post("/v1/" + endpoint, content=b"{broken")
                    unknown = await client.post("/v1/" + endpoint, json={"model": "missing"})
                    self.runtime.start.assert_not_awaited()
                    with patch.object(
                        self.runtime, "start", side_effect=RuntimeError("load failed")
                    ):
                        failed = await client.post("/v1/" + endpoint, json={"model": "local"})
                for response, status, kind in (
                    (denied, 401, "authentication_error"),
                    (invalid, 400, "invalid_request_error"),
                    (unknown, 404, "not_found_error"),
                    (failed, 503, "overloaded_error"),
                ):
                    self.assertEqual(response.status_code, status)
                    self.assertIsInstance(response.json()["error"]["message"], str)
                    if endpoint == "messages":
                        self.assertEqual(response.json()["type"], "error")
                        self.assertEqual(response.json()["error"]["type"], kind)
                self.assertFalse(self.app.state.scheduler.lock.locked())

    async def test_worker_errors_are_forwarded_even_when_stream_is_requested(self):
        for endpoint in ("messages", "responses"):
            with self.subTest(endpoint=endpoint):
                body = (
                    b'{"error":{"message":"unsupported parameter","type":"invalid_request_error"}}'
                )
                transport = httpx.MockTransport(
                    lambda request: httpx.Response(
                        400, content=body, headers={"content-type": "application/json"}
                    )
                )
                async with self.browser() as client:
                    with patch(
                        "llama_on_demand.service.httpx.AsyncClient",
                        side_effect=lambda **kwargs: self.client_type(transport=transport),
                    ):
                        response = await client.post(
                            "/v1/" + endpoint, json={"model": "local", "stream": True}
                        )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.content, body)
                self.assertFalse(self.app.state.scheduler.lock.locked())

    async def test_counting_returns_json_and_queue_errors_preserve_retry_headers(self):
        for endpoint in ("messages/count_tokens", "responses/input_tokens"):
            with self.subTest(endpoint=endpoint):
                transport = httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"input_tokens": 10})
                )
                async with self.browser() as client:
                    with patch(
                        "llama_on_demand.service.httpx.AsyncClient",
                        side_effect=lambda **kwargs: self.client_type(transport=transport),
                    ):
                        response = await client.post(
                            "/v1/" + endpoint, json={"model": "local", "stream": True}
                        )
                    self.assertEqual(response.json(), {"input_tokens": 10})
                    self.assertEqual(response.headers["content-type"], "application/json")
                    self.app.state.scheduler.config["max_queue"] = 0
                    try:
                        full = await client.post("/v1/" + endpoint, json={"model": "local"})
                    finally:
                        self.app.state.scheduler.config.pop("max_queue")
                    self.assertEqual(full.status_code, 503)
                    self.assertEqual(full.headers["retry-after"], "5")
                    self.assertIn("error", full.json())

    async def test_responses_and_messages_share_the_same_streaming_queue(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        class Body(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                yield b"event: response.created\ndata: {}\n\n"
                await finish.wait()

        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, stream=Body(), headers={"content-type": "text/event-stream"}
            )
        )
        async with self.browser() as client:
            with patch(
                "llama_on_demand.service.httpx.AsyncClient",
                side_effect=lambda **kwargs: self.client_type(transport=transport),
            ):
                first = asyncio.create_task(
                    client.post("/v1/responses", json={"model": "local", "stream": True})
                )
                await asyncio.wait_for(entered.wait(), 5)
                second = asyncio.create_task(
                    client.post("/v1/messages", json={"model": "local", "stream": True})
                )
                try:
                    async with asyncio.timeout(5):
                        while self.app.state.scheduler.waiting != 1:
                            await asyncio.sleep(0.01)
                    self.assertEqual(self.runtime.start.await_count, 1)
                finally:
                    finish.set()
                results = await asyncio.wait_for(asyncio.gather(first, second), 5)
                self.assertEqual([r.status_code for r in results], [200, 200])
                self.assertEqual(self.runtime.start.await_count, 2)
                self.assertFalse(self.app.state.scheduler.lock.locked())

    async def test_cancelled_streams_stop_worker_and_release_slot_for_both_protocols(self):
        for endpoint in ("messages", "responses"):
            with self.subTest(endpoint=endpoint):
                entered = asyncio.Event()

                class Body(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        entered.set()
                        await asyncio.Event().wait()
                        yield b""

                transport = httpx.MockTransport(
                    lambda request: httpx.Response(
                        200, stream=Body(), headers={"content-type": "text/event-stream"}
                    )
                )
                self.runtime.stop.reset_mock()
                async with self.browser() as client:
                    with patch(
                        "llama_on_demand.service.httpx.AsyncClient",
                        side_effect=lambda **kwargs: self.client_type(transport=transport),
                    ):
                        task = asyncio.create_task(
                            client.post("/v1/" + endpoint, json={"model": "local", "stream": True})
                        )
                        await asyncio.wait_for(entered.wait(), 5)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                self.runtime.stop.assert_awaited()
                self.assertFalse(self.app.state.scheduler.lock.locked())
