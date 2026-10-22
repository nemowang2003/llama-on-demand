import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from llama_on_demand.service import INFERENCE_ENDPOINTS, create_app


class AuthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("LLAMA_ON_DEMAND_API_KEY", None)
        self.runtime = SimpleNamespace(
            last_used=0,
            base_url="http://worker",
            socket_path="/fake.sock",
            warning=None,
            start=AsyncMock(),
            stop=AsyncMock(),
            status=lambda: {"running": False},
        )
        self.enterContext(patch("llama_on_demand.service.catalog", return_value={"local": {}}))
        self.client_type = httpx.AsyncClient

    def app(self, **settings):
        return create_app(settings, self.runtime)

    def browser(self, app):
        # A remote source address ensures auth never depends on loopback detection.
        return self.client_type(
            transport=httpx.ASGITransport(app=app, client=("192.0.2.1", 12345)),
            base_url="http://service",
        )

    def test_default_and_explicit_required_auth_reject_missing_or_empty_server_key(self):
        for settings in ({}, {"auth_required": True}):
            for key in (None, ""):
                with self.subTest(settings=settings, key=key):
                    if key is None:
                        os.environ.pop("LLAMA_ON_DEMAND_API_KEY", None)
                    else:
                        os.environ["LLAMA_ON_DEMAND_API_KEY"] = key
                    with self.assertRaisesRegex(ValueError, "set LLAMA_ON_DEMAND_API_KEY"):
                        self.app(**settings)

    def test_auth_required_requires_a_real_boolean(self):
        for invalid in (None, 0, 1, "false", "true"):
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "auth_required must be a boolean"):
                    self.app(auth_required=invalid)

    async def test_required_auth_accepts_both_headers_and_rejects_spoofed_proxy_headers(self):
        os.environ["LLAMA_ON_DEMAND_API_KEY"] = "test-key"
        for settings in ({}, {"auth_required": True}):
            with self.subTest(settings=settings):
                app = self.app(**settings)
                async with self.browser(app) as client:
                    for headers, status in (
                        ({}, 401),
                        ({"Authorization": "Bearer wrong"}, 401),
                        ({"x-api-key": "wrong"}, 401),
                        (
                            {
                                "Forwarded": "for=127.0.0.1",
                                "X-Forwarded-For": "127.0.0.1",
                                "Tailscale-User-Login": "trusted@example.com",
                            },
                            401,
                        ),
                        ({"Authorization": "Bearer test-key"}, 200),
                        ({"x-api-key": "test-key"}, 200),
                    ):
                        with self.subTest(headers=headers):
                            for path in ("/status", "/v1/models"):
                                result = await client.get(path, headers=headers)
                                self.assertEqual(result.status_code, status, result.text)
                    self.runtime.start.assert_not_awaited()
                    # Rejection happens before model loading for every API format.
                    for endpoint in INFERENCE_ENDPOINTS:
                        result = await client.post("/v1/" + endpoint, json={"model": "local"})
                        self.assertEqual(result.status_code, 401, result.text)
                    self.runtime.start.assert_not_awaited()

    async def test_disabled_auth_allows_all_endpoints_without_credentials_even_if_key_is_set(self):
        for key in (None, "unused-server-key"):
            with self.subTest(key=key):
                if key is None:
                    os.environ.pop("LLAMA_ON_DEMAND_API_KEY", None)
                else:
                    os.environ["LLAMA_ON_DEMAND_API_KEY"] = key
                app = self.app(auth_required=False)
                transport = httpx.MockTransport(
                    lambda request: httpx.Response(200, json={"accepted": True})
                )
                async with self.browser(app) as client:
                    with patch(
                        "llama_on_demand.service.httpx.AsyncClient",
                        side_effect=lambda **kwargs: self.client_type(transport=transport),
                    ):
                        for headers in (
                            {},
                            {"Authorization": "Bearer ignored"},
                            {"x-api-key": "ignored"},
                        ):
                            with self.subTest(headers=headers):
                                for path in ("/status", "/v1/models"):
                                    result = await client.get(path, headers=headers)
                                    self.assertEqual(result.status_code, 200, result.text)
                                for endpoint in INFERENCE_ENDPOINTS:
                                    result = await client.post(
                                        "/v1/" + endpoint, headers=headers, json={"model": "local"}
                                    )
                                    self.assertEqual(result.status_code, 200, result.text)
                                    self.assertEqual(result.json(), {"accepted": True})
                                    self.assertFalse(app.state.scheduler.lock.locked())
