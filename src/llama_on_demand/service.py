"""Local OpenAI/Anthropic-compatible API with a global FIFO admission lock."""

import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager, suppress

import anyio
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import catalog
from .runtime import Runtime

INFERENCE_ENDPOINTS = {
    "chat/completions",
    "completions",
    "embeddings",
    "responses",
    "responses/input_tokens",
    "messages",
    "messages/count_tokens",
}
STREAMING_ENDPOINTS = {"chat/completions", "completions", "responses", "messages"}


class Scheduler:
    def __init__(self, config, runtime):
        self.config, self.runtime = config, runtime
        self.lock = asyncio.Lock()  # asyncio locks admit waiters in arrival order.
        self.waiting = 0

    async def acquire(self):
        if self.waiting >= self.config.get("max_queue", 128):
            raise HTTPException(503, "local inference queue is full", headers={"Retry-After": "5"})
        self.waiting += 1
        try:
            await asyncio.wait_for(self.lock.acquire(), self.config.get("queue_timeout", 3600))
        except TimeoutError as error:
            raise HTTPException(504, "local inference queue wait timed out") from error
        finally:
            self.waiting -= 1

    def release(self):
        self.runtime.last_used = time.monotonic()
        self.lock.release()

    async def reap(self):
        while True:
            await asyncio.sleep(0.5)
            if self.lock.locked() or self.waiting:
                continue
            async with self.lock:
                if time.monotonic() - self.runtime.last_used >= self.config.get("idle_seconds", 30):
                    await self.runtime.stop()


def create_app(config: dict, runtime=None):
    runtime = runtime or Runtime(config)
    scheduler = Scheduler(config, runtime)
    auth_required = config.get("auth_required", True)
    if not isinstance(auth_required, bool):
        raise ValueError("auth_required must be a boolean")
    key = os.environ.get("LLAMA_ON_DEMAND_API_KEY", "")
    if auth_required and not key:
        raise ValueError("set LLAMA_ON_DEMAND_API_KEY before starting the service")

    @asynccontextmanager
    async def lifespan(app):
        reaper = asyncio.create_task(scheduler.reap())
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
            await runtime.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.scheduler = scheduler
    app.state.runtime = runtime

    @app.exception_handler(StarletteHTTPException)
    async def api_error(request: Request, error: StarletteHTTPException):
        if request.url.path.startswith("/v1/messages"):
            kind = {
                400: "invalid_request_error",
                401: "authentication_error",
                403: "permission_error",
                404: "not_found_error",
                413: "request_too_large",
                429: "rate_limit_error",
                503: "overloaded_error",
            }.get(error.status_code, "api_error")
            content = {"type": "error", "error": {"type": kind, "message": str(error.detail)}}
        elif request.url.path.startswith("/v1/"):
            kind = "server_error" if error.status_code >= 500 else "invalid_request_error"
            content = {
                "error": {"message": str(error.detail), "type": kind, "param": None, "code": None}
            }
        else:
            content = {"detail": error.detail}
        return JSONResponse(content, status_code=error.status_code, headers=error.headers)

    def authorize(request):
        if not auth_required:
            return
        bearer = hmac.compare_digest(
            request.headers.get("authorization", "").encode(), f"Bearer {key}".encode()
        )
        api_key = hmac.compare_digest(request.headers.get("x-api-key", "").encode(), key.encode())
        if not (bearer or api_key):
            raise HTTPException(401, "invalid API key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/status")
    async def status(request: Request):
        authorize(request)
        return {**runtime.status(), "waiting": scheduler.waiting, "active": scheduler.lock.locked()}

    @app.get("/v1/models")
    async def models(request: Request):
        authorize(request)
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "owned_by": "local", "created": 0}
                for name in catalog(config)
            ],
        }

    @app.post("/v1/{endpoint:path}")
    async def infer(endpoint: str, request: Request):
        authorize(request)
        if endpoint not in INFERENCE_ENDPOINTS:
            raise HTTPException(404, "unsupported inference endpoint")
        try:
            payload = await request.json()
        except ValueError as error:
            raise HTTPException(400, "invalid JSON request") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            raise HTTPException(400, "request must contain a model name")
        model = payload.get("model")
        if model not in catalog(config):
            raise HTTPException(404, "unknown local model")
        await scheduler.acquire()
        client = response = None
        owns_lock = True
        try:
            if await request.is_disconnected():
                raise HTTPException(499, "client disconnected while queued")
            await runtime.start(model)
            client = httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(uds=str(runtime.socket_path)),
                timeout=httpx.Timeout(config.get("request_timeout", 3600), connect=10),
            )
            upstream = client.build_request(
                "POST",
                f"{runtime.base_url}/v1/{endpoint}",
                json=payload,
                headers={
                    name: request.headers[name]
                    for name in ("anthropic-version", "anthropic-beta")
                    if name in request.headers
                },
            )
            response = await client.send(upstream, stream=True)
            headers = {}
            if runtime.warning:
                headers["x-local-llm-warning"] = runtime.warning
            if (
                response.status_code >= 400
                or not payload.get("stream")
                or endpoint not in STREAMING_ENDPOINTS
            ):
                body = await response.aread()
                from fastapi.responses import Response

                return Response(
                    body,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.headers.get("content-type", "application/json"),
                )

            completed = False

            async def stream():
                nonlocal completed
                async for chunk in response.aiter_bytes():
                    yield chunk
                completed = True

            class InferenceResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Cover header sending too: the body iterator may never start.
                        with anyio.CancelScope(shield=True):
                            try:
                                try:
                                    await response.aclose()
                                finally:
                                    await client.aclose()
                            finally:
                                try:
                                    if not completed:
                                        await runtime.stop()
                                finally:
                                    scheduler.release()

            streaming_response = InferenceResponse(
                stream(),
                status_code=response.status_code,
                headers=headers,
                media_type="text/event-stream",
            )
            owns_lock = False
            return streaming_response
        except (ValueError, RuntimeError, OSError, TimeoutError, httpx.HTTPError) as error:
            await runtime.stop()
            raise HTTPException(503, str(error)) from error
        except BaseException:
            await runtime.stop()
            raise
        finally:
            if owns_lock:
                try:
                    if response:
                        await response.aclose()
                    if client:
                        await client.aclose()
                finally:
                    scheduler.release()

    return app
