import asyncio
import json
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


class DownloadCacheTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.registry = root / "models.json"
        self.enterContext(patch("llama_on_demand.models.registry_path", return_value=self.registry))
        self.weights = root / "model.gguf"
        self.weights.write_bytes(b"GGUFtest")
        self.download = self.enterContext(
            patch("llama_on_demand.download.download_file", return_value=str(self.weights))
        )
        self.definition = {
            "repo": "owner/model",
            "file": "model.gguf",
            "options": {"ctx-size": 4096},
        }
        self.config = {"models": {"m": self.definition}}

    def test_pull_keeps_options_declarative_and_caches_only_download_identity(self):
        models.pull("m", self.config)
        saved = json.loads(self.registry.read_text())["m"]
        self.assertEqual(saved["definition"], {})
        self.assertNotIn("options", saved["download"])
        self.definition["options"] = {"ctx-size": 8192}
        path, record = models.model_path("m", self.config)
        self.assertEqual(path, self.weights)
        self.assertEqual(record["options"], {"ctx-size": 8192})
        self.assertNotIn("path", self.definition)

    def test_each_source_change_requires_pull_and_downloads_from_the_new_source(self):
        for field, new_value in (
            ("repo", "owner/new"),
            ("file", "new.gguf"),
            ("revision", "abc123"),
        ):
            with self.subTest(field=field):
                self.registry.unlink(missing_ok=True)
                definition = {**self.definition, "path": "/old/configured.gguf"}
                config = {"models": {"m": definition}}
                models.pull("m", config)
                definition[field] = new_value
                with self.assertRaisesRegex(ValueError, "models pull m"):
                    models.model_path("m", config)
                models.pull("m", config)
                self.download.assert_called_with(
                    repo_id=definition["repo"],
                    filename=definition["file"],
                    revision=definition.get("revision", "main"),
                )
                self.assertEqual(models.model_path("m", config)[0], self.weights)

    def test_default_revision_matches_explicit_main(self):
        models.pull("m", self.config)
        self.definition["revision"] = "main"
        self.assertEqual(models.model_path("m", self.config)[0], self.weights)

    def test_cli_definitions_override_toml_and_survive_pull(self):
        override = {"repo": "owner/override", "file": "override.gguf", "revision": "commit"}
        models.save("m", override)
        models.pull("m", self.config)
        self.assertEqual(
            models.catalog(self.config)["m"],
            {**override, "options": self.definition["options"], "path": str(self.weights)},
        )
        self.assertEqual(models.catalog()["m"], {**override, "path": str(self.weights)})
        self.assertEqual(json.loads(self.registry.read_text())["m"]["definition"], override)

    def test_import_replaces_download_cache_with_explicit_path(self):
        models.pull("m", self.config)
        models.import_model("m", self.weights)
        self.definition["repo"] = "owner/new"
        self.assertEqual(models.model_path("m", self.config)[0], self.weights)

    def test_cli_source_change_cannot_inherit_toml_weights(self):
        self.definition["path"] = str(self.weights)
        for field, value in (("repo", "owner/new"), ("file", "new.gguf"), ("revision", "commit")):
            with self.subTest(field=field):
                models.save("m", {field: value})
                with self.assertRaisesRegex(ValueError, "models pull m"):
                    models.model_path("m", self.config)
                models.pull("m", self.config)
                self.assertEqual(models.model_path("m", self.config)[0], self.weights)
        models.import_model("m", self.weights)
        self.assertEqual(models.model_path("m", self.config)[0], self.weights)

    def test_import_during_download_preserves_the_newer_selection(self):
        imported = self.weights.with_name("imported.gguf")
        imported.write_bytes(b"GGUFimported")

        def concurrent_import(**kwargs):
            models.import_model("m", imported)
            return str(self.weights)

        self.download.side_effect = concurrent_import
        with self.assertRaisesRegex(RuntimeError, "changed while downloading"):
            models.pull("m", self.config)
        self.assertEqual(models.model_path("m", self.config)[0], imported)
        self.assertNotIn("download", json.loads(self.registry.read_text())["m"])

    def test_removed_declarative_alias_is_not_resurrected_by_download_cache(self):
        models.pull("m", self.config)
        self.assertNotIn("m", models.catalog())

    def test_switching_to_a_local_toml_path_does_not_require_download(self):
        models.pull("m", self.config)
        local = self.weights.with_name("local.gguf")
        local.write_bytes(b"GGUFtest")
        self.config["models"]["m"] = {"path": str(local)}
        self.assertEqual(models.model_path("m", self.config)[0], local)

    def test_legacy_snapshot_uses_current_toml_and_is_upgraded_on_pull(self):
        self.registry.write_text(json.dumps({"m": {**self.definition, "path": str(self.weights)}}))
        self.definition["options"] = {"ctx-size": 8192}
        self.assertEqual(models.model_path("m", self.config)[1]["options"], {"ctx-size": 8192})
        self.definition["repo"] = "owner/new"
        with self.assertRaisesRegex(ValueError, "models pull m"):
            models.model_path("m", self.config)
        models.pull("m", self.config)
        self.download.assert_called_with(
            repo_id="owner/new", filename="model.gguf", revision="main"
        )
        self.assertEqual(json.loads(self.registry.read_text())["m"]["definition"], {})

    def test_legacy_registry_only_model_keeps_its_definition(self):
        self.registry.write_text(json.dumps({"m": {**self.definition, "path": str(self.weights)}}))
        self.assertEqual(models.model_path("m", {})[0], self.weights)
        models.pull("m", {})
        self.assertEqual(json.loads(self.registry.read_text())["m"]["definition"], self.definition)

    def test_failed_download_preserves_previous_cache(self):
        models.pull("m", self.config)
        previous = self.registry.read_bytes()
        self.download.side_effect = OSError("offline")
        with self.assertRaises(OSError):
            models.pull("m", self.config)
        self.assertEqual(self.registry.read_bytes(), previous)

    def test_definition_changed_during_download_is_not_overwritten(self):
        models.save("m", self.definition)

        def concurrent_edit(**kwargs):
            models.save("m", {**self.definition, "repo": "owner/new"})
            return str(self.weights)

        self.download.side_effect = concurrent_edit
        with self.assertRaisesRegex(RuntimeError, "changed while downloading"):
            models.pull("m", self.config)
        self.assertEqual(models.catalog(self.config)["m"]["repo"], "owner/new")
        with self.assertRaisesRegex(ValueError, "models pull m"):
            models.model_path("m", self.config)


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

    async def check_options(self, runtime, extra_options=None):
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
                        {
                            "options": {
                                "embedding": True,
                                "mlock": False,
                                "ctx-size": 4096,
                                **(extra_options or {}),
                            }
                        },
                    ),
                ),
                patch("llama_on_demand.runtime.asyncio.create_subprocess_exec", spawn),
                patch(
                    "llama_on_demand.runtime.httpx.AsyncClient",
                    side_effect=lambda **kwargs: client_type(transport=transport),
                ),
            ):
                await runtime.start("new")
                await runtime.start("new")
                spawn.assert_awaited_once()
            await runtime._reader
            command = spawn.call_args.args
            self.assertIn("--embedding", command)
            self.assertNotIn("True", command)
            self.assertNotIn("False", command)
            self.assertNotIn("--mlock", command)
            self.assertEqual(command[command.index("--ctx-size") + 1], "4096")
            for key in extra_options or {}:
                self.assertNotIn(f"--{key}", command)
            self.assertEqual(runtime.model, "new")
            runtime.process = None
            await runtime.stop()

    async def test_boolean_options_are_flags_and_false_is_omitted(self):
        await self.check_options(Runtime({}))

    async def test_same_alias_reloads_when_model_definition_changes(self):
        runtime = Runtime({})
        runtime.model = "new"
        runtime._loaded_spec = (Path("old.gguf"), {"options": {"ctx-size": 1024}})
        with (
            patch.object(runtime, "process", SimpleNamespace(returncode=None)),
            patch.object(runtime, "stop", new_callable=AsyncMock) as stop,
        ):
            await self.check_options(runtime)
        self.assertEqual(stop.await_count, 2)

    async def test_same_alias_revalidates_source_before_reusing_worker(self):
        runtime = Runtime({})
        runtime.model = "m"
        with (
            patch.object(runtime, "process", SimpleNamespace(returncode=None)),
            patch(
                "llama_on_demand.runtime.model_path", side_effect=ValueError("run models pull m")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "models pull m"):
                await runtime.start("m")

    async def test_gpu_layer_options_warn_and_allow_start_with_automatic_fitting(self):
        options = {"gpu_layers": 0, "gpu-layers": 8, "n_gpu_layers": 16, "n-gpu-layers": 32}
        with self.assertLogs("llama_on_demand.runtime", level="WARNING") as logs:
            await self.check_options(Runtime({}), options)
        self.assertEqual(len(logs.output), len(options))
        for key in options:
            self.assertTrue(any(f"option {key}=" in entry for entry in logs.output))


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
