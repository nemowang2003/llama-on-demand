import hashlib
import shutil
import tempfile
import threading
import unittest
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub import hf_hub_download

from llama_on_demand import models
from llama_on_demand.download import download_file


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.data = b"GGUF" + b"weights" * 600000
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.blob = self.root / "models--owner--repo" / "blobs" / self.sha
        self.partial = self.blob.with_name(self.sha + ".aria2-part")
        self.ranges = []
        data, ranges = self.data, self.ranges

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers.get("Range")
                ranges.append(header)
                start = int(header.split("=")[1].split("-")[0]) if header else 0
                body = data[start:]
                self.send_response(206 if header else 200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                if header:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.end_headers()
                # aria2 may close its probe request before issuing a resume range.
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        self.url = f"http://127.0.0.1:{server.server_port}/weights"
        self.metadata = SimpleNamespace(
            etag=self.sha, commit_hash="a" * 40, size=len(self.data), location=self.url
        )
        self.enterContext(patch("llama_on_demand.download.constants.HF_HUB_CACHE", str(self.root)))
        self.enterContext(
            patch("llama_on_demand.download.get_hf_file_metadata", return_value=self.metadata)
        )
        self.finish = self.enterContext(
            patch("llama_on_demand.download.hf_hub_download", return_value=str(self.blob))
        )
        self.enterContext(
            patch("llama_on_demand.models.registry_path", return_value=self.root / "models.json")
        )
        self.config = {"models": {"m": {"repo": "owner/repo", "file": "weights.gguf"}}}

    def test_real_aria2_resumes_partial_and_publishes_verified_model(self):
        self.assertIsNotNone(shutil.which("aria2c"), "run tests with the Nix check environment")
        self.partial.parent.mkdir(parents=True)
        resume_at = 2 * 1024 * 1024  # aria2 resumes at complete piece boundaries.
        self.partial.write_bytes(self.data[:resume_at])
        path = models.pull("m", self.config)
        self.assertEqual(path.read_bytes(), self.data)
        self.assertTrue(
            any(header and header.startswith(f"bytes={resume_at}-") for header in self.ranges)
        )
        self.assertFalse(self.partial.exists())
        self.assertEqual(models.model_path("m", self.config)[0], self.blob)
        self.assertEqual((self.root / "models--owner--repo/refs/main").read_text(), "a" * 40)
        self.finish.assert_called_once_with(
            repo_id="owner/repo", filename="weights.gguf", revision="a" * 40
        )
        with patch("llama_on_demand.download.subprocess.run") as transfer:
            self.assertEqual(models.pull("m", self.config), self.blob)
            transfer.assert_not_called()

    def test_failed_transfer_preserves_partial_and_previous_registration(self):
        models.save("m", {"path": "/previous/model.gguf"})
        previous = models.registry_path().read_bytes()

        def fail(*args, **kwargs):
            self.partial.write_bytes(self.data[:1234])
            return SimpleNamespace(returncode=1)

        with patch("llama_on_demand.download.subprocess.run", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "rerun models pull to resume"):
                models.pull("m", self.config)
        self.assertEqual(self.partial.read_bytes(), self.data[:1234])
        self.assertFalse(self.blob.exists())
        self.assertEqual(models.registry_path().read_bytes(), previous)
        self.finish.assert_not_called()

    def test_invalid_download_is_not_published_and_can_be_retried(self):
        for body in (b"short", b"x" * len(self.data)):
            with self.subTest(size=len(body)):

                def corrupt(*args, **kwargs):
                    self.partial.write_bytes(body)
                    return SimpleNamespace(returncode=0)

                with patch("llama_on_demand.download.subprocess.run", side_effect=corrupt):
                    with self.assertRaises(ValueError):
                        models.pull("m", self.config)
                self.assertFalse(self.blob.exists())
                self.assertFalse(self.partial.exists())
                self.assertFalse(models.registry_path().exists())
        self.assertEqual(models.pull("m", self.config).read_bytes(), self.data)

    def test_corrupt_cache_is_repaired_by_real_aria2(self):
        self.blob.parent.mkdir(parents=True)
        self.blob.write_bytes(b"corrupt")
        self.assertEqual(models.pull("m", self.config).read_bytes(), self.data)

    def test_git_blob_hash_is_supported(self):
        self.metadata.etag = hashlib.sha1(
            f"blob {len(self.data)}\0".encode() + self.data
        ).hexdigest()
        path = self.blob.with_name(self.metadata.etag)
        self.finish.return_value = str(path)
        self.assertEqual(download_file("owner/repo", "weights.gguf").read_bytes(), self.data)

    def test_authenticated_corrupt_cache_is_redownloaded_with_the_hub_client(self):
        snapshot = self.root / "models--owner--repo" / "snapshots" / ("a" * 40) / "weights.gguf"
        snapshot.parent.mkdir(parents=True)
        self.blob.parent.mkdir(parents=True)
        snapshot.symlink_to(self.blob)

        def cached_or_download(**kwargs):
            if kwargs.get("force_download"):
                self.blob.write_bytes(self.data)
                return str(snapshot)
            # Exercise the actual Hub client's existing-snapshot fast path.
            return hf_hub_download(**kwargs, cache_dir=self.root)

        self.finish.side_effect = cached_or_download
        with (
            patch("llama_on_demand.download.hf_hub_url", return_value=self.url),
            patch(
                "llama_on_demand.download.build_hf_headers",
                return_value={"authorization": "Bearer dummy"},
            ),
            patch("llama_on_demand.download.subprocess.run") as aria2,
        ):
            for corrupt in (b"GGUFbad", b"x" * len(self.data)):
                with self.subTest(size=len(corrupt)):
                    self.finish.reset_mock()
                    self.blob.write_bytes(corrupt)
                    self.assertEqual(models.pull("m", self.config).read_bytes(), self.data)
                    self.assertEqual(models.model_path("m", self.config)[0], snapshot)
                    self.assertEqual(self.finish.call_count, 2)
                    self.finish.assert_called_with(
                        repo_id="owner/repo",
                        filename="weights.gguf",
                        revision="a" * 40,
                        force_download=True,
                    )
            aria2.assert_not_called()

    def test_authenticated_failed_repair_preserves_previous_registration(self):
        models.save("m", {"path": "/previous/model.gguf"})
        previous = models.registry_path().read_bytes()
        self.blob.parent.mkdir(parents=True)
        self.blob.write_bytes(b"GGUFbad")
        with (
            patch("llama_on_demand.download.hf_hub_url", return_value=self.url),
            patch(
                "llama_on_demand.download.build_hf_headers",
                return_value={"authorization": "Bearer dummy"},
            ),
            patch("llama_on_demand.download.subprocess.run") as aria2,
        ):
            with self.assertRaisesRegex(ValueError, "size does not match"):
                models.pull("m", self.config)
            self.assertEqual(self.finish.call_count, 2)
            self.assertEqual(models.registry_path().read_bytes(), previous)
            aria2.assert_not_called()

    def test_missing_aria2_has_actionable_error(self):
        with patch("llama_on_demand.download.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "use the Nix package"):
                models.pull("m", self.config)
        self.assertFalse(models.registry_path().exists())

    def test_auth_is_not_forwarded_to_aria2_or_cdn(self):
        for same_host in (True, False):
            with self.subTest(same_host=same_host):

                def transfer(command, *, input, text):
                    self.assertNotIn("private-token", " ".join(command))
                    self.assertNotIn("private-token", input)
                    self.partial.write_bytes(self.data)
                    return SimpleNamespace(returncode=0)

                with (
                    patch(
                        "llama_on_demand.download.hf_hub_url",
                        return_value=self.url
                        if same_host
                        else "https://huggingface.co/owner/repo/resolve/main/weights.gguf",
                    ),
                    patch(
                        "llama_on_demand.download.build_hf_headers",
                        return_value={"authorization": "Bearer private-token"},
                    ),
                    patch("llama_on_demand.download.subprocess.run", side_effect=transfer) as aria2,
                ):
                    if same_host:
                        self.blob.parent.mkdir(parents=True, exist_ok=True)
                        self.blob.write_bytes(self.data)
                    models.pull("m", self.config)
                    if same_host:
                        aria2.assert_not_called()
                    else:
                        aria2.assert_called_once()
                self.blob.unlink()
