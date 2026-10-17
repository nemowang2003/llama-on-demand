import asyncio
import json
import os
import signal
import socket
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

from llama_on_demand.server import unix_listener


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def cli(self, env, *args):
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "from llama_on_demand import main; main()",
            *args,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def test_random_port_is_published_before_ready_and_used_by_status(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            endpoint = directory / "endpoint"
            endpoint.write_text("http://127.0.0.1:9\n")
            notify = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.addCleanup(notify.close)
            notify.bind(str(directory / "notify.sock"))
            notify.setblocking(False)
            env = {
                **os.environ,
                "LLAMA_ON_DEMAND_API_KEY": "test",
                "XDG_RUNTIME_DIR": root,
                "NOTIFY_SOCKET": str(directory / "notify.sock"),
            }
            args = ("serve", "--endpoint-file", str(endpoint))
            process = await self.cli(env, *args)
            try:
                ready = await asyncio.wait_for(
                    asyncio.get_running_loop().sock_recv(notify, 1024), 10
                )
                self.assertEqual(ready, b"READY=1")
                url = endpoint.read_text().strip()
                self.assertTrue(url.startswith("http://127.0.0.1:"), url)
                self.assertGreater(int(url.rsplit(":", 1)[1]), 0)
                self.assertEqual(stat.S_IMODE(endpoint.stat().st_mode), 0o600)
                async with httpx.AsyncClient(trust_env=False) as client:
                    self.assertEqual((await client.get(url + "/health")).status_code, 200)
                query = await self.cli(env, "status", "--endpoint-file", str(endpoint))
                output, error = await asyncio.wait_for(query.communicate(), 10)
                self.assertEqual(query.returncode, 0, error.decode())
                self.assertFalse(json.loads(output)["running"])
                query = await self.cli(env, "endpoint", "--endpoint-file", str(endpoint))
                output, error = await asyncio.wait_for(query.communicate(), 10)
                self.assertEqual(query.returncode, 0, error.decode())
                self.assertEqual(output.decode().strip(), url)
                # A second instance must not replace the live instance's address.
                duplicate = await self.cli(env, *args)
                await asyncio.wait_for(duplicate.communicate(), 10)
                self.assertNotEqual(duplicate.returncode, 0)
                self.assertEqual(endpoint.read_text().strip(), url)
            finally:
                if process.returncode is None:
                    process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(process.communicate(), 10)
            self.assertEqual(process.returncode, 0)
            self.assertFalse(endpoint.exists())

    async def test_failed_bind_does_not_publish_an_endpoint_or_signal_readiness(self):
        with tempfile.TemporaryDirectory() as root, socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            endpoint = Path(root) / "endpoint"
            endpoint.write_text("stale\n")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify:
                notify.bind(str(Path(root) / "notify.sock"))
                notify.setblocking(False)
                env = {
                    **os.environ,
                    "LLAMA_ON_DEMAND_API_KEY": "test",
                    "NOTIFY_SOCKET": str(Path(root) / "notify.sock"),
                }
                process = await self.cli(
                    env,
                    "serve",
                    "--port",
                    str(occupied.getsockname()[1]),
                    "--endpoint-file",
                    str(endpoint),
                )
                await asyncio.wait_for(process.communicate(), 10)
                self.assertNotEqual(process.returncode, 0)
                self.assertFalse(endpoint.exists())
                with self.assertRaises(BlockingIOError):
                    notify.recv(1024)

    async def test_cli_socket_status_permissions_auth_and_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "backend.sock"
            env = {
                **os.environ,
                "LLAMA_ON_DEMAND_API_KEY": "test",
                "XDG_DATA_HOME": root,
                "XDG_RUNTIME_DIR": root,
            }
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "from llama_on_demand import main; main()",
                "--config",
                str(Path(root) / "missing.toml"),
                "serve",
                "--uds",
                str(path),
                env=env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                async with asyncio.timeout(10):
                    while not path.exists():
                        if process.returncode is not None:
                            self.fail(f"backend exited with {process.returncode}")
                        await asyncio.sleep(0.01)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                async with httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(uds=str(path)), base_url="http://local"
                ) as client:
                    async with asyncio.timeout(10):
                        while True:
                            try:
                                self.assertEqual((await client.get("/health")).status_code, 200)
                                break
                            except httpx.TransportError:
                                await asyncio.sleep(0.01)
                    self.assertEqual((await client.get("/status")).status_code, 401)
                query = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "from llama_on_demand import main; main()",
                    "status",
                    "--uds",
                    str(path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(query.communicate(), 10)
                self.assertEqual(query.returncode, 0, stderr.decode())
                self.assertFalse(json.loads(stdout)["running"])
                endpoint = Path(root) / "llama-on-demand/endpoint"
                self.assertEqual(endpoint.read_text().strip(), f"unix://{path}")
            finally:
                if process.returncode is None:
                    process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(process.wait(), 10)
            self.assertFalse(path.exists())
            self.assertFalse(endpoint.exists())

    async def test_existing_file_or_listener_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "backend.sock"
            path.write_text("keep")
            with self.assertRaises(OSError), unix_listener(path):
                pass
            self.assertEqual(path.read_text(), "keep")
            path.unlink()
            with unix_listener(path):
                inode = path.stat().st_ino
                with self.assertRaises(OSError), unix_listener(path):
                    pass
                self.assertEqual(path.stat().st_ino, inode)
            self.assertFalse(path.exists())
