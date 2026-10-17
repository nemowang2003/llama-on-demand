"""Single-process listener, endpoint discovery and systemd readiness."""

import logging
import os
import signal
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path

import uvicorn
from filelock import FileLock

logger = logging.getLogger(__name__)


def runtime_dir() -> Path:
    if directory := os.environ.get("XDG_RUNTIME_DIR"):
        return Path(directory) / "llama-on-demand"
    return Path(tempfile.gettempdir()) / f"llama-on-demand-{os.getuid()}"


def endpoint_path() -> Path:
    return runtime_dir() / "endpoint"


def read_endpoint(path: Path | None = None) -> str:
    path = path or endpoint_path()
    if not path.exists():
        raise ValueError(
            f"no running service endpoint at {path}; start the service or specify --url/--uds"
        )
    return path.read_text().strip()


@contextmanager
def unix_listener(path: Path):
    """Never replace an existing listener or file."""
    path = path.expanduser().absolute()
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        identity = path.lstat()
        try:
            path.chmod(0o600)
            listener.listen(2048)
            yield listener
        finally:
            if path.exists() and path.lstat().st_ino == identity.st_ino:
                path.unlink()


def notify_ready() -> None:
    if address := os.environ.get("NOTIFY_SOCKET"):
        if address.startswith("@"):
            address = "\0" + address[1:]
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notifier:
            notifier.connect(address)
            notifier.sendall(b"READY=1")


class Server(uvicorn.Server):
    def __init__(self, app, endpoint_file: Path, uds: Path | None, host: str, port: int):
        super().__init__(
            uvicorn.Config(
                app,
                host=host,
                port=port,
                workers=1,
                proxy_headers=False,
                timeout_graceful_shutdown=25,
            )
        )
        self.endpoint_file = endpoint_file
        self.uds = uds

    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        if not self.started:
            return
        try:
            if self.uds:
                endpoint = f"unix://{self.uds}"
            else:
                addresses = [
                    sock.getsockname() for server in self.servers for sock in server.sockets
                ]
                if len({address[1] for address in addresses}) != 1:
                    raise ValueError(
                        "host resolved to multiple random ports; use an explicit IP address"
                    )
                host, port, *_ = addresses[0]
                host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
                endpoint = f"http://{'[' + host + ']' if ':' in host else host}:{port}"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.endpoint_file.parent, delete=False
                ) as output:
                    temporary = Path(output.name)
                    output.write(endpoint + "\n")
                temporary.replace(self.endpoint_file)
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
            logger.info("Listening at %s; endpoint file: %s", endpoint, self.endpoint_file)
            notify_ready()
        except BaseException:
            await self.shutdown(sockets=sockets)
            raise


def serve(
    app, *, host="127.0.0.1", port=0, uds: Path | None = None, endpoint_file: Path | None = None
):
    endpoint_file = (endpoint_file or endpoint_path()).expanduser().absolute()
    uds = uds.expanduser().absolute() if uds else None
    if uds == endpoint_file:
        raise ValueError("socket and endpoint file must have different paths")
    endpoint_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # A stable discovery path belongs to exactly one service instance. Locking
    # also lets us discard a stale endpoint safely after a crash.
    with FileLock(str(endpoint_file) + ".lock", timeout=0):
        endpoint_file.unlink(missing_ok=True)

        def terminate(signum, frame):
            # Uvicorn re-emits SIGTERM after graceful shutdown. Unwind normally
            # so discovery files and the Unix listener are removed too.
            raise SystemExit(0)

        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            server = Server(app, endpoint_file, uds, host, port)
            if uds:
                with unix_listener(uds) as listener:
                    server.run(sockets=[listener])
            else:
                server.run()
        finally:
            endpoint_file.unlink(missing_ok=True)
            signal.signal(signal.SIGTERM, previous)
