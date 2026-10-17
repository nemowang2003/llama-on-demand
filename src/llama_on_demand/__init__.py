"""User-service entrypoint and GGUF management CLI."""

import argparse
import json
import logging
import os
from pathlib import Path

from . import models
from .config import load


def main():
    parser = argparse.ArgumentParser(
        description="On-demand llama.cpp user service and GGUF catalog"
    )
    parser.add_argument("--config", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve")
    listen = server.add_mutually_exclusive_group()
    listen.add_argument("--port", type=int, default=0, help="TCP port; 0 lets the OS allocate it")
    listen.add_argument("--uds", type=Path, help="Unix socket path (permissions 0600)")
    server.add_argument("--host")
    server.add_argument("--endpoint-file", type=Path)
    status = commands.add_parser("status")
    target = status.add_mutually_exclusive_group()
    target.add_argument("--url")
    target.add_argument("--uds", type=Path, help="Query the service through a Unix socket")
    status.add_argument("--endpoint-file", type=Path)
    endpoint = commands.add_parser("endpoint", help="Print the running service's endpoint")
    endpoint.add_argument("--endpoint-file", type=Path)
    catalog = commands.add_parser("models").add_subparsers(dest="model_command", required=True)
    catalog.add_parser("list")
    for command in ("pull", "inspect"):
        cmd = catalog.add_parser(command)
        cmd.add_argument("name")
    imp = catalog.add_parser("import")
    imp.add_argument("name")
    imp.add_argument("path", type=Path)
    add = catalog.add_parser("add")
    add.add_argument("name")
    add.add_argument("repo")
    add.add_argument("file")
    add.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.command == "serve" and args.uds and args.host:
        parser.error("--uds cannot be combined with --host")
    config = load(args.config)
    try:
        if args.command == "serve":
            from .server import serve
            from .service import create_app

            logging.basicConfig(level=logging.INFO)
            serve(
                create_app(config),
                host=args.host or "127.0.0.1",
                port=args.port,
                uds=args.uds,
                endpoint_file=args.endpoint_file,
            )
        elif args.command == "endpoint":
            from .server import read_endpoint

            print(read_endpoint(args.endpoint_file))
        elif args.command == "status":
            import httpx

            from .server import read_endpoint

            endpoint = args.url or (
                f"unix://{args.uds.expanduser()}" if args.uds else read_endpoint(args.endpoint_file)
            )
            if endpoint.startswith("unix://"):
                transport = httpx.HTTPTransport(uds=endpoint.removeprefix("unix://"))
                url = "http://localhost"
            else:
                transport, url = None, endpoint
            with httpx.Client(transport=transport, trust_env=False) as client:
                result = client.get(
                    url.rstrip("/") + "/status",
                    headers={"Authorization": "Bearer " + os.environ["LLAMA_ON_DEMAND_API_KEY"]},
                )
            result.raise_for_status()
            print(json.dumps(result.json(), indent=2))
        elif args.model_command == "list":
            print(json.dumps(models.catalog(config), indent=2))
        elif args.model_command == "inspect":
            path, record = models.model_path(args.name, config)
            print(
                json.dumps(
                    {**record, "path": str(path), "size_bytes": path.stat().st_size}, indent=2
                )
            )
        elif args.model_command == "pull":
            print(models.pull(args.name, config))
        elif args.model_command == "import":
            print(models.import_model(args.name, args.path))
        else:
            models.save(
                args.name, {"repo": args.repo, "file": args.file, "revision": args.revision}
            )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"llama-on-demand: {error}\n")
