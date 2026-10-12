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
    server.add_argument("--port", type=int, default=4001)
    server.add_argument("--host", default="127.0.0.1")
    status = commands.add_parser("status")
    status.add_argument("--url", default="http://127.0.0.1:4001")
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
    config = load(args.config)
    try:
        if args.command == "serve":
            import uvicorn

            from .service import create_app

            logging.basicConfig(level=logging.INFO)
            uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1)
        elif args.command == "status":
            import httpx

            result = httpx.get(
                args.url + "/status",
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
