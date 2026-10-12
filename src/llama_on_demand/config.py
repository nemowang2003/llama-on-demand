import os
from pathlib import Path

import tomllib


def load(path: Path | None = None) -> dict:
    path = path or Path(
        os.environ.get(
            "LLAMA_ON_DEMAND_CONFIG",
            Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
            / "llama-on-demand/config.toml",
        )
    )
    return tomllib.loads(path.read_text()) if path.exists() else {}
