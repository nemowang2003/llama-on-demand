"""GGUF catalog. Existing files are registered without being copied."""

import json
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "llama-on-demand"


def registry_path() -> Path:
    return data_dir() / "models.json"


def catalog(config: dict | None = None) -> dict[str, dict]:
    result = dict((config or {}).get("models", {}))
    path = registry_path()
    if path.exists():
        for name, record in json.loads(path.read_text()).items():
            result[name] = {**result.get(name, {}), **record}
    return result


def save(name: str, record: dict) -> None:
    path = registry_path()
    records = json.loads(path.read_text()) if path.exists() else {}
    records[name] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(path)


def import_model(name: str, path: Path) -> Path:
    path = path.expanduser().resolve()
    validate(path)
    save(name, {"path": str(path)})
    return path


def validate(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing GGUF model: {path}; use llama-on-demand models pull or import")
    with path.open("rb") as model:
        if model.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF model: {path}")


def pull(name: str, config: dict) -> Path:
    record = catalog(config)[name]
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=record["repo"], filename=record["file"], revision=record.get("revision", "main")
        )
    )
    validate(path)
    save(name, {**record, "path": str(path)})
    return path


def model_path(name: str, config: dict) -> tuple[Path, dict]:
    record = catalog(config)[name]
    if not record.get("path"):
        raise ValueError(f"model {name} not downloaded; run llama-on-demand models pull {name}")
    path = Path(record["path"]).expanduser()
    validate(path)
    return path, record
