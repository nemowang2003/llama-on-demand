"""GGUF catalog. Existing files are registered without being copied."""

import json
import os
import tempfile
from pathlib import Path

from filelock import FileLock


def data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "llama-on-demand"


def registry_path() -> Path:
    return data_dir() / "models.json"


def _source(record: dict) -> dict:
    return {
        key: record.get(key, "main" if key == "revision" else None)
        for key in ("repo", "file", "revision")
    }


def _entry(record: dict, declarative: dict | None = None) -> dict:
    if "definition" in record:
        return record
    # Older pulls saved a merged definition alongside the downloaded path.
    # Its provenance is unknown: prefer TOML for declarative aliases, retaining
    # the old definition for aliases that exist only in the user registry.
    if record.get("path") and record.get("repo") and record.get("file"):
        return {
            "definition": {}
            if declarative is not None
            else {k: v for k, v in record.items() if k != "path"},
            "download": {"source": _source(record), "path": record["path"]},
        }
    return {"definition": record}


def _read_registry() -> dict:
    path = registry_path()
    return json.loads(path.read_text()) if path.exists() else {}


def _catalog(config: dict | None, records: dict) -> dict[str, dict]:
    declared = (config or {}).get("models", {})
    result = {name: dict(record) for name, record in declared.items()}
    for name, raw in records.items():
        entry = _entry(raw, declared.get(name))
        definition = entry["definition"]
        if name not in declared and not definition:
            continue
        base = result.get(name, {})
        record = {**base, **definition}
        if "path" not in definition and _source(record) != _source(base):
            # A CLI source override cannot inherit weights from the TOML source.
            record.pop("path", None)
        download = entry.get("download")
        if download and record.get("repo") and record.get("file"):
            if download["source"] == _source(record):
                record["path"] = download["path"]
            else:
                # Do not fall back to an old configured path after the source changes.
                record.pop("path", None)
        result[name] = record
    return result


def catalog(config: dict | None = None) -> dict[str, dict]:
    return _catalog(config, _read_registry())


def _update(name: str, update) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Lock a separate, stable file: the registry inode changes on replacement.
    with FileLock(str(path.with_suffix(".lock"))):
        records = _read_registry()
        records[name] = update(records.get(name, {}))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(json.dumps(records, indent=2) + "\n")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def save(name: str, record: dict) -> None:
    _update(name, lambda previous: {"definition": record})


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
    records = _read_registry()
    previous_entry = records.get(name, {})
    record = _catalog(config, records)[name]
    from .download import download_file

    path = Path(
        download_file(
            repo_id=record["repo"], filename=record["file"], revision=record.get("revision", "main")
        )
    )
    validate(path)
    declared = config.get("models", {})

    def publish(previous):
        if previous != previous_entry:
            raise RuntimeError(
                f"model {name} changed while downloading; registry unchanged; rerun models pull {name} if needed"
            )
        return {
            "definition": _entry(previous, declared.get(name))["definition"],
            "download": {"source": _source(record), "path": str(path)},
        }

    _update(name, publish)
    return path


def model_path(name: str, config: dict) -> tuple[Path, dict]:
    record = catalog(config)[name]
    if not record.get("path"):
        raise ValueError(f"model {name} not downloaded; run llama-on-demand models pull {name}")
    path = Path(record["path"]).expanduser()
    validate(path)
    return path, record
