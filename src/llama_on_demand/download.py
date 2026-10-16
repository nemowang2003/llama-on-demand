"""Transfer Hub weights with aria2, publishing only verified cache blobs."""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from filelock import FileLock
from huggingface_hub import constants, get_hf_file_metadata, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers


def _verify(path: Path, size: int, etag: str) -> None:
    if path.stat().st_size != size:
        raise ValueError("downloaded model size does not match Hugging Face metadata")
    with path.open("rb") as source:
        if len(etag) == 64:
            digest = hashlib.file_digest(source, "sha256")
        else:
            # Git stores the hash of the blob header followed by its contents.
            digest = hashlib.sha1(f"blob {size}\0".encode())
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    if digest.hexdigest() != etag:
        raise ValueError("downloaded model checksum does not match Hugging Face metadata")


def download_file(repo_id: str, filename: str, revision: str = "main") -> Path:
    url = hf_hub_url(repo_id, filename, revision=revision)
    metadata = get_hf_file_metadata(url)
    etag, commit, size = metadata.etag, metadata.commit_hash, metadata.size
    if (
        not etag
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", etag)
        or not commit
        or not re.fullmatch(r"[0-9a-f]{40}", commit)
        or size is None
    ):
        raise ValueError("Hugging Face did not provide a verifiable model size, hash and revision")
    location = metadata.location
    if urlparse(location).scheme not in ("http", "https") or "\n" in location or "\r" in location:
        raise ValueError("invalid Hugging Face download URL")
    # aria2 forwards custom Authorization headers through cross-host redirects.
    # Signed CDN URLs need no token; rare authenticated files served by the Hub
    # itself use its client so credentials remain scoped to the Hub host.
    if urlparse(location).netloc == urlparse(url).netloc and build_hf_headers().get(
        "authorization"
    ):
        path = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=commit))
        try:
            _verify(path, size, etag)
        except ValueError:
            # The Hub client trusts existing snapshots; retry a corrupt cache
            # through the same authenticated client instead of reusing it again.
            path = Path(
                hf_hub_download(
                    repo_id=repo_id, filename=filename, revision=commit, force_download=True
                )
            )
            _verify(path, size, etag)
        return path
    cache = Path(constants.HF_HUB_CACHE)
    folder = "models--" + repo_id.replace("/", "--")
    blob = cache / folder / "blobs" / etag
    lock = cache / ".locks" / folder / f"{etag}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock):
        if blob.exists():
            try:
                _verify(blob, size, etag)
            except ValueError:
                blob.unlink()
        if not blob.exists():
            aria2 = shutil.which("aria2c")
            if aria2 is None:
                raise RuntimeError("aria2c is required for models pull; use the Nix package")
            blob.parent.mkdir(parents=True, exist_ok=True)
            partial = blob.with_name(etag + ".aria2-part")
            # Signed URLs travel over stdin, not the process arguments.
            # --out is per-download when aria2 reads an input file.
            transfer = location + f"\n out={partial.name}\n"
            command = [
                aria2,
                "--no-conf=true",
                "--continue=true",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--max-connection-per-server=16",
                "--split=16",
                "--min-split-size=16M",
                "--file-allocation=none",
                "--max-tries=10",
                "--retry-wait=3",
                "--connect-timeout=15",
                "--timeout=30",
                f"--dir={blob.parent}",
                "--input-file=-",
            ]
            if certificate := os.environ.get("SSL_CERT_FILE") or os.environ.get(
                "NIX_SSL_CERT_FILE"
            ):
                command.append(f"--ca-certificate={certificate}")
            result = subprocess.run(command, input=transfer, text=True)
            if result.returncode:
                raise RuntimeError(
                    f"aria2c exited with {result.returncode}; rerun models pull to resume"
                )
            try:
                _verify(partial, size, etag)
                from .models import validate

                validate(partial)
            except ValueError:
                # A fully transferred but invalid file cannot be resumed safely.
                partial.unlink(missing_ok=True)
                partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
                raise
            partial.replace(blob)
    # Let the Hub library maintain snapshot links, including Windows fallbacks.
    # Pin to the resolved commit so a moving branch cannot trigger another transfer.
    snapshot = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=commit))
    if revision != commit:
        refs = cache / folder / "refs"
        reference = refs / revision
        if reference.resolve() == refs.resolve() or not reference.resolve().is_relative_to(
            refs.resolve()
        ):
            raise ValueError("invalid Hugging Face revision")
        reference.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=reference.parent, delete=False
            ) as output:
                temporary = Path(output.name)
                output.write(commit)
            temporary.replace(reference)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return snapshot
