# llama-on-demand

A user-wide systemd service for occasional local LLM use. A permanent LiteLLM
endpoint forwards requests to a lightweight backend. The backend starts a single
llama.cpp worker on demand, serializes requests across all models, and exits the
worker after 30 idle seconds. Exiting releases model weights, KV cache and CUDA
resources. No llama.cpp worker is started by model listings or health checks.

Whisper and other applications remain independent. llama.cpp `--fit on` adapts
GPU offloading to currently available VRAM, reserving 1024 MiB by default. It can
use CPU offload; this is reported in logs, backend status and the non-fatal HTTP
response header `x-local-llm-warning: cpu-offload; gpu-layers=N/T`. LiteLLM uses
our response-header callback to preserve this warning for both normal and
streaming responses. Clients may ignore it without changing the response body.
Automatic fitting cannot guarantee success if VRAM changes during loading or
inference. Loading failures return an actionable error and release the queue.

## Primary deployment: Home Manager user services

The flake exports `homeManagerModules.default` and package builders. For example:

```nix
{ inputs, pkgs, config, ... }: {
  imports = [ inputs.llama-on-demand.homeManagerModules.default ];
  services.llama-on-demand = {
    enable = true;
    package = inputs.llama-on-demand.lib.mkPackage { inherit pkgs; cudaSupport = true; };
    environment-file = "${config.xdg.configHome}/llama-on-demand/service.env";
    settings = { idle_seconds = 30; fit_target_mib = 1024; };
    models.sakura = {
      path = "/path/to/sakura.gguf";
      repo = "SakuraLLM/Sakura-GalTransl-7B-v3.7";
      file = "Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf";
      options = { ctx-size = 4096; };
    };
  };
}
```

Create the environment file outside the Nix store, mode 0600, containing a random
`LLAMA_ON_DEMAND_API_KEY=sk-…`. Both services read it; clients use the same key.
The module starts `llama-on-demand.service` on loopback port 4001 and
`litellm.service` on loopback port 4000. It generates TOML and LiteLLM YAML,
provides restart behavior and kills the worker process group on service stop.
The user manager must be running; enable lingering if operation outside login
sessions is required. CUDA builds support x86_64 Linux, including NixOS WSL.
The CLI and model catalog are portable; systemd deployment requires Linux.

```sh
systemctl --user status llama-on-demand litellm
journalctl --user -u llama-on-demand -u litellm
set -a; . ~/.config/llama-on-demand/service.env; set +a
llama-on-demand status
```

Client base URL: `http://127.0.0.1:4000/v1`. Use the configured alias as `model`.
Detailed worker logs: `${XDG_STATE_HOME:-~/.local/state}/llama-on-demand/llama.log`.

## GGUF management

```sh
llama-on-demand models list
llama-on-demand models import my-model /path/to/model.gguf
llama-on-demand models add my-model owner/repo file.gguf --revision COMMIT
llama-on-demand models pull my-model
llama-on-demand models inspect my-model
```

Import validates the GGUF magic and registers the existing path without copying.
`models pull` resolves the file version and download URL through Hugging Face,
then uses aria2c with up to 16 connections, retries and resumable partial files.
Rerun the same command after an interruption to resume. The file size, content
hash and GGUF magic are checked before updating the registry; aria2 transfers
are verified before publishing to the standard Hugging Face cache.
A failed download leaves the previous registered
model intact. Nix packages include aria2c; uv users must provide it on PATH.
Cached weights are verified and reused without starting aria2c.
Signed CDN URLs work with private repositories too. Authenticated files served
directly by the Hub use its client to avoid forwarding tokens through aria2 redirects.
The user registry is stored at
`${XDG_DATA_HOME:-~/.local/share}/llama-on-demand/models.json`. Each entry keeps
explicit CLI definitions separate from its download cache. CLI definitions augment
or override the declarative TOML catalog; `pull` caches only the downloaded path
and its repo, filename and revision, so subsequent TOML option changes take effect.
Changing any of those source fields invalidates the cached path and requires
another `models pull`. An omitted revision is equivalent to `main`; changing the
remote contents of `main` still requires an explicit pull to refresh the cache.
`models add` replaces the CLI definition and clears its download cache; import
registers an explicit local path. A CLI source override cannot inherit a path
from a different TOML source. If the same registry entry changes during a pull,
the downloaded file remains cached but registration is rejected to preserve the
newer selection; rerun the pull if still needed.
The backend checks the resolved definition on
each request and reloads a running worker when it changes.

Legacy downloaded entries are read using the current TOML definition when the
alias is declared there; registry-only aliases retain their old definitions.
Since legacy records did not distinguish explicit overrides from copied TOML,
reapply any intentional CLI source overrides with `models add` and `models pull`.
New writes use separate `definition` and `download` fields in each registry entry.
Add the alias to the Home Manager `models` configuration to expose it through
LiteLLM as well. Files are never automatically deleted.

Runtime settings: `idle_seconds` (30), `fit_target_mib` (1024),
`startup_timeout` (300), `request_timeout` (3600), `queue_timeout` (3600),
`max_queue` (128), `server_command` (packaged llama-server). The queue is FIFO
and shared across models, held until an entire response finishes. A disconnected
stream kills its worker before admitting the next request. Overflow returns 503;
queue timeout returns 504. These limits make resource exhaustion explicit.
GPU layer options (`gpu-layers`, `n-gpu-layers`, and their underscore spellings)
are ignored with a log warning so automatic fitting remains in control.
Other reserved llama.cpp options, including parallelism, cannot be overridden
in model options because they would bypass these guarantees.
Boolean model options enable a command-line switch when `true` and omit it when
`false` (for example, `embedding = true` adds `--embedding`). For options that
take values, use strings or numbers, such as `flash-attn = "on"` or `ctx-size = 4096`.

`llama-on-demand serve` is the foreground entrypoint used by systemd. It is also
available for debugging. `uv sync`, `uv sync --extra proxy`, `nix build .#cuda`,
`nix build .#proxy` and `nix flake check` are supported. uv installs Python
packages; llama-server and aria2c must be provided separately, preferably with Nix.

## Locked Nix builds

Nix builds resolve the project's `uv.lock` with uv2nix. Python dependencies,
including optional extras, come from that lock; nixpkgs supplies the interpreter,
system libraries and llama.cpp where applicable. Build backend versions are
pinned by the `pyproject-build-systems` flake input because uv does not lock them.
Package metadata is generated from `pyproject.toml` by pyproject-nix and inherited
by the CLI, so descriptions and the main program do not need separate Nix definitions.

All three build inputs (`pyproject-nix`, `uv2nix`, `pyproject-build-systems`) may
follow a consuming flake's shared inputs. The standalone flake pins them itself.
`lib.mkPythonSet { pkgs; python; }` exposes the locked Python package scope for
SDK consumers; `lib.mkPackage { pkgs; python; cudaSupport; }` ships the CLI with
its isolated environment. The CLI's `python-env` passthru exposes that environment
for installation and runtime checks. Development shells supply Python and uv;
use `uv sync` / `uv run` for development, and `nix build` for deployment.

`lib.mkProxy { pkgs; python; }` builds the locked `proxy` extra, including LiteLLM
and the warning callback, in one uv2nix environment. The `proxy-import` flake check
imports the actual installed service and callback modules.

`lib.runtimeOverlay { pkgs; }` supplies libsndfile for the locked soundfile
wheel without changing its version. Checks provide a CA bundle inside the Nix sandbox.

## Development checks

Ruff, ty and pre-commit are dev dependencies pinned in `uv.lock`. Install the
Git hook once per clone:

```sh
uv sync --extra proxy --locked
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

The local hooks use the locked tools through uv for lock consistency, Ruff lint
fixes, and Ruff formatting. The official `astral-sh/ty-pre-commit` hook pins ty
separately and checks the whole project with locked project dependencies. The ty hook also enables the `proxy` extra
to resolve the optional LiteLLM callback. Dev dependencies are excluded from
the deployed application environments.
