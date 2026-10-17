# llama-on-demand

An OpenAI-compatible user service for local GGUF models. Clients connect directly
and choose any registered model by name. A single llama.cpp worker is loaded on
demand; requests across all models share a FIFO queue. Switching models stops
the old process and loads the requested weights. After 30 idle seconds the worker
exits, releasing model weights, KV cache and GPU resources. Model listings and
health checks never load a model. No LiteLLM or other model gateway is required.

The service talks to llama.cpp over HTTP through a private Unix socket. Worker
sockets live in a unique mode-0700 runtime directory and are removed when the
worker stops. The packaged llama.cpp supports Unix listeners via `--host *.sock`.

Whisper and other applications remain independent. llama.cpp `--fit on` adapts
GPU offloading to currently available VRAM, reserving 1024 MiB by default. CPU
offload is reported in logs, status and the non-fatal HTTP response header
`x-local-llm-warning: cpu-offload; gpu-layers=N/T`, including streaming responses.
Loading failures return an actionable error and release the queue.

## Home Manager user service

The flake exports `homeManagerModules.default` and package builders:

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
`LLAMA_ON_DEMAND_API_KEY=sk-…`. Clients use that key. The module generates TOML
and starts only `llama-on-demand.service`. It restarts failed services and kills
the worker process group on service stop. The user manager must be running;
enable lingering if operation outside login sessions is required. CUDA builds
support x86_64 Linux, including NixOS WSL. Service and worker listeners require
Unix; systemd deployment requires Linux.

```sh
systemctl --user status llama-on-demand
journalctl --user -u llama-on-demand
set -a; . ~/.config/llama-on-demand/service.env; set +a
llama-on-demand endpoint
llama-on-demand status
curl -H "Authorization: Bearer $LLAMA_ON_DEMAND_API_KEY" \
  "$(llama-on-demand endpoint)/v1/models"
```

For TCP, the client base URL is `$(llama-on-demand endpoint)/v1`. Use the alias as
`model`. Worker logs: `${XDG_STATE_HOME:-~/.local/state}/llama-on-demand/llama.log`.

### Listeners and endpoint discovery

The default listener is `127.0.0.1` with port `0`: the OS allocates the port while
binding the actual server socket. There is no probe-and-release race and no fixed
port to reserve. Set `host` and `port` in the module to override this explicitly.
A wildcard bind address is reported as its local loopback URL; publication to
other machines is the deployment's responsibility.

Once HTTP startup completes, the service atomically writes its actual base URL,
without `/v1`, to `$XDG_RUNTIME_DIR/llama-on-demand/endpoint` (mode 0600). Only then
does it send systemd `READY=1`. The unit uses `Type=notify`, so `ExecStartPost` and
services ordered after it can read the file without polling or parsing logs.
`llama-on-demand endpoint` prints the file; `status` uses it automatically.

The module's `endpoint-file` option defaults to `%t/llama-on-demand/endpoint`.
`%t` is expanded by systemd in unit command arguments. Outside systemd, the CLI
uses `$XDG_RUNTIME_DIR/llama-on-demand/endpoint`, falling back to a user-specific
directory in the system temporary directory if `XDG_RUNTIME_DIR` is unset.
`serve`, `endpoint` and `status` accept `--endpoint-file` for a different instance.
Only one running instance can own a given endpoint file. Normal shutdown removes
it; the next startup discards a stale file after acquiring the instance lock.

For a Unix API listener instead of TCP:

```nix
services.llama-on-demand.socket = "%t/llama-on-demand/api.sock";
```

The API socket is mode 0600 and still requires the API key. Its endpoint file
contains `unix:///absolute/path/to/api.sock`; `status` also understands this.
Keep Unix socket paths short (typically under 108 bytes including the terminating
byte). Existing files and listeners are never replaced. If a crash leaves a stale
socket outside the managed runtime directory, remove it after confirming the old
service has stopped. The module's `host` and `port` are unused in socket mode.

Foreground use:

```sh
llama-on-demand serve                      # OS-allocated loopback TCP port
llama-on-demand serve --port 12345         # explicit TCP port
llama-on-demand serve --uds "$XDG_RUNTIME_DIR/llama-on-demand/api.sock"
llama-on-demand status --url http://127.0.0.1:12345
llama-on-demand status --uds "$XDG_RUNTIME_DIR/llama-on-demand/api.sock"
```

`serve --uds` cannot be combined with `--host` or `--port`.

### Extending deployment from another flake

Import the module and extend the generated unit using normal Nix module merging;
there is no need to patch the package or fork this flake. For example, a local
hook can receive the ready endpoint file as an argument:

```nix
{ config, lib, pkgs, ... }: let
  cfg = config.services.llama-on-demand;
  report = pkgs.writeShellScript "report-llm-endpoint" ''
    url=$(cat "$1")
    # Replace this with your machine's reverse-proxy/Windows publication command.
    printf 'Local LLM is ready at %s\n' "$url"
  '';
in {
  systemd.user.services.llama-on-demand.Service.ExecStartPost = lib.mkAfter [
    "${report} ${lib.escapeShellArg cfg.endpoint-file}"
  ];
}
```

Systemd expands `%t` in that argument, not inside the script body. A required
`ExecStartPost` failure fails the service startup. If publication has its own
retry and cleanup lifecycle, use a separate service instead, with `After` and
`BindsTo` pointing to `llama-on-demand.service`, and add it to the main unit's
`Wants` so it is started again with each service restart. Each restart may allocate
a different port; the publication step must rerun. Host-specific Tailscale, WSL,
Windows and DNS configuration belongs in the consuming flake.

### Migration from the LiteLLM deployment

There is now one API and one user service. Remove local `litellm.service`
overrides, LiteLLM launchers and references to `proxy-package`, `lib.mkProxy`,
`lib.runtimeOverlay`, `packages.<system>.proxy`, or the `proxy` Python extra.
The flake checks are now `cli` and `unit`; replace downstream `proxy-import`
references with `cli`. `backend-port` is removed; `port` now selects the direct
API port and defaults to 0. `backend-socket` is a deprecated alias for `socket`.
Point clients and publication hooks at the endpoint file instead of port 4000.

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
Every alias in the current local catalog can be requested immediately, including
aliases registered with the CLI after the service starts. No Home Manager change
or service restart is needed for a new alias. For example:

```sh
llama-on-demand models import another-model /path/to/another.gguf
# Send an inference request with model="another-model".
```

The first inference request loads that model. A request for a different model
unloads the current worker and loads the requested weights, through the same
global queue. Only registered local GGUF files are loaded; unknown aliases return
404, and models without downloaded weights require an explicit `models pull`.
Requests never trigger downloads or accept arbitrary filesystem paths as model
names. Files are never automatically deleted.

The authenticated `GET /v1/models` and `models list` CLI read the complete current
catalog without loading weights.

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
available for debugging. `uv sync`, `nix build`, `nix build .#cuda` and
`nix flake check` are supported. uv installs Python
packages; llama-server and aria2c must be provided separately, preferably with Nix.

## Locked Nix builds

Nix builds resolve the project's `uv.lock` with uv2nix. Python dependencies
come from that lock; nixpkgs supplies the interpreter,
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

`packages.<system>.default` / `llama-on-demand` and `apps.<system>.default` /
`llama-on-demand` expose the direct service CLI; `cuda` selects the NVIDIA build
on x86_64 Linux. The `cli` check runs the installed application and imports its
modules, while `unit` runs the tests against the installed Python environment.
Checks provide a CA bundle inside the Nix sandbox.

## Development checks

Ruff, ty and pre-commit are dev dependencies pinned in `uv.lock`. Install the
Git hook once per clone:

```sh
uv sync --locked
uv run --locked pre-commit install
uv run --locked pre-commit run --all-files
```

The local hooks use the locked tools through uv for lock consistency, Ruff lint
fixes, and Ruff formatting. The official `astral-sh/ty-pre-commit` hook pins ty
separately and checks the whole project with locked project dependencies.
Dev dependencies are excluded from the deployed application environments.
