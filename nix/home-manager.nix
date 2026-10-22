{mkPackage}: {
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.llama-on-demand;
  toml = pkgs.formats.toml {};
  model-config = toml.generate "llama-on-demand.toml" (cfg.settings // {models = cfg.models;});
  runtime-path = path: lib.hasPrefix "/" path || lib.hasPrefix "%t/" path;
in {
  imports = [
    (lib.mkRemovedOptionModule ["services" "llama-on-demand" "proxy-package"] "LiteLLM has been removed; clients now connect directly to llama-on-demand.")
    (lib.mkRemovedOptionModule ["services" "llama-on-demand" "backend-port"] "Use services.llama-on-demand.port for the direct API; its default is now 0 (OS allocated).")
    (lib.mkRenamedOptionModule ["services" "llama-on-demand" "backend-socket"] ["services" "llama-on-demand" "socket"])
  ];

  options.services.llama-on-demand = {
    enable = lib.mkEnableOption "queued, on-demand local llama.cpp inference";
    package = lib.mkOption {
      type = lib.types.package;
      default = mkPackage {inherit pkgs;};
      description = "Service package; choose a CUDA build on NVIDIA hosts.";
    };
    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "TCP bind address. Unused when socket is set.";
    };
    port = lib.mkOption {
      type = lib.types.ints.between 0 65535;
      default = 0;
      description = "TCP port; 0 lets the OS allocate an available port. Unused when socket is set.";
    };
    socket = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Unix socket path instead of TCP. Accepts an absolute path or systemd's %t/ runtime directory prefix.";
    };
    endpoint-file = lib.mkOption {
      type = lib.types.str;
      default = "%t/llama-on-demand/endpoint";
      description = "File containing the actual HTTP URL or unix:// socket address. Published atomically before systemd readiness, removed on shutdown. %t expands to the user runtime directory.";
    };
    environment-file = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional runtime environment file for LLAMA_ON_DEMAND_API_KEY or other variables. Required authentication still needs a key at runtime; never put secrets in Nix settings.";
    };
    settings = lib.mkOption {
      type = toml.type;
      default = {};
      description = "Runtime settings, including auth_required (default true), idle_seconds, fit_target_mib and queue_timeout.";
    };
    models = lib.mkOption {
      type = toml.type;
      default = {};
      description = "Model aliases and GGUF path or Hugging Face repo/file, revision and llama.cpp options.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = pkgs.stdenv.hostPlatform.isLinux;
        message = "llama-on-demand user services require Linux/systemd";
      }
      {
        assertion = cfg.socket == null || runtime-path cfg.socket;
        message = "llama-on-demand socket must be absolute or start with %t/";
      }
      {
        assertion = runtime-path cfg.endpoint-file;
        message = "llama-on-demand endpoint-file must be absolute or start with %t/";
      }
    ];
    home.packages = [cfg.package];
    xdg.configFile."llama-on-demand/config.toml".source = model-config;
    systemd.user.services.llama-on-demand = {
      Unit.Description = "On-demand llama.cpp models and global inference queue";
      Service =
        {
          Type = "notify";
          NotifyAccess = "main";
          ExecStart =
            "${cfg.package}/bin/llama-on-demand --config ${model-config} serve "
            + lib.escapeShellArgs (["--endpoint-file" cfg.endpoint-file]
              ++ (
                if cfg.socket != null
                then ["--uds" cfg.socket]
                else ["--host" cfg.host "--port" (toString cfg.port)]
              ));
          Restart = "on-failure";
          RestartSec = 3;
          KillMode = "control-group";
          TimeoutStartSec = 90;
          TimeoutStopSec = 45;
          UMask = "0077";
          RuntimeDirectory = "llama-on-demand";
          RuntimeDirectoryMode = "0700";
        }
        // lib.optionalAttrs (cfg.environment-file != null) {
          EnvironmentFile = cfg.environment-file;
        };
      Install.WantedBy = ["default.target"];
    };
  };
}
