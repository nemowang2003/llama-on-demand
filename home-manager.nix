self: {
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.llama-on-demand;
  toml = pkgs.formats.toml {};
  yaml = pkgs.formats.yaml {};
  model-config = toml.generate "llama-on-demand.toml" (cfg.settings // {models = cfg.models;});
  backend-url = "http://127.0.0.1:${toString cfg.backend-port}";
  proxy-config = yaml.generate "llama-on-demand-proxy.yaml" {
    model_list =
      lib.mapAttrsToList (name: _: {
        model_name = name;
        litellm_params = {
          model = "openai/${name}";
          api_base = "${backend-url}/v1";
          api_key = "os.environ/LLAMA_ON_DEMAND_API_KEY";
          timeout = 7200;
        };
      })
      cfg.models;
    general_settings.master_key = "os.environ/LLAMA_ON_DEMAND_API_KEY";
    litellm_settings = {
      callbacks = ["llama_on_demand.litellm_callback.callback"];
      return_response_headers = true;
      request_timeout = 7200;
      num_retries = 0;
      telemetry = false;
    };
    router_settings = {
      num_retries = 0;
      disable_cooldowns = true;
    };
  };
  common-service = {
    EnvironmentFile = cfg.environment-file;
    Restart = "on-failure";
    RestartSec = 3;
    KillMode = "control-group";
    TimeoutStopSec = 30;
    UMask = "0077";
  };
in {
  options.services.llama-on-demand = {
    enable = lib.mkEnableOption "queued on-demand llama.cpp and a permanent LiteLLM user endpoint";
    package = lib.mkOption {
      type = lib.types.package;
      default = self.lib.mkPackage {inherit pkgs;};
      description = "Backend package; choose a CUDA build on NVIDIA hosts.";
    };
    proxy-package = lib.mkOption {
      type = lib.types.package;
      default = self.lib.mkProxy {inherit pkgs;};
      description = "LiteLLM environment with the offload warning callback.";
    };
    port = lib.mkOption {
      type = lib.types.port;
      default = 4000;
      description = "Permanent LiteLLM endpoint port.";
    };
    backend-port = lib.mkOption {
      type = lib.types.port;
      default = 4001;
      description = "Internal backend port.";
    };
    environment-file = lib.mkOption {
      type = lib.types.str;
      description = "Runtime environment file containing LLAMA_ON_DEMAND_API_KEY; never put the key in Nix settings.";
    };
    settings = lib.mkOption {
      type = toml.type;
      default = {};
      description = "Backend settings, including idle_seconds, fit_target_mib and queue_timeout.";
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
        assertion = cfg.port != cfg.backend-port;
        message = "llama-on-demand frontend and backend ports must differ";
      }
    ];
    home.packages = [cfg.package cfg.proxy-package];
    xdg.configFile."llama-on-demand/config.toml".source = model-config;
    systemd.user.services.llama-on-demand = {
      Unit = {Description = "On-demand llama.cpp model worker and global request queue";};
      Service =
        common-service
        // {
          ExecStart = "${cfg.package}/bin/llama-on-demand --config ${model-config} serve --port ${toString cfg.backend-port}";
        };
      Install.WantedBy = ["default.target"];
    };
    systemd.user.services.litellm = {
      Unit = {
        Description = "Local LLM API with on-demand llama.cpp backend";
        Wants = ["llama-on-demand.service"];
        After = ["llama-on-demand.service"];
      };
      Service =
        common-service
        // {
          ExecStart = "${cfg.proxy-package}/bin/litellm --config ${proxy-config} --host 127.0.0.1 --port ${toString cfg.port} --num_workers 1";
        };
      Install.WantedBy = ["default.target"];
    };
  };
}
