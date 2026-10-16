{
  description = "Queued, on-demand llama.cpp user service with GGUF management";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    pyproject-nix.url = "github:pyproject-nix/pyproject.nix";
    pyproject-nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.url = "github:pyproject-nix/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.url = "github:pyproject-nix/build-system-pkgs";
    pyproject-build-systems.inputs.nixpkgs.follows = "nixpkgs";
    pyproject-build-systems.inputs.pyproject-nix.follows = "pyproject-nix";
    pyproject-build-systems.inputs.uv2nix.follows = "uv2nix";
  };
  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
    project = pyproject-nix.lib.project.loadPyproject {projectRoot = self;};
    workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = self;};
    overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
    mk-application = pkgs: (pkgs.callPackages pyproject-nix.build.util {}).mkApplication;
  in {
    lib.runtimeOverlay = {pkgs}: final: prev: {
      # The locked generic wheel loads libsndfile with ctypes, so autoPatchelf
      # cannot discover this dependency. Keep the Python wheel/version unchanged.
      soundfile = prev.soundfile.overrideAttrs (old: {
        postInstall =
          (old.postInstall or "")
          + ''
            substituteInPlace "$out/${final.python.sitePackages}/soundfile.py" \
              --replace-fail "_find_library('sndfile')" "'${pkgs.lib.getLib pkgs.libsndfile}/lib/libsndfile${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}'"
          '';
      });
    };
    lib.mkPythonSet = {
      pkgs,
      python ? pkgs.python314,
    }:
      (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (pkgs.lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        overlay
        (_final: prev: {
          llama-on-demand = prev.llama-on-demand.overrideAttrs (old: {
            meta = (old.meta or {}) // project.renderers.meta {};
          });
        })
        (self.lib.runtimeOverlay {inherit pkgs;})
      ]);
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      python-set = self.lib.mkPythonSet {inherit pkgs python;};
      environment = python-set.mkVirtualEnv "llama-on-demand-env" {llama-on-demand = [];};
      llama-cpp = pkgs.llama-cpp.override {inherit cudaSupport;};
      app = mk-application pkgs {
        venv = environment;
        package = python-set.llama-on-demand;
      };
    in
      app.overrideAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.makeWrapper];
        postFixup =
          (old.postFixup or "")
          + ''
            wrapProgram "$out/bin/llama-on-demand" --set-default LLAMA_SERVER_COMMAND ${llama-cpp}/bin/llama-server ${pkgs.lib.optionalString cudaSupport "--prefix LD_LIBRARY_PATH : /run/opengl-driver/lib:/usr/lib/wsl/lib"}
          '';
        passthru =
          (old.passthru or {})
          // {
            inherit llama-cpp;
            python-env = environment;
          };
      });
    lib.mkProxy = {
      pkgs,
      python ? pkgs.python314,
    }: let
      python-set = self.lib.mkPythonSet {inherit pkgs python;};
      environment = python-set.mkVirtualEnv "llama-on-demand-proxy-env" {llama-on-demand = ["proxy"];};
    in
      (pkgs.writeShellScriptBin "litellm" ''
        export LITELLM_LOCAL_MODEL_COST_MAP=True
        exec ${environment}/bin/litellm "$@"
      '').overrideAttrs (_: {passthru.python-env = environment;});
    packages = each-system (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
      {
        default = self.lib.mkPackage {inherit pkgs;};
        llama-on-demand = self.packages.${system}.default;
        proxy = self.lib.mkProxy {inherit pkgs;};
      }
      // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
        cuda = self.lib.mkPackage {
          inherit pkgs;
          cudaSupport = true;
        };
      });
    checks = each-system (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      environment = self.packages.${system}.proxy.python-env;
    in {
      proxy-import = pkgs.runCommand "llama-on-demand-proxy-import" {} ''
        export LITELLM_LOCAL_MODEL_COST_MAP=True
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        ${environment}/bin/python -c 'from llama_on_demand.litellm_callback import callback; from litellm.integrations.custom_logger import CustomLogger; assert isinstance(callback, CustomLogger); import litellm.proxy.proxy_server'
        touch "$out"
      '';
      unit = pkgs.runCommand "llama-on-demand-unit" {} ''
        export PYTHONDONTWRITEBYTECODE=1
        export LITELLM_LOCAL_MODEL_COST_MAP=True
        export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        ${environment}/bin/python -m unittest discover -s ${self}/tests
        touch "$out"
      '';
    });
    homeManagerModules.default = {
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
    };
    devShells = each-system (system: let pkgs = nixpkgs.legacyPackages.${system}; in {default = pkgs.mkShell {packages = [pkgs.uv pkgs.python314];};});
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
