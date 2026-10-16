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
    lib.runtimeOverlay = import ./nix/runtime-overlay.nix;
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
    homeManagerModules.default = import ./home-manager.nix self;
    devShells = each-system (system: let pkgs = nixpkgs.legacyPackages.${system}; in {default = pkgs.mkShell {packages = [pkgs.uv pkgs.python314];};});
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
