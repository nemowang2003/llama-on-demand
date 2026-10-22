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
    lib.mkPythonSet = {
      pkgs,
      python ? pkgs.python314,
    }:
      (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (pkgs.lib.composeManyExtensions [
        pyproject-build-systems.overlays.wheel
        overlay
        (final: prev: {
          llama-on-demand = prev.llama-on-demand.overrideAttrs (old: {
            # Use the backend from uv.lock instead of uv's bundled version.
            uvBuildFlags = (old.uvBuildFlags or []) ++ ["--force-pep517"];
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {uv-build = [];};
            meta = (old.meta or {}) // project.renderers.meta {};
          });
        })
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
            wrapProgram "$out/bin/llama-on-demand" --prefix PATH : ${pkgs.lib.makeBinPath [pkgs.aria2]} --set-default LLAMA_SERVER_COMMAND ${llama-cpp}/bin/llama-server ${pkgs.lib.optionalString cudaSupport "--prefix LD_LIBRARY_PATH : /run/opengl-driver/lib:/usr/lib/wsl/lib"}
          '';
        passthru =
          (old.passthru or {})
          // {
            inherit llama-cpp;
            python-env = environment;
          };
      });
    packages = each-system (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
      {
        default = self.lib.mkPackage {inherit pkgs;};
        llama-on-demand = self.packages.${system}.default;
      }
      // pkgs.lib.optionalAttrs (system == "x86_64-linux") {
        cuda = self.lib.mkPackage {
          inherit pkgs;
          cudaSupport = true;
        };
      });
    checks = each-system (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      environment = self.packages.${system}.default.python-env;
    in
      {
        cli = pkgs.runCommand "llama-on-demand-cli" {} ''
          ${self.packages.${system}.default}/bin/llama-on-demand --help
          ${environment}/bin/python -c 'from llama_on_demand.service import create_app; from llama_on_demand.runtime import Runtime'
          touch "$out"
        '';
        unit = pkgs.runCommand "llama-on-demand-unit" {nativeBuildInputs = [pkgs.aria2];} ''
          export PYTHONDONTWRITEBYTECODE=1
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          ${environment}/bin/python -m unittest discover -s ${self}/tests
          touch "$out"
        '';
      }
      // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
        home-manager = let
          configs = import ./tests/home-manager.nix {
            inherit pkgs;
            module = self.homeManagerModules.default;
          };
        in
          pkgs.runCommand "llama-on-demand-home-manager" {} ''
            ${environment}/bin/python - <<'PY'
            import tomllib
            from pathlib import Path
            assert tomllib.loads(Path("${configs.unauthenticated}").read_text())["auth_required"] is False
            assert tomllib.loads(Path("${configs.authenticated}").read_text())["auth_required"] is True
            PY
            touch "$out"
          '';
      });
    apps = each-system (system: let
      app = {
        type = "app";
        meta = project.renderers.meta {};
        program = "${self.packages.${system}.default}/bin/llama-on-demand";
      };
    in {
      default = app;
      llama-on-demand = app;
    });
    homeManagerModules.default = import ./nix/home-manager.nix {mkPackage = self.lib.mkPackage;};
    devShells = each-system (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      default = pkgs.mkShell {
        packages = [pkgs.uv pkgs.python314 pkgs.aria2];
        UV_PYTHON = "${pkgs.python314}/bin/python";
        UV_PYTHON_DOWNLOADS = "never";
      };
    });
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
