{
  description = "Queued, on-demand llama.cpp user service with GGUF management";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  outputs = {
    self,
    nixpkgs,
  }: let
    systems = ["x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin"];
    each-system = nixpkgs.lib.genAttrs systems;
  in {
    lib.mkPackage = {
      pkgs,
      cudaSupport ? false,
      python ? pkgs.python314,
    }: let
      llama-cpp = pkgs.llama-cpp.override {inherit cudaSupport;};
      app = python.pkgs.buildPythonPackage {
        pname = "llama-on-demand";
        version = "0.1.1";
        src = self;
        pyproject = true;
        build-system = [python.pkgs.uv-build];
        dependencies = with python.pkgs; [fastapi httpx uvicorn huggingface-hub];
        pythonImportsCheck = ["llama_on_demand" "llama_on_demand.service"];
        nativeBuildInputs = [pkgs.makeWrapper];
        postFixup = ''
          wrapProgram "$out/bin/llama-on-demand" --set-default LLAMA_SERVER_COMMAND ${llama-cpp}/bin/llama-server ${pkgs.lib.optionalString cudaSupport "--prefix LD_LIBRARY_PATH : /run/opengl-driver/lib:/usr/lib/wsl/lib"}
        '';
        passthru = {inherit llama-cpp;};
      };
    in
      app;
    lib.mkProxy = {
      pkgs,
      python ? pkgs.python314,
    }: let
      sdk = self.lib.mkPackage {inherit pkgs python;};
      environment = python.withPackages (ps: [sdk ps.litellm] ++ ps.litellm.optional-dependencies.proxy ++ ps.litellm.optional-dependencies.proxy-runtime);
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
      python = pkgs.python314.withPackages (ps: [ps.fastapi ps.httpx ps.litellm]);
    in {
      proxy-import = pkgs.runCommand "llama-on-demand-proxy-import" {} ''
        export LITELLM_LOCAL_MODEL_COST_MAP=True
        ${self.packages.${system}.proxy.python-env}/bin/python -c 'from llama_on_demand.litellm_callback import callback; from litellm.integrations.custom_logger import CustomLogger; assert isinstance(callback, CustomLogger)'
        touch "$out"
      '';
      unit = pkgs.runCommand "llama-on-demand-unit" {} ''
        export PYTHONPATH=${self}/src
        export PYTHONDONTWRITEBYTECODE=1
        export LITELLM_LOCAL_MODEL_COST_MAP=True
        ${python}/bin/python -m unittest discover -s ${self}/tests
        touch "$out"
      '';
    });
    homeManagerModules.default = import ./home-manager.nix self;
    devShells = each-system (system: let pkgs = nixpkgs.legacyPackages.${system}; in {default = pkgs.mkShell {packages = [pkgs.uv pkgs.python314];};});
    formatter = each-system (system: nixpkgs.legacyPackages.${system}.alejandra);
  };
}
