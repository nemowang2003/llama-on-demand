{
  pkgs,
  module,
}: let
  inherit (pkgs) lib;
  evaluate = settings: environment-file: let
    evaluated = lib.evalModules {
      specialArgs = {inherit pkgs;};
      modules = [
        {
          # Only the Home Manager fields written by this module are needed.
          options = {
            assertions = lib.mkOption {type = lib.types.listOf lib.types.anything;};
            home.packages = lib.mkOption {type = lib.types.listOf lib.types.package;};
            xdg.configFile = lib.mkOption {type = lib.types.attrsOf lib.types.anything;};
            systemd.user.services = lib.mkOption {type = lib.types.attrsOf lib.types.anything;};
          };
        }
        module
        {
          services.llama-on-demand =
            {
              enable = true;
              package = pkgs.hello;
              inherit settings;
            }
            // lib.optionalAttrs (environment-file != null) {inherit environment-file;};
        }
      ];
    };
  in
    evaluated.config;
  authenticated = evaluate {} null;
  unauthenticated = evaluate {auth_required = false;} null;
  with-environment = evaluate {auth_required = false;} "/run/llama-on-demand.env";
  authenticated-with-environment = evaluate {auth_required = true;} "/run/llama-on-demand.env";
in
  assert builtins.all (entry: entry.assertion) unauthenticated.assertions;
  assert !(authenticated.systemd.user.services.llama-on-demand.Service ? EnvironmentFile);
  assert !(unauthenticated.systemd.user.services.llama-on-demand.Service ? EnvironmentFile);
  assert with-environment.systemd.user.services.llama-on-demand.Service.EnvironmentFile == "/run/llama-on-demand.env";
  assert authenticated-with-environment.systemd.user.services.llama-on-demand.Service.EnvironmentFile == "/run/llama-on-demand.env"; {
    unauthenticated = unauthenticated.xdg.configFile."llama-on-demand/config.toml".source;
    authenticated = authenticated-with-environment.xdg.configFile."llama-on-demand/config.toml".source;
  }
