{
  description = "ideafindr - social listening + deep research pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    uv2nix,
    pyproject-build-systems,
    ...
  }: let
    inherit (nixpkgs) lib;
    systems = ["x86_64-linux" "aarch64-linux"];
    eachSystem = f: lib.genAttrs systems (system: f (import nixpkgs {inherit system;}));

    # A Python environment built from uv.lock. The workspace is the project
    # itself; overrides come from the build-system set, then the uv overlay.
    mkPythonSet = pkgs: let
      workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};
      # "wheel" keeps builds fast and avoids compiling from source. If a
      # dependency publishes no wheel for your platform, switch this to "sdist"
      # or add a per-package override.
      overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};
    in
      (pkgs.callPackage pyproject-nix.build.packages {
        python = pkgs.python312;
      })
      .overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
        ]
      );
  in {
    packages = eachSystem (pkgs: let
      pythonSet = mkPythonSet pkgs;
      # Core dependencies only. The `engines` and `scrapers` extras are heavy and
      # opt-in; the service does not need them.
      env = pythonSet.mkVirtualEnv "ideafindr-env" {
        ideafindr = [];
      };
      # The venv is named "ideafindr-env", so `nix run` would look for a binary of
      # that name. Point it at the real console script instead.
      app = env.overrideAttrs (old: {
        meta = (old.meta or {}) // {mainProgram = "ideafindr";};
      });
    in {
      default = app;
      app = app;
    });

    # `nix develop` -- test extras only, so the shell stays reliable. Add
    # "scrapers" or "social" here if you want TikTok/Instagram/Mastodon support.
    devShells = eachSystem (pkgs: let
      pythonSet = mkPythonSet pkgs;
      env = pythonSet.mkVirtualEnv "ideafindr-dev" {
        ideafindr = ["dev"];
      };
    in {
      default = pkgs.mkShell {
        packages = [env pkgs.uv pkgs.git];
        env = {
          UV_NO_SYNC = "1";
          UV_PYTHON = pythonSet.python.interpreter;
          UV_PYTHON_DOWNLOADS = "never";
        };
        shellHook = ''
          unset PYTHONPATH
          echo "ideafindr dev shell ready"
        '';
      };
    });

    # NixOS module: a hardened systemd service with a state directory for the
    # database, reports and UI-entered secrets. The Ollama key is NOT configured
    # in Nix; it is entered in the web UI and stored under StateDirectory.
    nixosModules.default = {
      config,
      lib,
      pkgs,
      ...
    }: let
      cfg = config.services.ideafindr;
    in {
      options.services.ideafindr = {
        enable = lib.mkEnableOption "ideafindr web UI";

        package = lib.mkOption {
          type = lib.types.package;
          default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
          description = "The ideafindr package to run.";
        };

        host = lib.mkOption {
          type = lib.types.str;
          default = "0.0.0.0";
          description = ''
            Bind address. Defaults to 0.0.0.0 so the UI is reachable from the
            LAN (the homelab use case). The UI has no authentication and holds
            API keys and session cookies, so restrict the firewall to your
            subnet or put a reverse proxy/VPN in front of it before exposing it.
          '';
        };

        port = lib.mkOption {
          type = lib.types.port;
          default = 8000;
        };

        stateDir = lib.mkOption {
          type = lib.types.path;
          default = "/var/lib/ideafindr";
          description = ''
            Where the database, reports and settings live. Realised as a systemd
            `StateDirectory` owned by `user`/`group`, so it is created on first
            start and keeps the secrets it holds private.
          '';
        };

        user = lib.mkOption {
          type = lib.types.str;
          default = "ideafindr";
        };

        group = lib.mkOption {
          type = lib.types.str;
          default = "ideafindr";
        };

        environmentFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          description = ''
            Optional file of Environment= lines (OLLAMA_BASE_URL, etc.). Keys set
            here override anything the web UI stored.
          '';
        };

        authPasswordFile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = "/run/secrets/ideafindr-password";
          description = ''
            File containing the single password that gates the web UI, so it is
            kept out of the Nix store. Loaded with systemd `LoadCredential`, and
            read by `auth_password` at startup. When null the UI is open to
            anyone who can reach `port` -- set this (or `auth_password` in the UI)
            before exposing the service beyond a trusted subnet.
          '';
        };

        openFirewall = lib.mkOption {
          type = lib.types.bool;
          default = false;
        };
      };

      config = lib.mkIf cfg.enable {
        users.users.${cfg.user} = {
          isSystemUser = true;
          group = cfg.group;
          home = cfg.stateDir;
        };
        users.groups.${cfg.group} = {};

        networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [cfg.port];

        systemd.services.ideafindr = {
          description = "ideafindr web UI";
          wantedBy = ["multi-user.target"];
          after = ["network-online.target"];
          wants = ["network-online.target"];

          # `%S` expands to the state root that matches StateDirectory. Deriving
          # the path here rather than hardcoding cfg.stateDir keeps the two in
          # sync if an operator changes stateDir to something outside /var/lib.
          environment.IDEAFINDR_STATE_DIR = "%S/ideafindr";

          serviceConfig = {
            ExecStart = "${cfg.package}/bin/ideafindr web --host ${cfg.host} --port ${toString cfg.port}";
            User = cfg.user;
            Group = cfg.group;
            EnvironmentFile = lib.optional (cfg.environmentFile != null) cfg.environmentFile;
            StateDirectory = "ideafindr";
            StateDirectoryMode = "0700";
            WorkingDirectory = cfg.stateDir;

            # A homelab UI that stays up across reboots and transient failures.
            Restart = "on-failure";
            RestartSec = 10;

            # Hardening: the service needs only its own state dir and network.
            NoNewPrivileges = true;
            PrivateTmp = true;
            PrivateDevices = true;
            ProtectSystem = "strict";
            ProtectHome = true;
            ProtectKernelTunables = true;
            ProtectKernelModules = true;
            ProtectControlGroups = true;
            RestrictAddressFamilies = ["AF_INET" "AF_INET6" "AF_UNIX"];
            RestrictNamespaces = true;
            LockPersonality = true;
            RestrictRealtime = true;
            SystemCallArchitectures = "native";
            ReadWritePaths = [cfg.stateDir];

            # systemd's default (90s) SIGTERM timeout is far longer than the
            # server needs to stop; don't make a restart wait on it.
            TimeoutStopSec = 15;
          }
          // lib.optionalAttrs (cfg.authPasswordFile != null) {
            # The password is delivered as a systemd credential, kept out of the
            # Nix store and the environment of unrelated processes. The wrapper
            # reads it into AUTH_PASSWORD, which settings.py picks up.
            LoadCredential = "auth-password:${cfg.authPasswordFile}";
            ExecStart = lib.mkForce (
              pkgs.writeShellScript "ideafindr-serve" ''
                if [ -r "$CREDENTIALS_DIRECTORY/auth-password" ]; then
                  AUTH_PASSWORD="$(cat "$CREDENTIALS_DIRECTORY/auth-password")"
                  export AUTH_PASSWORD
                fi
                exec ${cfg.package}/bin/ideafindr web --host ${cfg.host} --port ${toString cfg.port}
              ''
            );
          };
        };
      };
    };
  };
}
