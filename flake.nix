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

      # Fold the authPasswordFile shorthand into the generic credential map,
      # without letting it override an explicit credentials.AUTH_PASSWORD.
      credentials =
        lib.filterAttrs (_: v: v != null) cfg.credentials
        // lib.optionalAttrs (!(cfg.credentials ? AUTH_PASSWORD) && cfg.authPasswordFile != null) {
          AUTH_PASSWORD = cfg.authPasswordFile;
        };
      credentialLines = lib.mapAttrsToList (name: path: "${name}:${path}") credentials;
      credentialNames = lib.attrNames credentials;
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
            kept out of the Nix store. Shorthand for
            `credentials.AUTH_PASSWORD`. Loaded with systemd `LoadCredential`,
            and read by `auth_password` at startup. When null the UI is open to
            anyone who can reach `port` -- set this (or `auth_password` in the UI)
            before exposing the service beyond a trusted subnet.
          '';
        };

        credentials = lib.mkOption {
          type = lib.types.attrsOf lib.types.path;
          default = {};
          example = {
            OLLAMA_API_KEY = "/run/secrets/ideafindr-ollama-key";
            X_AUTH_TOKEN = "/run/secrets/ideafindr-x-auth-token";
            X_CT0 = "/run/secrets/ideafindr-x-ct0";
            INSTAGRAM_SESSIONID = "/run/secrets/ideafindr-ig-sessionid";
            REDDIT_CLIENT_ID = "/run/secrets/ideafindr-reddit-id";
            REDDIT_CLIENT_SECRET = "/run/secrets/ideafindr-reddit-secret";
          };
          description = ''
            Secrets delivered to the service as systemd credentials, mapped
            `ENV_VAR_NAME = path-to-file`. Each file is read into the matching
            environment variable (e.g. `X_AUTH_TOKEN`), so the value never lands
            in the world-readable Nix store and the UI is not the only way to
            configure a headless box. These take precedence over anything stored
            in the web UI, matching the usual environment-over-settings rule.

            The keys are the pydantic field names upper-cased, which is exactly
            what `ideafindr/config.py` reads from the environment.
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
          // lib.optionalAttrs (credentials != {}) {
            # Deliver each secret as a systemd credential and read it into the
            # matching environment variable. Kept out of the Nix store, and out
            # of the environment of every unrelated process.
            LoadCredential = credentialLines;
            ExecStart = lib.mkForce (
              pkgs.writeShellScript "ideafindr-serve" ''
                ${lib.concatMapStrings (name: ''
                  if [ -r "$CREDENTIALS_DIRECTORY/${name}" ]; then
                    export ${name}="$(cat "$CREDENTIALS_DIRECTORY/${name}")"
                  fi
                '') credentialNames}
                exec ${cfg.package}/bin/ideafindr web --host ${cfg.host} --port ${toString cfg.port}
              ''
            );
          };
        };
      };
    };
  };
}
