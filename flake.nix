{
    description = "Hanzi Friend in Python";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        (flake-utils.lib.eachDefaultSystem(system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                };

                hanzipy = pkgs.python312Packages.buildPythonPackage {
                    pname = "hanzi-py";
                    version = "1.0.4";

                    pyproject = true;
                    build-system = [ pkgs.python312Packages.setuptools ];

                    src = pkgs.fetchzip {
                        url = "https://github.com/Synkied/hanzipy/archive/refs/tags/v1.0.4.zip";
                        sha256 = "sha256-PXIitMnzL6BETKu4waYUWB45W5N5VAfY/Gno41cf3Rg=";
                    };
                };

                fsrs = pkgs.python312Packages.buildPythonPackage {
                    pname = "fsrs";
                    version = "6.1.0";

                    pyproject = true;
                    build-system = [ pkgs.python312Packages.setuptools ];

                    src = pkgs.fetchzip {
                        url = "https://github.com/open-spaced-repetition/py-fsrs/archive/refs/tags/v6.1.0.zip";
                        sha256 = "sha256-zN0Ga6Jb0qTwlwlSr61jIGgafRWrbPHCMJT1km46RFA=";
                    };
                };
                appPackages = ps: with ps; [
                    flask
                    pypinyin
                    hanzipy
                    fsrs
                    tinydb
                    openai
                    python-dotenv
                    requests
                    waitress
                ];

                # The dev shell adds the character-confusability experiment's
                # deps; the server package shouldn't drag torch around.
                pythonEnv = pkgs.python312.withPackages (ps: appPackages ps ++ (with ps; [
                    pillow
                    numpy
                    torch
                    transformers
                ]));

                appEnv = pkgs.python312.withPackages appPackages;

                # Just the app; our db, caches, and experiments stay out
                # of the nix store.
                appSrc = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.unions [
                        ./app.py
                        ./auth.py
                        ./db.py
                        ./enrich.py
                        ./hanzi.py
                        ./quiz.py
                        ./serve.py
                        ./translation.py
                        ./tts.py
                        ./filter_defs.py
                        ./loach_word_order.py
                        ./radicals.py
                        ./migrate
                        ./templates
                        ./static/style.css
                        ./static/fonts
                        ./static/success.wav
                        ./static/failure.wav
                    ];
                };
            in {
                devShells.default = pkgs.mkShell {
                    name = "hanzipy-dev-shell";

                    buildInputs = [
                        pythonEnv
                        pkgs.git
                        pkgs.which
                    ];

                    shellHook = ''
                        echo "Entering python dev shell."
                    '';
                };

                packages.default = pkgs.writeShellApplication {
                    name = "hanzi-friend";
                    runtimeInputs = [ appEnv ];
                    text = ''
                        cd ${appSrc}
                        exec python serve.py
                    '';
                };
            }
        )) // {
            nixosModules.default = { config, lib, pkgs, ... }:
                let
                    cfg = config.services.hanzi-friend;
                in {
                    options.services.hanzi-friend = {
                        enable = lib.mkEnableOption "Hanzi Friend";

                        host = lib.mkOption {
                            type = lib.types.str;
                            default = "127.0.0.1";
                            description = "Address to bind; front with your reverse proxy.";
                        };

                        port = lib.mkOption {
                            type = lib.types.port;
                            default = 8089;
                            description = "Port to listen on.";
                        };

                        environmentFile = lib.mkOption {
                            type = lib.types.nullOr lib.types.path;
                            default = null;
                            description = ''
                                Secrets: OPENAI_API_KEY, SPEECH_KEY,
                                FLASK_SECRET, and INVITE_CODE (unset
                                keeps registration closed).
                            '';
                        };
                    };

                    config = lib.mkIf cfg.enable {
                        systemd.services.hanzi-friend = {
                            wantedBy = [ "multi-user.target" ];
                            after = [ "network.target" ];

                            environment = {
                                HANZI_DATA_DIR = "/var/lib/hanzi-friend";
                                HOST = cfg.host;
                                PORT = toString cfg.port;
                            };

                            serviceConfig = {
                                ExecStart = "${self.packages.${pkgs.stdenv.hostPlatform.system}.default}/bin/hanzi-friend";
                                DynamicUser = true;
                                StateDirectory = "hanzi-friend";
                                Restart = "on-failure";
                            } // lib.optionalAttrs (cfg.environmentFile != null) {
                                EnvironmentFile = cfg.environmentFile;
                            };
                        };
                    };
                };
        };
}
