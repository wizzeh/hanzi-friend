{
    description = "Hanzi Friend in Python";
    
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem(system:
            let
                pkgs = import nixpkgs {
                    inherit system;
                };
               
                hanzipy = pkgs.python312Packages.buildPythonPackage {
                    pname = "hanzi-py";
                    version = "1.0.4";

                    src = pkgs.fetchzip {
                        url = "https://github.com/Synkied/hanzipy/archive/refs/tags/v1.0.4.zip";
                        sha256 = "sha256-PXIitMnzL6BETKu4waYUWB45W5N5VAfY/Gno41cf3Rg=";
                    };
                };

                fsrs = pkgs.python312Packages.buildPythonPackage {
                    pname = "fsrs";
                    version = "6.1.0";

                    src = pkgs.fetchzip {
                        url = "https://github.com/open-spaced-repetition/py-fsrs/archive/refs/tags/v6.1.0.zip";
                        sha256 = "sha256-zN0Ga6Jb0qTwlwlSr61jIGgafRWrbPHCMJT1km46RFA=";
                    };
                };
                pythonEnv = pkgs.python312.withPackages (ps: with ps; [
                    flask
                    pypinyin
                    hanzipy
                    fsrs
                    tinydb
                    openai
                    python-dotenv
                    requests
                    pillow
                    numpy
                    torch
                    transformers
                ]);
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
            }
        );
}
