{
  description = "tdseries — tick-exact time-series frames with named, indexed dimensions";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = { self, nixpkgs, flake-utils, git-hooks }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            ruff = {
              enable = true;
              package = pkgs.ruff;
            };
            ruff-format = {
              enable = true;
              package = pkgs.ruff;
            };
            pyright = {
              enable = true;
            };
          };
        };
      in
      {
        formatter =
          let
            inherit (pre-commit-check.config) package configFile;
            script = ''
              ${pkgs.lib.getExe package} run --all-files --config ${configFile}
            '';
          in
          pkgs.writeShellScriptBin "pre-commit-run" script;

        checks = {
          inherit pre-commit-check;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = pre-commit-check.enabledPackages ++ (with pkgs; [
            python
            uv
            # C++ standard library for NumPy and other native dependencies
            stdenv.cc.cc.lib
            zlib
          ]);

          shellHook = ''
            ${pre-commit-check.shellHook}
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH
            export UV_PYTHON=${python}/bin/python
            uv sync --quiet
            source .venv/bin/activate
          '';
        };
      });
}
