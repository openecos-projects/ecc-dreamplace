{
  inputs.self.submodules = true;
  # pinning nixpkgs for old cmake and gcc
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/f4b140d5b253f5e2a1ff4e5506edbf8267724bde";
  outputs = inputs@{
    self, nixpkgs, flake-parts,
  }: let
    ecc-dreamplace = {
      lib,
      python3Packages,
      cmake,
      ninja,
      cairo,
      bison,
      flex,
      pkg-config,
    }: python3Packages.buildPythonPackage rec {
      name = "dreamplace";
      format = "pyproject";

      src = with lib.fileset; toSource {
        root = ./.;
        fileset = unions [
          ./thirdparty
          ./dreamplace
          ./unittest
          ./benchmarks
          ./test
          ./cmake
          ./CMakeLists.txt
          ./pyproject.toml
          ./uv.lock
        ];
      };

      build-system = [
        python3Packages.scikit-build-core
      ];

      # The torch>=2.11,<2.12 pin exists to select a PyPI CPU wheel in upstream
      # CI (14577a56); the pinned nixpkgs ships torch 2.8.0 which dreamplace
      # builds and runs against fine. Relax it for the nix build. The CMake
      # gate guards the same wheel-ABI contract, so relax its lower bound too.
      postPatch = ''
        substituteInPlace pyproject.toml \
          --replace-fail 'torch>=2.11,<2.12' 'torch'
        substituteInPlace cmake/TorchExtension.cmake \
          --replace-fail 'VERSION_LESS 2.11 OR' 'VERSION_LESS 2.8 OR'
      '';

      dependencies = with python3Packages; [
        cairocffi
        distutils
        matplotlib
        numpy
        patool
        pkgconfig
        scipy
        setuptools
        shapely
        torch
        wheel
      ];

      buildInputs = [ cairo flex ];
      nativeBuildInputs = [ bison flex cmake ninja pkg-config ];

      dontUseCmakeConfigure = true;
      dontCheckRuntimeDeps = true;

      pythonImportsCheck = [
        "dreamplace"
        "dreamplace.Params"
      ];

      passthru.rawBuildInputs = buildInputs;
      passthru.rawNativeBuildInputs = nativeBuildInputs;
    };
  in flake-parts.lib.mkFlake { inherit inputs; } {
    systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    perSystem = { self', pkgs, system, config, ... }: {
      # Re-export packages and the devShell as checks so CI
      # (`nix flake check`) builds them.
      checks = config.packages // {
        devShell = config.devShells.default;
      };
      packages.default = pkgs.callPackage ecc-dreamplace {};
      devShells.default = pkgs.mkShell.override {} {
        buildInputs = self'.packages.default.rawBuildInputs;
        nativeBuildInputs = self'.packages.default.rawNativeBuildInputs ++ (with pkgs; [ uv ]);
      };
    };
  };
}
