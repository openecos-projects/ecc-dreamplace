{
  inputs.self.submodules = true;
  # pinning nixpkgs for old cmake and gcc
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/f4b140d5b253f5e2a1ff4e5506edbf8267724bde";
  outputs = inputs@{
    self, nixpkgs, flake-parts,
  }: let
    eccJobs = builtins.getEnv "ECC_JOBS";
    ecc-dreamplace = {
      lib,
      python3Packages,
      cmake,
      ninja,
      cairo,
      bison,
      flex,
      pkg-config,
      eccJobs ? "",
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

      preConfigure = ''
        JOBS="${eccJobs}"
        if [ -n "$JOBS" ]; then
          echo "dreamplace: injecting -j$JOBS"
          sed -i '/build\.tool-args/d' pyproject.toml
          sed -i "/^\[tool\.scikit-build\]/a build.tool-args = [\"-j$JOBS\"]" pyproject.toml
        fi
      '';

      postBuild = ''
        sed -i '/build\.tool-args/d' pyproject.toml 2>/dev/null || true
      '';
    };
  in flake-parts.lib.mkFlake { inherit inputs; } {
    systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
    perSystem = { self', pkgs, system, ... }: {
      packages.default = pkgs.callPackage ecc-dreamplace { inherit eccJobs; };
      devShells.default = pkgs.mkShell.override {} {
        buildInputs = self'.packages.default.rawBuildInputs;
        nativeBuildInputs = self'.packages.default.rawNativeBuildInputs ++ (with pkgs; [ uv ]);
      };
    };
  };
}
