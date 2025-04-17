{
  description = "RyuLang";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            gdb
            zig
            llvm
            gcc13
            clang
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.llvm}/lib:$LD_LIBRARY_PATH
            export LIBRARY_PATH=${pkgs.llvm}/lib:$LIBRARY_PATH
            export C_INCLUDE_PATH=${pkgs.llvm}/include:$C_INCLUDE_PATH
            export LLVM_PATH=${pkgs.llvm}
            export LLVM_CONFIG=${pkgs.llvm}/bin/llvm-config
            export PATH=${pkgs.gcc13}/bin:$PATH
          '';
        };
      });
}
