let
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/abd6d48f8c77bea7dc51beb2adfa6ed3950d2585.tar.gz") { };
in
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
      numpy
      matplotlib
      scipy
      torch-bin
      desktop-notifier
      numba
      z3
      ipython
      pip
      black
      isort
      python-lsp-server
      python-lsp-ruff
      pylsp-rope
      pynvim
      ueberzug
    ]))
    pkgs.neovim
  ];
}
