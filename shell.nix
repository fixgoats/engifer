{ pkgs ? import <nixpkgs> { } }:
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
