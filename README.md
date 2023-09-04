# Numerical Schrödinger equation solvers

This repo provides some classes to numerically simulate the Schrödinger
equation with periodic and Dirichlet boundary conditions in 2D, and
example usage of these classes.

# Usage
Clone the project and change directories into the project:
```
git clone https://github.com/fixgoats/engifer.git
cd engifer
```
Ideally, create and activate a virtual environment:
```
python3 -m venv env
source env/bin/activate
```
Dependencies are specified in `requirements.txt` and can be installed after
activating the virtual environment:
```
pip install -r requirements.txt
```
Now you can run `torchtest.py`. You need an Nvidia GPU to run it, and you supply
the flags `-a` and `-d` to specify the output files for the animation of the system
in R and k space and the animation for the dispersion relation, respectively.
```
python torchtest.py -a animations/system.mp4 -d animations/dispersion.mp4
```
If either flag is 
