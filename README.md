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
Ideally, create and activate a virtual environment and install the dependencies:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
Now you can run `movingpumps.py`. You need an Nvidia GPU to run it and it takes 
at least a few seconds to run.
```
python movingpumps.py
```
