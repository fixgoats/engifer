# Numerical Schrödinger and GP equation solvers

This repo provides some classes to numerically simulate the Schrödinger
and GP equation in 2D space, and example usage of these classes. Most of the
classes in `src/solvers.py` are probably flawed in some way, but `SsfmGPGPU`
is usable. Although it has GPU in the name it may be possible to make it run on
cpu by passing the cpu in as a device.

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
Now you can run the example script to simulate the Gross-Pitaevskii equation,
`example.py` with the example configuration, `example.toml`. It should be possible
to run it with any GPU supported by Pytorch if the appropriate device is passed
to the GPU based class, so if you use an AMD GPU you'll have to modify the script
accordingly.  The command to run it, assuming you have an active virtual environment
is:
```
python example.py configs/example.toml
```
Since this project has a lot of files specific to my needs I recommend just copying
`src/solvers.py` if you want to use this functionality in your project.
