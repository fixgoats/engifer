#!/usr/bin/bash
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/thin.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/thick.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/thickthin.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/thinthickthin.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/thickthinthick.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/penroselarge.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/penrosemed.toml
env/bin/python rhombcombinationsfromscratch.py rhombconfigs/penrosesmall.toml
