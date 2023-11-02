# Code taken from Hastings Greer's post on generating an aperiodic hat tiling.
# http://www.hgreer.com/HatTile/
# Although the code works miracles it is very slow, so the resulting points are
# dumped to a file for quick access.

import z3
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

grid_size = 21
x, y = np.mgrid[-2:grid_size - 2, -2:grid_size - 2]
hexagon_centers = x + .5 * y + 1j * np.sqrt(3) / 2 * y
six_rotations = np.exp(1j * (np.pi /3 * np.arange(6)))

kite = np.array([0, .5, 1 / np.sqrt(3) * np.exp(1j * np.pi / 6), .5 * np.exp(1j * np.pi / 3), 0])
kites = kite[None, None, None, :] * six_rotations[None, None, :, None]
kites = kites + hexagon_centers[:, :, None, None]



indices = [
    [2, 2, 2, 2, 2, 2, 1, 1], #row
    [2, 2, 2, 2, 1, 1, 2, 2], #col
    [1, 2, 3, 4, 0, 1, 5, 4]  #rotation
    ]
hat = kites[indices[0], indices[1], indices[2], :]

hats = hat[None, :, :] * six_rotations[:, None, None]
hats = np.concatenate([hats, np.real(hats) - 1j * np.imag(hats)])
hats = hats[None, None, :, :, :] + hexagon_centers[:, :, None, None, None]
hats = np.reshape(hats, (-1, len(hat), 5))
hat_centers = np.mean(hats, axis=-1)
hat_centers = np.round(hat_centers, 2)
hats_with_point = defaultdict(lambda: [])
for hat_index, centers in enumerate(hat_centers):
  for loc in centers:
    hats_with_point[loc] += [hat_index]
max_pop = max(len(c) for p, c in hats_with_point.items())
full_points = np.array([p for p, c in hats_with_point.items() if len(c) == max_pop])
all_points = np.array([p for p, c in hats_with_point.items()])
def atleastone(solver, bools):
  solver.add(z3.Or(bools))
def atmostone(solver, bools):
  #solver.add(z3.PbLe([(x,1) for x in bools], 1))
  for i, b1 in enumerate(bools):
    for j, b2 in enumerate(bools):
      if i > j:
        solver.add(z3.Not(z3.And(b1, b2)))
hat_present = [z3.Bool(f"hat{i}") for i in range(len(hats))]
s = z3.Solver()
for p in all_points:
  atmostone(s, [hat_present[i] for i in hats_with_point[p]])
for p in full_points:
  atleastone(s, [hat_present[i] for i in hats_with_point[p]])
print(s.check())
m = s.model()
chosen_hats = np.array([z3.is_true(m[h]) for h in hat_present])
hat = np.round(hat, 2)
segments = np.concatenate([hat[:, 1:, None], hat[:, :-1, None]], axis=2)
segments = segments.reshape(-1, 2)
reversed_segments = set(((seg[1], seg[0]) for seg in segments))
outline = np.array([l for l in segments if tuple(l) not in reversed_segments])
outlines = outline[None, :] * six_rotations[:, None, None]
outlines = np.concatenate([outlines, np.real(outlines) - 1j * np.imag(outlines)])
outlines = outlines[None, None, :, :, :] + hexagon_centers[:, :, None, None, None]
outlines = np.reshape(outlines, (-1, len(outline), 2))
result = outlines[np.array(chosen_hats), :]
result = result.reshape(-1, 2)
plt.figure(figsize=(10, 10))


def npModSqr(x):
    return x.real * x.real + x.imag * x.imag


def remove_duplicates(array, tol):
    retarray = np.ndarray((0), dtype=complex)
    for element in array:
        if all(npModSqr(element - other) > tol for other in retarray):
            retarray = np.append(retarray, element)
    return retarray


grid = remove_duplicates(result.flatten(), 0.01)
np.save("montilegrid.npy", grid)
