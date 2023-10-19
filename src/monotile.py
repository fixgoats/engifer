import numpy as np
import matplotlib.pyplot as plt


def rotate(array, theta):
    return np.array([
        [vec[0]*np.cos(theta) - vec[1]*np.sin(theta),
         vec[1]*np.cos(theta) + vec[0]*np.sin(theta)]
        for vec in array
        ])


def makegrid():
    vectors = np.array([
            [0, 0],
            [0, 1.73205081],
            [-1,    0],
            [-0.5,  0.86602540],
            [-1.5,  -0.86602540],
            [0, -1.73205081],
            [-1, 0],
            [-0.5,  -0.86602540],
            [1.5, -0.86602540],
            [1.5, 0.86602540],
            [0.5, -0.86602540],
            [2, 0],
            [0.5, 0.86602540]
            ])
    tile = np.cumsum(vectors, axis=0)
    mirrortile = np.array([[-x[0], x[1]] for x in tile])
    tile2 = rotate(tile, np.radians(60))
    tile2 = tile2 - (tile2[4, :])
    tile3 = rotate(tile, np.radians(60))
    tile3 = tile3 + tile[8, :]
    tile4 = rotate(tile - tile[11, :], np.radians(180)) + (tile[10, :])
    tile5 = mirrortile + tile3[8, :]
    grid = np.concatenate((tile, tile2, tile3, tile4, tile5))
    returngrid = np.array([grid[0, :]])
    for row in grid:
        for rrow in returngrid:
            if any(np.abs(rrow - row) > 0.1):
                np.append(returngrid, [[0, 1]], axis=0)
                print(returngrid)
    return returngrid
