import numpy as np
import matplotlib.pyplot as plt


def rotate(array, theta):
    return np.array([
        [vec[0]*np.cos(theta) - vec[1]*np.sin(theta),
         vec[1]*np.cos(theta) + vec[0]*np.sin(theta)]
        for vec in array
        ])


def makegrid(scale):
    vectors = scale*np.array([
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
    duplicates = []
    for i, row in enumerate(grid):
        for j, otherrow in enumerate(grid[i+1:, :]):
            if all(np.abs(row-otherrow) < 0.1*scale):
                duplicates.append(j + i + 1)
    return np.delete(grid, duplicates, axis=0)


def makerawgrid(scale):
    vectors = scale*np.array([
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
    return grid

if __name__ == '__main__':
    a = makerawgrid(1)
    fig, ax = plt.subplots()
    ax.scatter(a[:, 0], a[:, 1])
    plt.show()
