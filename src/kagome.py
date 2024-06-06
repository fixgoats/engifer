import numpy as np


def kagome(nrows=1, ncols=1, a=1):
    """
    Produce the vertices of a kagome (trihexagonal) tiling. This function always
    includes the adjacent triangles of the edge hexagons of the tiling. The
    vertices are calculated assuming the leftmost corner of the bottom leftmost
    triangle is the origin, growing to the right and upwards.

    Parameters
    ----------
    nrows : int
        Number of rows to be produced
    ncols : int
        Number of columns to be produced
    a : float
        Sidelengths of the hexagons
    """
    baselayer = np.array([[0, 0], [a, 0], [2 * a, 0], [3 * a, 0]])
    baselayer = np.concatenate(
        (
            baselayer,
            np.array([[(4 + i) * a, 0] for i in range((ncols - 1) * 2)]),
        )
    )
    midlayer = np.array(
        [[0.5 * a, (np.sqrt(3) / 2) * a], [2.5 * a, (np.sqrt(3) / 2) * a]]
    )
    midlayer = np.concatenate(
        (
            midlayer,
            np.array(
                [[(2.5 + 2.0 * i) * a, (np.sqrt(3) / 2) * a] for i in range(ncols - 1)]
            ),
        )
    )
    bottomlayer = np.array(
        [[(1.5 + i) * a, -(np.sqrt(3) / 2) * a] for i in range(ncols)]
    )
