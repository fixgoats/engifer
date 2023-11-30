import cmath
import math
import numpy as np

goldenRatio = (1 + math.sqrt(5)) / 2


class Triangle:
    __slots__ = ('kind', 'a', 'b', 'c')

    def __init__(self, kind, a, b, c):
        self.kind = kind
        self.a = a
        self.b = b
        self.c = c

    def xs(self):
        return [self.a.real, self.b.real, self.c.real]

    def ys(self):
        return [self.a.imag, self.b.imag, self.c.imag]


def subdivide(triangles):
    result = []
    for t in triangles:
        if t.kind == 0:
            p = t.a + (t.b - t.a) / goldenRatio
            result += [Triangle(0, t.c, p, t.b), Triangle(1, p, t.c, t.a)]
        else:
            q = t.b + (t.a - t.b) / goldenRatio
            r = t.b + (t.c - t.b) / goldenRatio
            result += [Triangle(1, r, t.c, t.a),
                       Triangle(1, q, r, t.b),
                       Triangle(0, r, q, t.a)]
    return result


def makeSunGrid(radius, steps):
    s = np.ndarray((0, 2))
    triangles = []
    for i in range(5):
        b = cmath.rect(radius, math.radians(i * 72))
        c = cmath.rect(radius, math.radians(i * 72 + 36))
        d = cmath.rect(radius, math.radians((i + 1) * 72))
        triangles += [Triangle(0, 0j, b, c), Triangle(0, 0j, d, c)]
    for _ in range(steps):
        triangles = subdivide(triangles)
    for t in triangles:
        if not any([all(np.isclose(v, np.array([t.a.real, t.a.imag]), atol=0.01)) for v in s]):
            s = np.concatenate((s, [[t.a.real, t.a.imag]]))
        if not any([all(np.isclose(v, np.array([t.b.real, t.b.imag]), atol=0.01)) for v in s]):
            s = np.concatenate((s, [[t.b.real, t.b.imag]]))
        if not any([all(np.isclose(v, np.array([t.c.real, t.c.imag]), atol=0.01)) for v in s]):
            s = np.concatenate((s, [[t.c.real, t.c.imag]]))
    return s


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    grid1 = makeSunGrid(5, 4)
    # grid2 = makeSunGrid(5, 3)
    # grid3 = makeSunGrid(5, 2)
    # grid4 = makeSunGrid(5, 1)
    # grid5 = makeSunGrid(5, 0)
    ax.set_aspect('equal')
    # ax.scatter(grid1[:, 0], grid1[:, 1], color='r')
    # ax.scatter(grid2[:, 0], grid2[:, 1], color='g')
    ax.scatter(grid1[:, 0], grid1[:, 1], color='b')
    # ax.scatter(grid4[:, 0], grid4[:, 1], color='orange')
    # ax.scatter(grid5[:, 0], grid5[:, 1], color='m')
    plt.show()
