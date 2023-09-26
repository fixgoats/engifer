import cmath
import math

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
    s = set()
    triangles = []
    for i in range(5):
        b = cmath.rect(radius, math.radians(i*72))
        c = cmath.rect(radius, math.radians(i*72 + 36))
        d = cmath.rect(radius, math.radians((i + 1)*72))
        triangles += [Triangle(0, 0j, b, c), Triangle(0, 0j, c, d)]
    for _ in range(steps):
        triangles = subdivide(triangles)
    for t in triangles:
        s.add(t.a)
        s.add(t.b)
        s.add(t.c)
    return s
