from numba import njit
import matplotlib.pyplot as plt
import cmath as m
import numpy as np

T = 0
H = 1
P = 2
F = 3

sq3 = np.sqrt(3)

class MTile:
    __slots__ = ["type", "rotation", "translation", "scale"]

    def __init__(self, type, rotation, translation, scale):
        self.type = type
        self.rotation = rotation
        self.translation = translation
        self.scale = scale

p1 = 0
p2 = sq3 * np.exp(-1j * np.pi / 6)
p3 = 2 * np.exp(-1j * np.pi / 3)
p4 = 2 * np.exp(-2j * np.pi / 3)
p5 = sq3 * np.exp(-5j * np.pi / 6)
p6 = 2*sq3 * np.exp(-5j * np.pi / 6)
p7 = p6 + sq3 * np.exp(5j * np.pi / 6)
p8 = p6 + 2 * np.exp(2j * np.pi / 3)
p9 = p6 + sq3 * np.exp(1j * np.pi / 2)
p10 = 2 * sq3 * np.exp(5j * np.pi / 6)
p11 = p10 + sq3 * np.exp(1j * np.pi / 6)
p12 = p10 + 2
p13 = p1 + sq3 * 1j


hat = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13])

reflected_hat = np.array([-p.real + 1j*p.imag for p in hat])

connections = ((1, 12), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12))

htile = np.array([np.exp(2j*np.pi / 3) * reflected_hat - 2j * sq3, reflected_hat + 2 * sq3 * np.exp(-1j * np.pi / 6), np.exp(2j*np.pi / 3) * reflected_hat + 2*sq3*np.exp(1j*np.pi / 6), -1 * hat])
ttile = np.array([np.exp(2j*np.pi / 3) * reflected_hat])
ptile = np.array([-1 * reflected_hat, np.exp(2j*np.pi/3) * reflected_hat + 2*sq3*np.exp(-1j * np.pi / 6)])
ftile = np.array([reflected_hat, np.exp(-1j * np.pi / 3) * reflected_hat + 2*sq3*np.exp(5j*np.pi/6)])

hpftile = np.concatenate((htile,
                          np.exp(-1j*np.pi/3)*ptile + 4*sq3*np.exp(1j*np.pi / 6),
                              np.exp(1j*np.pi/3)*ftile + 3+5j*sq3,#4*sq3*np.exp(1j*np.pi/3)
                          np.exp(1j*np.pi/3)*ptile + 2*sq3*np.exp(5j*np.pi/6),
                          -1*ftile + 4*sq3*np.exp(7j*np.pi/6),
                          ptile + 3 - 3j*sq3,
                          np.exp(-1j*np.pi/3)*ftile + 6*sq3*np.exp(-1j*np.pi/6)
                          ))
print(hpftile)

for h in hpftile:
        plt.plot(h.real, h.imag)

plt.show()

def expand(mtile):
    if mtile.type == T:
        return [MTile(H, mtile.rotation, mtile.translation), MTile(H, mtile.rotation, mtile.translation)]
