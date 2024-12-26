from scipy.interpolate import interp1d
import numpy as np


def gauss(x, loc, scale):
    return 1/(scale * np.sqrt(2*np.pi)) * np.exp(-(x-loc)**2 / (2*scale**2))


def lorentz(x, loc, scale):
    return 1/(np.pi * scale * (1 + ((x - loc) / scale) ** 2))


def pseudo_voight(x, loc, scale, c, r):
    return c * (r * gauss(x, loc, scale / np.sqrt(2*np.log(2))) + (1 - r) * lorentz(x, loc, scale))


def peak_sum(n):
    def f(x, *p):
        y = np.zeros_like(x, dtype=np.float32)
        for i in range(n):
            y += pseudo_voight(x, p[4 * i], p[4 * i + 1], p[4 * i + 2], p[4 * i + 3])
        return y
    return f     


def interpolate(x, y, num=256):
    f = interp1d(x, y, kind='linear')
    new_x = np.linspace(x[0], x[-1], num, dtype=np.float32)
    new_y = f(new_x)
    return new_x, new_y
