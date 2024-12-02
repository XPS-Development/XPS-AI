from pathlib import Path

from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def view_point(x, y, point_x):
    plt.plot(x, y, 'k')
    idx = (np.abs(x - point_x)).argmin()
    plt.plot(x[idx], y[idx], 'ro')
    plt.show()


def view_labeled_data(
        x, 
        y, 
        masks=(),
        mask_params=({'color': 'b', 'alpha': 0.2}, {'color': 'r'}),
        lines=(),
        save_path=None
):

    plt.plot(x, y, 'k')

    min_to_fill = y.min()
    for mask, mask_param in zip(masks, mask_params):
        plt.fill_between(x, y, min_to_fill, where=mask > 0, **mask_param)

    for line in lines:
        plt.plot(x, line)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()