import numpy as np
from numpy.random import random, normal, choice
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt

from utils import view_labeled_data, create_mask


def create_gauss(x, loc, scale, c, base=True):
    gauss = stats.norm(loc=loc, scale=scale)
    y = c * gauss.pdf(x)
    if base:
        y += c * 0.1 * gauss.cdf(x)
    return y


def generate_synth():

    x_start = random() * 2000
    x_stop = x_start + 0.1 * 256
    x = np.arange(x_start, stop=x_stop, step=0.1)
    if len(x) > 256:
        x = x[:256]
    y = np.zeros_like(x)
    peak_mask = np.zeros_like(x)
    max_mask = np.zeros_like(x)

    num_of_peaks = int(np.ceil(random() * 2 ))
    # last_peak = None

    for _ in range(num_of_peaks):
        loc = choice(x[32:224])
        scale = random() + 1
        c = random()

        y += create_gauss(x, loc, scale, c)

        peak_mask += create_mask(x, y, from_x=loc+4*scale, to_x=loc-4*scale)
        peak_mask[peak_mask > 1] = 1
        max_mask += create_mask(x, y, from_x=loc+scale/4, to_x=loc-scale/4)
        max_mask[max_mask > 1] = 1
        
        # last_peak = sorted([loc+4*scale, loc-4*scale])

    noise = normal(0, 0.1, x.shape) / 8
    y += noise
    y -= y.min()
    y = y / y.max()
    return x, y, peak_mask, max_mask


if __name__ == '__main__':
    counter = 85
    while True:
        x, y, peak_mask, max_mask = generate_synth()
        view_labeled_data(x, y, (peak_mask, max_mask))
        q = input('Save? ')
        # if q == 'y':
        #     array = np.stack((x, y, peak_mask, max_mask), axis=1)
        #     df = pd.DataFrame(array)
        #     df.to_csv(f'data/data_to_train/synth_{counter}.csv', index=False, header=False)
        #     counter += 1
        # elif q == 'stop':
        #     break
