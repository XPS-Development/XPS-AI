#TODO: rename module
#TODO: ordered input to model

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import numpy as np
from numpy import trapz
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import XPSModel
from dataset import XPSDataset
from utils import view_labeled_data, view_point

PATH_TO_VAL = 'data/data_to_val'
TRESHOLD = 0.5


def gauss(x, loc, scale, c):
    return c * np.exp(-(x-loc)**2 / (2*scale**2))


def g_summ(n):

    def f(x, *p):
        y = np.zeros_like(x, dtype=np.float32)
        for i in range(n):
            y += gauss(x, p[3 * i], p[3 * i + 1], p[3 * i + 2])
        return y

    return f


def find_region_borders(mask):
    # return idxs of borders in mask
    separators = np.diff(mask)
    idxs = np.argwhere(separators == True).reshape(-1)
    return idxs


def calc_shirley_background(x, y, i_1, i_2):
    # i_2 > i_1
    background = np.zeros_like(x, dtype=np.float32)

    while True:
        y_adj = y - background
        s_adj = trapz(y_adj, x)
        shirley_to_i = lambda i: i_1 + (i_2 - i_1) * trapz(y_adj[:i+1], x[:i+1]) / s_adj
        
        points = [shirley_to_i(i) for i in range(len(x))]
        if np.allclose(background, points, rtol=1e-1):
            return points
        else:
            background = points


if __name__ == '__main__':
    model = XPSModel()
    model.load_state_dict(torch.load('WEIGHTS_OLD'))
    model.eval()

    dataset_val = XPSDataset(PATH_TO_VAL)
    dataloader_val = DataLoader(dataset_val, shuffle=False)

    for x, peak_mask, max_mask in dataloader_val:
        # get model output
        pred_peak_mask, pred_max_mask = model(x)
        y = x.view(-1).detach().cpu().numpy()
        x = np.arange(1, 257, step=1)
        # reduce noise
        y_filtered = savgol_filter(y, 50, 4)

        pred_peak_mask = pred_peak_mask.view(-1).detach().cpu().numpy()
        pred_peak_mask = (pred_peak_mask > TRESHOLD)

        pred_max_mask = pred_max_mask.view(-1).detach().cpu().numpy()
        pred_max_mask = (pred_max_mask > TRESHOLD)
        view_labeled_data(x, y, (pred_peak_mask, pred_max_mask))

        # save data for each region
        spec_data = {'regions_info': []}
        # find separated peak regions
        peak_borders_idx = find_region_borders(pred_peak_mask)
        for n in range(len(peak_borders_idx) // 2):
            f = peak_borders_idx[2 * n]
            t = peak_borders_idx[2 * n + 1] + 1
            l = t - f
            # skip too small regions
            if l < 20:
                continue

            curr_y = y[f:t]
            curr_x = x[f:t]
            curr_y_filtered = y_filtered[f:t]
            curr_max_mask = pred_max_mask[f:t]

            # average over 10% points
            i_1 = np.mean(curr_y_filtered[:l//10])
            i_2 = np.mean(curr_y_filtered[-l//10:])
            
            #TODO: delete this
            if i_1 > i_2:
                i_1, i_2 = i_2, i_1
                curr_y = curr_y[::-1]
                curr_y_filtered = curr_y_filtered[::-1]
                curr_x = curr_x[::-1]

            background = calc_shirley_background(curr_x, curr_y, i_1, i_2)

            # find idxs of max regions in each peak region
            # idxs belong to curr_max_mask
            max_borders = find_region_borders(curr_max_mask)
            n_peaks = len(max_borders) // 2
            # find x-borders from max_mask
            max_borders = curr_x[max_borders]

            # create initial values for each gaussian
            p0 = []
            for i in range(n_peaks):
                # values: loc=mean(borders), scale=10, c=0.5
                p0.extend([(max_borders[2*i] + max_borders[2*i + 1])/2, 10, 0.5])

            popt, pcov = curve_fit(g_summ(n_peaks), curr_x, curr_y - background, p0)

            spec_data['regions_info'].append((curr_x, background, n_peaks, popt))
        
        plt.plot(x, y, color='k', alpha=0.4)
        for region in spec_data['regions_info']:
            r_x = region[0]
            r_back = region[1]
            n_peaks = region[2]
            peak_params = region[3]
            plt.plot(r_x, r_back, color='k')
            for i in range(n_peaks):
                plt.plot(r_x, gauss(r_x, peak_params[3*i], peak_params[3*i+1], peak_params[3*i+2]) + r_back)
            min_to_fill = y.min()
            plt.fill_between(x, y, min_to_fill, where=pred_peak_mask > 0, color='b', alpha=0.2)
        plt.show()

