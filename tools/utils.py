from pathlib import Path

from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def save_train_log_png(path: Path):
    with open(path, 'r') as f:
        log_data = pd.read_csv(f)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    metrics = log_data.columns.to_list()[3:]
    log_data.plot(x='epoch', y=['train_loss', 'val_loss'], kind='line', ax=axs[0])
    log_data.plot(x='epoch', y=metrics, kind='line', ax=axs[1])
    for a in axs:
        a.legend(loc='upper left', bbox_to_anchor=(1,1), frameon=False)

    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')        


def load_data_from_casa(path):
    array = pd.read_csv(path, sep='\t', skiprows=3, header=None).drop(columns=2).to_numpy()
    return array


def interpolate(x, y, num=256):
    f = interp1d(x, y, kind='linear')
    new_x = np.linspace(x[0], x[-1], num)
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
        params=({'color': 'b', 'alpha': 0.2}, {'color': 'r'}),
        save_path=None
):

    plt.plot(x, y, 'k')

    min_to_fill = y.min()
    
    for mask, param in zip(masks, params):
        plt.fill_between(x, y, min_to_fill, where=mask > 0, **param)
    # plt.axis('off')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
