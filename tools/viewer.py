import numpy as np
from matplotlib import pyplot as plt


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


def view_point(x, y, point_x):
    plt.plot(x, y, 'k')
    idx = (np.abs(x - point_x)).argmin()
    plt.plot(x[idx], y[idx], 'ro')
    plt.show()
