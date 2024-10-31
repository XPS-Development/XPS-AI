from copy import deepcopy

from scipy.interpolate import interp1d
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
        params=({'color': 'b', 'alpha': 0.2}, {'color': 'r'})
):

    plt.plot(x, y, 'k')

    min_to_fill = y.min()
    
    for mask, param in zip(masks, params):
        plt.fill_between(x, y, min_to_fill, where=mask > 0, **param)
    # plt.axis('off')
    plt.show()


def create_mask(x, from_x, to_x):  
    zeros = np.zeros_like(x)
    zeros[(x > from_x) & (x < to_x)] = 1
    return zeros

#TODO: complete
def labeling(array):

    # file_name = path.split('/')[-1]

    # data_to_masking = pd.read_csv(path, sep=';').to_numpy()
    x, y = interpolate(array[:, 2], array[:, 1])

    masks = [np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]

    for mask_num in range(3):
        print(f'Mask_num: {mask_num}')
        view_labeled_data(x, y)

        while True:
            mask = np.zeros_like(x)
            while True:
                from_x = float(input('from '))
                view_point(x, y, from_x)
                q = input('Continue? ')
                
                try:
                    to_x = float(q)
                    mask += create_mask(x, y, from_x, to_x)
                    view_labeled_data(x, y, mask)
                    break
                except ValueError:
                    pass

            q = input('Next mask or peak? ')
            if q == 'm':
                masks[mask_num] += mask
                break
            elif q == 'p':
                masks[mask_num] += mask
                
    print('Final result:')

    mask_1, mask_2, mask_3 = masks[0], masks[1], masks[2]
    view_labeled_data(x, y, mask_1, mask_2, mask_3)

    array = np.stack(
        (x, y, mask_1, mask_2, mask_3),
        axis=1
    )
    return array

# if __name__ == '__main__':
#     # for i in range(8, 12):
#     #     path = f'data/data_from_casa/{i}.csv'
#     #     labeling(path)
#     data = pd.read_csv('data/data_to_train/1.csv').to_numpy()
#     view_point(data, 133)


if __name__ == '__main__':
    # for i in range(1, 18):
    #     path = f'data/data_to_train/{i}.csv'
    #     array = np.loadtxt(path, delimiter=',')
    #     x, y, mask_1, mask_2 = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
    #     view_labeled_data(
    #         x, y, [mask_1, mask_2]
    #     )
    print('ok')
