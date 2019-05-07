# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 2: IMAGE ENHANCEMENT
# =====================================================

import numpy as np
import scipy
import imageio


def normalize(img, new_min, new_max):
    old_min=np.min(img)
    old_max=np.max(img)

    norm_img = (img - old_min) * ((new_max - new_min)/(old_max - old_min)) \
        + new_min

    return norm_img


def compute_error(reference, generated):
    return np.sqrt(np.mean(np.square(generated - reference)))


def read_params():
    params = {}

    params['reference'] = input().rstrip()
    params['degradated'] = input().rstrip()
    filter_choice = int(input().rstrip())
    params['gamma'] = float(input().rstrip())

    # Size of kernel if denoising
    # Size of degradation function if deblurring
    params['size'] = int(input().rstrip())

    # Filter choice comes as integer, assign function
    if filter_choice == 1:
        params['filter_func'] = denoising
        params['denoising_mode'] = input().rstrip()
    elif filter_choice == 2:
        params['filter_func'] = deblurring
        params['sigma'] = int(input().rstrip())
    else:
        raise ValueError('Filter choice must be either 1 or 2')

    return params


#========================= FILTERING FUNCTIONS ===========================


def denoising(params):
    degradated = imageio.imread(params['degradated'])
    generated = imageio.imread(params['degradated'])

    # TODO: something like this, not exaclty
    for x in range(generated.shape[0]):
        for y in range(generated.shape[1]):
            dispn = dispersion(x, y, degradated)

    return normalize(generated, 0, max(degradated))


def deblurring(params):
    pass


if __name__ == '__main__':
    params = read_params()

    generated = params['filter_func'](params)

    reference = imageio.imread(params['reference'])
    error = compute_error(reference, generated)

    print('{0:.3f}', format(error))



