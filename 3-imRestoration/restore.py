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



#========================= AUX FUNCTIONS ===========================

# calculates disp_n
def general_dispersion():
    ans = 0

    if ans == 0:
        return 1

    return ans


# calculates disp_l
def local_dispersion(x, y, image, disp_n):
    ans = 0

    if ans == 0:
        return disp_n

    return ans


# calculates centr_l
def local_centralization(x, y, image, mode):
    ans = 0

    if mode == 'robust':
        pass
    elif mode == 'average':
        pass
    else:
        raise ValueError('Invalid centralization mode')

    return ans



#========================= FILTERING FUNCTIONS ===========================


def denoising(params):
    degradated = imageio.imread(params['degradated'])
    generated = imageio.imread(params['degradated'])

    # unpack parameters
    gamma = params['gamma']
    mode = params['denoising_mode']

    disp_n = general_dispersion(degradated)

    for x in range(generated.shape[0]/6, generated.shape[0] - generated.shape[0]/6):
        for y in range(generated.shape[1]/6, generated.shape[1] - generated.shape[1]/6):
            disp_l = local_dispersion(x, y, degradated)
            centr_l = local_centralization(x, y, degradated, mode)
            generated[x][y] = degradated[x][y] - (gamma * (disp_n/disp_l) * (degradated[x][y] - centr_l))

    return normalize(generated, 0, max(degradated))


def deblurring(params):
    pass


if __name__ == '__main__':
    params = read_params()

    generated = params['filter_func'](params)

    reference = imageio.imread(params['reference'])
    error = compute_error(reference, generated)

    print('{0:.3f}', format(error))



