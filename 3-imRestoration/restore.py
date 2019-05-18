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
    params['size_k'] = int(input().rstrip())

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



#========================= DENOISING AUX FUNCTIONS ===========================

def local_disp_centr(x, y, image, disp_n, size_k, mode):
    """
    Returns a tuple: (disp_l, centr_l)
    """

    border_pixels = image[x-size_k : x+size_k+1, y-size_k : y+size_k+1]

    if mode == 'robust':
        percentiles = np.percentile(border_pixels, [75, 50, 25])
        disp_l = percentiles[2] - percentiles[0] # interquartile range
        centr_l = percentiles[1] # the 50 percentile is the median
    elif mode == 'average':
        disp_l = np.std(border_pixels)
        centr_l = np.mean(border_pixels)
    else:
        raise ValueError('Invalid denoising mode')

    if disp_l == 0:
        disp_l = disp_n

    return disp_l, centr_l


def general_dispersion(image, mode):
    """
    Returns disp_n calculated related to the borders of the image
    """
    # border paddings
    rows = image.shape[0]
    cols = image.shape[1]
    m = rows//6
    n = cols//6

    # border arrays
    right = image[::, 0:n].flatten()
    left = image[::, cols-n:cols].flatten()
    top = image[0:m, ::].flatten()
    bot = image[rows-m:rows, ::].flatten()

    border_pixels = np.concatenate((right, left, top, bot))

    # calculate dispersion measure according to mode
    if mode == 'robust':
        percentiles = np.percentile(border_pixels, [75, 50, 25])
        disp_n = percentiles[2] - percentiles[0]
    elif mode == 'average':
        disp_n = np.std(border_pixels)
    else:
        raise ValueError('Invalid denoising mode')

    if disp_n == 0:
        return 1

    return disp_n


#========================= FILTERING FUNCTIONS ===========================


def denoising(params):
    degradated = imageio.imread(params['degradated'])
    generated = imageio.imread(params['degradated'])

    # unpack parameters
    gamma = params['gamma']
    mode = params['denoising_mode']
    size_k = params['size_k']

    disp_n = general_dispersion(degradated, mode)

    for x in range(generated.shape[0]//6, \
                       generated.shape[0] - generated.shape[0]//6):
        for y in range(generated.shape[1]//6, \
                       generated.shape[1] - generated.shape[1]//6):
            disp_l, centr_l = \
                local_disp_centr(x, y, degradated, disp_n, size_k, mode)
            generated[x][y] = degradated[x][y] \
                    - (gamma * (disp_n/disp_l) * (degradated[x][y] - centr_l))

    return normalize(generated, 0, np.max(degradated))


def deblurring(params):
    pass


if __name__ == '__main__':
    params = read_params()

    generated = params['filter_func'](params)

    reference = imageio.imread(params['reference'])
    error = compute_error(reference, generated)

    print('{0:.3f}'.format(error))



