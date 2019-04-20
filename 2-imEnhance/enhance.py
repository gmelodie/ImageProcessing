# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 2: IMAGE ENHANCEMENT
# =====================================================

import numpy as np
import imageio


def compute_error(reference, generated):
    return np.sqrt(np.mean(np.square(generated - reference)))


def read_params():
    params = {}

    # read filename
    params['reference'] = input().rstrip()

    params['choice'] = int(input())
    choice = params['choice']

    if choice == 1:
        params['initial_th'] = float(input())
    elif choice == 2:
        params['filter_size'] = float(input())
        params['weights'] = [float(a) for a in input().split()]
    elif choice == 3:
        params['filter_size'] = float(input())

        # Read matrix filter_size X filter_size
        params['weights_mat'] = []
        for i in range(int(params['filter_size'])):
            params['weights_mat'].append([float(a) for a in input().split()])

        params['initial_th'] = float(input())
    elif choice == 4:
        params['filter_size'] = float(input())
    else:
        raise ValueError('Not a valid method number')


    return params


def convolve1d(img, mask, mask_size):
    mask_half = int(mask_size // 2)

    for i in range(len(img)):
        s = 0
        for j in range(-mask_half, mask_half):
            s += img[i - mask_half + j]
        img[i] = s

    return img


def convolve2d(img, mask, pad_size):
    ans = np.zeros((img.shape[0]-2*pad_size, img.shape[1]-2*pad_size))
    # mask is originally read as a normal python list
    # need numpy array
    mask = np.array(mask)

    for i in range(pad_size, img.shape[0] - pad_size):
        for j in range(pad_size, img.shape[1] - pad_size):
            # crop part of image the same shape as mask
            sub_arr = img[i-pad_size : i+pad_size+1, j-pad_size : j+pad_size+1]
            # calculate sum between cropped array and mask
            ans[i-pad_size][j-pad_size] = np.einsum('ij,ij->', sub_arr, mask)

    return ans


def get_median_neigh(img, i, j, shift):
    neigh = []

    for x in range(i-shift, i+shift+1):
        for y in range(j-shift, j+shift+1):
            neigh.append(img[x][y])

    neigh.sort()
    neigh_mid = int(len(neigh) // 2) + 1
    return neigh[neigh_mid]


def convolve_median(pad_img, pad_size):
    ans = np.zeros((pad_img.shape[0]-2*pad_size, pad_img.shape[1]-2*pad_size))

    for i in range(pad_size, pad_img.shape[0] - pad_size):
        for j in range(pad_size, pad_img.shape[1] - pad_size):
            ans[i-pad_size][j-pad_size] = get_median_neigh(pad_img, i, j, pad_size)

    return ans


# ======================= ENHANCEMENT FUNCTIONS =======================
def limiarization(params, other_img=None):
    # unpack parameters
    img = imageio.imread(params['reference'])
    initial_th = params['initial_th']

    # used when called by filtering2
    if other_img is not None:
        img = other_img

    cur_th = initial_th

    # forces following while to execute once
    past_th = initial_th + 1

    # Calculate threshold
    while abs(cur_th - past_th) >= 0.5:
        # group pixels
        lower = img[img <= cur_th]
        higher = img[img > cur_th]

        # update estimate
        past_th = cur_th
        cur_th = 0.5 * (np.mean(higher) + np.mean(lower))

    # Apply calculated threshold
    img[img < cur_th] = 0

    return img


def filtering1(params):
    # unpack parameters
    img = imageio.imread(params['reference'])
    filter_size = params['filter_size']
    weights = params['weights']

    # Save shape for later
    orig_shape = img.shape

    # Apply filter to flatten image
    flat_img = convolve1d(img.flatten(), weights, filter_size)

    # Reshape img back to 2D array
    img = flat_img.reshape(orig_shape)

    return img


def filtering2(params):
    # unpack parameters
    img = imageio.imread(params['reference'])
    filter_size = int(params['filter_size'])
    weights_mat = params['weights_mat']
    initial_th = params['initial_th']

    # We want to augment 2 lines if n == 5
    # And 3 lines if n == 7...
    pad_size = int(filter_size // 2)
    pad_img = np.pad(img, pad_width=pad_size, mode='symmetric')

    img = convolve2d(pad_img, weights_mat, pad_size)

    img = limiarization(params, other_img=img)

    return img


def median(params):
    # unpack parameters
    img = imageio.imread(params['reference'])
    filter_size = int(params['filter_size'])

    pad_size = int(filter_size // 2)
    pad_img = np.pad(img, pad_width=pad_size, mode='constant', constant_values=(0))

    img = convolve_median(pad_img, pad_size)
    return img


# =====================================================================


if __name__ == '__main__':

    methods = {
        1: limiarization,
        2: filtering1,
        3: filtering2,
        4: median,
    }

    params = read_params()

    # apply filter
    generated = methods[params['choice']](params)
    reference = imageio.imread(params['reference'])

    print('{0:.2f}'.format(compute_error(reference, generated)))






