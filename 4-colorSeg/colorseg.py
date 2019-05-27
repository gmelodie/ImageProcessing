# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 4: Color image processing and segmentation
# =====================================================

import numpy as np
import imageio, random, math


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

    params['original'] = input().rstrip()
    params['reference'] = input().rstrip()
    params['pixel_attr']= int(input().rstrip())

    if params['pixel_attr'] > 4 or params['pixel_attr'] < 1:
        raise ValueError('Option for pixel attributes must be an integer \
                          between 0 and 1')

    params['k_clusters'] = int(input().rstrip())
    params['n_iterations'] = int(input().rstrip())
    params['seed'] = int(input().rstrip())

    return params


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)




#================================= K MEANS ===================================

def kmeans(seed, k_clusters, n_iterations, dset):
    random.seed(seed)
    # TODO: What is m?
    centroids = np.sort(random.sample(range(0, dset.shape[0], dset.shape[1]),\
                                      k_clusters))
    pass


#=========================== PROCESSING FUNCTIONS ============================


def rgb(params):
    original = imageio.imread(params['original'])
    pass


def rgbxy(params):
    original = imageio.imread(params['original'])
    pass


def luminance(params):
    original = imageio.imread(params['original'])
    pass


def luminancexy(params):
    original = imageio.imread(params['original'])
    pass


#=============================================================================


if __name__ == '__main__':
    params = read_params()

    processing_opts = {
        1: rgb,
        2: rgbxy,
        3: luminance,
        4: luminancexy,
    }

    # call processing function
    option = params['pixel_attr']
    generated = processing_opts[option](params)

    reference = imageio.imread(params['reference'])

    error = compute_error(reference, generated)
    print('{0:.4f}'.format(error))



