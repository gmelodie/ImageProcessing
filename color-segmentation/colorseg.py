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
    return np.sqrt(np.mean(np.square(generated.astype(float) - \
                                     reference.astype(float))))


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


def euclidean_distance(centroid, pixel):
    return np.linalg.norm(centroid - pixel)




#================================= K MEANS ===================================

def kmeans(params, dset):

    # unpack parameters
    seed = params['seed']
    k_clusters = params['seed']
    n_iterations = params['seed']

    # define initial centroids
    random.seed(seed)
    centroids = np.sort(random.sample(range(0, dset.shape[0]*dset.shape[1]), \
                                      k_clusters))

    for i in range(n_iterations):
        for pixel in dset:
            get_distances = np.vectorize(euclidean_distance)
            distances = get_distances(centroids, pixel)

            # TODO: change to new array
            pixel = np.argmin(distances)

        # TODO: use np.mean(new_array[new_array == 0], axis=0)...
    pass


#=========================== PROCESSING FUNCTIONS ============================


def rgb(params):
    original = imageio.imread(params['original'])
    segmented = kmeans(params, original)
    pass


def rgbxy(params):
    original = imageio.imread(params['original'])
    pass


def luminance(params):
    original = imageio.imread(params['original'])
    luminance = np.dot(original, [0.299, 0.587, 0.114])
    segmented = kmeans(params, luminance)
    pass


def luminancexy(params):
    original = imageio.imread(params['original'])
    luminancexy = np.dot(original, [0.299, 0.587, 0.114])
    #TODO: add x and y in luminancexy
    segmented = kmeans(params, luminancexy)
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

    # normalize to 0 - 255
    generated = normalize(generated, 0, 255)

    reference = imageio.imread(params['reference'])

    error = compute_error(reference, generated)
    print('{0:.4f}'.format(error))



