# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 4: Color image processing and segmentation
# =====================================================

import numpy as np
import imageio, random, sys



def normalize(img, new_min, new_max):
    old_min=np.min(img)
    old_max=np.max(img)

    if debug:
        print('old min: ', old_min)
        print('new min: ', new_min)
        print('old max: ', old_max)
        print('new max: ', new_max)

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





#================================= K MEANS ===================================

def update_centroids(centroids, classifications, dset):
    newcentroids = np.copy(centroids)

    # TODO: use np.mean(new_array[new_array == 0], axis=0)...
    for centroid_idx in range(len(centroids)):
        mean_count_aux = 0
        mean_sum_aux = np.zeros(dset[0].shape)

        # calculate the mean of the elements in each cluster
        for entry_idx in range(len(dset)):
            if classifications[entry_idx] == centroid_idx:
                mean_count_aux += 1
                mean_sum_aux += dset[entry_idx]

        if mean_count_aux != 0:
            newcentroids[centroid_idx] = mean_sum_aux / mean_count_aux

    return newcentroids


def kmeans(params, dset):
    """
    dset: numpy array (m*n, j)
    where m = number of rows in original image
    n = number of cols in original image
    j = number of features (1, 2, 3 depending on the case)
    """

    # unpack parameters
    seed = params['seed']
    k_clusters = params['k_clusters']
    n_iterations = params['n_iterations']

    # define initial centroids
    random.seed(seed)
    centroids_idx = np.sort(random.sample(range(0,dset.shape[0]),k_clusters))
    centroids = dset[centroids_idx]

    # 1D array, each value contains the cluster to which that entry belongs
    classifications = np.zeros(len(dset))

    for i in range(n_iterations):

        for entry in range(len(dset)):
            distances = np.sqrt(np.square((centroids - dset[entry])).sum(axis=1))
            classifications[entry] = np.argmin(distances)

        if debug:
            print('classifications')
            print(classifications)
            print('centroids')
            print(centroids)

        newcentroids = update_centroids(centroids, classifications, dset)

        # check for convergence
        if np.array_equal(newcentroids, centroids): # kmeans already converged
            if debug:
                print('new controids')
                print(newcentroids)
            return classifications
        else:
            centroids = newcentroids

    return classifications


#=========================== PROCESSING FUNCTIONS ============================

def segment_dset(params, dset, original):
    dset = dset.astype(np.float64)
    classifications = kmeans(params, dset)
    segmented = np.reshape(classifications, \
                           (original.shape[0], original.shape[1]))
    return segmented


def rgb(params, original):
    dset = np.reshape(original, (original.shape[0] * original.shape[1], 3))
    if debug:
        print(dset)
    return segment_dset(params, dset, original)


def rgbxy(params, original):
    dset = np.reshape(original, (original.shape[0] * original.shape[1], 3))

    # include xy fields
    dset_xy = np.zeros((original.shape[0] * original.shape[1], 5))
    dset_xy[:,:-2] = dset
    for x in range(original.shape[0]):
        for y in range(original.shape[1]):
            dset_xy[x*original.shape[1] + y][3] = x
            dset_xy[x*original.shape[1] + y][4] = y

    return segment_dset(params, dset_xy, original)


def luminance(params, original):
    reshaped_original = np.reshape(original, \
                                   (original.shape[0] * original.shape[1], 3))
    luminance = np.dot(reshaped_original, [0.299, 0.587, 0.114])
    luminance = np.reshape(luminance, (len(luminance), 1))

    return segment_dset(params, luminance, original)


def luminancexy(params, original):
    reshaped_original = np.reshape(original, \
                                   (original.shape[0] * original.shape[1], 3))
    luminance = np.dot(reshaped_original, [0.299, 0.587, 0.114])
    luminance = np.reshape(luminance, (len(luminance), 1))

    # include xy fields
    luminance_xy = np.zeros((original.shape[0] * original.shape[1], 3))
    luminance_xy[:,:-2] = luminance
    for x in range(original.shape[0]):
        for y in range(original.shape[1]):
            luminance_xy[x*original.shape[1] + y][1] = x
            luminance_xy[x*original.shape[1] + y][2] = y

    return segment_dset(params, luminance_xy, original)


#=============================================================================


if __name__ == '__main__':

    global debug
    if len(sys.argv) > 1 and sys.argv[1] == 'debug':
        debug = True
    else:
        debug = False

    params = read_params()
    processing_opts = {
        1: rgb,
        2: rgbxy,
        3: luminance,
        4: luminancexy,
    }

    # segment image
    option = params['pixel_attr']
    original = imageio.imread(params['original'])
    generated = processing_opts[option](params, original)

    norm_generated = normalize(generated, 0, 255)

    if debug:
        import matplotlib.pyplot as plt
        plt.imshow(norm_generated, cmap='gray')
        plt.show()

    reference = np.load(params['reference'])
    error = compute_error(reference, norm_generated)
    print('{0:.4f}'.format(error))



