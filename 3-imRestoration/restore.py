# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 3: IMAGE FILTERING
# =====================================================

import numpy as np
from scipy import fftpack
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
        params['sigma'] = float(input().rstrip())
    else:
        raise ValueError('Filter choice must be either 1 or 2')

    return params



#========================= DENOISING AUX FUNCTIONS ===========================

def local_disp_centr(x, y, image, disp_n, size_k, mode):
    """
    Returns a tuple: (disp_l, centr_l)
    """

    border_pixels = image[x-size_k//2 : x+size_k//2+1, y-size_k//2 : y+size_k//2+1]

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



#======================= DEBLURRING AUX FUNCTIONS =========================


def gaussian_filter(k=3, sigma=1.0):
    arx = np.arange((-k//2) + 1.0, (k//2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma))

    return filt/np.sum(filt)


def pad_mask(mask, img_dimensions):
    """
    Pad a matrix (mask) to be the same size as img_dimensions
    """
    # Assume both dimensions are even
    pad_dimensions = [(img_dimensions[0]//2 - 2, img_dimensions[0]//2 - 1), \
                      (img_dimensions[1]//2 - 2, img_dimensions[1]//2 - 1)]

    # Check if dimensions are odd (initial assumption was wrong
    if img_dimensions[0] % 2 != 0:
        pad_dimensions[0] = (img_dimensions[0]//2-1, img_dimensions[0]//2-1)

    if img_dimensions[1] % 2 != 0:
        pad_dimensions[1] = (img_dimensions[1]//2-1, img_dimensions[1]//2-1)


    pad_mask = np.pad(mask, pad_dimensions, 'constant', constant_values=0)

    return pad_mask


#========================= FILTERING FUNCTIONS ===========================


def denoising(params):
    degradated = imageio.imread(params['degradated'])
    generated = imageio.imread(params['degradated'])

    # unpack parameters
    gamma = params['gamma']
    mode = params['denoising_mode']
    size_k = params['size_k']
    border_ignore_size = size_k//2

    disp_n = general_dispersion(degradated, mode)

    for x in range(generated.shape[0]//border_ignore_size, \
                       generated.shape[0] - generated.shape[0]//border_ignore_size):
        for y in range(generated.shape[1]//border_ignore_size, \
                       generated.shape[1] - generated.shape[1]//border_ignore_size):
            disp_l, centr_l = \
                local_disp_centr(x, y, degradated, disp_n, size_k, mode)
            generated[x][y] = degradated[x][y] \
                    - (gamma * (disp_n/disp_l) * (degradated[x][y] - centr_l))

    return normalize(generated, 0, np.max(degradated))


def deblurring(params):

    # unpack parameters
    sigma = params['sigma']
    gamma = params['gamma']
    degradated = imageio.imread(params['degradated'])
    fft_degradated = fftpack.fft2(degradated)

    # prepare degradation function
    degrad_func = gaussian_filter()
    pad_degrad_func = pad_mask(degrad_func, degradated.shape)
    fft_degrad_func = fftpack.fft2(pad_degrad_func)
    prep_degrad_func = np.square(np.abs(fft_degrad_func))

    # also prepare complex conjugate
    conjugate_fft_degrad_func = np.conjugate(fft_degrad_func)

    # prepare laplatian operator
    lapl_op = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    pad_lapl_op = pad_mask(lapl_op, degradated.shape)
    fft_lapl_op = fftpack.fft2(pad_lapl_op)
    prep_lapl_op = np.square(np.abs(fft_lapl_op))

    generated = ( conjugate_fft_degrad_func \
                / (prep_degrad_func + gamma*prep_lapl_op) ) \
                * fft_degradated

    generated = np.real(fftpack.fftshift(fftpack.ifft2(generated)))

    return normalize(generated, 0, np.max(degradated))


if __name__ == '__main__':
    params = read_params()

    generated = params['filter_func'](params)

    reference = imageio.imread(params['reference'])
    error = compute_error(reference, generated)

    print('{0:.3f}'.format(error))



