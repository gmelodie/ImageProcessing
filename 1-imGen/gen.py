# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 1: IMAGE GENERATION
# =====================================================
import numpy as np
import random
import imageio
import math
from matplotlib import pyplot as plt


# ================= IMAGE GENERATION FUNCTIONS ================================
def simple(parameters, x, y):
    return x*y + 2*y


def sincos(parameters, x, y):
    Q = parameters['Q']
    return abs(math.cos(x/Q) + 2*math.sin(y/Q))

def root(parameters, x, y):
    Q = parameters['Q']
    return abs(3*(x/Q) - (y/Q)**(1./3.))


def rand(parameters, x, y):
    return random.uniform(0, 1)


# Works a bit differently than the other generation functions
def randomwalk(parameters):
    C = parameters['C']
    f = np.zeros((C, C), dtype=float)

    x = 0
    y = 0
    f[x, y] = 1

    for _ in range(1+(C*C)):
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        x = (x + dx) % C
        y = (y + dy) % C
        f[x, y] = 1

    return f

# =============================================================================

def normalize_img(img, new_min, new_max):
    old_min = np.min(img)
    old_max = np.max(img)

    norm_img = (img - old_min) * ((new_max - new_min)/(old_max - old_min)) \
        + new_min

    return norm_img


def sceneImgGen(parameters):

    # call specified function
    sceneImgGen = {1: simple,
                   2: sincos,
                   3: root,
                   4: rand,
                   5: randomwalk,
    }

    # Load variables, for reading's sake
    sceneImgGenFunc = parameters['sceneImgGenFunc']
    C = parameters['C']
    S = parameters['S']
    f = np.zeros((C, C), dtype=float)

    # initialize seed (if ever needed)
    random.seed(S)

    # Actual img generation
    if sceneImgGenFunc == 5: # randomwalk works differently than other functions
        f = randomwalk(parameters)
    else:
        for x in range(C):
            for y in range(C):
                f[x, y] = sceneImgGen[sceneImgGenFunc](parameters, x, y)

    # visualization stuff, remove later
    # print(f)
    # plt.imshow(f, cmap='gray')
    # plt.show()

    # normalize scene image
    norm_f = normalize_img(f, 0, 2**16 - 1)

    return norm_f


def downsample(f, C, N):
    dsampled_f = np.zeros((N, N), dtype=float)
    ratio = C/N

    # rounds  if ratio is non integer
    # ratio = 1.4
    # dsampled_f[3, 2] = f[4, 3]
    for x in range(N):
        for y in range(N):
            dsampled_f[x, y] = f[round(ratio*x), round(ratio*y)]

    return dsampled_f


def digitalImgGen(parameters, f):
    N = parameters['N']
    C = parameters['C']
    B = parameters['B']

    # Downsampling
    dsampled_f = downsample(f, C, N)

    # Quantisation
    g = normalize_img(dsampled_f, 0, 2**8 - 1).astype(np.uint8)
    g = np.right_shift(g, 8 - B) # shift to use only B most significant bits

    # visualization stuff, remove later
    plt.imshow(g, cmap='gray')
    plt.show()

    return g


def compare(parameters, g):
    r = parameters['img']
    return np.sqrt(np.mean(np.square(g - r)))


def read_input():
    parameters = {}
    parameters['inputfile'] = str(input())
    parameters['C'] = int(input())
    parameters['sceneImgGenFunc'] = int(input())
    parameters['Q'] = int(input())
    parameters['N'] = int(input())
    parameters['B'] = int(input())
    parameters['S'] = int(input())
    return parameters


if __name__ == '__main__':

    parameters = read_input()

    # load image
    parameters['img'] = np.load(parameters['inputfile'])
    # test if img reading is fine
    plt.imshow(parameters['img'], cmap='gray')
    plt.show()

    # generate normalized scene image
    norm_f = sceneImgGen(parameters)

    # generate digital image
    g = digitalImgGen(parameters, norm_f)

    # compare images
    error = compare(parameters, g)

    # print error
    print(error)

