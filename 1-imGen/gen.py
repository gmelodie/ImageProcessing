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
        dx = random.choice([-1, 0, 1])
        dy = random.choice([-1, 0, 1])
        x = (x + dx) % C
        y = (y + dy) % C
        f[x, y] = 1

    return f

# =============================================================================

def normalize_img(parameters, img, new_min, new_max):
    C = parameters['C']
    old_min = np.min(img)
    old_max = np.max(img)

    norm_img = (img - old_min) * ((new_max - new_min)/(old_max - old_min)) \
        + new_min

    return norm_img


def sceneImgGen(parameters, f):
    pass


def digitalImgGen(parameters, g):
    pass


def compare(parameters, g):
    pass


if __name__ == '__main__':

    parameters = {}

    # read input parameters
    parameters['inputfile'] = str(input())
    parameters['C'] = int(input())
    sceneImgGenFunc = int(input())
    parameters['Q'] = int(input())
    parameters['N'] = int(input())
    parameters['B'] = int(input())
    parameters['S'] = int(input())

    # load image
    # parameters['img'] = imageio.imread(parameters['inputfile'])

    # call specified function
    sceneImgGen = {1: simple,
                   2: sincos,
                   3: root,
                   4: rand,
                   5: randomwalk,
    }

    # generate scene image
    C = parameters['C']
    f = np.zeros((C, C), dtype=float)

    # initialize seed (if ever needed)
    S = parameters['S']
    random.seed(S)

    if sceneImgGenFunc == 5: # randomwalk works differently than other functions
        f = randomwalk(parameters)
    else:
        for x in range(C):
            for y in range(C):
                f[x, y] = sceneImgGen[sceneImgGenFunc](parameters, x, y)

    # visualization stuff, remove later
    print(f)
    plt.imshow(f, cmap='gray')
    plt.show()

    # normalize scene image
    norm_f = normalize_img(parameters, f, 0, 2**16 - 1)
    print(norm_f)

    # generate digital image
    g = digitalImgGen(parameters, norm_f)

    # compare images
    error = compare(parameters, g)

    # print error
    print(error)

