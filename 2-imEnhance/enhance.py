# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 2: IMAGE ENHANCEMENT
# =====================================================

import numpy as np
import imageio


# TODO: check this function for correctness
def compute_error(reference, generated):
    return np.sqrt(np.mean(np.square(generated - reference)))


def read_params():
    params = {}

    params['choice'] = int(input())

    if choice == 1:
        params['initial_th'] = float(input())
    elif choice == 2:
        params['filter_size'] = float(input())
        params['weights'] = [float(a) for a in input().split()]
    elif choice == 3:
        params['filter_size'] = float(input())

        # TODO: read matrix filter_size X filter_size
        # weights_mat = 

        params['initial_th'] = float(input())
    elif choice == 4:
        params['filter_size'] = float(input())
    else:
        raise ValueError('Not a valid method number')


    return params


def limiarization(params):
    pass


def filtering1(params):
    pass


def filtering1(params):
    pass


def median(params):
    pass


if __name__ == '__main__':

    methods = {
        1: limiarization
        2: filtering1
        3: filtering2
        4: median
    }

    filename = input()

    params = read_params()

    # calculate image
    generated = methods[params['choice']](params)
    reference = np.load(filename)

    print(compute_error(reference, generated))






