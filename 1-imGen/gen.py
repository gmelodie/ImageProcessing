# =====================================================
# Name: Gabriel de Melo Cruz
# NUSP: 9763043
# Course code: SCC0251
# Period: 2019/1
# Assignment 1: IMAGE GENERATION
# =====================================================
import numpy as np
import random



def simple(parameters):
    pass


def sincos(parameters):
    pass


def root(parameters):
    pass


def rand(parameters):
    pass


def randomwalk(parameters):
    pass


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



    # call specified function
    sceneImgGen = {1: simple,
                   2: sincos,
                   3: root,
                   4: rand,
                   5: randomwalk,
    }

    # generate scene image
    f = sceneImgGen[sceneImgGenFunc](parameters)

    # generate digital image
    g = digitalImgGen(parameters, f)

    # compare images
    error = compare(parameters, g)

    # print error
    print(error)

