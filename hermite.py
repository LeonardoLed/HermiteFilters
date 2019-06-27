# Hermite Fuctions
"""
This function calculate the Hermite filters given the order and scale, the filters are based on continuos hermite
transform matrix and DoG

Note:
    This function uses the formula

    Hn(x) = (2x)^n - n(n-1)/1!(2x)^(n-2) + n(n-1)(n-2)(n-3)/2!(2x)^(n-4)-...

    to generate the coefficients for the hermite of order n.

paramaters: from DCCN we have :
            M numbers of orientations =>  select several filters in several orders (where maximum order is n)
                                            the order n depends of filter size, (n-1) <= h,w
            h, w size of filters
            nScale, the scale of filtersnn

#version 0.1 n is static (n=4, to have 5x5 filters), the selected orientation is (2,0),(1,1),(1,1)', (0,2)

@autor: Leonardo Ledesma Domínguez


"""
import math
from math import pi, sqrt
import numpy as np
import tensorflow as tf
# HERMITE - Compute the Hermite polynomials
def hermite(D, x):
    """

    :param n: degree of polynomial
    :param x: evaluates the polynomial at x. It uses numpy POLYVAL
    :return: H (Hermite operarator of general filters equation)
    """

    dmax = max(D)
    nup = len(D)
    sf = (len(x), nup)
    H = np.zeros(sf)

    for i in range(0, nup):
        n = D[i]
        k = np.arange(n, -1, -2)
        s = np.power(-1, np.floor((n - k) / 2))
        c = np.zeros(len(k))
        for j in range(0, len(k)):
            z = n - (2 * j - 1)
            a = np.arange(n, z - 1, -1)
            b = np.arange(1, j + 1, 1)
            c[j] = np.prod(a) / np.prod(b)
            # print (n, j, k, s, z, a, b, c, np.prod (a), np.prod (b))

        h = np.zeros(n + 1)
        h[n - k] = s * (np.power(2, k)) * c
        H[:, i] = np.polyval(h, x)

    return H



# HERMITE - Continuos Hermite transform matrix
def hermite_filters(nScale, M, h, w):
    """

    :param nScale: scale value of filter
    :param M: number of orientantions
    :param h: size filter in x
    :param w: size filter in y
    :return: the bank of hermite filters given Mz
    """

    # the value to start scale in the given order

    s_min = (w-1) / 8
    # the next value for the next order
    s_max = w / 8
    dif_s = (s_max - s_min) / 4
    s = s_min + (dif_s * nScale)
    # print s
    # print dif_s, s, w, s_min, s_max
    x = np.arange(-4 * s, (4 * s) + s_min, 1)
    x = x[:w]
    n = np.arange(0, h, 1)
    d = np.cumprod(n[1:])
    e = np.array([1])
    f = np.concatenate((e, d))
    c = np.power(-1, n) * np.sqrt(np.power(2, n) * f)
    # guassian window
    g = np.exp(-1 * np.power(x, 2) / (4 * s)) / np.sqrt(4 * s * pi)
    # hermite polynomials
    p = hermite(D=n, x=x / sqrt(4 * s))
    g_ones = np.transpose(np.tile(g, (h, 1)))
    c_ones = np.tile(c, (len(x), 1))
    # print (c_ones, x)
    h = p * g_ones / c_ones

    return h


def normalize(matrix):

    #normalize the Hermite filters
    #xymin = 1e309
    #xymax = -1e309
    xymax = np.max(matrix)
    xymin = np.min(matrix)
    #xymax = max(xymax, max_A)
    #xymin = min(xymin,min_A)

    #print("valores")
    #print(xymax, xymin, matrix)
    matrix = (matrix-xymin)/(xymax-xymin)
    #print(matrix)
    return matrix


def getHermiteFilterBank(nScale, M, h,w):
    """
    Selected Filters for Convolutional Neuronal Network

    v1.0 the selected filters for four orientantions are:
    (0,0) | (1,0) | (2,0)[1]
    ---------------------
    (0,1) | (1,1)[2] -> (1,1)' [4}
    -------------
    (0,2)[3] |

    :param M: number of orientantions
    :return: bank of Hermite Filter Bank given M
    """
    nScale = nScale -1
    A = hermite_filters(nScale, M, h, w)
    #print(A)

    f1 = A[:, 0].reshape((-1, 1))
    f2 = A[:, 1].reshape((-1, 1))
    f3 = A[:, 2].reshape((-1, 1))

    # orientations
    of1 = A[:, 0] * f3
    of2 = A[:, 1] * f2
    of3 = A[:, 2] * f1
    of4 = np.rot90(of2)


    # normalize
    of1 = normalize(of1)
    of2 = normalize(of2)
    of3 = normalize(of3)
    of4 = normalize(of4)

    # print(of2)

    gfilter_real = np.stack((of1, of2, of3, of4))
    #gfilter_real = torch.from_numpy(gfilter_real)
    #gfilter_real = torch.tensor(gfilter_real.tolist())
    gfilter_real = tf.constant(gfilter_real)
    #print gfilter_real
    gfilter_real = tf.cast(gfilter_real, tf.float32)

    return gfilter_real

def main():
    tf.enable_eager_execution()
    A = getHermiteFilterBank(1,4,5,5)
    print(A)
    print(type(A))

if __name__  == '__main__':
    main()
