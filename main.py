#!/usr/bin/env python3

import argparse

import numpy as np
from overlapadd2_fn import overlapadd2_int16
from convolve_pycuda_fn import convolve_pycuda

MAX_RANDOM_INT = 255 # equal to np.iinfo(np.uint8).max
MAX_RANDOM_INT = 3 # TODO remove

FILTER = np.array([[-1, 0, 1]])




def convolutions_np(M, T):
    """
    Convolve M and T horizontally and vertically, using np.convolve.
    """
    Dx = np.apply_along_axis(
            lambda row: np.convolve(row, T[0]),
            axis=1, arr=M)
    Dy = np.apply_along_axis(
            lambda col: np.convolve(col, T[0]),
            axis=0, arr=M)
    return Dx, Dy

def convolutions_overlap_add(M, T):
    """
    Convolve M and T horizontally and vertically, using overlap-add algorithm.
    """
    Dx = overlapadd2_int16(M, T)
    Dy = overlapadd2_int16(M, T.T)
    return Dx, Dy


def positive_int(s):
    """
    Convert string to int. Raise an error if string is nonpositive.
    """
    val = int(s)
    if val <= 0:
        raise argparse.ArgumentTypeError("Argument must be positive integer")
    return val

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=positive_int, help="Number of rows of M")
    parser.add_argument('cols', type=positive_int, help="Number of columns of M")
    args = parser.parse_args()

    M = np.random.randint(0, MAX_RANDOM_INT+1, size=(args.rows, args.cols), dtype=np.uint8)
        # +1 because upper endpoint is exclusive
    #if args.cols == 1:
    #    M = M.T
    # M is Hmat.shape, Na is Amat.shape

    Dx, Dy = convolutions_overlap_add(M, FILTER)

    print(M)
    print()
    print(Dx)
    print()
    print(Dy)
    print()

    #Dx, Dy = convolutions_np(M, FILTER)

    #print(M)
    #print()
    #print(Dx)
    #print()
    #print(Dy)


    Dx, Dy = convolve_pycuda(M, FILTER)

    print(M)
    print()
    print(Dx)
    print()
    print(Dy)
