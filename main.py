#!/usr/bin/env python3

import argparse

import numpy as np
from overlapadd2_function import overlapadd2_int16

MAX_RANDOM_INT = 255 # equal to np.iinfo(np.uint8).max
#MAX_RANDOM_INT = 3

FILTER = np.array([[-1, 0, 1]])




def convolutions_np(M, T):
    Dx = np.apply_along_axis(
            lambda row: np.convolve(row, T[0]),
            axis=1, arr=M)
    Dy = np.apply_along_axis(
            lambda col: np.convolve(col, T[0]),
            axis=0, arr=M)
    return Dx, Dy

def convolutions_overlap_add(M, T):
    Dx = overlapadd2_int16(M, T)
    Dy = overlapadd2_int16(M, T.T)
    return Dx, Dy

# TODO implement PyCuda version


def positive_int(s):
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

    Dx, Dy = convolutions_np(M, FILTER)

    print(M)
    print()
    print(Dx)
    print()
    print(Dy)

