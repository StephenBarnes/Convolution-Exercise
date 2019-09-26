#!/usr/bin/env python3

"""
Runs the methods on a randomly-generated matrix.
Takes rows and cols as command-line arguments.
Also takes --method argument; default is to run cuda method.
"""

import argparse

import numpy as np

from numpy_fns import convolutions_np, minmax_streaming
from overlapadd2_fn import convolutions_overlap_add
from pycuda_fns import convolutions_pycuda, minmax_pycuda

MAX_RANDOM_INT = 255 # equal to np.iinfo(np.uint8).max

FILTER = np.array([[-1, 0, 1]])

METHODS = ('np', 'overlap_add', 'cuda')


def positive_int(s):
    """
    Convert string to int. Raise an error if string is nonpositive.
    """
    val = int(s)
    if val <= 0:
        raise argparse.ArgumentTypeError("Argument must be positive integer")
    return val


def run_method(method, M):
    print(f"Running method {method}:")
    convolution_fn = {
            'np':           convolutions_np,
            'overlap_add':  convolutions_overlap_add,
            'cuda':         convolutions_pycuda,
            }[method]
    minmax_fn = {
            'np':           minmax_streaming,
            'overlap_add':  minmax_streaming,
            'cuda':         minmax_pycuda,
            }[method]

    Dx, Dy = convolution_fn(M, FILTER)

    print("Dx minimum %d, maximum %d\nDy minimum %d, maximum %d" % minmax_fn(Dx, Dy))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('rows', type=positive_int, help="Number of rows of M")
    parser.add_argument('cols', type=positive_int, help="Number of columns of M")
    parser.add_argument('--method', help="Method to use for computing convolutions",
            choices=METHODS + ('all',), default='cuda')
    args = parser.parse_args()

    M = np.random.randint(0, MAX_RANDOM_INT+1, size=(args.rows, args.cols), dtype=np.uint8)

    if args.method != 'all':
        run_method(args.method, M)
    else:
        print("Running all methods...")
        for method in METHODS:
            run_method(method, M)

