#!/usr/bin/env python3

"""Plots time taken by each method."""

from timeit import default_timer as timer
import signal
import time

import numpy as np
import pandas as pd

from numpy_fns import convolutions_np, minmax_streaming
from overlapadd2_fn import convolutions_overlap_add
from pycuda_fns import convolutions_pycuda, minmax_pycuda

import matplotlib.style as style
style.use('seaborn-poster')

MAX_RANDOM_INT = 255 # equal to np.iinfo(np.uint8).max

FILTER = np.array([[-1, 0, 1]])

METHODS = ('np', 'overlap_add', 'cuda')
SIZES = np.logspace(1, 4.5, 8, dtype=int)

MAXIMUM_SECONDS = 10


class Timeout(Exception):
    pass

def raise_timeout(*_):
    raise Timeout

def ignore_signal(*_):
    pass

def time_method(method, M):
    """
    Calculate how long the given convolution method takes on M.
    """
    print(f"\tTiming method {method}")
    convolution_fn = {
            'np':           convolutions_np,
            'overlap_add':  convolutions_overlap_add,
            'cuda':         convolutions_pycuda,
            }[method]

    # Register timeout signal, start alarm
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(MAXIMUM_SECONDS)
    # Time function, give up on timeout
    start = timer()
    try:
        convolution_fn(M, FILTER)
    except Timeout:
        print("\t\tTimed out")
        return np.NaN
    except MemoryError:
        print("\t\tNot enough memory")
        return np.NaN
    signal.signal(signal.SIGALRM, ignore_signal) # undo signal handler registration
    end = timer()
    return end - start


if __name__ == "__main__":

    results = pd.DataFrame(index=pd.Index(SIZES, name="M side length"),
                           columns=pd.Index(METHODS, name="Method used"))

    for size in SIZES:
        print(f"For size {size}:")
        M = np.random.randint(0, MAX_RANDOM_INT+1, size=(size, size), dtype=np.uint8)

        for method in METHODS:
            time_taken = time_method(method, M)
            results.ix[size, method] = time_taken

    print(results)
    ax = results.plot.line(loglog=True, legend=True, style='.-')
    ax.set_ylabel("Time (seconds)")
    fig = ax.get_figure()
    fig.savefig('plot.png')
