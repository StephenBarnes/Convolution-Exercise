"""Implement the convolutions and the streaming min/max using NumPy."""

import numpy as np

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

def minmax_streaming(Dx, Dy):
    """
    Find min and max of Dx and Dy, using linear-time streaming algorithm.
    """
    return Dx.min(), Dx.max(), Dy.min(), Dy.max()
