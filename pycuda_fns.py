"""Implement the convolutions using PyCUDA, and a function to find the min/max
given two GPUArrays using a parallel reduction."""

import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

MAX_THREADS_PER_BLOCK = 1024 # for p2.xlarge, which has NVidida GK210 GPUs
BLOCK_DIM = (32, 32, 1)

def convolutions_pycuda(M, T):
    """
    Compute convolutions using PyCuda.
    Returns GPUArray objects representing arrays stored in GPU - doesn't load them to main memory.
    """
    assert M.dtype == np.uint8
    M_gpu = gpuarray.to_gpu(M)

    Dx = np.empty((M.shape[0], M.shape[1] + T.shape[1] - 1), dtype=np.int16)
    Dx_gpu = gpuarray.to_gpu(Dx)

    Dy = np.empty((M.shape[0] + T.shape[1] - 1, M.shape[1]), dtype=np.int16)
    Dy_gpu = gpuarray.to_gpu(Dy)

    mod = SourceModule("""
        #define DX_WIDTH %d
        #define DX_HEIGHT %d
        #define DY_WIDTH %d
        #define DY_HEIGHT %d
        #define M_WIDTH %d

        __global__ void get_Dx(short *Dx, unsigned char *M) // TODO M is char, not short
        {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;

            if ((idx >= DX_WIDTH) || (idy >= DX_HEIGHT))
                return;

            int id = idx + idy * DX_WIDTH;
            short pos, neg; // the parts added and subtracted in the convolution
                // NOTE this assumes short are 2 bytes, which is not
                // specified by the standard but is generally true


            if (idx < 2)
                pos = 0;
            else
                pos = M[idx + idy * M_WIDTH - 2];

            if (idx >= DX_WIDTH - 2)
                neg = 0;
            else
                neg = M[idx + idy * M_WIDTH];

            Dx[id] = pos - neg;
        }

        __global__ void get_Dy(short *Dy, unsigned char *M) // TODO M is char, not short
        {
            int idx = threadIdx.x + blockDim.x * blockIdx.x;
            int idy = threadIdx.y + blockDim.y * blockIdx.y;

            if ((idx >= DY_WIDTH) || (idy >= DY_HEIGHT))
                return;

            int id = idx + idy * DY_WIDTH;
            short pos, neg; // the parts added and subtracted in the convolution
                // NOTE this assumes short are 2 bytes, which is not
                // specified by the standard but is generally true

            if (idy < 2)
                pos = 0;
            else
                pos = M[idx + (idy - 2) * M_WIDTH];

            if (idy >= DY_HEIGHT - 2)
                neg = 0;
            else
                neg = M[idx + idy * M_WIDTH];
            
            Dy[id] = pos - neg;
        }
        """ % (Dx.shape[1], Dx.shape[0], Dy.shape[1], Dy.shape[0], M.shape[1]))

    # Compute the grid size needed
    dx, mx = divmod(Dx.shape[1], BLOCK_DIM[1])
    dy, my = divmod(Dx.shape[0], BLOCK_DIM[0])
    grid_dim = ((dx + (mx>0)), (dy + (my>0)))
        # from talonmies and DLH at:
        # https://stackoverflow.com/questions/14504580/pycuda-blocks-and-grids-to-work-with-big-datas
    # Run compiled function get_Dx
    get_Dx = mod.get_function("get_Dx")
    get_Dx(Dx_gpu, M_gpu, block=BLOCK_DIM, grid=grid_dim, shared=0)

    # Repeat for Dy
    dx, mx = divmod(Dy.shape[1], BLOCK_DIM[1])
    dy, my = divmod(Dy.shape[0], BLOCK_DIM[0])
    grid_dim = ((dx + (mx>0)), (dy + (my>0)))
    get_Dy = mod.get_function("get_Dy")
    get_Dy(Dy_gpu, M_gpu, block=BLOCK_DIM, grid=grid_dim, shared=0)

    return Dx_gpu, Dy_gpu

def minmax_pycuda(Dx_gpu, Dy_gpu):
    """
    Given two GPUArrays, finds and returns their mins and maxes.
    This is all done in the GPU; only the mins/maxes are sent back to the host.
    """
    return (gpuarray.min(Dx_gpu).get(), gpuarray.max(Dx_gpu).get(),
            gpuarray.min(Dy_gpu).get(), gpuarray.max(Dy_gpu).get())
