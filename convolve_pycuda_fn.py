import numpy as np

import pycuda.driver as cuda
import pycuda.autoinit  # noqa
from pycuda.compiler import SourceModule

def convolve_pycuda(M, T):
    M = M.astype(np.int16) # TODO
    M_gpu = cuda.mem_alloc(M.size * M.dtype.itemsize)
    cuda.memcpy_htod(M_gpu, M)

    Dx = np.empty((M.shape[0], M.shape[1] + T.shape[1] - 1), dtype=np.int16)
    Dx_gpu = cuda.mem_alloc(Dx.size * Dx.dtype.itemsize)

    Dy = np.empty((M.shape[0] + T.shape[1] - 1, M.shape[1]), dtype=np.int16)
    Dy_gpu = cuda.mem_alloc(Dy.size * Dy.dtype.itemsize)

    mod = SourceModule("""
        #define DX_WIDTH %d
        #define DX_HEIGHT %d
        #define DY_WIDTH %d
        #define DY_HEIGHT %d
        #define M_WIDTH %d

        __global__ void get_Dx(short *Dx, short *M) // TODO M is char, not short
        {
            int idx = threadIdx.x + threadIdx.y * DX_WIDTH;
            short pos, neg; // the parts added and subtracted in the convolution
                // NOTE this assumes short are 2 bytes, which is not
                // specified by the standard but is generally true

            if (threadIdx.x < 2)
                pos = 0;
            else
                pos = M[threadIdx.x + threadIdx.y * M_WIDTH - 2];

            if (threadIdx.x >= DX_WIDTH - 2)
                neg = 0;
            else
                neg = M[threadIdx.x + threadIdx.y * M_WIDTH];
            
            Dx[idx] = pos - neg;
        }

        __global__ void get_Dy(short *Dy, short *M) // TODO M is char, not short
        {
            int idx = threadIdx.x + threadIdx.y * DY_WIDTH;
            short pos, neg; // the parts added and subtracted in the convolution
                // NOTE this assumes short are 2 bytes, which is not
                // specified by the standard but is generally true

            if (threadIdx.y < 2)
                pos = 0;
            else
                pos = M[threadIdx.x + (threadIdx.y - 2) * M_WIDTH];

            if (threadIdx.y >= DY_HEIGHT - 2)
                neg = 0;
            else
                neg = M[threadIdx.x + threadIdx.y * M_WIDTH];
            
            Dy[idx] = pos - neg;
        }
        """ % (Dx.shape[1], Dx.shape[0], Dy.shape[1], Dy.shape[0], M.shape[1]))

    get_Dx = mod.get_function("get_Dx")
    get_Dx(Dx_gpu, M_gpu, block=(Dx.shape[1], Dx.shape[0], 1), grid=(1, 1), shared=0)
    cuda.memcpy_dtoh(Dx, Dx_gpu)

    get_Dy = mod.get_function("get_Dy")
    get_Dy(Dy_gpu, M_gpu, block=(Dy.shape[1], Dy.shape[0], 1), grid=(1, 1), shared=0)
    cuda.memcpy_dtoh(Dy, Dy_gpu)

    return Dx, Dy

