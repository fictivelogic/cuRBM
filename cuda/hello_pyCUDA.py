import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import ctypes
import os
import numpy as np

#library_file_path = 'lib/test_createMovieRatings.so'
#test_lib = ctypes.cdll.LoadLibrary(library_file_path)
#returned = test_lib.brah()

mod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
    const int i = threadIdx.x;
    dest[i] = a[i] * b[i];
}""")

multiply_them = mod.get_function("multiply_them")

a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

dest = np.zeros_like(a)
multiply_them(
    drv.Out(dest), drv.In(a), drv.In(b),
    block=(400,1,1), grid=(1,1))

print(dest-a*b)
