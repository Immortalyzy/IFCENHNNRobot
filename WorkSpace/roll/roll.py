import numpy as np
from numba import jit
import time


A = np.asarray(range(2400000))
A = A.reshape(30, 400, 200)
axis = 1
Ndim = A.ndim
shift = 3

start_time = time.time()
a = 1
for i in range(Ndim - axis):
    j = i + 1
    a *= A.shape[Ndim - j]

b = (int)(a / A.shape[axis])


B = np.empty(A.shape, dtype=np.int16)
for i in range(A.size):
    # Get x_axis
    x_axis = (int)((i % a) / b)
    idx = (i + ((x_axis + shift) % A.shape[axis] - x_axis) * b) % A.size
    B.flat[idx] = A.flat[i]
print("A method took " + str(time.time() - start_time) + " s.")


@jit(nopython=True)
def rol(A, shift, axis):
    return np.roll(A, shift, axis=axis)


start_time = time.time()
S = np.roll(A, shift, axis=axis)
print("B method took " + str(time.time() - start_time) + " s.")
start_time = time.time()
R1 = rol(A, shift, axis)
print("C1 method took " + str(time.time() - start_time) + " s.")
start_time = time.time()
R2 = rol(A, shift, axis)
print("C2 method took " + str(time.time() - start_time) + " s.")
