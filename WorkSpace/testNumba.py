from numba import jit
import numpy as np
import time


@jit(nopython=True)
def go_fast(): # Function is compiled and runs in machine code
    trace = 0
    for i in range(1000000):
        trace += np.tanh(i)
    return trace

def go_fast_withoutNumba(): # Function is compiled and runs in machine code
    trace = 0
    for i in range(1000000):
        trace += np.tanh(i)
    return trace



# WITHOUT NUMBA
start = time.time()
go_fast_withoutNumba()
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast()
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()
go_fast()
end = time.time()
print("Elapsed (after compilation) = %s" % (end - start))


# RESULT ON MAC Pro 2015  (RUN TIME : UNIT SECOND)
# 1.65 - 0.36 - 0.0045