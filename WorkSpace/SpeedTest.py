import math
import numpy as np
import time

example = np.random.randint(1000, size=212576400)

start_time = time.time()
for i in example:
    j = 5 * i
e_time = time.time() - start_time

print("Calculation sqrt took : " + str(e_time) + " s.")


start_time = time.time()
for i in example:
    j = 5 / i
e_time = time.time() - start_time
print("Calculation sqrt took : " + str(e_time) + " s.")

print("Done")
