import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import numba
from numba import jit
from scipy.special import comb, perm
from itertools import combinations, permutations

start_time = time.time()
# Declare space size
envx = 192
envy = 108
env = np.zeros((envx, envy))
envx2 = int(envx/2)
envy2 = int(envy/2)

# Declare neuraonal space size
dimension = 4
blocks = np.power(3, dimension)
x_size = int(192 / 3)
y_size = int(108 / 2)
a1_size = 36
a2_size = 36
max_step = 1e4

# Define destination coordinates and initial coordinate
target = np.array([155, 50], dtype=int)
nx0 = 0
ny0 = int(y_size / 2)
na10 = int(a1_size / 2)
na20 = int(a1_size / 2)

# Calculate step size and angle array
dx = 1. * envx / x_size
dy = 1. * envy / y_size
da1 = np.pi * 2 / a1_size
da2 = np.pi * 2 / a1_size


@jit(nopython=True)
def NxToX(nx):
    return int(nx * dx)


@jit(nopython=True)
def NyToY(ny):
    return int(ny * dy)


@jit(nopython=True)
def XToNx(x):
    return (int)(x / dx)


@jit(nopython=True)
def NaToA(na):
    return -np.pi + na * da1


@jit(nopython=True)
def AToNa(a):
    return (int)((a + np.pi / 2) / da1)


l = int(envy / 4)
w = int(envy / 40)


@jit(nopython=True)
def Presence(x__, y__, a1, a2=0):
    """ Calculate the presence of the robot at configuration config,
        return an matrix of the same size of environment"""
    pres = np.zeros((envx, envy), dtype=np.bool_)
    x1 = x__
    y1 = y__
    x2 = x1 + l * np.cos(a1)
    y2 = y1 + l * np.sin(a1)
    for x_ in range(l):
        for y_ in range(l):
            # check if in arm1
            x = int(x1 + x_ * (- 1 + 2 * int(np.abs(a1) <= np.pi/2)))
            y = int(y1 + y_ * (- 1 + 2 * int(a1 >= 0)))
            if x >= 0 and y >= 0:
                d = np.sqrt((x - x1) ** 2 + (y-y1) ** 2)
                if d < l:
                    A = np.tan(a1)
                    dl = np.abs(A * x - y + y1 - x1 * A) / \
                        np.sqrt(A ** 2 + 1)
                    if dl < w:
                        pres[x, y] = 1
                        continue
            # check if in arm2
            x = int(x2 + x_ * (- 1 + 2 * int(np.abs(a2) <= np.pi/2)))
            y = int(y2 + y_ * (- 1 + 2 * int(a2 >= 0)))
            if x >= 0 and y >= 0:
                d = np.sqrt((x - x2) ** 2 + (y-y2) ** 2)
                if d < l:
                    A = np.tan(a2)
                    dl = np.abs(A * x - y + y2 - x2 * A) / \
                        np.sqrt(A ** 2 + 1)
                    if dl < w:
                        pres[x, y] = 1
                        continue
    return pres


def storePresences():
    x = envx2
    y = envy2
    presences = np.zeros((a1_size, a2_size, envx, envy))
    for na1 in range(a1_size):
        for na2 in range(a2_size):
            presences[na1, na2, :, :] = Presence(x, y, NaToA(na1), NaToA(na2))
    return presences

# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True)
def rPresence(nx_, ny_, na1_, na2_, presences_):
    # return np.roll(presences[na1_, :, :], nx_, axis=0)
    # return np.roll(presences_[na1, :, :], nx)
    result = np.zeros((envx, envy), dtype=np.bool_)
    Dx = envx2 - NxToX(nx_)
    aDx = np.abs(Dx)
    Dy = envy2 - NyToY(ny_)
    aDy = np.abs(Dy)
    if Dx >= 0 and Dy >= 0:
        result[0:envx-aDx, 0:envy-aDy] = presences[na1_,
                                                   na2_, aDx:envx, aDy:envy]
    elif Dx >= 0 and Dy <= 0:
        result[0:envx-aDx, aDy:envy] = presences[na1_,
                                                 na2_, aDx:envx, 0:envy-aDy]
    elif Dx <= 0 and Dy <= 0:
        result[aDx:envx, aDy:envy] = presences[na1_,
                                               na2_, 0:envx-aDx, 0:envy-aDy]
    else:
        result[aDx:envx, 0:envy-aDy] = presences[na1_,
                                                 na2_, 0:envx-aDx, aDy:envy]
    return result


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True)
def Feasable(nx, ny, na1, na2, presences_, env_):
    overlap = rPresence(nx, ny, na1, na2, presences_) * env_
    result = overlap.sum() <= 2
    return result


@jit(nopython=True)
def generateNeuronalSpace(presences):
    feasible_area = np.zeros(
        (x_size * y_size * a1_size * a2_size, 4), dtype=np.int16)
    feasible_count = 0
    space = np.zeros((x_size, y_size, a1_size, a2_size))
    for nx in range(x_size):
        for ny in range(y_size):
            for na1 in range(a1_size):
                for na2 in range(a2_size):
                    if not Feasable(nx, ny, na1, na2, presences, env):
                        space[nx, ny, na1, na2] = -1
                    else:
                        feasible_area[feasible_count, 0] = nx
                        feasible_area[feasible_count, 1] = ny
                        feasible_area[feasible_count, 2] = na1
                        feasible_area[feasible_count, 3] = na2
                        feasible_count += 1
    return feasible_area, feasible_count, space


@jit(nopython=True)
def findTargetArea(feasible_area, feasible_count, presences, space):
    target_area = np.zeros(
        (x_size * y_size * a1_size * a2_size, 4), dtype=np.int16)
    target_count = 0

    for i in range(feasible_count):
        if rPresence(feasible_area[i, 0], feasible_area[i, 1], feasible_area[i, 2], feasible_area[i, 3], presences)[target[0], target[1]] == 1:
            space[feasible_area[i, 0], feasible_area[i, 1],
                  feasible_area[i, 2]] = 1
            target_area[target_count, 0] = feasible_area[i, 0] + 1
            target_area[target_count, 1] = feasible_area[i, 1] + 1
            target_area[target_count, 2] = feasible_area[i, 2] + 1
            target_count += 1
    return space, target_area, target_count


@jit(nopython=True)
def spread(A, dimension):
    As = np.copy(A)
    for i in range(x_size):
        for j in range(y_size):
            for k in range(a1_size):
                for l in range(a2_size):
                    As[i+1, j+1, k+1, l +
                        1] = np.sum(A[i:i+3, j:j+3, k:k+3, l:l+3])

    return As

    # @jit(nopython=True)


@jit(nopython=True)
def findPath(A, ta, tc):
    # generate A with surrondings
    A_ = np.zeros((x_size + 2, y_size + 2, a1_size +
                   2, a2_size + 2), dtype=np.float64)
    A_b = np.zeros((x_size + 2, y_size + 2, a1_size +
                    2, a2_size + 2), dtype=np.float64)
    A_[1:x_size+1, 1:y_size+1, 1:a1_size+1, 1:a2_size+1] = A
    A_b[1:x_size+1, 1:y_size+1, 1:a1_size +
        1, 1:a2_size+1] = np.ones((x_size, y_size, a1_size, a2_size), dtype=np.float64)
    A_p = A_ >= 0
    # Remove obstacles
    A_s = A_ * A_p

    iteration = 0
    while A_s[nx0+1, ny0+1, na10+1, na20+1] < 0.0000000000000001:
        A_ss = spread(A_s, dimension)
        A_ss *= A_b
        A_ss *= A_p
        A_s = np.tanh(A_ss / blocks)
        A_s[ta[0, 0], ta[0, 1], ta[0, 2], ta[0, 3]] = 1.0
        iteration += 1
        if(iteration > max_step):
            break
        # drawNeuronalSpace(A_s)

    return A_s, iteration


def drawSpace(pres):
    img = np.ones((envy, envx, 3))
    for x in range(envx):
        for y in range(envy):
            if env[x, y] == 1:
                img[y, x, 0] = 0
                img[y, x, 1] = 0
                img[y, x, 2] = 0
            if pres[x, y] == 1:
                img[y, x, 0] = 1
                img[y, x, 1] = 0
                img[y, x, 2] = 0
    for x in range(5):
        for y in range(5):
            img[target[1] + x, target[0] + y, 2] = 0
    imgplot = plt.imshow(img)
    plt.show()

# Build test environment (with obstacles)


def readEnvironment():
    # read obstacles
    global env
    envp = mpimg.imread("./Environment.png")
    for x in range(envx):
        for y in range(envy):
            if envp[y, x, 1] < 0.99:
                env[x, y] = 1
            else:
                env[x, y] = 0
    # read target


def drawNeuronalSpace(space_):
    img = np.ones((x_size, a1_size, 3))
    for nx in range(x_size):
        for na1 in range(a1_size):
            img[nx, na1, 1] = 1 - space_[nx, ny0, na1]
            img[nx0, na10, 2] = 1
            if space_[nx, ny0, na1] < 0:
                img[nx, na1, 0] = 0
                img[nx, na1, 1] = 0
                img[nx, na1, 2] = 0
    plt.imshow(img)
    plt.show()


def animate(space, presences_):
    result = []
    # current position
    cu_po = [nx0 + 1, ny0 + 1, na10 + 1, na20 + 1]
    # find next position
    v_max = np.amax(space)
    while space[cu_po[0], cu_po[1], cu_po[2], cu_po[3]] < v_max:
        sub_matrix = space[cu_po[0]-1:cu_po[0]+2,
                           cu_po[1]-1:cu_po[1]+2,
                           cu_po[2]-1:cu_po[2]+2,
                           cu_po[3]-1:cu_po[3]+2]
        ne_po = np.where(sub_matrix >= np.amax(sub_matrix))
        ne_po = [ne_po[0][0] + cu_po[0] - 1,
                 ne_po[1][0] + cu_po[1] - 1,
                 ne_po[2][0] + cu_po[2] - 1,
                 ne_po[3][0] + cu_po[3] - 1]
        if cu_po == ne_po:
            break
        cu_po = ne_po
        result.append(cu_po)

    for po in result:
        drawSpace(rPresence(po[0] - 1, po[1] - 1,
                            po[2] - 1, po[3] - 1,  presences_))

    return


readEnvironment()
presences = storePresences()
# for na1 in range(a1_size):
#     for na2 in range(a2_size):
#         drawSpace(rPresence(20, 20, na1, na2, presences))
# drawSpace(rPresence(nx0 + 20, ny0+10, 0, presences))
preparation_time = time.time() - start_time
print("Preparation took " + str(preparation_time) + " s.")
fa, fc, space = generateNeuronalSpace(presences)
space, ta, tc = findTargetArea(fa, fc, presences, space)
# drawNeuronalSpace()
generation_time = time.time() - start_time - preparation_time
print("Generation of neuronal space took " + str(generation_time) + " s.")
s_r, iteration = findPath(space, ta, tc)
if(iteration >= max_step):
    print("No path found in " + str(max_step) + " steps.")
finding_time = time.time() - start_time - generation_time - preparation_time
print("Path finding took " + str(finding_time) + " s.")
animate(s_r, presences)
print("Done")
#
# start_time = time.time()
# readEnvironment()
# presences = storePresences()
# drawSpace(rPresence(20, 1))
# preparation_time = time.time() - start_time
# print("Preparation took " + str(preparation_time) + " s.")
# fa, fc, space = generateNeuronalSpace(presences)
# space = findTargetArea(fa, fc, presences, space)
# drawNeuronalSpace()
# generation_time = time.time() - start_time - preparation_time
# print("Generation of neuronal space took " + str(generation_time) + " s.")
# s_r = findPath(space)
# finding_time = time.time() - start_time - generation_time - preparation_time
# print("Path finding took " + str(finding_time) + " s.")
# animate(s_r)
# print("Done")
#
