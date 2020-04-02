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
dimension = 3
blocks = np.power(3, dimension)
x_size = int(192 / 2)
y_size = int(108 / 2)
a1_size = 72

# Define destination coordinates and initial coordinate
target = np.array([155, 20], dtype=int)
nx0 = 0
ny0 = int(y_size / 2)
na10 = int(a1_size / 2)

# Calculate step size and angle array
dx = 1. * envx / x_size
dy = 1. * envy / y_size
da1 = np.pi * 2 / a1_size


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


# Define robot
robot = {}
robot['armlength'] = int(envy / 4)
robot['armwidth'] = int(envy / 40)
w = int(robot['armwidth'])
# Calculate the local coordinate space for robot arm
l = int(robot['armlength'])


@jit(nopython=True)
def Presence(x__, y__, a1):
    """ Calculate the presence of the robot at configuration config,
        return an matrix of the same size of environment"""
    pres = np.zeros((envx, envy), dtype=np.bool_)
    x1 = x__
    y1 = y__
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
    return pres


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True)
def rPresence(nx_, ny_, na1_, presences_):
    # return np.roll(presences[na1_, :, :], nx_, axis=0)
    # return np.roll(presences_[na1, :, :], nx)
    result = np.zeros((envx, envy), dtype=np.bool_)
    Dx = envx2 - NxToX(nx_)
    aDx = np.abs(Dx)
    Dy = envy2 - NyToY(ny_)
    aDy = np.abs(Dy)
    if Dx >= 0 and Dy >= 0:
        result[0:envx-aDx, 0:envy-aDy] = presences[na1_, aDx:envx, aDy:envy]
    elif Dx >= 0 and Dy <= 0:
        result[0:envx-aDx, aDy:envy] = presences[na1_, aDx:envx, 0:envy-aDy]
    elif Dx <= 0 and Dy <= 0:
        result[aDx:envx, aDy:envy] = presences[na1_, 0:envx-aDx, 0:envy-aDy]
    else:
        result[aDx:envx, 0:envy-aDy] = presences[na1_, 0:envx-aDx, aDy:envy]
    return result


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True)
def Feasable(nx, ny, na1, presences_, env_):
    overlap = rPresence(nx, ny, na1, presences_) * env_
    result = overlap.sum() <= 2
    # if not result: (draw space if overlap)
    #    drawSpace(overlap)
    return result


@jit(nopython=True)
def generateNeuronalSpace(presences):
    feasible_area = np.zeros((x_size * y_size * a1_size, 3), dtype=np.int16)
    feasible_count = 0
    space = np.zeros((x_size, y_size, a1_size))
    for nx in range(x_size):
        for na1 in range(a1_size):
            for ny in range(y_size):
                if not Feasable(nx, ny, na1, presences, env):
                    space[nx, ny, na1] = -1
                else:
                    feasible_area[feasible_count, 0] = nx
                    feasible_area[feasible_count, 1] = ny
                    feasible_area[feasible_count, 2] = na1
                    feasible_count += 1
    return feasible_area, feasible_count, space


@jit(nopython=True)
def findTargetArea(feasible_area, feasible_count, presences, space):
    target_area = np.zeros((x_size * y_size * a1_size, 3), dtype=np.int16)
    target_count = 0

    for i in range(feasible_count):
        if rPresence(feasible_area[i, 0], feasible_area[i, 1], feasible_area[i, 2], presences)[target[0], target[1]] == 1:
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
                As[i+1, j+1, k+1] = np.sum(A[i:i+3, j:j+3, k:k+3])

    return As

    # @jit(nopython=True)


@jit(nopython=True)
def findPath(A, ta, tc):
    # generate A with surrondings
    A_ = np.zeros((x_size + 2, y_size + 2, a1_size + 2), dtype=np.float64)
    A_b = np.zeros((x_size + 2, y_size + 2, a1_size + 2), dtype=np.float64)
    A_[1:x_size+1, 1:y_size+1, 1:a1_size+1] = A
    A_b[1:x_size+1, 1:y_size+1, 1:a1_size +
        1] = np.ones((x_size, y_size, a1_size), dtype=np.float64)
    A_p = A_ >= 0
    # Remove obstacles
    A_s = A_ * A_p

    iteration = 0
    while A_s[nx0+1, ny0+1, na10+1] < 0.0000000000000001:
        A_ss = spread(A_s, dimension)
        A_ss *= A_b
        A_ss *= A_p
        A_s = np.tanh(A_ss / blocks)
        for i in range(tc):
            A_s[ta[i, 0], ta[i, 1], ta[i, 2]] = 1.0
        iteration += 1
        # drawNeuronalSpace(A_s)

    A_r = np.zeros((x_size, y_size, a1_size))
    for nx in range(x_size):
        for ny in range(y_size):
            for na1 in range(a1_size):
                A_r[nx][ny][na1] = A_s[nx + 1][ny+1][na1 + 1]
    return A_s


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


def storePresences():
    x = envx2
    y = envy2
    presences = np.zeros((a1_size, envx, envy))
    for na in range(a1_size):
        presences[na, :, :] = Presence(x, y, NaToA(na))
    return presences


def animate(space, presences_):
    result = []
    # current position
    cu_po = [nx0 + 1, ny0 + 1, na10 + 1]
    # find next position
    v_max = np.amax(space)
    while space[cu_po[0], cu_po[1], cu_po[2]] < v_max:
        sub_matrix = space[cu_po[0]-1:cu_po[0]+2,
                           cu_po[1]-1:cu_po[1]+2, cu_po[2]-1:cu_po[2]+2]
        ne_po = np.where(sub_matrix >= np.amax(sub_matrix))
        ne_po = [ne_po[0][0] + cu_po[0]-1, ne_po[1]
                 [0] + cu_po[1] - 1, ne_po[2][0]+cu_po[2]-1]
        if cu_po == ne_po:
            break
        cu_po = ne_po
        result.append(cu_po)

    for po in result:
        drawSpace(rPresence(po[0] - 1, po[1] - 1, po[2] - 1,  presences_))

    return


readEnvironment()
presences = storePresences()
# drawSpace(rPresence(nx0 + 20, ny0+10, 0, presences))
preparation_time = time.time() - start_time
print("Preparation took " + str(preparation_time) + " s.")
fa, fc, space = generateNeuronalSpace(presences)
space, ta, tc = findTargetArea(fa, fc, presences, space)
# drawNeuronalSpace()
generation_time = time.time() - start_time - preparation_time
print("Generation of neuronal space took " + str(generation_time) + " s.")
s_r = findPath(space, ta, tc)
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
