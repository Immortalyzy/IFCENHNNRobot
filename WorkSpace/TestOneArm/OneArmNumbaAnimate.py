import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math
import numba
from numba import jit

start_time = time.time()
# Declare space size
envx = 192
envy = 108
env = np.zeros((envx, envy))

# Declare neuraonal space size and generate corresponding coordinate space
x_size = 192
a1_size = 180

# Calculate step size and angle array
dx = 1. * envx / x_size
da1 = np.pi * 2 / a1_size


def NxToX(nx):
    return nx * dx


def XToNx(x):
    return (int)(x / dx)


def NaToA(na):
    return -np.pi / 2 + na * da1


def AToNa(a):
    return (int)((a + np.pi / 2) / da1)


# Define destination coordinates and initial coordinate
target = np.array([155, 60], dtype=int)
nx0 = 0
na10 = 9

# Define robot
robot = {}
robot['armlength'] = int(envy / 4)
robot['armwidth'] = int(envy / 40)
w = int(robot['armwidth'])
# Calculate the local coordinate space for robot arm
l = int(robot['armlength'])


@jit(nopython=True)
def Presence(x, a1):
    """ Calculate the presence of the robot at configuration config,
        return an matrix of the same size of environment"""
    # calculate second joint position
    pres = np.zeros((envx, envy), dtype=np.bool_)
    x1 = x
    y1 = int(envy/2)
    for x_ in range(l):
        for y_ in range(l):
            # check if in arm1
            x = int(x1 + x_ * (- 1 + 2 * int(np.abs(a1) <= np.pi/2)))
            y = int(y1 + y_ * (- 1 + 2 * int(a1 >= 0)))
            if x >= 0 and y >= 0 and x < envx:
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
def rPresence(nx_, na1_, presences_):
    # return np.roll(presences[na1_, :, :], nx_, axis=0)
    # return np.roll(presences_[na1, :, :], nx)
    result = np.zeros((envx, envy), dtype=np.bool_)
    for x in range(envx - nx_):
        for y in range(envy):
            result[x + nx_, y] = presences_[na1_, x, y]
    return result


# @jit(nopython=True, cache=True, parallel=True)
@jit(nopython=True)
def Feasable(nx, na1, presences_, env_):
    overlap = rPresence(nx, na1, presences_) * env_
    result = overlap.sum() <= 2
    # if not result: (draw space if overlap)
    #    drawSpace(overlap)
    return result


@jit(nopython=True)
def generateNeuronalSpace(presences):
    feasible_area = np.zeros((x_size * a1_size, 2), dtype=np.int16)
    feasible_count = 0
    space = np.zeros((x_size, a1_size))
    for nx in range(x_size):
        for na1 in range(a1_size):
            if not Feasable(nx, na1, presences, env):
                space[nx, na1] = -1
            else:
                feasible_area[feasible_count, 0] = nx
                feasible_area[feasible_count, 1] = na1
                feasible_count += 1
    return feasible_area, feasible_count, space


@jit(nopython=True)
def findTargetArea(feasible_area, feasible_count, presences, space):
    target_area = np.zeros((x_size * a1_size, 2), dtype=np.int16)
    target_count = 0

    for i in range(feasible_count):
        if rPresence(feasible_area[i, 0], feasible_area[i, 1], presences)[target[0], target[1]] == 1:
            space[feasible_area[i, 0], feasible_area[i, 1]] = 1
            target_area[target_count, 0] = feasible_area[i, 0] + 1
            target_area[target_count, 1] = feasible_area[i, 1] + 1
            target_count += 1
    return space, target_area, target_count


# @jit(nopython=True)
def findPath(A, ta, tc):
    # generate A with surrondings
    A_ = np.zeros((x_size + 2, a1_size + 2), dtype=np.float64)
    A_b = np.zeros((x_size + 2, a1_size + 2), dtype=np.float64)
    for nx in range(x_size):
        for na1 in range(a1_size):
            A_[nx + 1][na1 + 1] = A[nx][na1]
            A_b[nx + 1][na1 + 1] = 1
    A_p = A_ >= 0
    # Remove obstacles
    A_s = A_ * A_p

    iteration = 0
    while A_s[nx0+1, na10+1] < 0.0000000000000001:
        A_ss = np.copy(A_s)
        A_ss += np.roll(A_s, 1, axis=0)
        A_ss += np.roll(A_s, -1, axis=0)
        A_ss += np.roll(A_s, 1, axis=1)
        A_ss += np.roll(A_s, -1, axis=1)
        A_ss += np.roll(np.roll(A_s, 1, axis=0), 1, axis=1)
        A_ss += np.roll(np.roll(A_s, -1, axis=0), 1, axis=1)
        A_ss += np.roll(np.roll(A_s, 1, axis=0), -1, axis=1)
        A_ss += np.roll(np.roll(A_s, -1, axis=0), -1, axis=1)
        A_ss *= A_b
        A_ss *= A_p
        A_s = np.tanh(A_ss / 9)
        for i in range(tc):
            A_s[ta[i, 0], ta[i, 1]] = 1.0
        iteration += 1
#        """
#        space_ = np.zeros([x_size, a1_size])
#        for nx in range(x_size):
#            for na1 in range(a1_size):
#                space_[nx, na1] = A_s[nx + 1][na1 + 1]
#
#        drawNeuronalSpace(space_)
#        """

    print("Path found with " + str(iteration) + " iteration")
    A_r = np.zeros((x_size, a1_size))
    for nx in range(x_size):
        for na1 in range(a1_size):
            A_r[nx][na1] = A_s[nx + 1][na1 + 1]
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
            img[nx, na1, 1] = 1 - space_[nx, na1]
            img[nx0, na10, 2] = 1
            if space_[nx, na1] < 0:
                img[nx, na1, 0] = 0
                img[nx, na1, 1] = 0
                img[nx, na1, 2] = 0
    plt.imshow(img)
    plt.show()


def storePresences():
    x = 0
    presences = np.zeros((a1_size, envx, envy))
    for na in range(a1_size):
        presences[na, :, :] = Presence(x, NaToA(na))
    return presences


def animate(space, presences_):
    result = []
    # current position
    cu_po = [nx0 + 1, na10 + 1]
    # find next position
    v_max = np.amax(space)
    while space[cu_po[0], cu_po[1]] < v_max:
        sub_matrix = space[cu_po[0]-1:cu_po[0]+2, cu_po[1]-1:cu_po[1]+2]
        ne_po = np.where(sub_matrix >= np.amax(sub_matrix))
        ne_po = [ne_po[0][0] + cu_po[0], ne_po[1][0] + cu_po[1]]
        cu_po = ne_po
        result.append(cu_po)

    for po in result:
        drawSpace(rPresence(po[0] - 1, po[1] - 1, presences_))

    return


readEnvironment()
presences = storePresences()
# drawSpace(rPresence(20, 1))
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
