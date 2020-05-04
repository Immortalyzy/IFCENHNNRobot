
import numpy as np
import matplotlib.pyplot as plt

# 10e5 - 10e6
target = [[0], [0]]
origin = [[99], [99]]
x_size = 100
y_size = 100
Tv = np.asarray([1, 0.60, 0.20, 1])


def buildEnv():
    env = np.zeros((x_size, y_size), dtype=np.int)
    for i in range(10):
        for j in range(50):
            x = i + 10
            y = j + 10
            env[x, y] = 2

    for i in range(50):
        for j in range(10):
            x = i + 30
            y = j + 10
            env[x, y] = 3

    for i in range(10):
        for j in range(40):
            x = i + 30
            y = j + 20
            env[x, y] = 3

    for i in range(30):
        for j in range(10):
            x = i + 50
            y = j + 30
            env[x, y] = 2
    return env


def generateNeuronalSpace(Env, target):
    space = -1.0 * (Env >= 3)
    T = np.ones((x_size, y_size, 8), dtype=np.float)
    A = np.abs(np.roll(Env, 1, axis=0) - Env)
    T[:, :, 0] = Tv[A]
    T[:, :, 0] = Tv[np.abs(np.roll(Env, 1, axis=0) - Env)]
    T[:, :, 1] = Tv[np.abs(np.roll(Env, -1, axis=0) - Env)]
    T[:, :, 2] = Tv[np.abs(np.roll(Env, 1, axis=1) - Env)]
    T[:, :, 3] = Tv[np.abs(np.roll(Env, -1, axis=1) - Env)]
    T[:, :, 4] = Tv[np.abs(np.roll(np.roll(Env, 1, axis=0), 1, axis=1) - Env)]
    T[:, :, 5] = Tv[np.abs(np.roll(np.roll(Env, -1, axis=0), 1, axis=1) - Env)]
    T[:, :, 6] = Tv[np.abs(np.roll(np.roll(Env, 1, axis=0), -1, axis=1) - Env)]
    T[:, :, 7] = Tv[np.abs(
        np.roll(np.roll(Env, -1, axis=0), -1, axis=1) - Env)]
    space[target] = 1.0
    return space, T


def findPath(A, T):
    A_ = np.zeros((x_size + 2, y_size + 2))
    A_b = np.zeros((x_size + 2, y_size + 2))
    T_ = np.zeros((x_size + 2, y_size + 2, 8))
    for nx in range(x_size):
        for na1 in range(y_size):
            A_[nx + 1][na1 + 1] = A[nx][na1]
            A_b[nx + 1][na1 + 1] = 1
            T_[nx + 1, na1 + 1, :] = T[nx, na1, :]
    A_p = A_ >= 0
    # Remove obstacles
    A_s = A_ * A_p

    iteration = 0
    dA1 = 10
    dA2 = 50 
    # while A_s[origin[0][0]+1, origin[1][0]+1] < 0.001:
    # while A_s[origin[0][0]+1, origin[1][0]+1] < 1e-14:
    while np.abs(dA2 - dA1) > 1e-15:
        A_ss = np.copy(A_s)
        A_ss += T_[:, :, 0] * np.roll(A_s, 1, axis=0)
        A_ss += T_[:, :, 1] * np.roll(A_s, -1, axis=0)
        A_ss += T_[:, :, 2] * np.roll(A_s, 1, axis=1)
        A_ss += T_[:, :, 3] * np.roll(A_s, -1, axis=1)
        A_ss += T_[:, :, 4] * np.roll(np.roll(A_s, 1, axis=0), 1, axis=1)
        A_ss += T_[:, :, 5] * np.roll(np.roll(A_s, -1, axis=0), 1, axis=1)
        A_ss += T_[:, :, 6] * np.roll(np.roll(A_s, 1, axis=0), -1, axis=1)
        A_ss += T_[:, :, 7] * np.roll(np.roll(A_s, -1, axis=0), -1, axis=1)
        A_ss *= A_b
        A_ss *= A_p
        A_ss = A_ss/9
        dA1 = dA2
        dA2 = np.sum(np.abs(A_ss - A_s))
        A_s = np.copy(A_ss)
#        A_s = 1 / (1 + np.exp(-A_ss))
        A_s[target[0][0] + 1, target[1][0] + 1] = 1.0
        iteration += 1
#        """
#        space_ = np.zeros([x_size, y_size])
#        for nx in range(x_size):
#            for na1 in range(y_size):
#                space_[nx, na1] = A_s[nx + 1][na1 + 1]
#
#        drawNeuronalSpace(space_)
#        """
    print("Path found with " + str(iteration) + " iteration")

    return A_s


def draw(env, position):
    img = np.ones((x_size, y_size, 3))
    for x in range(x_size):
        for y in range(y_size):
            if env[x, y] >= 3:
                img[y, x, 0] = 1
                img[y, x, 1] = 0
                img[y, x, 2] = 0
            if env[x, y] == 2:
                img[y, x, 0] = 1
                img[y, x, 1] = 0
                img[y, x, 2] = 1
            if env[x, y] == 1:
                img[y, x, 0] = 0
                img[y, x, 1] = 1
                img[y, x, 2] = 0
            img[position[1], position[0], 0] = 0
            img[position[1], position[0], 1] = 0
            img[position[1], position[0], 2] = 1

    for x in range(5):
        for y in range(5):
            img[target[1][0] + x, target[0][0] + y, 2] = 0
    imgplot = plt.imshow(img)
    plt.show()


def drawS(env, positions):
    img = np.ones((x_size, y_size, 3))
    for x in range(x_size):
        for y in range(y_size):
            if env[x, y] >= 3:
                img[y, x, 0] = 1
                img[y, x, 1] = 0
                img[y, x, 2] = 0
            if env[x, y] == 2:
                img[y, x, 0] = 1
                img[y, x, 1] = 0
                img[y, x, 2] = 1
            if env[x, y] == 1:
                img[y, x, 0] = 0
                img[y, x, 1] = 1
                img[y, x, 2] = 0

    for position in positions:
        img[position[1], position[0], 0] = 0
        img[position[1], position[0], 1] = 0
        img[position[1], position[0], 2] = 1

    for x in range(5):
        for y in range(5):
            img[target[1][0] + x, target[0][0] + y, 2] = 0
    imgplot = plt.imshow(img)
    plt.show()


def animate(env, space):
    result = []
    # current position
    cu_po = [origin[0][0], origin[1][0]]
    # find next position
    v_max = np.amax(space)
    while space[cu_po[0], cu_po[1]] < v_max:
        sub_matrix = space[cu_po[0]-1:cu_po[0]+2, cu_po[1]-1:cu_po[1]+2]
        ne_po = np.where(sub_matrix >= np.amax(sub_matrix))
        ne_po = [ne_po[0][0] + cu_po[0]-1, ne_po[1][0] + cu_po[1]-1]
        cu_po = ne_po
        result.append(cu_po)

    drawS(env, result)


env = buildEnv()
draw(env, [99, 99])
space, T = generateNeuronalSpace(env, target)
res = findPath(space, T)
animate(env, res)
