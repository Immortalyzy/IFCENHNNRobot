
import numpy as np
import matplotlib.pyplot as plt

# 10e5 - 10e6
target = [[0], [0]]
origin = [[99], [99]]
x_size = 100
y_size = 100


def buildEnv():
    env = np.zeros((x_size, y_size), dtype=np.int)
    for i in range(10):
        for j in range(50):
            x = i + 10
            y = j + 10
            env[x, y] = 1

    for i in range(50):
        for j in range(10):
            x = i + 30
            y = j + 10
            env[x, y] = 3

    for i in range(10):
        for j in range(30):
            x = i + 30
            y = j + 30
            env[x, y] = 1

    for i in range(30):
        for j in range(10):
            x = i + 40
            y = j + 30
            env[x, y] = 2
    return env


def generateNeuronalSpace(Env, target):
    space = -1.0 * (Env >= 3)
    T
    space[target] = 1.0
    return space


def findPath(A):
    A_ = np.zeros((x_size + 2, y_size + 2))
    A_b = np.zeros((x_size + 2, y_size + 2))
    for nx in range(x_size):
        for na1 in range(y_size):
            A_[nx + 1][na1 + 1] = A[nx][na1]
            A_b[nx + 1][na1 + 1] = 1
    A_p = A_ >= 0
    # Remove obstacles
    A_s = A_ * A_p

    iteration = 0
    while A_s[origin[0][0]+1, origin[1][0]+1] < 0.01:
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
        A_s = np.tanh(A_ss)
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
    A_r = np.zeros((x_size, y_size))
    for nx in range(x_size):
        for na1 in range(y_size):
            A[nx][na1] = A_s[nx + 1][na1 + 1]
    return A_r


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
                img[y, x, 1] = 1
                img[y, x, 2] = 0
            if env[x, y] == 1:
                img[y, x, 0] = 0
                img[y, x, 1] = 1
                img[y, x, 2] = 0
    for x in range(5):
        for y in range(5):
            img[target[1] + x, target[0] + y, 2] = 0
    imgplot = plt.imshow(img)
    plt.show()


env = buildEnv()
space = generateNeuronalSpace(env, target)
findPath(space)
