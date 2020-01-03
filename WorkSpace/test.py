import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math

start_time = time.time()

envx = 192
envy = 108
zer = np.zeros((envx, envy), dtype=bool)
one = np.ones((envx, envy), dtype=bool)
x_size = 192
a1_size = 18
a2_size = 18

robot = {}
robot['armlength'] = int(envy / 4)
robot['armwidth'] = int(envy / 40)

space = np.zeros([x_size, a1_size, a2_size], dtype=float)
env = zer


def Presence(x, a1, a2):
    """ Calculate the presence of the robot at configuration config, 
        return an matrix of the same size of environment"""
    # calculate second joint position
    pres = np.zeros((envx, envy), dtype=bool)
    l = int(robot['armlength'])
    w = int(robot['armwidth'])
    x1 = x
    y1 = int(envy/2)
    x2 = int(x1 + l * np.cos(a1))
    y2 = int(y1 + l * np.sin(a1))
    for x_ in range(l):
        for y_ in range(l):
            # check if in arm1
            x = int(x1 + x_ * (- 1 + 2 * int(np.abs(a1) <= np.pi/2)))
            y = int(y1 + y_ * (- 1 + 2 * int(a1 >= 0)))
            if x >= 0 and y >= 0 and x < envx:
                d = math.sqrt((x - x1) ** 2 + (y-y1) ** 2)
                if d < l:
                    A = np.tan(a1)
                    dl = np.abs(A * x - y + y1 - x1 * A) / \
                        np.sqrt(A ** 2 + 1)
                    if dl < w:
                        pres[x, y] = 1
            # check if in arm2
            x = int(x2 + x_ * (- 1 + 2 * int(np.abs(a2) <= np.pi/2)))
            y = int(y2 + y_ * (- 1 + 2 * int(a2 >= 0)))
            if x >= 0 and y >= 0 and x < envx:
                d = np.sqrt((x - x2) ** 2 + (y-y2) ** 2)
                if d < l:
                    A = np.tan(a2)
                    dl = np.abs(A * x - y + y2 - x2 * A) / np.sqrt(A ** 2 + 1)
                    if dl < w:
                        pres[x, y] = 1
                        continue

    return pres


def drawSpace(pres=zer):
    img = np.zeros((envy, envx, 3))
    for x in range(envx):
        for y in range(envy):
            if env[x, y] == 1:
                img[y, x, 0] = 1
                img[y, x, 1] = 1
                img[y, x, 2] = 1
            if pres[x, y] == 1:
                img[y, x, 0] = 1
                img[y, x, 1] = 0
                img[y, x, 2] = 0
    imgplot = plt.imshow(img)
    plt.show()


def Feasable(x, a1, a2):
    overlap = Presence(x, a1, a2) * env
    return (overlap.sum() > 0)


def generateNeuronalSpace():
    dx = 1. * envx / x_size
    da1 = np.pi * 2 / a1_size
    da2 = np.pi * 2 / a2_size
    for nx in range(x_size):
        print(str(nx))
        for na1 in range(a1_size):
            for na2 in range(a2_size):
                x = nx * dx
                a1 = float(- np.pi + na1 * da1)
                a2 = float(- np.pi + na2 * da2)
                space[nx, na1, na2] = Feasable(x, a1, a2)


def findPath():
    print("Path found")


# Build test environment (with obstacles)
envp = mpimg.imread("e:/Projects/Robot2019/WorkSpace/Environment.png")
env = np.zeros((envx, envy), dtype=int)
for x in range(envx):
    for y in range(envy):
        if envp[y, x, 1] < 0.99:
            env[x, y] = 0
        else:
            env[x, y] = 1

a1 = float(np.pi/3)
a2 = float(np.pi/5)
x = int(envx / 2)

drawSpace(Presence(x, a1, a2))
preparation_time = time.time() - start_time
print("Preparation took " + str(preparation_time) + " s.")
generateNeuronalSpace()
generation_time = time.time() - preparation_time
print("Generation of neuronal space took " + str(generation_time) + " s.")
findPath()
finding_time = time.time() - generation_time - preparation_time
print("Path finding took " + str(generation_time) + " s.")
print("Done")
