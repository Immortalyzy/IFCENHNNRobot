import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math

start_time = time.time()

envx = 192
envy = 108
target = [155, 0]
zer = np.zeros((envx, envy), dtype=bool)
one = np.ones((envx, envy), dtype=bool)
x_size = 192
a1_size = 18
a2_size = 18
dx = 1. * envx / x_size
da1 = np.pi * 2 / a1_size
da2 = np.pi * 2 / a2_size
nx0 = 0
na10 = 9
na20 = 9

robot = {}
robot['armlength'] = int(envy / 4)
robot['armwidth'] = int(envy / 40)
l = int(robot['armlength'])
w = int(robot['armwidth'])

space = np.zeros([x_size, a1_size, a2_size], dtype=float)
space2 = space
env = zer
obstacle_aera = []
feasible_aera = []
target_aera = []


def Presence(x, a1, a2):
    """ Calculate the presence of the robot at configuration config, 
        return an matrix of the same size of environment"""
    # calculate second joint position
    pres = np.zeros((envx, envy), dtype=bool)
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
                        continue

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
    for x in range(5):
        for y in range(5):
            img[target[1] + x, target[0] + y, 2] = 1
    imgplot = plt.imshow(img)
    plt.show()


def Feasable(x, a1, a2):
    overlap = np.multiply(Presence(x, a1, a2), env)
    return (overlap.sum() <= 2)


def generateNeuronalSpace():
    for nx in range(x_size):
        print(str(nx))
        for na1 in range(a1_size):
            for na2 in range(a2_size):
                x = nx * dx
                a1 = float(- np.pi + na1 * da1)
                a2 = float(- np.pi + na2 * da2)
                if Feasable(x, a1, a2):
                    feasible_aera.append([nx, na1, na2])
                else:
                    obstacle_aera.append([nx, na1, na2])


def findTargetArea():
    for config in feasible_aera:
        x = config[0] * dx
        a1 = float(- np.pi + config[1] * da1)
        a2 = float(- np.pi + config[2] * da2)
        if abs(x - target[0]) > 2 * l:
            continue
        else:
            if Presence(x, a1, a2)[target[0], target[1]] == 1:
                target_aera.append(config)
                space[config[0], config[1], config[2]] = 1


def checkReach():
    if space[nx0, na10, na20] > 0:
        return True
    else:
        return False


def findPath():
    space2 = space
    iteration = 0
    while not checkReach():
        iteration += 1
        for config in feasible_aera:
            nx = config[0]
            na1 = config[1]
            na2 = config[2]
            delta = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        delta += space[min(nx - 1 + i, 0),
                                       min(na1 - 1 + j, 0), min(na2 - 1 + k, 0)]
            space2[nx, na1, na2] = np.tanh(delta)

            delta = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        delta += space2[min(nx - 1 + i, 0),
                                        min(na1 - 1 + j, 0), min(na2 - 1 + k, 0)]
            space[nx, na1, na2] = np.tanh(delta)

    print("Path found")


# Build test environment (with obstacles)
def readEnvironment():
    # read obstacles
    envp = mpimg.imread("./Environment.png")
    for x in range(envx):
        for y in range(envy):
            if envp[y, x, 1] < 0.99:
                env[x, y] = 0
            else:
                env[x, y] = 1
    # read target


#a1 = float(np.pi/3)
#a2 = float(np.pi/5)
#x = int(envx / 2)

readEnvironment()
#drawSpace(Presence(x, a1, a2))
preparation_time = time.time() - start_time
print("Preparation took " + str(preparation_time) + " s.")
generateNeuronalSpace()
findTargetArea()
generation_time = time.time() - start_time - preparation_time
print("Generation of neuronal space took " + str(generation_time) + " s.")
findPath()
finding_time = time.time() - start_time - generation_time - preparation_time
print("Path finding took " + str(generation_time) + " s.")
print("Done")
