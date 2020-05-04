import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math

start_time = time.time()
# Declare space size
envx = 192
envy = 108

# Frequently used arrays
zer = np.zeros((envx, envy), dtype=bool)
one = np.ones((envx, envy), dtype=bool)

# Declare neuraonal space size and generate corresponding coordinate space
x_size = 192
a1_size = 18
basic_space = np.zeros((x_size, a1_size, 2), dtype=int)
for nx in range(x_size):
    for na1 in range(a1_size):
        basic_space[nx, na1, 0] = nx
        basic_space[nx, na1, 1] = na1
presences = []

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


a_array = []
for na1 in range(a1_size):
    a_array.append(- np.pi / 2 + na1 * da1)

# Define destination coordinates and initial coordinate
target = [155, 54]
nx0 = 0
na10 = 9

# Define robot
robot = {}
robot['armlength'] = int(envy / 4)
robot['armwidth'] = int(envy / 40)
w = int(robot['armwidth'])
# Calculate the local coordinate space for robot arm
l = int(robot['armlength'])
robot_space = np.zeros((l, l, 2), dtype=int)
for x_ in range(l):
    for y_ in range(l):
        robot_space[x_, y_, 0] = x_
        robot_space[x_, y_, 1] = y_

# Allocate memory for neuronal space iteration
space = np.zeros([x_size, a1_size], dtype=float)
space2 = space
env = zer

obstacle_area = []
feasible_area = []
target_area = []


def Presence(x, a1):
    """ Calculate the presence of the robot at configuration config, 
        return an matrix of the same size of environment"""
    # calculate second joint position
    pres = np.zeros((envx, envy), dtype=bool)
    x1 = x
    y1 = int(envy/2)
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
    return pres


def rPresence(nx, na1):
    return Presence(NxToX(nx), NaToA(na1))


def Feasable(nx, na1):
    overlap = np.multiply(rPresence(nx, na1), env)
    result = overlap.sum() <= 2
    # if not result: (draw space if overlap)
    #    drawSpace(overlap)
    return result


def generateNeuronalSpace():
    global feasible_area
    global obstacle_area
    for nx in range(x_size):
        # print(str(nx))
        for na1 in range(a1_size):
            if Feasable(nx, na1):
                feasible_area.append(np.array([nx, na1]))
            else:
                obstacle_area.append(np.array([nx, na1]))

    feasible_area = np.array(feasible_area)
    obstacle_area = np.array(obstacle_area)


def findTargetArea():
    for config in feasible_area:
        if rPresence(config[0], config[1])[target[0], target[1]] == 1:
            #drawSpace(rPresence(config[0], config[1]))
            target_area.append(config)
            space[config[0], config[1]] = 1


def checkReach():
    if space[nx0, na10] > 0.00000001:
        return True
    else:
        return False


def findPath():
    global space
    global space2
    space2 = space
    iteration = 0
    while not checkReach():
        iteration += 1
        for config in feasible_area:
            nx = config[0]
            na1 = config[1]
            delta = 0
            for i in range(3):
                for j in range(3):
                    delta += space[min(max(nx - 1 + i, 0), 191),
                                   min(max(na1 - 1 + j, 0), 17)]
            space2[nx, na1] = np.tanh(delta)

            delta = 0
            for i in range(3):
                for j in range(3):
                    delta += space2[min(max(nx - 1 + i, 0), 191),
                                    min(max(na1 - 1 + j, 0), 17)]
            space[nx, na1] = np.tanh(delta)

        # drawNeuronalSpace()

    print("Path found")


def drawSpace(pres=zer):
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


def drawNeuronalSpace():
    img = np.ones((x_size, a1_size, 3))
    for nx in range(x_size):
        for na1 in range(a1_size):
            img[nx, na1, 1] = 1 - space[nx, na1]
    for config in obstacle_area:
        img[config[0], config[1], 1] = 0
        img[config[0], config[1], 2] = 0
    imgplot = plt.imshow(img)
    plt.show()


def storePresences():
    x = 0
    global presences
    for a in a_array:
        presences.append(Presence(x, a))
    return 0


readEnvironment()
storePresences()
#drawSpace(rPresence(50, 1))
preparation_time = time.time() - start_time
print("Preparation took " + str(preparation_time) + " s.")
generateNeuronalSpace()
findTargetArea()
# drawNeuronalSpace()
generation_time = time.time() - start_time - preparation_time
print("Generation of neuronal space took " + str(generation_time) + " s.")
findPath()
finding_time = time.time() - start_time - generation_time - preparation_time
print("Path finding took " + str(finding_time) + " s.")
print("Done")
