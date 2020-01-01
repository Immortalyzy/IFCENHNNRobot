import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

envx = 192
envy = 108
zer = np.zeros((envx, envy), dtype=int)
one = np.ones((envx, envy), dtype=int)

robot = {}
robot['armlength'] = int(envy / 4)
robot['armwidth'] = int(envy / 40)


def Presence(x, a1, a2):
    """ Calculate the presence of the robot at configuration config, 
        return an matrix of the same size of environment"""
    # calculate second joint position
    pres = np.zeros((envx, envy), dtype=int)
    l = robot['armlength']
    w = robot['armwidth']
    x1 = x
    y1 = int(envy/2)
    x2 = x1 + l * np.cos(a1)
    y2 = y1 + l * np.sin(a1)
    for x in range(envx):
        for y in range(envy):
            # check if in arm2
            deltax = x - x2
            deltay = y - y2
            if deltax <= l and deltay <= l:
                # four lines that define the arm2
                dd1 = y - y2 - np.tan(a2) * (x - (x2 + w / (2 * np.sin(a2))))
                dd2 = y - y2 - np.tan(a2) * (x - (x2 - w / (2 * np.sin(a2))))
                dd3 = y - y2 - np.tan(a2 + np.pi / 2) * (x - x2)
                dd4 = y - y2 - np.tan(a2 + np.pi / 2) * \
                    (x - (x2 + l / np.cos(a2)))
                if dd1 * dd2 <= 0 and dd3 * dd4 <= 0:
                    pres[x, y] = 1
                # check if in arm1
                deltax = x - x1
                deltay = y - y1
                if deltax <= l and deltay <= l:
                    # four lines that define the arm2
                    dd1 = y - y1 - np.tan(a1) * \
                        (x - (x1 + w / (2 * np.sin(a1))))
                    dd2 = y - y1 - np.tan(a1) * \
                        (x - (x1 - w / (2 * np.sin(a1))))
                    dd3 = y - y1 - np.tan(a1 + np.pi / 2) * (x - x1)
                    dd4 = y - y1 - np.tan(a1 + np.pi / 2) * \
                        (x - (x1 + l / np.cos(a1)))
                    if dd1 * dd2 <= 0 and dd3 * dd4 <= 0:
                        pres[x, y] = 1
    return pres


def drawSpace(env, pres=zer):
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


def Feasable(x, a1, a2, env):
    overlap = Presence(x, a1, a2) * env
    return (overlap.sum() > 0)


def generateNeuronalSpace(env, x_size, a1_size, a2_size):
    space = np.zeros([x_size, a1_size, a2_size])
    dx = 1. * envx / x_size
    da1 = np.pi * 2 / a1_size
    da2 = np.pi * 2 / a2_size
    for nx in range(x_size):
        for na1 in range(a1_size):
            for na2 in range(a2_size):
                x = nx * dx
                a1 = - np.pi + na1 * da1
                a2 = - np.pi + na2 * da2
                space[nx, na1, na2] = Feasable(x, a1, a2, env)


# Build test environment (with obstacles)
envp = mpimg.imread("e:/Projects/Robot2019/WorkSpace/Environment.png")
env = np.zeros((envx, envy), dtype=int)
for x in range(envx):
    for y in range(envy):
        if envp[y, x, 1] < 0.99:
            env[x, y] = 0
        else:
            env[x, y] = 1

config = {
    "a1": np.pi/4,
    "a2": np.pi/2,
    "x": int(envx / 2)
}

#drawSpace(env, Presence(config))
generateNeuronalSpace(env, 192, 18, 18)
print("Done")
