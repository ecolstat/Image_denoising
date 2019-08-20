# Author: TienTian
# Date: 2019/8/20
# Cite: 《Salt-and-Pepper Noise Removal by Median-Type Noise Detectors and Detail-Preserving Regularization》
# Detail: 2005 IEEE TRANSACTIONS ON IMAGE PROCESSING

import matplotlib.pyplot as plt
import numpy as np
import copy
from Genetic_Algorithm import GA


pic = plt.imread('C:/Users\Tian\Desktop\去噪大作业/Lena-salt-pepper-noise.bmp')
ori = plt.imread('C:/Users\Tian\Desktop\去噪大作业/Lena.bmp')
pic_restored = copy.copy(pic)
pic_restored_2 = copy.copy(pic)


def window(pic, center, win_size):
    """
    :param pic: ndarray with shape of m*n
    :param center: list:[i, j]
    :param win_size: number
    :return: ndarray with shape of win_size^2
    """
    low_bound_raw = 0 if int(center[0] - (win_size - 1) / 2) < 0 else int(center[0] - (win_size - 1) / 2)
    high_bound_raw = 0 if int(center[0] + (win_size - 1) / 2) < 0 else int(center[0] + (win_size - 1) / 2)
    low_bound_col = 0 if int(center[1] - (win_size - 1) / 2) < 0 else int(center[1] - (win_size - 1) / 2)
    high_bound_col = 0 if int(center[1] + (win_size - 1) / 2) < 0 else int(center[1] + (win_size - 1) / 2)
    return pic[low_bound_raw: high_bound_raw + 1, low_bound_col: high_bound_col + 1]


# detect Noise pixels
N_set = set()   # The set of Noise Pixels
win_size_max = 9  # User-Defined
for i in range(pic.shape[0]):
    for j in range(pic.shape[1]):
        print(i, j)
        pixel = pic[i, j]
        win_size = 3
        while win_size <= win_size_max:
            win = window(pic, [i, j], win_size)
            min_win = np.min(win)
            med_win = np.median(win)
            max_win = np.max(win)
            if min_win < med_win < max_win:
                if min_win < pixel < max_win:
                    break
                else:
                    N_set.add((i, j, pixel))
                    pic_restored[i, j] = med_win
                    break
            else:
                win_size += 2

        if win_size > win_size_max:
            N_set.add((i, j, pixel))
            pic_restored[i, j] = med_win


pic_set = set((i, j, pic[i, j]) for i in range(pic.shape[0]) for j in range(pic.shape[1]))
correct_set = pic_set - N_set

dic_noise = {}
for n in N_set:
    dic_noise[n[:2]] = n[2]

dic_correct = {}
for c in correct_set:
    dic_correct[c[:2]] = c[2]


def phi_func(t, a):      # φ:edge-preserving potential function
    assert 0 < a <= 2
    if 1 < a <= 2:
        return np.abs(t)**a
    return np.sqrt(a+t**2)


def Vij_func(p):
    """
    :param p: Tuple, such as (10,12)
    :return:
    """
    left = None if p[1] - 1 < 0 else (p[0], p[1] - 1, pic[p[0], p[1] - 1])
    right = None if p[1] + 1 == pic.shape[1] else (p[0], p[1] + 1, pic[p[0], p[1] + 1])
    up = None if p[0] - 1 < 0 else (p[0] - 1, p[1], pic[p[0] - 1, p[1]])
    down = None if p[0] + 1 == pic.shape[0] else (p[0] + 1, p[1], pic[p[0] + 1, p[1]])
    return {left, right, up, down}


N = list(N_set)  # convert set of Noise Pixels to list
local = [i[:2] for i in N]


def F_u(u_set):
    """
    :param u_set: List such as [21, 42, 93, 95, 12, 43 ...]
    u_set is the restored pixel values of noise pixels
    :return:
    """
    assert len(u_set) == len(N)  # ensure the nums of Noise pixels are same with u_set
    belta = 5
    FyNu = 0
    for t in range(len(N)):
        ind = [N[t][0], N[t][1]]
        print(t)
        Vij_of_Nt = Vij_func(ind)  # To generate the four closest neighboors
        # Calculate S1
        Vij_and_Nc = Vij_of_Nt & correct_set
        S1 = 0
        for i in Vij_and_Nc:
            S1 += 2*phi_func(t=(u_set[t] - i[2]), a=1)
        S2 = 0
        Vij_and_N = Vij_of_Nt & N_set
        for m in Vij_and_N:
            S2 += phi_func(t=(u_set[t] - u_set[local.index(m[:2])]), a=1)
        FyNu += (np.abs(u_set[t] - N[t][2]) + (belta/2)*(S1+S2))
    return FyNu


# Initial the random values of GA.
u_set = []
for i in range(0, 256):
    u = [i]*len(N)
    u = tuple(u)
    u_set.append(u)


bound = [(0, 255)]*len(N)
DNA_SIZE = 20
cross_rate = 0.7
mutation = 0.01

ga = GA(nums=u_set, bound=bound, func=F_u, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate, mutation=mutation)

# Start Training.....a lots of epoch
print("Start GA....")
for epoch in range(500):
    print("Epoch: %s" % epoch)
    ga.evolution()

r = ga.log().max()

# replace the noise pixels by the restored pixels
i = 0
for new_pixel in r:
    pic_restored_2[local[i][0]][local[i][1]] = new_pixel
    i += 1


plt.subplot(131); plt.imshow(pic, cmap='gray'); plt.title('Original image')
plt.subplot(132); plt.imshow(pic_restored, cmap='gray'); plt.title('Adaptive median filter')
plt.subplot(133); plt.imshow(pic_restored_2, cmap='gray'); plt.title('Noise Detection and GA Optimizer')
plt.show()
