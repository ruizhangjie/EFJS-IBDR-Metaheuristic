import copy

import numpy as np


def ipox_crossover(parent_a, parent_b):
    num_jobs = len(set(parent_a))  # 工件的数量
    crossover_point = np.random.randint(1, num_jobs)  # 随机选择交叉点

    # Step 1: 将所有工件随机分成两部分 J1 和 J2
    J1 = set(np.random.choice(range(num_jobs), crossover_point, replace=False))
    J2 = set(range(num_jobs)) - J1

    # Step 2: 计算 C1
    C1 = np.full_like(parent_a, -1)
    # P1J1复制到C1保持位置
    for i, job in enumerate(parent_a):
        if job in J1:
            C1[i] = job
    j2_index = 0
    indices = [value for _, value in enumerate(parent_b) if value in J2]
    for i in range(len(C1)):
        if C1[i] == -1:
            C1[i] = indices[j2_index]
            j2_index += 1

    # Step 3: 计算 C2
    C2 = np.full_like(parent_a, -1)
    # P2J2复制到C2保持位置
    for i, job in enumerate(parent_b):
        if job in J2:
            C2[i] = job
    j1_index = 0
    indices = [value for _, value in enumerate(parent_a) if value in J1]
    for i in range(len(C2)):
        if C2[i] == -1:
            C2[i] = indices[j1_index]
            j1_index += 1

    return C1, C2


def mpx_crossover(parent_a, parent_b):
    position = np.random.randint(2, size=len(parent_a))
    C1 = copy.deepcopy(parent_a)
    C2 = copy.deepcopy(parent_b)
    C1[position == 1] = parent_b[position == 1]
    C2[position == 1] = parent_a[position == 1]
    return C1, C2


def swap_mutation(parent):
    a = copy.deepcopy(parent)
    position = np.random.choice(len(parent), 2)
    value = parent[position]
    reversed = position[::-1]
    a[reversed] = value
    return a


def single_point_mutation(parent, time, power):
    a = copy.deepcopy(parent)
    position = np.random.choice(len(parent), 2)
    time = copy.deepcopy(time)
    power = copy.deepcopy(power)
    time[time == 0] = 9999
    power[power == 0] = 9999
    index = np.zeros(2)
    min_time = np.where(time[position[0]] == np.min(time[position[0]]))[0]
    index[0] = np.random.choice(min_time)
    min_power = np.where(power[position[1]] == np.min(power[position[1]]))[0]
    index[1] = np.random.choice(min_power)
    a[position] = index
    return a


if __name__ == '__main__':
    p1 = np.array([0, 1, 2, 0, 0, 1, 1, 2])
    p2 = np.array([1, 0, 2, 0, 1, 2, 1, 0])
    c1,c2= ipox_crossover(p1, p2)
    d=1
