import copy
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from decode import Decode
from encode import Encode
from load_data import Get_Problem

from operators import ipox_crossover, mpx_crossover, swap_mutation, single_point_mutation
from pop import Pop


class Algorithms:
    def __init__(self, path, slot, matrix_sb_power, thr, fsl, num, pop_size=210, gene_size=400, pc_min=0.5, pc_max=1.0,
                 pm_min=0.25, pm_max=0.5):
        self.slot = slot
        self.thr = thr
        self.fsl = fsl
        self.pop_size = pop_size
        self.gene_size = gene_size
        self.pc_min = pc_min
        self.pm_min = pm_min
        self.pc_max = pc_max
        self.pm_max = pm_max
        self.pc = self.pc_max
        self.pm = self.pm_max
        self.Pop = []
        self.matrix_proc_time, self.matrix_proc_power, self.matrix_ope_ma_adj, self.nums_ope, self.num_ope_biases = Get_Problem(
            path, num)
        dim = np.shape(self.matrix_proc_time)[0]
        self.matrix_sb_power = np.tile(matrix_sb_power, (dim, 1)).T
        self.E = Encode(self.matrix_proc_time, self.matrix_proc_power, self.matrix_ope_ma_adj, self.num_ope_biases,
                        self.nums_ope)
        self.half_len_chromo = np.sum(self.nums_ope)
        self.percentage = []

    def offspring_Population(self):
        new_pop = []
        while len(new_pop) < self.pop_size:
            # 随机选择父代个体
            pop1, pop2 = random.sample(self.Pop, 2)
            P1, P2 = self.gen_pop(pop1, pop2)
            new_pop.extend([P1, P2])
        return new_pop

    def gen_pop(self, pop1, pop2):
        p1, p2 = copy.deepcopy(pop1.CHS), copy.deepcopy(pop2.CHS)
        # 交叉算子
        if random.random() < self.pc:
            p1_os = p1[:self.half_len_chromo]
            p1_ms = p1[self.half_len_chromo:]
            p2_os = p2[:self.half_len_chromo]
            p2_ms = p2[self.half_len_chromo:]
            c1_os, c2_os = ipox_crossover(p1_os, p2_os)
            c1_ms, c2_ms = mpx_crossover(p1_ms, p2_ms)
            p1 = np.concatenate((c1_os, c1_ms))
            p2 = np.concatenate((c2_os, c2_ms))
        if random.random() < self.pm:
            p1_os = p1[:self.half_len_chromo]
            p1_ms = p1[self.half_len_chromo:]
            c1_os = swap_mutation(p1_os)
            c1_ms = single_point_mutation(p1_ms, self.matrix_proc_time, self.matrix_proc_power)
            p1 = np.concatenate((c1_os, c1_ms))
        if random.random() < self.pm:
            p2_os = p2[:self.half_len_chromo]
            p2_ms = p2[self.half_len_chromo:]
            c2_os = swap_mutation(p2_os)
            c2_ms = single_point_mutation(p2_ms, self.matrix_proc_time, self.matrix_proc_power)
            p2 = np.concatenate((c2_os, c2_ms))
        P1 = Pop(p1, self.matrix_proc_time, self.matrix_proc_power, self.matrix_ope_ma_adj, self.num_ope_biases,
                 self.nums_ope, self.slot, self.matrix_sb_power, self.thr, self.fsl)
        P2 = Pop(p2, self.matrix_proc_time, self.matrix_proc_power, self.matrix_ope_ma_adj, self.num_ope_biases,
                 self.nums_ope, self.slot, self.matrix_sb_power, self.thr, self.fsl)
        return P1, P2

    def fast_non_dominated_sort(self, Pop):
        num_individuals = len(Pop)

        S = [set() for _ in range(num_individuals)]  # 存储 p 支配的个体索引
        n = [0] * num_individuals  # 记录每个个体被支配的次数
        rank = [0] * num_individuals  # 每个个体所属的 Pareto 前沿的层次

        front = [[]]  # 存储第 i 层的非支配解的个体索引

        for p in range(num_individuals):
            for q in range(num_individuals):
                if self.Tri_Dominate(Pop[p], Pop[q]):
                    S[p].add(q)
                elif self.Tri_Dominate(Pop[q], Pop[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                front[0].append(p)
        i = 0
        while front[i]:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            front.append(Q)
        NDSet = [[Pop[pi] for pi in Fi] for Fi in front[:-1]]
        return NDSet

    def Tri_Dominate(self, Pop1, Pop2):
        dominated = False
        for i in range(len(Pop1.fitness)):
            if Pop1.fitness[i] > Pop2.fitness[i]:
                return False
            if Pop1.fitness[i] < Pop2.fitness[i]:
                dominated = True
        return dominated

    def crowding_distance(self, NDSet):
        num_objs = len(NDSet[0].fitness)
        distances = [0] * len(NDSet)

        for obj_index in range(num_objs):
            # 按当前目标排序
            NDSet.sort(key=lambda x: x.fitness[obj_index])

            # 为边界个体设置极大值
            distances[0] = distances[-1] = float('inf')

            # 获取目标值的最大和最小值以用于归一化
            min_val = NDSet[0].fitness[obj_index]
            max_val = NDSet[-1].fitness[obj_index]

            if max_val == min_val:
                continue

            # 计算中间个体的拥挤距离
            for i in range(1, len(NDSet) - 1):
                distances[i] += (NDSet[i + 1].fitness[obj_index] - NDSet[i - 1].fitness[obj_index]) / (
                        max_val - min_val)

        # 重新根据距离排序索引
        sorted_indices = sorted(range(len(NDSet)), key=lambda i: distances[i], reverse=True)

        return sorted_indices

    def NSGA_main(self):
        self.Pop = []
        chs_list = self.E.Pop_Gene(self.pop_size)
        # 计算所有个体适应度值
        for i in range(self.pop_size):
            pop_i = Pop(chs_list[i], self.matrix_proc_time, self.matrix_proc_power, self.matrix_ope_ma_adj,
                        self.num_ope_biases, self.nums_ope, self.slot, self.matrix_sb_power, self.thr, self.fsl)
            self.Pop.append(pop_i)
        for ii in range(self.gene_size):
            self.pc = self.pc_max - ((self.pc_max - self.pc) / self.gene_size) * ii
            self.pm = self.pm_max - ((self.pm_max - self.pm) / self.gene_size) * ii
            # 交叉变异生成新种群
            new_pop = self.offspring_Population()
            R_pop = self.Pop + new_pop
            NDSet = self.fast_non_dominated_sort(R_pop)
            self.Pop = []
            j = 0
            while len(self.Pop) + len(NDSet[j]) <= self.pop_size:  # until parent population is filled
                self.Pop.extend(NDSet[j])
                j += 1
            if len(self.Pop) < self.pop_size:
                Ds = self.crowding_distance(copy.deepcopy(NDSet[j]))  # calcalted crowding-distance
                k = self.pop_size - len(self.Pop)
                for l in Ds[:k]:
                    self.Pop.append(NDSet[j][l])
            print('第%s次迭代' % (ii + 1))
        EP = self.fast_non_dominated_sort(self.Pop)[0]
        return EP


if __name__ == '__main__':
    path = 'benchmark/1_Brandimarte/BrandimarteMk7.fjs'
    name = path.split('/')[-1].split('.')[0][-3:]
    slot = np.arange(50, 90, dtype=np.float32)
    thr = 16
    matrix_sb_power = np.array([0.2, 0.1, 0.3, 0.2, 0.1])
    fsl = 20
    A = Algorithms(path, slot, matrix_sb_power, thr, fsl, 400)
    t1 = time.time()
    for i in range(1, 3):
        print('第%s次运行开始' % i)
        PF = A.NSGA_main()
        Makespan = np.array([PF[j].fitness[0] for j in range(len(PF))])
        TPC = np.array([PF[j].fitness[1] for j in range(len(PF))])
        IR = np.array([PF[j].fitness[2] for j in range(len(PF))])
        all_fit = np.stack((Makespan, TPC, IR), axis=1)
        fit = np.unique(all_fit, axis=0)
        np.savetxt(f'result/{name}_nsga2_{i}.txt', fit, fmt='%.2f', delimiter=' ')
    t2 = time.time()
    print("the CPU(s) time", t2 - t1)
