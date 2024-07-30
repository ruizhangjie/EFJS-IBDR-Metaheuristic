import copy
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from encode import Encode
from load_data import Get_Problem
from operators import ipox_crossover, mpx_crossover, swap_mutation, single_point_mutation
from pop import Pop


def Tri_VGM(H):
    delta = 1 / H
    w = []
    w1 = 0
    while w1 <= 1:
        w2 = 0
        while w2 + w1 <= 1:
            w3 = 1 - w1 - w2
            w.append([w1, w2, w3])
            w2 += delta
        w1 += delta
    outside = np.array(w)
    inner = 0.5 * outside + 1 / 6.0
    result = np.concatenate((outside, inner), axis=0)
    return result


def find_top_t_neighbors(weights, T):
    """
    找出每个权重向量的前T个最近的邻居权重向量的索引。

    参数:
    weights (np.array): n x 3 的权重矩阵。
    T (int): 要返回的邻居的数量。

    返回:
    list: 包含每个权重向量前T个最近邻居索引的列表。
    """
    n = weights.shape[0]
    top_t_neighbors = []

    # 计算每对权重向量之间的距离
    for i in range(n):
        distances = []
        for j in range(n):
            # 计算欧氏距离
            distance = np.linalg.norm(weights[i] - weights[j])
            distances.append((distance, j))

        # 排序并选取前T个最近的邻居
        distances.sort()  # 默认按照第一个元素，即距离排序
        neighbor_indices = [idx for _, idx in distances[:T]]  # 包括自身，选取前T个
        top_t_neighbors.append(neighbor_indices)

    return top_t_neighbors


def Tchebycheff(x, z, lambd):
    Gte = []
    for i in range(len(x.fitness)):
        Gte.append(np.abs(x.fitness[i] - z[i]) * lambd[i])
    return np.max(Gte)


def Tri_Dominate(Pop1, Pop2):
    dominated = False
    for i in range(len(Pop1.fitness)):
        if Pop1.fitness[i] > Pop2.fitness[i]:
            return False
        if Pop1.fitness[i] < Pop2.fitness[i]:
            dominated = True
    return dominated


def update_ep(EP, pop):
    if EP == []:
        EP.append(pop)
    else:
        dominateY = False  # 是否有支配Y的解
        _remove = []  # Remove from EP all the vectors dominated by y
        for ei in range(len(EP)):
            if Tri_Dominate(pop, EP[ei]):
                _remove.append(EP[ei])
            elif Tri_Dominate(EP[ei], pop):
                dominateY = True
                break
        # add y to EP if no vectors in EP dominated y
        if not dominateY:
            if len(_remove) > 0:
                for j in range(len(_remove)):
                    EP.remove(_remove[j])
            EP.append(pop)


class Algorithms:
    def __init__(self, path, slot, matrix_sb_power, thr, fsl, num, pop_size=210, gene_size=400, pc_min=0.5, pc_max=1.0,
                 pm_min=0.25, pm_max=0.5):
        self.slot = slot
        self.thr = thr
        self.fsl = fsl
        self.Z = 1e20 * np.ones(3)
        self.T = 10
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

    def MOEAD_main(self):
        lambd = Tri_VGM(13)  # 生成权重向量
        self.pop_size = lambd.shape[0]
        self.Pop = []
        chs_list = self.E.Pop_Gene(self.pop_size)
        for i in range(self.pop_size):
            pop_i = Pop(chs_list[i], self.matrix_proc_time, self.matrix_proc_power, self.matrix_ope_ma_adj,
                        self.num_ope_biases, self.nums_ope, self.slot, self.matrix_sb_power, self.thr, self.fsl)
            # 参考点设置
            for j in range(self.Z.shape[0]):
                if self.Z[j] > pop_i.fitness[j]:
                    self.Z[j] = pop_i.fitness[j]
            self.Pop.append(pop_i)
        B = find_top_t_neighbors(lambd, self.T)  # work out the T closest weight vectors to each weight vector
        EP = []  # EP is used to store non-dominated solutions found during the search
        for gi in range(self.gene_size):
            self.pc = self.pc_max - ((self.pc_max - self.pc) / self.gene_size) * gi
            self.pm = self.pm_max - ((self.pm_max - self.pm) / self.gene_size) * gi
            for i in range(len(self.Pop)):
                # Randomly select two indexes k,l from B(i)
                j = random.randint(0, self.T - 1)
                k = random.randint(0, self.T - 1)
                pop1, pop2 = self.gen_pop(self.Pop[B[i][j]], self.Pop[B[i][k]])
                if Tri_Dominate(pop1, pop2):
                    y = pop1
                elif Tri_Dominate(pop2, pop1):
                    y = pop2
                else:
                    y = random.choice([pop1, pop2])
                # update of the reference point z
                for zi in range(self.Z.shape[0]):
                    if self.Z[zi] > y.fitness[zi]:
                        self.Z[zi] = y.fitness[zi]
                # update of Neighboring solutions
                for bi in range(len(B[i])):
                    Ta = Tchebycheff(self.Pop[B[i][bi]], self.Z, lambd[B[i][bi]])
                    Tb = Tchebycheff(y, self.Z, lambd[B[i][bi]])
                    if Tb < Ta:
                        self.Pop[B[i][bi]] = y
                update_ep(EP, y)
            print('第%s次迭代' % (gi + 1))
        return EP


if __name__ == '__main__':
    path = 'benchmark/1_Brandimarte/BrandimarteMk3.fjs'
    name = path.split('/')[-1].split('.')[0][-3:]
    slot = np.arange(40, 60, dtype=np.float32)
    thr = 26
    matrix_sb_power = np.array([0.2, 0.1, 0.3, 0.2, 0.1, 0.3,0.1, 0.4])
    fsl = 32
    A = Algorithms(path, slot, matrix_sb_power, thr, fsl, 400)
    t1 = time.time()
    for i in range(1, 3):
        print('第%s次运行开始' % i)
        PF = A.MOEAD_main()
        Makespan = np.array([PF[j].fitness[0] for j in range(len(PF))])
        TPC = np.array([PF[j].fitness[1] for j in range(len(PF))])
        IR = np.array([PF[j].fitness[2] for j in range(len(PF))])
        all_fit = np.stack((Makespan, TPC, IR), axis=1)
        fit = np.unique(all_fit, axis=0)
        np.savetxt(f'result/{name}_moead_{i}.txt', fit, fmt='%.2f', delimiter=' ')
    t2 = time.time()
    print("the CPU(s) time", t2 - t1)

    # import numpy as np
    # from sklearn.kernel_ridge import KernelRidge
    # import matplotlib.pyplot as plt
    #
    # # 生成示例数据
    # np.random.seed(0)
    # X = np.sort(5 * np.random.rand(100, 1), axis=0)
    # y = np.sin(X).ravel()
    # y[::5] += 3 * (0.5 - np.random.rand(20))
    #
    # # 数据分段
    # X1, y1 = X[:50], y[:50]
    # X2, y2 = X[50:], y[50:]
    #
    # # 构建模型并拟合
    # model1 = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.1)
    # model1.fit(X1, y1)
    #
    # model2 = KernelRidge(kernel='rbf', gamma=0.1, alpha=0.1)
    # model2.fit(X2, y2)
    #
    # # 生成对应分段范围的测试数据
    # X_test1 = np.linspace(X1.min(), X1.max(), 100).reshape(-1, 1)
    # X_test2 = np.linspace(X2.min(), X2.max(), 100).reshape(-1, 1)
    #
    # # 在对应分段内进行预测
    # y_pred1 = model1.predict(X_test1)
    # y_pred2 = model2.predict(X_test2)
    #
    # # 绘图
    # plt.scatter(X1, y1, color='blue', s=30, marker='o', label='segment 1 data')
    # plt.plot(X_test1, y_pred1, color='red', label='segment 1 model')
    # plt.scatter(X2, y2, color='green', s=30, marker='o', label='segment 2 data')
    # plt.plot(X_test2, y_pred2, color='orange', label='segment 2 model')
    #
    # plt.xlabel('data')
    # plt.ylabel('target')
    # plt.title('Kernel Ridge Regression with Segmented Data')
    # plt.legend()
    # plt.show()
