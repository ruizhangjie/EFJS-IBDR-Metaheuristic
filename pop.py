import copy

import numpy as np

from decode import Decode
from plot import initialize_plt, plt_demand


class Pop:
    def __init__(self, CHS, matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, num_ope_biases, nums_ope, slot,
                 matrix_sb_power, thr, fsl):
        self.CHS = CHS
        self.num_ope_biases = num_ope_biases
        self.nums_ope = nums_ope
        maxp = fsl - thr
        D = Decode(matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, num_ope_biases, nums_ope, slot,
                   matrix_sb_power, maxp)
        self.start, self.end, self.sb_start, self.sb_end, self.power, self.x, self.op_idx = D.decode_Ma(self.CHS)
        s = copy.deepcopy(self.sb_start)
        s[s == -99] = 0  # 方便计算TPC
        Cmax = np.max(self.end)
        TPC = np.sum((self.end - self.start) * self.power) + np.sum((self.sb_end - s) * matrix_sb_power)
        x1 = (np.sum(self.x, axis=0))[slot.astype(int)]
        x2 = np.where(x1 <= maxp, x1, fsl)
        IR = np.sum(x2)
        self.fitness = np.array([Cmax, TPC, IR])

    def plot_gannt(self, path_plot):
        num_mas = self.start.shape[0]
        num_jobs = len(self.nums_ope)
        cm = float(self.fitness[0])
        tec = float(self.fitness[1])
        ir = float(self.fitness[2])
        path = f'{path_plot}/{round(cm, 2)}_{round(tec, 2)}_{round(ir, 2)}_gannt.jpg'
        max_time = initialize_plt(num_mas, num_jobs, self.op_idx, self.start, self.end, self.num_ope_biases,
                                  path)
        return max_time

    def plot_demand(self, max_time, path_plot):
        cm = float(self.fitness[0])
        tec = float(self.fitness[1])
        ir = float(self.fitness[2])
        path = f'{path_plot}/{round(cm, 2)}_{round(tec, 2)}_{round(ir, 2)}_demand.jpg'
        plt_demand(self.x, path, max_time)
