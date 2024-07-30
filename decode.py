import copy
import random

import numpy as np

from encode import Encode
from operators import ipox_crossover, mpx_crossover, swap_mutation, single_point_mutation

from plot import initialize_plt


class Decode:
    def __init__(self, matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, num_ope_biases, nums_ope, slot,
                 matrix_sb_power, maxp):
        self.nums_ope = nums_ope
        self.matrix_proc_time = matrix_proc_time
        self.matrix_proc_power = matrix_proc_power
        self.matrix_ope_ma_adj = matrix_ope_ma_adj
        self.num_ope_biases = num_ope_biases
        self.all_opes = np.sum(nums_ope)
        self.num_mas = matrix_proc_time.shape[-1]
        self.slot = slot
        self.matrix_sb_power = matrix_sb_power
        self.maxp = maxp

    def decode_Ma(self, chromosome):
        matrix = np.zeros((self.all_opes, 3), dtype=np.float32)
        start = -99 * np.ones((self.num_mas, self.all_opes), dtype=np.float32)
        sb_start = -99 * np.ones((self.num_mas, self.all_opes), dtype=np.float32)
        end = np.zeros_like(start, dtype=np.float32)
        sb_end = np.zeros_like(start, dtype=np.float32)
        demand = np.zeros_like(start, dtype=np.float32)
        op_idx = -99 * np.ones_like(start, dtype=np.int64)
        Os = chromosome[:self.all_opes]
        Ms = chromosome[self.all_opes:]
        # 先解码MS部分
        for i in range(self.all_opes):
            ma = Ms[i]
            time = self.matrix_proc_time[i, ma]
            power = self.matrix_proc_power[i, ma]
            matrix[i, 0] = ma
            matrix[i, 1] = time
            matrix[i, 2] = power
        step_ope_biases = copy.deepcopy(self.num_ope_biases)
        for i in range(self.all_opes):
            job = Os[i]
            op = step_ope_biases[job]
            ma = int(matrix[op, 0])
            process_time = matrix[op, 1]
            process_power = matrix[op, 2]
            start_a = start[ma]
            end_a = end[ma]
            sb_start_a = sb_start[ma]
            sb_end_a = sb_end[ma]
            op_idx_a = op_idx[ma]
            power_a = demand[ma]
            job_release = 0
            if op not in self.num_ope_biases:
                job_release = end[op_idx == op - 1]
            ma_release = np.max(end_a)
            possiblePos = np.where(job_release < start_a)[0]
            if len(possiblePos) == 0:
                self.putInTheEnd(op, job_release, ma_release, start_a, end_a, sb_start_a, sb_end_a, op_idx_a,
                                 power_a, process_time, process_power)
            else:
                idxLegalPos, legalPos, startTimeEarlst = self.calLegalPos(possiblePos, start_a, end_a, job_release,
                                                                          process_time)
                if len(legalPos) == 0:
                    self.putInTheEnd(op, job_release, ma_release, start_a, end_a, sb_start_a, sb_end_a,
                                     op_idx_a, power_a, process_time, process_power)
                else:
                    self.putInBetween(op, idxLegalPos, legalPos, startTimeEarlst, start_a, end_a,
                                                           sb_start_a, sb_end_a, op_idx_a, power_a, process_time,
                                                           process_power)
            step_ope_biases[job] += 1

        x = self.cal_demand(demand, start, end) + self.cal_demand(self.matrix_sb_power, sb_start, sb_end)
        return start, end, sb_start, sb_end, demand, x, op_idx

    def putInTheEnd(self, op, job_release, ma_release, start_a, end_a, sb_start_a, sb_end_a, op_idx_a, power_a,
                    process_time, process_power):
        index = np.where(start_a == -99)[0][0]
        startTime_a = max(job_release, ma_release)
        start_a[index] = startTime_a
        op_idx_a[index] = op
        power_a[index] = process_power
        end_a[index] = startTime_a + process_time
        if index > 0:
            st = end_a[index - 1]
            if st < startTime_a:
                idx1 = np.where(sb_start_a == -99)[0][0]
                sb_start_a[idx1] = st
                sb_end_a[idx1] = startTime_a
        return startTime_a

    def calLegalPos(self, possiblePos, start_a, end_a, job_release, process_time):
        part_start = end_a[possiblePos[:-1]]
        if possiblePos[0] != 0:
            t1 = end_a[possiblePos[0] - 1]
            t2 = max(job_release, t1)
            startTimeEarlst = np.insert(part_start, 0, t2)
        else:
            t = max(job_release, 0)
            startTimeEarlst = np.insert(part_start, 0, t)
        dur = start_a[possiblePos] - startTimeEarlst
        idxLegalPos = np.where(dur >= process_time)[0]  # possiblePos中的下标
        legalPos = np.take(possiblePos, idxLegalPos)  # start_a中的下标
        return idxLegalPos, legalPos, startTimeEarlst

    def putInBetween(self, op, idxLegalPos, legalPos, startTimeEarlst, start_a, end_a, sb_start_a, sb_end_a, op_idx_a,
                     power_a, process_time, process_power):
        earlstIdx = idxLegalPos[0]
        earlstPos = legalPos[0]
        startTime_a = startTimeEarlst[earlstIdx]
        start_a[:] = np.insert(start_a, earlstPos, startTime_a)[:-1]
        end_a[:] = np.insert(end_a, earlstPos, startTime_a + process_time)[:-1]
        op_idx_a[:] = np.insert(op_idx_a, earlstPos, op)[:-1]
        power_a[:] = np.insert(power_a, earlstPos, process_power)[:-1]
        st = startTime_a + process_time
        if earlstPos == 0:
            et = start_a[1]
            if st != et:
                pos = np.where(sb_start_a == -99)[0][0]
                sb_start_a[pos] = st
                sb_end_a[pos] = et
        else:
            pre_e = end_a[earlstPos - 1]
            sub_s = start_a[earlstPos + 1]
            pos = np.where(sb_start_a == pre_e)[0][0]
            if pre_e == startTime_a:
                if sub_s != st:
                    sb_start_a[pos] = st
                else:
                    sb_start_a[pos] = -99
                    sb_end_a[pos] = 0
            else:
                sb_end_a[pos] = startTime_a
                if sub_s != st:
                    pos1 = np.where(sb_start_a == -99)[0][0]
                    sb_start_a[pos1] = st
                    sb_end_a[pos1] = sub_s
        return startTime_a, earlstPos

    def cal_demand(self, power, start, end):
        dim0 = np.shape(start)[0]
        dim1 = np.shape(start)[1]
        a = np.tile(start, (96, 1, 1))
        b = np.tile(end, (96, 1, 1))
        c = np.tile(power, (96, 1, 1))
        steps = np.arange(96, dtype=np.float32)
        temp_array = np.tile(steps, (dim0, 1))
        d = np.tile(temp_array, (dim1, 1, 1)).transpose(2, 1, 0)
        e = np.where(a <= d, d, -1)
        e = np.where(e < b, e, -1)
        f = np.where(e != -1, 1, 0)
        g = f * c
        result = np.sum(g, axis=-1).T
        return result

