import copy
import random

import numpy as np


class Encode:

    def __init__(self, matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, num_ope_biases, nums_ope, p_T=0.3,
                 p_P=0.3, p_N=0.2, p_R=0.2,
                 p_L=0.2):
        self.nums_ope = nums_ope
        self.matrix_proc_time = matrix_proc_time
        self.matrix_proc_power = matrix_proc_power
        self.matrix_ope_ma_adj = matrix_ope_ma_adj
        self.num_ope_biases = num_ope_biases
        self.all_opes = np.sum(nums_ope)
        self.p_T, self.p_N, self.p_R, self.p_L, self.p_P = p_T, p_N, p_R, p_L, p_P

    def encode_ROS(self):
        b = [0] * self.all_opes
        start = 0
        for i, num in enumerate(self.nums_ope):
            b[start:start + num] = [i] * num
            start += num
        random.shuffle(b)
        return b

    def encode_TOS(self):
        # 平均加工时间最小
        result = []
        ope_step = copy.deepcopy(self.num_ope_biases)
        b = np.sum(self.matrix_proc_time, axis=1) / np.sum(self.matrix_proc_time != 0, axis=1).astype(float)
        for i in range(self.all_opes):
            c = b[ope_step]
            # 找到最小元素的索引
            min_indices = np.where(c == np.min(c))[0]
            random_min_index = np.random.choice(min_indices)
            result.append(random_min_index)
            b[ope_step[random_min_index]] = 999
            if random_min_index != len(ope_step) - 1:
                if ope_step[random_min_index] != self.num_ope_biases[random_min_index + 1] - 1:
                    ope_step[random_min_index] += 1
            else:
                if ope_step[random_min_index] != self.all_opes - 1:
                    ope_step[random_min_index] += 1
        return result

    def encode_POS(self):
        # 平均加工功率最小
        result = []
        ope_step = copy.deepcopy(self.num_ope_biases)
        b = np.sum(self.matrix_proc_power, axis=1) / np.sum(self.matrix_proc_power != 0, axis=1).astype(float)
        for i in range(self.all_opes):
            c = b[ope_step]
            # 找到最小元素的索引
            min_indices = np.where(c == np.min(c))[0]
            random_min_index = np.random.choice(min_indices)
            result.append(random_min_index)
            b[ope_step[random_min_index]] = 999
            if random_min_index != len(ope_step) - 1:
                if ope_step[random_min_index] != self.num_ope_biases[random_min_index + 1] - 1:
                    ope_step[random_min_index] += 1
            else:
                if ope_step[random_min_index] != self.all_opes - 1:
                    ope_step[random_min_index] += 1
        return result

    def encode_NOS(self):
        # 剩余工序最多
        result = []
        num_step = copy.deepcopy(self.nums_ope)
        for i in range(self.all_opes):
            max_value = np.max(num_step)
            max_indices = np.where(num_step == max_value)[0]
            random_min_index = np.random.choice(max_indices)
            result.append(random_min_index)
            num_step[random_min_index] -= 1
        return result

    def encode_RMS(self):
        result = []
        for i in range(self.all_opes):
            non_zero_indices = np.where(self.matrix_proc_time[i] > 0)[0]
            if len(non_zero_indices) > 0:
                result.append(np.random.choice(non_zero_indices))
        return result

    def encode_TMS(self):
        # 加工时间最小
        result = []
        for i in range(self.all_opes):
            arr = self.matrix_proc_time[i]
            non_zero_elements = arr[arr != 0]
            min_value = np.min(non_zero_elements)
            min_indices = np.where(arr == min_value)[0]
            result.append(np.random.choice(min_indices))
        return result

    def encode_PMS(self):
        # 加工时间最小
        result = []
        for i in range(self.all_opes):
            arr = self.matrix_proc_power[i]
            non_zero_elements = arr[arr != 0]
            min_value = np.min(non_zero_elements)
            min_indices = np.where(arr == min_value)[0]
            result.append(np.random.choice(min_indices))
        return result

    def encode_LMS(self):
        # 全局选择策略
        result = [0] * self.all_opes
        cum_load = np.zeros(self.matrix_proc_time.shape[-1], dtype=np.int64)
        job_list = [_ for _ in range(self.nums_ope.shape[0])]
        proc_time = copy.deepcopy(self.matrix_proc_time)
        proc_time[proc_time == 0] = 999
        all_job = len(job_list)
        while job_list:
            i = random.choice(job_list)  # 随机选择一个工件
            start_op = self.num_ope_biases[i]
            if i != all_job - 1:
                end_op = self.num_ope_biases[i + 1]
            else:
                end_op = self.all_opes
            a = proc_time[start_op:end_op]
            for j in range(a.shape[0]):
                b = a[j] + cum_load
                min_index_candidates = np.where(b == np.min(b))[0]
                random_min_index = np.random.choice(min_index_candidates)
                result[start_op + j] = random_min_index
                cum_load[random_min_index] += a[j, random_min_index]
            job_list.remove(i)  # 将选择后的工件从未选工件集中移除
        return result

    def Pop_Gene(self, size):
        Pop = []
        for i in range(size):
            methods = [self.encode_TOS, self.encode_POS, self.encode_NOS, self.encode_ROS]
            probabilities = [self.p_T, self.p_P, self.p_N, self.p_R]
            chosen_method = random.choices(methods, probabilities)[0]
            os_list = chosen_method()
            methods1 = [self.encode_TMS, self.encode_PMS, self.encode_LMS, self.encode_RMS]
            probabilities1 = [self.p_T, self.p_P, self.p_L, self.p_R]
            chosen_method1 = random.choices(methods1, probabilities1)[0]
            ms_list = chosen_method1()
            item = os_list + ms_list
            Pop.append(item)
        return np.array(Pop)


if __name__ == '__main__':
    import numpy as np
    from sklearn.kernel_ridge import KernelRidge
    import matplotlib.pyplot as plt

    path = 'testdata/1.txt'
