import os

import numpy as np


# 需要解决的问题
# 传入的path为当前文档所在路径之后的拼接路径
def Get_Problem(path,num=None):
    if os.path.exists(path):
        # 提取txt文件中的数据
        with open(path, 'r') as data:
            List = data.readlines()
            # flag = 0
            num_opes = 0
            line_split = List[0].strip().split()
            num_jobs = int(line_split[0])
            num_mas = int(line_split[1])
            for i in range(1, num_jobs + 1):
                num_opes += int(List[i].strip().split()[0])
            matrix_proc_time = np.zeros((num_opes, num_mas), dtype=np.float32)
            matrix_proc_power = np.zeros((num_opes, num_mas), dtype=np.float32)
            nums_ope = []  # A list of the number of operations for each job
            num_ope_biases = []  # The id of the first operation of each job
            # Parse data line by line
            for i in range(1, num_jobs + 1):
                num_ope_bias = int(sum(nums_ope))
                num_ope_biases.append(num_ope_bias)
                # Detect information of this job and return the number of operations
                num_ope = edge_detec(List[i], num_ope_bias, matrix_proc_time)
                nums_ope.append(num_ope)
            if num is not None:
                matrix_proc_power = np.round(num / (matrix_proc_time ** 2 + 1e-9), 2)
                matrix_proc_power[~(matrix_proc_time > 0)] = 0
            else:
                for i in range(num_jobs + 2, 2 * num_jobs + 2):
                    idx = i - num_jobs - 2
                    # Detect information of this job and return the number of operations
                    edge_detec(List[i], num_ope_biases[idx], matrix_proc_power)

            matrix_ope_ma_adj = np.where(matrix_proc_time > 0, 1, 0)
    else:
        print('路径有问题')
    return matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, np.array(nums_ope).astype(int), np.array(
        num_ope_biases).astype(int)


def edge_detec(line, num_ope_bias, matrix):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = float(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = int(x - 1)
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix[idx_ope + num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return int(num_ope)


if __name__ == '__main__':
    path = 'instance/5j_5m_001.fjs'
    matrix_proc_time, matrix_proc_power, matrix_ope_ma_adj, nums_ope, num_ope_biases = Get_Problem(path)
