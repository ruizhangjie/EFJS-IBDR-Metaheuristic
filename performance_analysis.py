import numpy as np
import os
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.gd_plus import GDPlus


def read_and_combine_txt_files(directory):
    all_data = []
    # 遍历目录中的所有文件
    for filename in directory:
        # 读取文件内容
        data = np.loadtxt(filename, dtype=float)
        # 将读取的数据添加到列表中
        all_data.append(data)
    # 将所有数据垂直堆叠成一个大的NumPy数组
    combined_array = np.vstack(all_data)
    return combined_array


def find_non_dominated_solutions(fitness_values):
    # 获取个体数量和目标数量
    pop_size, num_objectives = fitness_values.shape

    # 标记所有个体是否是非支配解，初始全为True
    is_non_dominated = np.ones(pop_size, dtype=bool)

    for i in range(pop_size):
        for j in range(pop_size):
            if i != j:
                # 检查个体j是否支配个体i
                if np.all(fitness_values[j] <= fitness_values[i]) and np.any(fitness_values[j] < fitness_values[i]):
                    is_non_dominated[i] = False
                    break

    # 返回非支配解
    return fitness_values[is_non_dominated]


def calculate_hypervolume(solutions, reference_point):
    ind = HV(ref_point=reference_point)
    # 计算给定解集的超体积
    hypervolume = int(ind(solutions))
    return hypervolume


def hv_all(directory, reference_point,epsilon=0.01):
    all_data = []
    for filename in directory:
        data = np.loadtxt(filename, dtype=float)
        solutions = np.array(data)
        hv = calculate_hypervolume(solutions, reference_point)
        all_data.append(hv)
    # min_val = min(all_data)
    # max_val = max(all_data)
    # range_val = max_val - min_val
    # normalized_data = [round((x - min_val + epsilon) / (range_val + 2 * epsilon), 2) for x in all_data]
    return np.array(all_data)

def spacing_all(directory):
    all_data = []
    for filename in directory:
        data = np.loadtxt(filename, dtype=float)
        solutions = np.array(data)
        spacing = calculate_spacing(solutions)
        all_data.append(spacing)
    return np.array(all_data)


def calculate_spacing(non_dominated_solutions):
    # Step 1: Calculate pairwise distances
    n = non_dominated_solutions.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i, j] = np.linalg.norm(non_dominated_solutions[i] - non_dominated_solutions[j])

    # Step 2: Find the nearest neighbor distance for each solution
    nearest_distances = np.min(distances + np.diag([np.inf] * n), axis=1)

    # Step 3: Calculate the mean of the nearest distances
    mean_nearest_distance = np.mean(nearest_distances)

    # Step 4: Calculate the spacing metric
    spacing = np.sqrt(np.sum((nearest_distances - mean_nearest_distance) ** 2) / (n - 1))

    return spacing
# def calculate_igd_plus(approximate_set, reference_set):
#     ind = IGDPlus(reference_set)
#     # 计算 IGD 值
#     igd_value = round(ind(approximate_set), 4)
#     return igd_value
#
#
# def igd_plus_all(directory, reference_set):
#     all_data = []
#     for filename in directory:
#         data = np.loadtxt(filename, dtype=float)
#         solutions = np.array(data)
#         igd_plus = calculate_igd_plus(solutions, reference_set)
#         all_data.append(igd_plus)
#     return np.array(all_data)
#
#
# def calculate_gd_plus(approximate_set, reference_set):
#     ind = GDPlus(reference_set)
#     # 计算 GD 值
#     gd_value = round(ind(approximate_set), 4)
#     return gd_value
#
#
# def c1r_all(directory, reference_set):
#     all_data = []
#     for filename in directory:
#         data = np.loadtxt(filename, dtype=float)
#         solutions = np.array(data)
#         c1r = calculate_c1r(solutions, reference_set)
#         all_data.append(c1r)
#     return np.array(all_data)
#
#
# def calculate_c1r(approximate_set, reference_set):
#     # 将数组a和b转换为集合，以找到重复的行
#     set_a = set(map(tuple, approximate_set))
#     set_b = set(map(tuple, reference_set))
#
#     # 计算重复的行数
#     repeated_rows = set_a.intersection(set_b)
#     num_repeated_rows = len(repeated_rows)
#
#     # 计算比值并保留两位小数
#     ratio = round(num_repeated_rows / reference_set.shape[0], 2)
#     return ratio
#
#
# def gd_plus_all(directory, reference_set):
#     all_data = []
#     for filename in directory:
#         data = np.loadtxt(filename, dtype=float)
#         solutions = np.array(data)
#         gd_plus = calculate_gd_plus(solutions, reference_set)
#         all_data.append(gd_plus)
#     return np.array(all_data)


if __name__ == '__main__':
    from pymoo.config import Config

    Config.warnings['not_compiled'] = False
    path = 'result/MK10/'
    file_list = os.listdir(path)
    file_read = []
    # num = '2'
    for filename in file_list:
        # if filename.split('_')[-1][0:1] == num:
        file_read.append(os.path.join(path, filename))
    data = read_and_combine_txt_files(file_read)
    all_pop = np.unique(data, axis=0)
    true_front = find_non_dominated_solutions(all_pop)
    # 计算每个目标的最大值
    max_values = np.max(true_front, axis=0)
    offset_ratio = 0.001
    # 设置参考点为最大值加上一个偏移量
    reference_point = max_values * (1 + offset_ratio)
    hv = hv_all(file_read, reference_point)
    spacing = spacing_all(file_read)
    # gd = gd_plus_all(file_read, true_front)
    # igd = igd_plus_all(file_read, true_front)
    # c1r = c1r_all(file_read, true_front)
    print("HV:", hv)
    print("\nspacing:", spacing)
    # print("\nGD:", gd)
    # print("\nIGD:", igd)
    # print("\nC1R:", c1r)
    # path = 'peak/'
    # file_list = os.listdir(path)
    # file_read = []
    # for filename in file_list:
    #     file_read.append(os.path.join(path, filename))
    # data = read_and_combine_txt_files(file_read)
    # all_pop = np.unique(data, axis=0)
    # true_front = find_non_dominated_solutions(all_pop)
    # np.savetxt(f'peak/pareto.txt', true_front, fmt='%.2f', delimiter=' ')
    # s = 2
