import math
import random

import numpy as np

from matplotlib import pyplot as plt, rcParams


def colour_gen():
    color1 = [99 / 255, 178 / 255, 238 / 255]
    color2 = [118 / 255, 218 / 255, 145 / 255]
    color3 = [248 / 255, 203 / 255, 127 / 255]
    color4 = [248 / 255, 149 / 255, 136 / 255]
    color5 = [124 / 255, 214 / 255, 207 / 255]
    color6 = [145 / 255, 146 / 255, 171 / 255]
    color7 = [120 / 255, 152 / 255, 225 / 255]
    color8 = [239 / 255, 166 / 255, 102 / 255]
    color9 = [237 / 255, 221 / 255, 134 / 255]
    color10 = [153 / 255, 135 / 255, 206 / 255]
    color11 = [0 / 255, 44 / 255, 83 / 255]
    color12 = [255 / 255, 165 / 255, 16 / 255]
    color13 = [12 / 255, 132 / 255, 198 / 255]
    color14 = [255 / 255, 189 / 255, 102 / 255]
    color15 = [247 / 255, 77 / 255, 77 / 255]
    color16 = [36 / 255, 85 / 255, 164 / 255]
    color17 = [65 / 255, 183 / 255, 172 / 255]
    color18 = [223 / 255, 122 / 255, 94 / 255]
    color19 = [60 / 255, 64 / 255, 91 / 255]
    color20 = [130 / 255, 178 / 255, 154 / 255]
    color = []
    color.append(color1)
    color.append(color2)
    color.append(color3)
    color.append(color4)
    color.append(color5)
    color.append(color6)
    color.append(color7)
    color.append(color8)
    color.append(color9)
    color.append(color10)
    color.append(color11)
    color.append(color12)
    color.append(color13)
    color.append(color14)
    color.append(color15)
    color.append(color16)
    color.append(color17)
    color.append(color18)
    color.append(color19)
    color.append(color20)
    return color


def initialize_plt(num_mas, num_jobs, ma_idx, ma_start, ma_end, start_op, path):
    y_value = list(range(1, num_mas + 1))
    max_time = 0
    # plt.figure(figsize=(3 / 2.54, 4 / 2.54))
    plt.figure(figsize=(6, 2))

    plt.yticks(y_value, fontproperties='Times New Roman', size=10)
    plt.xticks(fontproperties='Times New Roman', size=10)

    colors = colour_gen()

    font = {'family': 'Times New Roman',
            'color': 'black',
            'weight': 'normal',
            'size': 6,
            }

    for i in range(ma_end.shape[0]):
        end_idx = np.where(ma_end[i] > 0)[0]
        if len(end_idx) > 0:
            for k in range(len(end_idx)):
                # job号
                possible_pos = np.where(ma_idx[i, k] < start_op)[0]
                if (len(possible_pos) == 0):
                    job = num_jobs
                    op = ma_idx[i, k] - start_op[-1] + 1
                else:
                    job = possible_pos[0]
                    op = ma_idx[i, k] - start_op[possible_pos[0] - 1] + 1
                start_time = ma_start[i, k]
                if ma_end[i, k] > max_time:
                    max_time = ma_end[i, k]
                dur = ma_end[i, k] - ma_start[i, k]
                plt.barh(i + 1, dur, 0.5, left=start_time, color=colors[job - 1])
                plt.text(start_time, i + 1.3, '%s-%s' % (job, op), fontdict=font)

    # 获取当前的Axes对象
    ax = plt.gca()
    # 设置x轴的范围，使其起始位置为0,跨度为5
    max_time = find_multiple(max_time, 4)
    x = generate_sequence(max_time, 4)
    label = np.arange(max_time // 4 + 1)
    # 设置x轴刻度的位置
    ax.set_xticks(x)
    # 设置x轴刻度的标签
    ax.set_xticklabels(label)
    # 隐藏上方和右侧的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 只显示左侧和底部的刻度
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    # 设置刻度线朝内
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    return max_time


def generate_sequence(max_value, step):
    sequence = []
    a = 0
    while True:
        # 根据给定的规则生成元素
        if a == 0:
            value = 0
        else:
            value = step * a

        # 如果生成的元素大于最大值，跳出循环
        if value > max_value:
            break

        # 将生成的元素添加到序列中
        sequence.append(value)

        # 增加索引
        a += 1

    return sequence


def find_multiple(number, num):
    # 找到比给定数大的最小整数
    ceil_number = math.ceil(number)

    # 如果这个数是5的倍数，直接返回
    if ceil_number % num == 0:
        return ceil_number
    else:
        # 找到最近的大于这个数的5的倍数
        return ceil_number + num - ceil_number % num


def plt_demand(total_power, path, max_time):
    total_power = total_power[:, :max_time]

    # 提取最后一列
    last_column = total_power[:, -1].reshape(-1, 1)
    # 将最后一列与原数组进行水平堆叠
    total_power = np.hstack((total_power, last_column))

    y = total_power.shape[1]

    # 绘制
    rcParams['font.family'] = 'Times New Roman'
    # 创建一个用于堆叠图的数组
    stacked_data = np.cumsum(total_power, axis=0)
    # 创建x轴的数据
    x = np.arange(y)
    # 创建堆叠面积图
    # fig, ax = plt.subplots(figsize=(3 / 2.54, 4 / 2.54))
    fig, ax = plt.subplots(figsize=(6, 2))
    # fig, ax = plt.subplots()
    # color1 = [244 / 255, 111 / 255, 68 / 255]
    # color2 = [127 / 255, 203 / 255, 164 / 255]
    # color3 = [75 / 255, 101 / 255, 175 / 255]
    color = colour_gen()
    ax.fill_between(x, 0, stacked_data[0, :], color=color[0])
    for i in range(1, total_power.shape[0]):
        ax.fill_between(x, stacked_data[i - 1, :], stacked_data[i, :], color=color[i])
    ax.plot(x, stacked_data[-1, :], color='red', linewidth=2)
    # 设置x轴的范围，使其起始位置为0
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim([0, y])
    xticks = generate_sequence(y, 4)
    xlbls = np.arange(y // 4 + 1)
    # 设置x轴刻度的位置
    ax.set_xticks(xticks)
    # 设置x轴刻度的标签
    ax.set_xticklabels(xlbls)
    # 设置y轴刻度，以10为单位
    max_y_value = stacked_data.max()
    yticks = np.arange(0, max_y_value + 10, 10)
    ax.set_yticks(yticks)
    # 隐藏上方和右侧的边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 只显示左侧和底部的刻度
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    # 设置刻度线朝内
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')

    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
