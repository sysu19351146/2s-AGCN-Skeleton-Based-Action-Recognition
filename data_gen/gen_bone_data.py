import os
import numpy as np
from numpy.lib.format import open_memmap

dir = 'E:/大二下/人工智能实验/action_recognition/'

paris = {
    'ntu/xview': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),
    'ntu/xsub': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    ),

    # 'kinetics': ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
    #              (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
    'kinetics': ((10, 8), (8, 6), (9, 7), (7, 5), # arms
    (15, 13), (13, 11), (16, 14), (14, 12), # legs
    (11, 5), (12, 6), (11, 12), (5, 6), # torso
    (5, 0), (6, 0), (1, 0), (2, 0), (3, 1), (4, 2))
}

sets = {
    'train', 'val'
}

# 'ntu/xview', 'ntu/xsub',  'kinetics'
datasets = {
    # 'ntu/xview', 'ntu/xsub',
    'kinetics'
}
# bone
from tqdm import tqdm

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data = np.load('../data/{}/{}_data_joint.npy'.format(dataset, set))
        print('load successfully')
        #只能绝对路径？？？
        N, C, T, V, M = data.shape
        fp_sp = open_memmap(
            # 'E:/大二下/人工智能实验/action_recognition/2s-AGCN-master/data/{}/{}_data_joint.npy'.format(dataset, set),
            '../data/{}/{}_data_bone.npy'.format(dataset, set),
            dtype='float32',
            mode='w+',
            shape=(N, 3, T, V, M))

        fp_sp[:, :C, :, :, :] = data
        for v1, v2 in tqdm(paris[dataset]):
            if dataset != 'kinetics':
                v1 -= 1
                v2 -= 1
            # 计算骨骼向量
            fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

        # # 这行是自己写的，找不到data_bone.npy的保存
        # np.save('../data/{}/{}_data_bone.npy'.format(dataset, set), fp_sp)
