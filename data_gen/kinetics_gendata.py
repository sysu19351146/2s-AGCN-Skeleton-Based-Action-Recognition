import argparse
import os
import numpy as np
import json
from torch.utils.data import Dataset
import pickle
from tqdm import tqdm

num_joint = 17
max_frame = 500
num_person_out = 2
num_person_in = 5


class Feeder_kinetics(Dataset):
    """ Feeder for skeleton-based action recognition in kinetics-skeleton dataset
    # Joint index:
    # {0,  "Nose"}
    # {1,  "Neck"},
    # {2,  "RShoulder"},
    # {3,  "RElbow"},
    # {4,  "RWrist"},
    # {5,  "LShoulder"},
    # {6,  "LElbow"},
    # {7,  "LWrist"},
    # {8,  "RHip"},
    # {9,  "RKnee"},
    # {10, "RAnkle"},
    # {11, "LHip"},
    # {12, "LKnee"},
    # {13, "LAnkle"},
    # {14, "REye"},
    # {15, "LEye"},
    # {16, "REar"},
    # {17, "LEar"},
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
        num_person_in: The number of people the feeder can observe in the input sequence
        num_person_out: The number of people the feeder in the output sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 ignore_empty_sample=True,
                 window_size=-1,
                 num_person_in=5,
                 num_person_out=2):
        self.data_path = data_path
        self.label_path = label_path
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self):
        # load file list
        self.sample_name = os.listdir(self.data_path)

        # load label
        label_path = self.label_path
        with open(label_path) as f:
            label_info = json.load(f)

        # 根据json文件名直接给每个json样本命名id
        sample_id = [name.split('.')[0] for name in self.sample_name]
        # 索引得到label_index
        self.label = np.array([label_info[id]['label_index'] for id in sample_id])
        # 索引得到has_skeleton，确认是否有对应的数据
        has_skeleton = np.array([label_info[id]['has_skeleton'] for id in sample_id])
        print(type(label_info[sample_id[1]]['has_skeleton']))
        # ignore the samples which does not has skeleton sequence
        # 删除没有对应数据的标签元素
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        # output data shape (N, C, T, V, M)
        self.N = len(self.sample_name)  # sample
        self.C = 3  # channel
        self.T = max_frame  # frame
        self.V = num_joint  # joint
        self.M = self.num_person_out  # person

    def __len__(self):
        return len(self.sample_name)

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # output shape (C, T, V, M)
        # get data
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, 'r') as f:
            video_info = json.load(f)

        # fill data_numpy
        # output shape
        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        # 循环每个data
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            # frame_index和skeleton同级，可以直接循环每个skeleton
            # m是skeleton里面的每次循环的下标，e.g. 有两组pose & score时，表示有两个人
            # m的取值和下标开始有关，默认0开始
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                # 只读取最大人数的数据，之后不再读取
                if m >= self.num_person_in:
                    break
                pose = skeleton_info['pose']
                score = skeleton_info['score']
                # 第m个人的第frame_index帧在第一通道上的所有关节数据 = pose[0::2]
                # pose[0::2]表示从0开始每隔一位取得的列表
                # 同理 pose[1::2]表示从1开始每隔一位取得的列表
                # 左端 1*1*18*1 , 右端 1*18
                data_numpy[0, frame_index, :, m] = pose[0::2]
                print(frame_index)
                # 第m个人的第frame_index帧在第二通道上的所有关节数据 = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                # # 第m个人的第frame_index帧在第三通道上的所有关节数据 = score
                data_numpy[2, frame_index, :, m] = score

        # centralization
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        # 为什么要对二、三维取反？？？
        data_numpy[1:2] = -data_numpy[1:2]
        # 对于score为0的关节点，将前两个维度的数据都置为0
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        # get & check label index
        #根据每个json文件末尾的label_index来确认文件对应的正确性
        label = video_info['label_index']
        assert (self.label[index] == label)

        # sort by score
        # 第三维度的矩阵去除关节维度相加并排序
        # in another word 对与每个关节的score，所有frame_index维度上进行相加，再按照score总值来对person个体排序，参考line160注释
        # 随后返回排序下标，调换顺序
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, s in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, s].transpose((1, 2,
                                                                       0))
        # 根据num_person_out再次切片，选择出score较高的person
        data_numpy = data_numpy[:, :, :, 0:self.num_person_out]

        return data_numpy, label


def gendata(data_path, label_path,
            data_out_path, label_out_path,
            num_person_in=num_person_in,  # observe the first 5 persons
            num_person_out=num_person_out,  # then choose 2 persons with the highest score
            max_frame=max_frame):
    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = np.zeros((len(sample_name), 3, max_frame, num_joint, num_person_out), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data, label = feeder[i]
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    np.save(data_out_path, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/kinetics_raw')
    parser.add_argument(
        '--out_folder', default='../data/kinetics')
    arg = parser.parse_args()

    part = ['val', 'train']
    for p in part:
        print('kinetics ', p)
        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        data_path = '{}/kinetics_{}'.format(arg.data_path, p + '_new')  # 训练集或者验证集的文件夹名
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)  # 单个json文件的名字
        data_out_path = '{}/{}_data_joint.npy'.format(arg.out_folder, p)  # 输出的data的npy文件的名字
        label_out_path = '{}/{}_label.pkl'.format(arg.out_folder, p)  # 输出的label的pkl文件的名字

        gendata(data_path, label_path, data_out_path, label_out_path)  # 读取json文件生成关节数据
