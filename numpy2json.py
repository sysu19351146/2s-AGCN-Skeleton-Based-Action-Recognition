import json
from json import JSONEncoder
import numpy as np
import os

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# Serialization
dir = 'raw_npy_data/'
datasets = ['val', 'train']
folders = ['000', '001', '002', '003', '004']


def numpy2json (dataset, folder, sample_name):
        path = dir + dataset + '/' + folder + '/' +sample_name
        data = np.load(path)
        # print(data)
        N, C, T, V, M = data.shape
        # print(N, C, T, V, M)
        # d[0]即为每个sample的output
        # print(data[0])
        # data[:, 1, :, :, :] = 0

        # 第二个人的动作全为0是都会影响？？？
        if data[:, :, :, :, 1].all() == 0:
                M = 1

        data = data[0]
        sample_data = []
        for frame_index in range(0, T):
                skeleton = []
                for person in range(0, M):
                        data_joints = data[:, frame_index, :, person]
                        pose = np.zeros(2 * V)
                        score = np.ones(V)
                        for joint in range(0, V):
                                pose[joint * 2] = data_joints[0][joint]
                                pose[joint * 2 + 1] = data_joints[2][joint]

                        skeleton_single = {"pose": pose, "score": score}
                        skeleton.append(skeleton_single)
                        # print(skeleton_single)
                        # print(type(skeleton_single))
                # print("*************")
                # print(skeleton)
                frame = {"frame_index": frame_index + 1, "skeleton": skeleton}
                sample_data.append(frame)
        data_array = {"data": sample_data, "label": folder, "label_index": folders.index(folder)}
        encodedNumpyData = json.dumps(data_array, cls=NumpyArrayEncoder)  # use dump() to write array into file
        fileObject = open('data/kinetics_raw/kinetics_' + dataset
                          + '_new/' + sample_name + '.json', 'w')
        fileObject.write(encodedNumpyData)
        fileObject.close()
        return encodedNumpyData


for dataset in datasets:
        sample_label = {}
        for folder in folders:
                sample_name = os.listdir(dir + dataset + '/' + folder)
                sample_id = [name.split('.')[0] for name in sample_name]
                for name in sample_name:
                        data_json = numpy2json(dataset, folder, name)
                        sample_info = {"has_skeleton": True, "label": folder, "label_index": folders.index(folder)}
                        sample_label[name.split('.')[0]] = sample_info
        encodedNumpyData = json.dumps(sample_label, cls=NumpyArrayEncoder)  # use dump() to write array into file
        fileObject = open('data/kinetics_raw/' + 'kinetics_'
                          + dataset + '_label' + '.json', 'w')
        fileObject.write(encodedNumpyData)
        fileObject.close()
        print(sample_label)
