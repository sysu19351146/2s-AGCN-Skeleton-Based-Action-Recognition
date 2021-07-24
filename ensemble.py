import argparse
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', default='kinetics', choices={'kinetics', 'ntu/xsub', 'ntu/xview'},
                    help='the work folder for storing results')
parser.add_argument('--alpha', default=0, help='weighted summation')
arg = parser.parse_args()

dataset = arg.datasets
label = open('data/' + dataset + '/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('work_dir/' + dataset +
          '/agcn_joint_test/epoch1_test_score.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('work_dir/' + dataset +
          '/agcn_test_bone/epoch1_test_score.pkl', 'rb')
r2 = list(pickle.load(r2).items())
right_num = total_num = right_num_5 = 0
acc_5=np.zeros(5)
total_5=np.zeros(5)
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    r = r11 + r22 * arg.alpha
    r = np.argmax(r)
    if r == int(l):
        right_num += 1
        acc_5[r]+=1
    total_num += 1
    total_5[int(l)]+=1
acc = right_num / total_num
acc_5 = acc_5 / total_5
average=np.sum(acc_5)/5
print("数量平均准确率：{:.4f}\n种类平均准确率：{:.4f}".format(acc,average))
