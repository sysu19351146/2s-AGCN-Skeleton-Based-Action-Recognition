#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"


# class GradualWarmupScheduler(_LRScheduler):
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(_):
    #随机种子设置
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # 初始参数设置
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/kinetics-skeleton/test_joint.yaml',
        # default='./config/kinetics-skeleton/train_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=0,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        
        
        self.arg = arg   #初始参数
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        
        self.global_step = 0
        self.load_model()   #载入模型
        self.load_optimizer()  #载入优化器
        self.load_data()      #载入数据
        self.lr = self.arg.base_lr
        self.best_acc = 0   

    def load_data(self):
        """
        载入数据
        """
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                # num_workers=self.arg.num_worker,
                num_workers=0,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            # num_workers=self.arg.num_worker,
            num_workers=0,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        """
        导入模型
        """
        
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        """
        载入优化器
        """
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        """
        保存参数设置成yaml文件
        """
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        """
        更新学习率
        """
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        """
        打印时间和各种操作，并选择写入那些文件里
        """
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        """
        更新当前时间
        """
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        """
        计算时间差
        """
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        """
        训练过程
        """
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))           #训练轮数
        loader = self.data_loader['train']                               #导入数据
        self.adjust_learning_rate(epoch)                                 #自适应学习率
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()                                               #记录单腔时间
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)    #记录各种操作的时间
        process = tqdm(loader)                                           #数据集和进度条
        
        #训练时是否到了不需要梯度的轮次
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                
        
        for batch_idx, (data, label, index) in enumerate(process):
            self.global_step += 1
            
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)   #数据
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)  #标签
            timer['dataloader'] += self.split_time()                      #载入数据所用的时间

            # forward
            output = self.model(data)

            #只取分类结果
            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1   

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()            #载入数据所用的时间反向传播所用时间

            value, predict_label = torch.max(output.data, 1)   
            acc = torch.mean((predict_label == label.data).float())

            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_l1', l1, self.global_step)


            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()    #记录数据所用时间
            

        # 记录时间消耗和loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))
        
        #保存模型参数
        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1],
                                    v.cpu()] for k, v in state_dict.items()])

            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        """
        测试部分代码
        """
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
            
        #开始测试
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))   #轮数
        
        for ln in loader_name:      
            loss_value = []                         #loss大小
            score_frag = []                         #样本在不同分类的得分
            right_num_total = 0                     #总共的正确个数
            total_num = 0                           #样本总数量
            loss_total = 0                          #总loss
            step = 0                                #轮数
            process = tqdm(self.data_loader[ln])                               #数据集和进度条
            for batch_idx, (data, label, index) in enumerate(process):
                with torch.no_grad():
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)                                         #数据
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)                                         #标签
                    output = self.model(data)                                  #结果
                    
                    #只取分类结果
                    if isinstance(output, tuple):            
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())
                    _, predict_label = torch.max(output.data, 1)               #分类预测
                    step += 1

                #记录分类正确和分类错误的索引到txt文件中
                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
                            
                        
            score = np.concatenate(score_frag)    
            loss = np.mean(loss_value)                                         #平均loss
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)            #正确率
            if accuracy > self.best_acc:                                       #记录最高的正确率
                self.best_acc = accuracy
            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)  #打印准确率
            
            
            #如果在训练过程中测试则需记录到文本中
            if self.arg.phase == 'train':                                      
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('loss_l1', l1, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            
            #将样本在不同类别的评分保存，便于进行ensemble操作
            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)
                    
                    
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))

            #打印准确率
            for k in [1]:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))   

            

    def start(self):
        """
        开始训练或测试
        """

        #导入权重
        load_weight=True
        if load_weight:
            self.model.load_state_dict(torch.load(self.arg.weights))
        #训练部分
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))   #初始参数打印
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size  #训练轮数
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-3:
                    break

                self.train(epoch, save_model=True)   #训练
                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

            print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)  #打印准确率以及模型名称


        #测试部分
        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '_wrong.txt'
                rf = self.arg.model_saved_name + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))   #所使用的模型
            self.print_log('Weights: {}.'.format(self.arg.weights))  #权重文件的位置
            self.model.load_state_dict(torch.load(self.arg.weights)) #载入权重
            
            #测试
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    """
    将字符转化为bool型
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):       
    """
    载入模型
    """
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()    
    
    #获得参数值
    p = parser.parse_args()    
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)
    arg = parser.parse_args()
    init_seed(0)   #设置种子
    processor = Processor(arg)  
    processor.start()           #开始训练或测试
