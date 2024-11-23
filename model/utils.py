import os
import time
import pprint
import torch
import argparse
import numpy as np
from model import file_dir

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

# function to calculate class accuracies
# some way of getting class names in each iteration 

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
#     print('pred ', pred)
#     print('label ', label)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def postprocess_args(args):
    if args.dataset == 'MiniImageNet':
        args.mem_len = 64
        args.top5s = file_dir.mini_top5
    elif args.dataset == 'TieredImageNet_og':
        args.mem_len = 351
        args.top5s = file_dir.tiered_top5
    elif args.dataset == 'CUB':
        args.mem_len = 100
        args.top5s = file_dir.cub_top5
    else:
        raise KeyError
    if args.backbone_class == 'ConvNet':
        args.dim_model = 64
        if args.dataset == 'MiniImageNet':
            args.init_weights = file_dir.mini_pre_conv4
            args.memory = file_dir.mini_mem_conv4
        elif args.dataset == 'CUB':
            args.init_weights = file_dir.cub_pre_conv4
            args.memory = file_dir.cub_mem_conv4
        elif args.dataset == 'TieredImageNet_og':
            args.init_weights = file_dir.tiered_pre_conv4
            args.memory = file_dir.tiered_mem_conv4
        
    elif args.backbone_class == 'Res12':
        args.dim_model = 640
        if args.dataset == 'MiniImageNet':
            args.init_weights = file_dir.mini_pre_res12
            args.memory = file_dir.mini_mem_res12
        elif args.dataset == 'TieredImageNet_og':
            args.init_weights = file_dir.tiered_pre_res12
            args.memory = file_dir.tiered_mem_res12
        elif args.dataset == 'CUB':
            args.init_weights = file_dir.cub_pre_res12
            args.memory = file_dir.cub_mem_res12
        
    if args.use_simclr:
        args.return_simclr = 2
    extra_path = str(args.dataset) + '_' + str(args.backbone_class) + '_' + str(args.shot) + 'shot' + '/'
    print(extra_path)
    if os.path.exists('/output'):
        args.save_path = '/output/' + '_{}'.format(str(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()+28800))))
    else:
        args.save_path = 'checkpoints/' + extra_path + str(time.strftime('%Y_%m_%d_%H%M%S', time.localtime(time.time()+28800)))
    os.makedirs(args.save_path)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # 训练参数
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])
    parser.add_argument('--metric', type=str, default='eu', choices=['dot', 'proj', 'cos', 'eu'])
    parser.add_argument('--spatial_dim', type=int, default=5)
    parser.add_argument('--pooling', type=str, default=None, choices=['max', 'mean'])
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--step_size', type=str, default='40')
    
    # MemoryAttention
    parser.add_argument('--mem_init', type=str, default='pre_train', choices=['pre_train', 'random'])
    parser.add_argument('--mem_grad', action='store_true', default=False) # memory是否学习
    parser.add_argument('--mem_share', type=int, default=0)
    parser.add_argument('--mem_2d_norm', action='store_true', default=False) # 是否在640维度归一化
    parser.add_argument('--mem_sample', type=float, default=None) # 是否将memory看作分布，暂时弃用
    # 如果spatial_dim是1，需要pooling
    parser.add_argument('--mem_init_pooling', type=str, default=None, choices=['max', 'mean'])
    parser.add_argument('--temperature', type=float, default=0.1) # 缩放小样本logits
    parser.add_argument('--n_heads', type=int, default=1) # self-attention heads
    # simclr
    parser.add_argument('--use_simclr', action='store_true', default=False)
    parser.add_argument('--return_simclr', type=int, default=None) # number of views in simclr
    parser.add_argument('--balance', type=float, default=0.1)
    parser.add_argument('--temperature2', type=float, default=0.1)
    # BatchNorm
    parser.add_argument('--bn2d', action='store_true', default=False)

    # infoNCE
    parser.add_argument('--use_infoNCE', action='store_true', default=False)
    parser.add_argument('--pool_before_lstm', action='store_true', default=False)
    parser.add_argument('--tasker', type=str, default='blstm', choices=['blstm', 'attention']) # 目前只有blstm
    parser.add_argument('--task_feat', type=str, default='output_max', choices=['hn_mean', 'output_max', 'cls_token'])
    parser.add_argument('--T', type=float, default=0.07) # temperature for infoNCE loss
    parser.add_argument('--K', type=int, default=256) # （负样本）队列长度
    parser.add_argument('--D', type=int, default=256)
    parser.add_argument('--M', type=float, default=0.99) # 没用，目前并没有动量编码器

    # blstm分类
    parser.add_argument('--use_blstm_meta', action='store_true', default=False)
    parser.add_argument('--blstm_metric', type=str, default='dot', choices=['dot', 'proj', 'cos', 'eu'])
    parser.add_argument('--temperature3', type=float, default=0.1) # 缩放logits
    parser.add_argument('--logits_mix', type=float, default=0.05) # 没用，目前会遍历这个参数

    # 全是这个设置
    parser.add_argument('--z_norm', type=str, default='before_tx', choices=['before_tx', 'before_euclidian', 'both', None])
    
    # 训练，测试，记录相关
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--minitest_interval', type=int, default=1)
    parser.add_argument('--test100k_interval', type=int, default=20)
    
    # 模型
    parser.add_argument('--model_class', type=str, default='SETC', 
                        choices=['MatchNet', 'ProtoNet', 'FEAT','SETC'])
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet_og', 'CUB'])
    # 这里的resize分两步，先128再84，不知道为什么——128是放进cache的，要不然就直接84
    parser.add_argument('--im_size', type=int, default=128)
    parser.add_argument('--use_im_cache', action='store_true', default=False)
    
    # 学习率
    parser.add_argument('--lr_mul', type=float, default=10)    
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine', 'onecycle', 'cyclic'])
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--momentum', type=float, default=0.9) #SGD momentum
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # 训练设置
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mixed_precision', type=str, default=None)

    return parser
