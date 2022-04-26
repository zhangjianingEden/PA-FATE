import os

os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from collections import deque
import copy
import datetime
from datetime import datetime
import glob
import imageio
import importlib
import joblib
import math
import matplotlib as mpl
import matplotlib.path as mplpath
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
import os.path
import pandas as pd
import paramiko
from PIL import Image
import random
import re
from scipy.stats import pearsonr
import shutil
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import socket
import stat
from statistics import mean
import sys
import time
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import traceback

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

project_name = __file__.split('/')[-2]


def global_dict_init():  # 初始化
    global _global_dict
    _global_dict = {}


def set_global_dict_value(key, value):
    # 定义一个全局变量
    _global_dict[key] = value


def get_global_dict_value(key):
    # 获得一个全局变量，不存在则提示读取对应变量失败
    try:
        return _global_dict[key]
    except:
        print(datetime.now(), '读取' + key + '失败\r\n')



def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def gen_p_list(tr_lb_sth_list):
    if tr_lb_sth_list[0] == 'Nan':
        tr_lb_num = len(tr_lb_sth_list)
        p_list = [1 / tr_lb_num] * tr_lb_num
    else:
        tr_lb_sth_array = np.array(tr_lb_sth_list, dtype=np.float64)
        min_sth = np.min(tr_lb_sth_array)
        tr_lb_sth_array -= min_sth
        tr_lb_sth_array += 1e-4
        sum_sth = np.sum(tr_lb_sth_array)
        tr_lb_sth_array /= sum_sth
        p_list = list(tr_lb_sth_array)
    return p_list


def set_dict_value(mydict, keys, val):
    mydict_tmp = mydict
    lastkey = keys[-1]
    for key in keys[:-1]:
        mydict_tmp = mydict_tmp[key]
    if val == 'True':
        mydict_tmp[lastkey] = True
    elif val == 'False':
        mydict_tmp[lastkey] = False
    else:
        mydict_tmp[lastkey] = type(mydict_tmp[lastkey])(val)


def check_dict_key(mydict, keys):
    mydict_tmp = mydict
    flag = True
    for key in keys:
        if not isinstance(mydict_tmp, dict) or key not in mydict_tmp:
            flag = False
            break
        else:
            mydict_tmp = mydict_tmp[key]
    return flag


def gen_conf(args, conf_temp):
    conf = copy.deepcopy(conf_temp)
    for attr in dir(args):
        if attr == 'param_name':
            param_list = getattr(args, attr).split('___')
            for param in param_list:
                if param == 'param_name':
                    if check_dict_key(conf, [param]):
                        set_dict_value(conf, [param], getattr(args, attr))
                    continue
                keys = param.split('__')[:-1]
                val = param.split('__')[-1]
                if check_dict_key(conf, keys):
                    set_dict_value(conf, keys, val)
        else:
            keys = attr.split('__')
            if check_dict_key(conf, keys):
                set_dict_value(conf, keys, getattr(args, attr))
    return conf
