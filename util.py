import os

os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from collections import deque
import copy
import importlib
import math
import numpy as np
import pandas as pd
import random
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def global_dict_init():
    global _global_dict
    _global_dict = {}


def set_global_dict_value(key, value):
    _global_dict[key] = value


def get_global_dict_value(key):
    return _global_dict[key]


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
        if getattr(args, attr) is not None:
            keys = attr.split('__')
            if check_dict_key(conf, keys):
                set_dict_value(conf, keys, getattr(args, attr))
    return conf
