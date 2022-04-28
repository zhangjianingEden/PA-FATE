from util import *


def gen_reremap_fea_list(remap_fea_list):
    reremap_fea_list = []
    for i in range(len(remap_fea_list)):
        for j in range(len(remap_fea_list)):
            if remap_fea_list[j] == i:
                reremap_fea_list.append(j)
                break
    return reremap_fea_list


fea_list = ['std_temp', 'std_pH', 'conduct', 'o2', 'o2sat', 'co2', 'hardness', 'no2', 'no3', 'nh4', 'po4', 'cl', 'sio2',
            'kmno4', 'k2cr2o7', 'bod']
lb_list = ['25400', '29600', '30400', '33400', '17300', '19400', '34500', '38100', '49700', '50390', '55800', '57500',
           '59300', '37880']

tr_lb_list = [
    '25400', '29600', '30400', '33400', '17300', '19400', '34500'
]
te_lb_list = [
    '38100', '49700', '50390', '55800', '57500', '59300', '37880'
]
remap_fea_list = [0, 3, 1, 6, 12, 11, 2, 4, 5, 15, 7, 8, 13, 10, 14, 9]
reremap_fea_list = gen_reremap_fea_list(remap_fea_list)

DATASET_CONF = {
    'dataset_name': __file__.split('/')[-2],
    'dataset_path': '',
    'metric_type_list': ['auc', 'f1_score'],
    'fea_list': fea_list,
    'fea_num': len(fea_list),
    'lb_list': tr_lb_list + te_lb_list,
    'tr_lb_list': tr_lb_list,
    'te_lb_list': te_lb_list,
    'use_remap': True,
    'remap_fea_list': remap_fea_list,
    'reremap_fea_list': reremap_fea_list,
    'PARAM_PRE_TRAIN': {
        'SVM': {
            '25400': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '29600': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '30400': {
                'C': 3,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '33400': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '17300': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '19400': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '34500': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '38100': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '49700': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '50390': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '55800': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '57500': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '59300': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '37880': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
        }

    }
}
