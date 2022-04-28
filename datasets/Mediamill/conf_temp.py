from util import *


def gen_reremap_fea_list(remap_fea_list):
    reremap_fea_list = []
    for i in range(len(remap_fea_list)):
        for j in range(len(remap_fea_list)):
            if remap_fea_list[j] == i:
                reremap_fea_list.append(j)
                break
    return reremap_fea_list


fea_list = ['Att1', 'Att2', 'Att3', 'Att4', 'Att5', 'Att6', 'Att7', 'Att8', 'Att9', 'Att10', 'Att11', 'Att12', 'Att13',
            'Att14', 'Att15', 'Att16', 'Att17', 'Att18', 'Att19', 'Att20', 'Att21', 'Att22', 'Att23', 'Att24', 'Att25',
            'Att26', 'Att27', 'Att28', 'Att29', 'Att30', 'Att31', 'Att32', 'Att33', 'Att34', 'Att35', 'Att36', 'Att37',
            'Att38', 'Att39', 'Att40', 'Att41', 'Att42', 'Att43', 'Att44', 'Att45', 'Att46', 'Att47', 'Att48', 'Att49',
            'Att50', 'Att51', 'Att52', 'Att53', 'Att54', 'Att55', 'Att56', 'Att57', 'Att58', 'Att59', 'Att60', 'Att61',
            'Att62', 'Att63', 'Att64', 'Att65', 'Att66', 'Att67', 'Att68', 'Att69', 'Att70', 'Att71', 'Att72', 'Att73',
            'Att74', 'Att75', 'Att76', 'Att77', 'Att78', 'Att79', 'Att80', 'Att81', 'Att82', 'Att83', 'Att84', 'Att85',
            'Att86', 'Att87', 'Att88', 'Att89', 'Att90', 'Att91', 'Att92', 'Att93', 'Att94', 'Att95', 'Att96', 'Att97',
            'Att98', 'Att99', 'Att100', 'Att101', 'Att102', 'Att103', 'Att104', 'Att105', 'Att106', 'Att107', 'Att108',
            'Att109', 'Att110', 'Att111', 'Att112', 'Att113', 'Att114', 'Att115', 'Att116', 'Att117', 'Att118',
            'Att119', 'Att120']

lb_list = ['Class12', 'Class25', 'Class32', 'Class34', 'Class44', 'Class52', 'Class66', 'Class67',
           'Class68', 'Class76', 'Class79', 'Class85', 'Class94', 'Class96', 'Class97', 'Class98']

tr_lb_list = [
    'Class12', 'Class32', 'Class44', 'Class68', 'Class76', 'Class94', 'Class96',
]
te_lb_list = [
    'Class25', 'Class34', 'Class52', 'Class66', 'Class67', 'Class79', 'Class85', 'Class97', 'Class98'
]
remap_fea_list = [
    102, 93, 97, 71, 63, 18, 28, 79, 13, 74, 64, 19, 34, 41, 87, 42, 72, 86, 25, 119, 118, 117, 116, 115,
    114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 101, 100, 99, 98, 96, 95, 94, 92, 91, 90,
    89, 88, 85, 84, 83, 82, 81, 80, 78, 77, 76, 75, 73, 70, 69, 68, 67, 66, 65, 62, 61, 60, 59, 58, 57,
    56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 40, 39, 38, 37, 36, 35, 33, 32, 31, 30, 29,
    27, 26, 24, 23, 22, 21, 20, 17, 16, 15, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
]

reremap_fea_list = gen_reremap_fea_list(remap_fea_list)

DATASET_CONF = {
    'dataset_name': __file__.split('/')[-2],
    'dataset_path': '',
    'metric_type_list': ['auc', 'f1_score'],
    'fea_list': fea_list,
    'fea_num': len(fea_list),
    'lb_list': lb_list,
    'tr_lb_list': tr_lb_list,
    'te_lb_list': te_lb_list,
    'use_remap': True,
    'remap_fea_list': remap_fea_list,
    'reremap_fea_list': reremap_fea_list,
    'PARAM_PRE_TRAIN': {
        'SVM': {
            'Class12': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class25': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class32': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class34': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class44': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class52': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class66': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class67': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class68': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class76': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class79': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class85': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class94': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class96': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class97': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'Class98': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
        }

    }
}
