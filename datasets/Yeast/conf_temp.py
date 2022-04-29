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
            'Att98', 'Att99', 'Att100', 'Att101', 'Att102', 'Att103']
lb_list = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10',
           'Class11', 'Class12', 'Class13', 'Class14']
tr_lb_list = [
    'Class1',
    'Class3',
    'Class6',
    'Class7',
    'Class9',
    'Class10',
    'Class12',

]
te_lb_list = [
    'Class2',
    'Class4',
    'Class5',
    'Class8',
    'Class11',
    'Class13',
    'Class14',
]
remap_fea_list = [
    62, 83, 21, 87, 95, 59, 94, 29, 37, 102, 22, 2, 76, 97, 57, 88, 65, 96, 24, 53, 80, 64, 8, 79, 60, 36, 17, 61, 47,
    1, 41, 98, 90, 30, 50, 86, 101, 100, 99, 93, 92, 91, 89, 85, 84, 82, 81, 78, 77, 75, 74, 73, 72, 71, 70, 69, 68, 67,
    66, 63, 58, 56, 55, 54, 52, 51, 49, 48, 46, 45, 44, 43, 42, 40, 39, 38, 35, 34, 33, 32, 31, 28, 27, 26, 25, 23, 20,
    19, 18, 16, 15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 4, 3, 0
]
reremap_fea_list = gen_reremap_fea_list(remap_fea_list)

DATASET_CONF = {
    'dataset_name': os.path.abspath(__file__).split('/')[-2],
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
            'Class1': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class2': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class3': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'Class4': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'Class5': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class6': {
                'C': 3,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class7': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class8': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class9': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'Class10': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'Class11': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'Class12': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'Class13': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'Class14': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'auto',
            },

        },
    }
}
