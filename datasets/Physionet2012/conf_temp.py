from util import *


def gen_reremap_fea_list(remap_fea_list):
    reremap_fea_list = []
    for i in range(len(remap_fea_list)):
        for j in range(len(remap_fea_list)):
            if remap_fea_list[j] == i:
                reremap_fea_list.append(j)
                break
    return reremap_fea_list


fea_list = ['Age', 'Gender', 'Height', 'ICUType', 'Weight', 'Albumin', 'ALP', 'ALT',
            'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2',
            'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
            'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets',
            'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC']

tr_lb_list = [
    'los_less_5_days',
    'los_less_7_days',
    'los_less_14_days',

    'survival_less_7_days',
    'survival_less_21_days',
    'survival_less_70_days',

    'sofa_less_than_2',
    'sofa_less_than_6',
    'sofa_less_than_10',

    'saps_less_than_10',
    'saps_less_than_16',
    'saps_less_than_20',
]
te_lb_list = [
    'In-hospital_death',

    'los_less_6_days',
    'los_less_8_days',
    'los_less_10_days',
    'los_less_12_days',
    'los_less_16_days',

    'survival_less_5_days',
    'survival_less_10_days',
    'survival_less_15_days',
    'survival_less_30_days',
    'survival_less_35_days',

    'sofa_less_than_3',
    'sofa_less_than_7',
    'sofa_less_than_12',

    'saps_less_than_14',
    'saps_less_than_18',
    'saps_less_than_21',
]
remap_fea_list = [24, 15, 18, 3, 0, 10, 35, 28, 25, 21, 27, 19, 11, 5, 40, 39, 38, 37, 36, 34, 33, 32, 31, 30, 29, 26,
                  23, 22, 20, 17, 16, 14, 13, 12, 9, 8, 7, 6, 4, 2, 1]
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
            'In-hospital_death': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'los_less_5_days': {
                'C': 3,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'los_less_6_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'los_less_7_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'los_less_8_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'los_less_10_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'los_less_12_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'los_less_14_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'los_less_16_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'survival_less_5_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_7_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_10_days': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_15_days': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_21_days': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_30_days': {
                'C': 15,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_35_days': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'survival_less_70_days': {
                'C': 3,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'sofa_less_than_2': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'sofa_less_than_3': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'sofa_less_than_6': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'sofa_less_than_7': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'auto',
            },
            'sofa_less_than_10': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'sofa_less_than_12': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'saps_less_than_10': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'saps_less_than_14': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'saps_less_than_16': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'saps_less_than_18': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'saps_less_than_20': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'scale',
            },
            'saps_less_than_21': {
                'C': 5,
                'kernel': 'rbf',
                'gamma': 'scale',
            },

        },
        'DNN': {
            'In-hospital_death': {
                'lr': 1e-3,
                'eps': 1e-8,
                'betas': (0.9, 0.999),
                'batch_size': 128,
            },
        }
    }
}
