from util import *


def gen_reremap_fea_list(remap_fea_list):
    reremap_fea_list = []
    for i in range(len(remap_fea_list)):
        for j in range(len(remap_fea_list)):
            if remap_fea_list[j] == i:
                reremap_fea_list.append(j)
                break
    return reremap_fea_list


fea_list = ['Mean_Acc1298_Mean_Mem40_Centroid', 'Mean_Acc1298_Mean_Mem40_Rolloff', 'Mean_Acc1298_Mean_Mem40_Flux',
            'Mean_Acc1298_Mean_Mem40_MFCC_0', 'Mean_Acc1298_Mean_Mem40_MFCC_1', 'Mean_Acc1298_Mean_Mem40_MFCC_2',
            'Mean_Acc1298_Mean_Mem40_MFCC_3', 'Mean_Acc1298_Mean_Mem40_MFCC_4', 'Mean_Acc1298_Mean_Mem40_MFCC_5',
            'Mean_Acc1298_Mean_Mem40_MFCC_6', 'Mean_Acc1298_Mean_Mem40_MFCC_7', 'Mean_Acc1298_Mean_Mem40_MFCC_8',
            'Mean_Acc1298_Mean_Mem40_MFCC_9', 'Mean_Acc1298_Mean_Mem40_MFCC_10', 'Mean_Acc1298_Mean_Mem40_MFCC_11',
            'Mean_Acc1298_Mean_Mem40_MFCC_12', 'Mean_Acc1298_Std_Mem40_Centroid', 'Mean_Acc1298_Std_Mem40_Rolloff',
            'Mean_Acc1298_Std_Mem40_Flux', 'Mean_Acc1298_Std_Mem40_MFCC_0', 'Mean_Acc1298_Std_Mem40_MFCC_1',
            'Mean_Acc1298_Std_Mem40_MFCC_2', 'Mean_Acc1298_Std_Mem40_MFCC_3', 'Mean_Acc1298_Std_Mem40_MFCC_4',
            'Mean_Acc1298_Std_Mem40_MFCC_5', 'Mean_Acc1298_Std_Mem40_MFCC_6', 'Mean_Acc1298_Std_Mem40_MFCC_7',
            'Mean_Acc1298_Std_Mem40_MFCC_8', 'Mean_Acc1298_Std_Mem40_MFCC_9', 'Mean_Acc1298_Std_Mem40_MFCC_10',
            'Mean_Acc1298_Std_Mem40_MFCC_11', 'Mean_Acc1298_Std_Mem40_MFCC_12', 'Std_Acc1298_Mean_Mem40_Centroid',
            'Std_Acc1298_Mean_Mem40_Rolloff', 'Std_Acc1298_Mean_Mem40_Flux', 'Std_Acc1298_Mean_Mem40_MFCC_0',
            'Std_Acc1298_Mean_Mem40_MFCC_1', 'Std_Acc1298_Mean_Mem40_MFCC_2', 'Std_Acc1298_Mean_Mem40_MFCC_3',
            'Std_Acc1298_Mean_Mem40_MFCC_4', 'Std_Acc1298_Mean_Mem40_MFCC_5', 'Std_Acc1298_Mean_Mem40_MFCC_6',
            'Std_Acc1298_Mean_Mem40_MFCC_7', 'Std_Acc1298_Mean_Mem40_MFCC_8', 'Std_Acc1298_Mean_Mem40_MFCC_9',
            'Std_Acc1298_Mean_Mem40_MFCC_10', 'Std_Acc1298_Mean_Mem40_MFCC_11', 'Std_Acc1298_Mean_Mem40_MFCC_12',
            'Std_Acc1298_Std_Mem40_Centroid', 'Std_Acc1298_Std_Mem40_Rolloff', 'Std_Acc1298_Std_Mem40_Flux',
            'Std_Acc1298_Std_Mem40_MFCC_0', 'Std_Acc1298_Std_Mem40_MFCC_1', 'Std_Acc1298_Std_Mem40_MFCC_2',
            'Std_Acc1298_Std_Mem40_MFCC_3', 'Std_Acc1298_Std_Mem40_MFCC_4', 'Std_Acc1298_Std_Mem40_MFCC_5',
            'Std_Acc1298_Std_Mem40_MFCC_6', 'Std_Acc1298_Std_Mem40_MFCC_7', 'Std_Acc1298_Std_Mem40_MFCC_8',
            'Std_Acc1298_Std_Mem40_MFCC_9', 'Std_Acc1298_Std_Mem40_MFCC_10', 'Std_Acc1298_Std_Mem40_MFCC_11',
            'Std_Acc1298_Std_Mem40_MFCC_12', 'BH_LowPeakAmp', 'BH_LowPeakBPM', 'BH_HighPeakAmp', 'BH_HighPeakBPM',
            'BH_HighLowRatio', 'BHSUM1', 'BHSUM2', 'BHSUM3']

lb_list = ['amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still', 'sad-lonely', 'angry-aggresive']

tr_lb_list = [
    'amazed-suprised', 'happy-pleased', 'relaxing-calm', 'quiet-still',
]
te_lb_list = [
    'sad-lonely', 'angry-aggresive',
]
remap_fea_list = [4, 3, 0, 39, 57, 17, 51, 64, 56, 71, 7, 44, 36, 58, 70, 69, 68, 67, 66, 65, 63, 62, 61, 60, 59, 55,
                  54, 53, 52, 50, 49, 48, 47, 46, 45, 43, 42, 41, 40, 38, 37, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26,
                  25, 24, 23, 22, 21, 20, 19, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 6, 5, 2, 1]

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
            'amazed-suprised': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            'happy-pleased': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            'relaxing-calm': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            'quiet-still': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'sad-lonely': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            'angry-aggresive': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
        }

    }
}
